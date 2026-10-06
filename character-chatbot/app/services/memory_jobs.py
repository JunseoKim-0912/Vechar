"""DB-authoritative memory outbox, short fenced claims and retryable workers."""

import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
from uuid import uuid4

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..memory_config import MemoryConfigurationError
from ..models import Character, Conversation, MemoryDeletion, MemoryIngestion, Message, MessageRole
from . import memory_service
from .chat_actions import normalize_assistant_actions
from .training_queue import MEMORY_DELETE_TOPIC, MEMORY_INGEST_TOPIC, publish_memory_operation

logger = logging.getLogger(__name__)
MEMORY_WORK_LEASE_SECONDS = 1200
MEMORY_QUEUE_RETRY_SECONDS = 60
MEMORY_QUEUE_DELIVERY_ATTEMPTS = 30


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def schedule_ingestion(db: Session, *, user_id: str, character_id: str,
                       conversation_id: str, user_message_id: str,
                       assistant_message_id: str) -> MemoryIngestion | None:
    """Call before committing the assistant turn; raw text stays in Messages."""
    if memory_service.PROVIDER_NAME == "noop":
        return None
    existing = db.query(MemoryIngestion).filter(
        MemoryIngestion.provider == memory_service.PROVIDER_NAME,
        MemoryIngestion.assistant_message_id == assistant_message_id,
    ).first()
    if existing:
        return existing
    row = MemoryIngestion(
        user_id=user_id, character_id=character_id, conversation_id=conversation_id,
        user_message_id=user_message_id, assistant_message_id=assistant_message_id,
        provider=memory_service.PROVIDER_NAME,
    )
    db.add(row)
    db.flush()
    return row


def schedule_character_deletion(db: Session, *, user_id: str, character_id: str) -> MemoryDeletion | None:
    """Durable intent in the same transaction as DB character deletion."""
    provider = memory_service.PROVIDER_NAME
    if provider == "noop":
        # Preserve cleanup duty after a temporary production fail-closed switch
        # from MemMachine to noop. Pre-activation noop has no external memory.
        prior = db.query(MemoryIngestion.provider).filter(
            MemoryIngestion.user_id == user_id,
            MemoryIngestion.character_id == character_id,
            MemoryIngestion.provider != "noop",
        ).first()
        if not prior:
            return None
        provider = prior[0]
    row = db.query(MemoryDeletion).filter(
        MemoryDeletion.provider == provider,
        MemoryDeletion.character_id == character_id,
    ).first()
    if row is None:
        row = MemoryDeletion(user_id=user_id, character_id=character_id,
                             provider=provider)
        db.add(row)
        db.flush()
    db.query(MemoryIngestion).filter(
        MemoryIngestion.user_id == user_id,
        MemoryIngestion.character_id == character_id,
        MemoryIngestion.status == "queued",
    ).update({MemoryIngestion.status: "cancelled", MemoryIngestion.completed_at: _now()},
             synchronize_session=False)
    return row


def publish_safe(topic: str, operation_id: str | None) -> None:
    if operation_id is None:
        return
    try:
        publish_memory_operation(topic, operation_id)
    except Exception as exc:
        # The row is authoritative; the scheduled reconciler will republish.
        logger.warning("memory queue publish failed operation_id=%s type=%s error=%s",
                       operation_id, topic, type(exc).__name__)


def _claim(db: Session, model, operation_id: str, max_attempts: int) -> tuple[str, object] | None:
    token = str(uuid4())
    now = _now()
    changed = db.query(model).filter(
        model.id == operation_id,
        model.attempt_count < max_attempts,
        or_(model.status == "queued",
            (model.status == "processing") & (model.lease_expires_at <= now)),
    ).update({
        model.status: "processing", model.lease_token: token,
        model.lease_expires_at: now + timedelta(seconds=MEMORY_WORK_LEASE_SECONDS),
        model.attempt_count: model.attempt_count + 1,
        model.started_at: now, model.updated_at: now,
    }, synchronize_session=False)
    db.commit()
    if changed != 1:
        return None
    return token, db.get(model, operation_id)


def _finish(db: Session, model, operation_id: str, token: str, *, status: str,
            error_code: str | None = None) -> bool:
    now = _now()
    changed = db.query(model).filter(model.id == operation_id, model.lease_token == token,
                                     model.status == "processing").update({
        model.status: status, model.last_error_code: error_code,
        model.lease_token: None, model.lease_expires_at: None,
        model.completed_at: now if status in {"completed", "failed", "cancelled"} else None,
        model.updated_at: now,
    }, synchronize_session=False)
    db.commit()
    return changed == 1


def _failure_status(exc: Exception, attempt: int, maximum: int) -> str:
    if isinstance(exc, (MemoryConfigurationError, memory_service.MemoryScopeError,
                        memory_service.MemoryInvalidResult)):
        return "failed"
    if isinstance(exc, memory_service.MemoryDeletionError) and not exc.retryable:
        return "failed"
    return "failed" if attempt >= maximum else "queued"


def _log(operation_id: str, operation_type: str, provider: str, attempt: int,
         started: float, result: str, error: str | None = None) -> None:
    logger.info("memory operation id=%s type=%s provider=%s attempt=%d duration_ms=%.1f result=%s error_category=%s",
                operation_id, operation_type, provider, attempt, (perf_counter() - started) * 1000,
                result, error or "none")


def process_ingestion(operation_id: str) -> str:
    started = perf_counter()
    with SessionLocal() as db:
        pending = db.get(MemoryIngestion, operation_id)
        if pending and pending.status in {"queued", "processing"} and pending.provider != memory_service.PROVIDER_NAME:
            return "busy"  # Provider intentionally disabled; keep the duty visible.
        maximum = memory_service.MEMORY_CONFIG.max_ingestion_attempts
        claimed = _claim(db, MemoryIngestion, operation_id, maximum)
        if not claimed:
            row = db.get(MemoryIngestion, operation_id)
            return "busy" if row and row.status == "processing" else "done"
        token, row = claimed
        provider, attempt = row.provider, row.attempt_count
        # No FK on the outbox: deletion survives the target's DB cascade.
        deleted = db.query(MemoryDeletion.id).filter(
            MemoryDeletion.user_id == row.user_id, MemoryDeletion.character_id == row.character_id,
        ).first() is not None
        character = db.query(Character.id).filter(Character.id == row.character_id,
                                                   Character.user_id == row.user_id).first()
        conversation = db.query(Conversation.id).filter(
            Conversation.id == row.conversation_id, Conversation.character_id == row.character_id,
            Conversation.user_id == row.user_id,
        ).first()
        user_turn = db.get(Message, row.user_message_id)
        assistant_turn = db.get(Message, row.assistant_message_id)
        if deleted or not character or not conversation:
            _finish(db, MemoryIngestion, operation_id, token, status="cancelled")
            _log(operation_id, "ingest", provider, attempt, started, "cancelled")
            return "done"
        if (not user_turn or not assistant_turn or user_turn.conversation_id != row.conversation_id
                or assistant_turn.conversation_id != row.conversation_id
                or user_turn.role != MessageRole.USER or assistant_turn.role != MessageRole.CHARACTER
                or user_turn.is_correction_cmd):
            _finish(db, MemoryIngestion, operation_id, token, status="failed", error_code="invalid_turn")
            _log(operation_id, "ingest", provider, attempt, started, "failed", "invalid_turn")
            return "done"
        scope = dict(user_id=row.user_id, character_id=row.character_id,
                     conversation_id=row.conversation_id, user_message_id=row.user_message_id,
                     assistant_message_id=row.assistant_message_id,
                     user_message=user_turn.content,
                     assistant_message=normalize_assistant_actions(assistant_turn.content))
    try:
        memory_service._provider_record_completed_turn(**scope)
    except Exception as exc:
        outcome = _failure_status(exc, attempt, maximum)
        with SessionLocal() as db:
            _finish(db, MemoryIngestion, operation_id, token, status=outcome,
                    error_code=type(exc).__name__)
        _log(operation_id, "ingest", provider, attempt, started, outcome, type(exc).__name__)
        return "retry" if outcome == "queued" else "done"
    with SessionLocal() as db:
        # If deletion committed during the provider call, the deletion worker
        # waits for this lease to finish and then purges any written episodes.
        tombstone = db.query(MemoryDeletion.id).filter(
            MemoryDeletion.user_id == scope["user_id"],
            MemoryDeletion.character_id == scope["character_id"],
        ).first()
        _finish(db, MemoryIngestion, operation_id, token,
                status="cancelled" if tombstone else "completed")
    _log(operation_id, "ingest", provider, attempt, started, "completed")
    return "done"


def process_deletion(operation_id: str) -> str:
    started = perf_counter()
    with SessionLocal() as db:
        pending = db.get(MemoryDeletion, operation_id)
        if pending and pending.status in {"queued", "processing"} and pending.provider != memory_service.PROVIDER_NAME:
            return "busy"  # Reconciler will retry after explicit reactivation.
        maximum = memory_service.MEMORY_CONFIG.max_deletion_attempts
        claimed = _claim(db, MemoryDeletion, operation_id, maximum)
        if not claimed:
            row = db.get(MemoryDeletion, operation_id)
            return "busy" if row and row.status == "processing" else "done"
        token, row = claimed
        provider, attempt = row.provider, row.attempt_count
        scope = dict(user_id=row.user_id, character_id=row.character_id)
        active = db.query(MemoryIngestion.id).filter(
            MemoryIngestion.user_id == row.user_id,
            MemoryIngestion.character_id == row.character_id,
            MemoryIngestion.status == "processing",
            MemoryIngestion.lease_expires_at > _now(),
        ).first() is not None
        if active:
            # Waiting for an in-flight write is not a provider attempt.
            db.query(MemoryDeletion).filter(MemoryDeletion.id == operation_id,
                                            MemoryDeletion.lease_token == token).update({
                MemoryDeletion.status: "queued", MemoryDeletion.lease_token: None,
                MemoryDeletion.lease_expires_at: None,
                MemoryDeletion.attempt_count: MemoryDeletion.attempt_count - 1,
            }, synchronize_session=False)
            db.commit()
            _log(operation_id, "delete", provider, attempt, started, "waiting")
            return "retry"
    try:
        result = memory_service.delete_character_memories(**scope)
        if not result.success:
            raise memory_service.MemoryDeletionError("Provider deletion unconfirmed",
                                                      retryable=result.retryable,
                                                      partial=result.partial)
    except Exception as exc:
        outcome = _failure_status(exc, attempt, maximum)
        with SessionLocal() as db:
            _finish(db, MemoryDeletion, operation_id, token, status=outcome,
                    error_code=type(exc).__name__)
        _log(operation_id, "delete", provider, attempt, started, outcome, type(exc).__name__)
        return "retry" if outcome == "queued" else "done"
    with SessionLocal() as db:
        _finish(db, MemoryDeletion, operation_id, token, status="completed")
    _log(operation_id, "delete", provider, attempt, started, "completed")
    return "done"


def reconcile_pending(limit: int = 100) -> dict[str, int]:
    """Republish committed intents after HTTP publish failures or lost delivery."""
    now = _now()
    published = {"ingest": 0, "delete": 0}
    with SessionLocal() as db:
        for model, topic, key in (
            (MemoryIngestion, MEMORY_INGEST_TOPIC, "ingest"),
            (MemoryDeletion, MEMORY_DELETE_TOPIC, "delete"),
        ):
            rows = db.query(model.id).filter(or_(
                model.status == "queued",
                (model.status == "processing") & (model.lease_expires_at <= now),
            )).limit(limit).all()
            for (operation_id,) in rows:
                try:
                    publish_memory_operation(topic, operation_id)
                    published[key] += 1
                except Exception as exc:
                    logger.warning("memory reconcile publish failed operation_id=%s type=%s error=%s",
                                   operation_id, key, type(exc).__name__)
    return published
