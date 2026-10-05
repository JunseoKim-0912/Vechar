"""Durable training orchestration. Queue payloads contain identifiers only."""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import HTTPException
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError

from ..database import SessionLocal
from ..llm import make_training_text_counter
from ..llm_config import CHARACTER_EXTRACTION_OUTPUT_TOKENS, MAX_ADAPTIVE_SPLIT_DEPTH
from ..llm_failures import FailureKind, classify_llm_failure
from ..llm_operation import LLMOperation, llm_operation
from ..models import (
    Character, IngestStatus, SourceType, TrainingJob, TrainingJobChunk,
    TrainingSource, World, WorldSource, WorldSourceType,
)
from ..schemas import CharacterProfileData, WorldProfileData
from .character_profile_service import get_profile
from .extraction_service import extract_profile_from_text
from .training_aggregation import ChunkEvidence
from .training_chunker import TrainingChunk, normalize_training_text, split_training_chunk
from .training_job_state import (
    ACTIVE_JOBS, TERMINAL_JOBS, ChunkStatus, JobStatus, MAX_CHUNK_ATTEMPTS,
    MAX_STAGE_ATTEMPTS, WORK_LEASE_SECONDS, fail_job, transition,
)
from .training_pipeline import (
    ExtractionRun, finalize_character_run, finalize_world_run, plan_training_source,
)
from .training_queue import TrainingQueue
from .world_extraction_service import extract_world_profile_from_text
from .world_profile_service import get_world_profile


logger = logging.getLogger(__name__)


def _now() -> datetime:
    # Existing DateTime columns are timezone-naive in PostgreSQL and SQLite.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _lease_deadline() -> datetime:
    return _now() + timedelta(seconds=WORK_LEASE_SECONDS)


def _target_exists(db, job: TrainingJob) -> bool:
    model = Character if job.target_type == "character" else World
    return db.query(model.id).filter(model.id == job.target_id, model.user_id == job.user_id).first() is not None


def _lock_job(db, job_id: str) -> TrainingJob | None:
    return db.query(TrainingJob).filter(TrainingJob.id == job_id).with_for_update().first()


async def submit_job(
    db, queue: TrainingQueue, *, user_id: str, target_type: str, target_id: str,
    source_type: str, training_source_type: str, raw_text: str,
    series_name: str | None = None, episode_number: int | None = None,
) -> TrainingJob:
    normalized = normalize_training_text(raw_text)
    existing = db.query(TrainingJob).filter(
        TrainingJob.user_id == user_id, TrainingJob.target_type == target_type,
        TrainingJob.target_id == target_id, TrainingJob.status.in_([s.value for s in ACTIVE_JOBS]),
    ).first()
    if existing:
        return existing
    job = TrainingJob(
        user_id=user_id, target_type=target_type, target_id=target_id,
        source_type=source_type, training_source_type=training_source_type,
        source_text=normalized, source_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        source_char_count=len(normalized), series_name=series_name, episode_number=episode_number,
        status=JobStatus.QUEUED.value, stage=JobStatus.QUEUED.value,
    )
    db.add(job)
    try:
        db.commit()
        db.refresh(job)
    except IntegrityError:
        db.rollback()
        existing = db.query(TrainingJob).filter(
            TrainingJob.user_id == user_id, TrainingJob.target_type == target_type,
            TrainingJob.target_id == target_id, TrainingJob.status.in_([s.value for s in ACTIVE_JOBS]),
        ).first()
        if existing:
            return existing
        raise
    try:
        await queue.publish_plan(job.id)
    except Exception as exc:
        db.rollback()
        locked = _lock_job(db, job.id)
        if locked and JobStatus(locked.status) == JobStatus.QUEUED:
            fail_job(locked, "queue_unavailable", "Training could not be queued. Please try again.")
            locked.completed_at = _now()
            db.commit()
        logger.warning("training queue publish failed job_id=%s error_type=%s", job.id, type(exc).__name__)
        raise HTTPException(status_code=503, detail={"code": "training_queue_unavailable"}) from exc
    logger.info("training submitted job_id=%s target_type=%s status=queued", job.id, target_type)
    return job


def cancel_target_jobs(db, *, user_id: str, target_type: str, target_id: str) -> None:
    jobs = db.query(TrainingJob).filter(
        TrainingJob.user_id == user_id, TrainingJob.target_type == target_type,
        TrainingJob.target_id == target_id, TrainingJob.status.in_([s.value for s in ACTIVE_JOBS]),
    ).with_for_update().all()
    for job in jobs:
        transition(job, JobStatus.CANCELLED)
        job.source_text = None
        job.completed_at = _now()
        for chunk in db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job.id):
            chunk.extraction_result = None
    # Caller commits cancellation and target deletion in one transaction.


def _handle_stage_failure(job_id: str, token: str, exc: Exception) -> str:
    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        if not job or job.lease_token != token or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        classification = classify_llm_failure(exc)
        if not classification.retryable or job.attempt_count >= MAX_STAGE_ATTEMPTS:
            fail_job(job, classification.code)
            job.completed_at = _now()
            for chunk in db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job_id):
                chunk.extraction_result = None
            outcome = "done"
        else:
            job.lease_token = None
            job.lease_expires_at = None
            outcome = "retry"
        attempt, status = job.attempt_count, job.status
        db.commit()
    logger.warning("training stage failed job_id=%s attempt=%s status=%s error_type=%s retry_class=%s",
                   job_id, attempt, status, type(exc).__name__, classification.kind.value)
    return outcome


async def process_plan(job_id: str, queue: TrainingQueue) -> str:
    """Plan once, then publish chunk IDs. Redelivery republishes checkpointed chunks."""
    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        if not job or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        if JobStatus(job.status) == JobStatus.EXTRACTING:
            chunks = db.query(TrainingJobChunk).filter(
                TrainingJobChunk.job_id == job_id, TrainingJobChunk.status != ChunkStatus.SPLIT.value,
            ).all()
            chunk_ids = [c.id for c in chunks]
            db.rollback()
            for chunk_id in chunk_ids:
                await queue.publish_chunk(job_id, chunk_id)
            return "done"
        if JobStatus(job.status) == JobStatus.SYNTHESIZING:
            return "done"
        if JobStatus(job.status) != JobStatus.QUEUED and job.lease_expires_at and job.lease_expires_at > _now():
            return "busy"
        if job.attempt_count >= MAX_STAGE_ATTEMPTS:
            fail_job(job, "planning_failed")
            job.completed_at = _now()
            db.commit()
            return "done"
        if JobStatus(job.status) == JobStatus.QUEUED:
            transition(job, JobStatus.CHUNKING)
            job.started_at = job.started_at or _now()
        token = str(uuid4())
        job.lease_token = token
        job.lease_expires_at = _lease_deadline()
        job.attempt_count += 1
        user_id, raw_text, kind = job.user_id, job.source_text, job.target_type
        db.commit()
    try:
        if raw_text is None:
            raise RuntimeError("Training source unavailable")
        with SessionLocal() as db:
            normalized, source_tokens, chunks, direct_mode = plan_training_source(
                db, user_id, raw_text,
                "character_extraction" if kind == "character" else "world_extraction",
                CHARACTER_EXTRACTION_OUTPUT_TOKENS if kind == "character" else 2000,
            )
        with SessionLocal() as db:
            job = _lock_job(db, job_id)
            if not job or job.lease_token != token or JobStatus(job.status) != JobStatus.CHUNKING:
                return "done"
            if not _target_exists(db, job):
                transition(job, JobStatus.CANCELLED)
                job.source_text = None
                job.completed_at = _now()
                db.commit()
                return "done"
            job.source_text = normalized
            job.source_tokens = source_tokens
            job.direct_mode = direct_mode
            job.total_chunks = len(chunks)
            for chunk in chunks:
                db.add(TrainingJobChunk(
                    job_id=job_id, chunk_index=chunk.index, status=ChunkStatus.QUEUED.value,
                    source_start=chunk.source_start, core_start=chunk.core_start,
                    core_end=chunk.core_end, token_start=chunk.token_start, token_end=chunk.token_end,
                    token_count=max(0, chunk.token_end - chunk.token_start + chunk.overlap_tokens),
                    overlap_tokens=chunk.overlap_tokens,
                ))
            transition(job, JobStatus.EXTRACTING)
            job.attempt_count = 0
            job.lease_token = None
            job.lease_expires_at = None
            db.commit()
        with SessionLocal() as db:
            chunk_ids = [c.id for c in db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job_id).all()]
        for chunk_id in chunk_ids:
            await queue.publish_chunk(job_id, chunk_id)
        logger.info("training planned job_id=%s chunks=%d status=extracting", job_id, len(chunk_ids))
        return "done"
    except Exception as exc:
        with SessionLocal() as db:
            current = db.get(TrainingJob, job_id)
            if current and JobStatus(current.status) == JobStatus.EXTRACTING:
                logger.warning("training chunk publishing failed job_id=%s error_type=%s",
                               job_id, type(exc).__name__)
                return "retry"
        return _handle_stage_failure(job_id, token, exc)


def _chunk_as_value(job: TrainingJob, chunk: TrainingJobChunk) -> TrainingChunk:
    return TrainingChunk(
        index=chunk.chunk_index, total=max(job.total_chunks, chunk.chunk_index),
        source_start=chunk.source_start,
        core_start=chunk.core_start, core_end=chunk.core_end,
        token_start=chunk.token_start, token_end=chunk.token_end,
        overlap_tokens=chunk.overlap_tokens,
        text=(job.source_text or "")[chunk.source_start:chunk.core_end],
    )


async def _split_output_limited_chunk(
    job_id: str, chunk_id: str, token: str, queue: TrainingQueue,
) -> str | None:
    """Replace just the failed character leaf; None means bounded fallback is exhausted."""
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        parent = db.get(TrainingJobChunk, chunk_id)
        if (not job or not parent or parent.job_id != job_id
                or JobStatus(job.status) != JobStatus.EXTRACTING
                or parent.status != ChunkStatus.PROCESSING.value or parent.lease_token != token):
            return "done"
        if job.target_type != "character" or parent.split_depth >= MAX_ADAPTIVE_SPLIT_DEPTH:
            return None
        if job.source_text is None or not _target_exists(db, job):
            return "done"
        source = job.source_text
        parent_value = _chunk_as_value(job, parent)
        count_tokens = make_training_text_counter(
            db, job.user_id, "character_extraction", CHARACTER_EXTRACTION_OUTPUT_TOKENS,
        )
        children = split_training_chunk(source, parent_value, count_tokens)
        if children is None:
            return None

    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        parent = db.query(TrainingJobChunk).filter(
            TrainingJobChunk.id == chunk_id, TrainingJobChunk.job_id == job_id,
        ).with_for_update().first()
        if (not job or not parent or JobStatus(job.status) != JobStatus.EXTRACTING
                or parent.status != ChunkStatus.PROCESSING.value or parent.lease_token != token
                or job.source_text is None or not _target_exists(db, job)):
            return "done"
        next_index = (db.query(func.max(TrainingJobChunk.chunk_index))
                      .filter(TrainingJobChunk.job_id == job_id).scalar() or 0) + 1
        child_ids = []
        for order, child in enumerate(children, 1):
            child_id = str(uuid5(NAMESPACE_URL, f"training:{job_id}:parent:{parent.id}:child:{order}"))
            db.add(TrainingJobChunk(
                id=child_id, job_id=job_id, chunk_index=next_index + order - 1,
                parent_chunk_id=parent.id, split_depth=parent.split_depth + 1,
                child_order=order, status=ChunkStatus.QUEUED.value,
                source_start=child.source_start, core_start=child.core_start,
                core_end=child.core_end, token_start=child.token_start,
                token_end=child.token_end,
                token_count=max(0, child.token_end - child.token_start + child.overlap_tokens),
                overlap_tokens=child.overlap_tokens,
            ))
            child_ids.append(child_id)
        parent.status = ChunkStatus.SPLIT.value
        parent.error_code = "llm_output_limit"
        parent.lease_token = None
        parent.lease_expires_at = None
        parent.completed_at = _now()
        job.total_chunks += 1  # Two active children replace one active parent.
        job.direct_mode = False
        job.progress = int(85 * job.completed_chunks / job.total_chunks)
        db.commit()
        logger.warning(
            "training chunk split job_id=%s parent_chunk_id=%s child_ids=%s depth=%d "
            "parent_token_range=%d-%d child_token_ranges=%s trigger=output_limit",
            job_id, chunk_id, child_ids, parent.split_depth + 1,
            parent.token_start, parent.token_end,
            [(child.token_start, child.token_end) for child in children],
        )
    try:
        for child_id in child_ids:
            await queue.publish_chunk(job_id, child_id)
    except Exception as exc:
        logger.warning("training split child publish failed job_id=%s parent_chunk_id=%s error_type=%s",
                       job_id, chunk_id, type(exc).__name__)
        return "retry"
    return "done"


def _handle_chunk_failure(job_id: str, chunk_id: str, token: str, exc: Exception) -> str:
    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        chunk = db.query(TrainingJobChunk).filter(TrainingJobChunk.id == chunk_id,
                                                   TrainingJobChunk.job_id == job_id).first()
        if not job or not chunk or chunk.lease_token != token or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        classification = classify_llm_failure(exc)
        if not classification.retryable or chunk.attempt_count >= MAX_CHUNK_ATTEMPTS:
            chunk.status = ChunkStatus.FAILED.value
            chunk.error_code = classification.code
            fail_job(job, classification.code)
            job.completed_at = _now()
            for item in db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job_id):
                item.extraction_result = None
            outcome = "done"
        else:
            chunk.status = ChunkStatus.QUEUED.value
            chunk.error_code = "retryable_error"
            chunk.lease_token = None
            chunk.lease_expires_at = None
            outcome = "retry"
        db.commit()
        attempt, status = chunk.attempt_count, chunk.status
    logger.warning("training chunk failed job_id=%s chunk_id=%s attempt=%d status=%s error_type=%s "
                   "retry_class=%s", job_id, chunk_id, attempt, status, type(exc).__name__,
                   classification.kind.value)
    return outcome


async def process_chunk(job_id: str, chunk_id: str, queue: TrainingQueue) -> str:
    """Atomic short claim, unmetered duplicate no-op, metered extraction, fenced save."""
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        chunk = db.get(TrainingJobChunk, chunk_id)
        if not job or not chunk or chunk.job_id != job_id or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        if chunk.status == ChunkStatus.COMPLETED.value:
            all_done = (job.completed_chunks == job.total_chunks and job.total_chunks > 0)
            if all_done:
                await queue.publish_finalize(job_id)
            return "done"
        if chunk.status == ChunkStatus.SPLIT.value:
            if JobStatus(job.status) != JobStatus.EXTRACTING:
                return "done"
            child_ids = [row.id for row in db.query(TrainingJobChunk).filter(
                TrainingJobChunk.job_id == job_id,
                TrainingJobChunk.parent_chunk_id == chunk_id,
            ).order_by(TrainingJobChunk.child_order).all()]
            db.rollback()
            try:
                for child_id in child_ids:
                    await queue.publish_chunk(job_id, child_id)
            except Exception:
                return "retry"
            return "done"
        if JobStatus(job.status) != JobStatus.EXTRACTING:
            return "busy"
        if chunk.status == ChunkStatus.PROCESSING.value and chunk.lease_expires_at and chunk.lease_expires_at > _now():
            return "busy"
        if chunk.attempt_count >= MAX_CHUNK_ATTEMPTS:
            return _handle_chunk_failure(job_id, chunk_id, chunk.lease_token or "", RuntimeError("Attempt limit"))
        token = str(uuid4())
        changed = db.query(TrainingJobChunk).filter(
            TrainingJobChunk.id == chunk_id,
            TrainingJobChunk.status.in_([ChunkStatus.QUEUED.value, ChunkStatus.PROCESSING.value]),
            or_(TrainingJobChunk.lease_expires_at.is_(None), TrainingJobChunk.lease_expires_at <= _now()),
            TrainingJobChunk.attempt_count < MAX_CHUNK_ATTEMPTS,
        ).update({
            TrainingJobChunk.status: ChunkStatus.PROCESSING.value,
            TrainingJobChunk.lease_token: token,
            TrainingJobChunk.lease_expires_at: _lease_deadline(),
            TrainingJobChunk.attempt_count: TrainingJobChunk.attempt_count + 1,
            TrainingJobChunk.started_at: _now(),
        }, synchronize_session=False)
        db.commit()
        if changed != 1:
            return "busy"
    try:
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            chunk = db.get(TrainingJobChunk, chunk_id)
            if not job or not chunk or job.source_text is None or not _target_exists(db, job):
                raise RuntimeError("Training target/source unavailable")
            plan = _chunk_as_value(job, chunk)
            with llm_operation(LLMOperation(
                job.id, "extract", chunk.attempt_count, chunk.chunk_index,
                chunk.id if chunk.parent_chunk_id else None,
            )):
                if job.target_type == "character":
                    character = db.get(Character, job.target_id)
                    existing = get_profile(db, job.target_id)
                    canonical = CharacterProfileData.model_validate(existing.data) if existing else None
                    extracted = extract_profile_from_text(
                        db, job.user_id, plan.text, SourceType(job.training_source_type),
                        character.name, canonical, chunk=None if job.direct_mode else plan,
                    )
                else:
                    existing = get_world_profile(db, job.target_id)
                    canonical = WorldProfileData.model_validate(existing.data) if existing else None
                    extracted = extract_world_profile_from_text(
                        db, job.user_id, plan.text, WorldSourceType(job.training_source_type),
                        job.series_name, job.episode_number, canonical,
                        chunk=None if job.direct_mode else plan,
                    )
            extraction_result = extracted.model_dump()
        with SessionLocal() as db:
            job = _lock_job(db, job_id)
            chunk = db.query(TrainingJobChunk).filter(TrainingJobChunk.id == chunk_id,
                                                       TrainingJobChunk.job_id == job_id).first()
            if not job or not chunk or JobStatus(job.status) != JobStatus.EXTRACTING or chunk.lease_token != token:
                return "done"
            if not _target_exists(db, job):
                transition(job, JobStatus.CANCELLED)
                job.source_text = None
                job.completed_at = _now()
                db.commit()
                return "done"
            chunk.extraction_result = extraction_result
            chunk.status = ChunkStatus.COMPLETED.value
            chunk.completed_at = _now()
            chunk.lease_token = None
            chunk.lease_expires_at = None
            db.flush()
            completed = db.query(TrainingJobChunk).filter(
                TrainingJobChunk.job_id == job_id, TrainingJobChunk.status == ChunkStatus.COMPLETED.value,
            ).count()
            job.completed_chunks = min(completed, job.total_chunks)
            job.progress = int(85 * job.completed_chunks / job.total_chunks)
            all_done = job.completed_chunks == job.total_chunks
            if all_done:
                transition(job, JobStatus.SYNTHESIZING)
                job.attempt_count = 0
            db.commit()
            attempt = chunk.attempt_count
        if all_done:
            await queue.publish_finalize(job_id)
        logger.info("training chunk completed job_id=%s chunk_index=%d attempt=%d status=completed",
                    job_id, plan.index, attempt)
        return "done"
    except Exception as exc:
        with SessionLocal() as db:
            current = db.get(TrainingJobChunk, chunk_id)
            if current and current.status == ChunkStatus.COMPLETED.value:
                logger.warning("training finalization publishing failed job_id=%s chunk_id=%s error_type=%s",
                               job_id, chunk_id, type(exc).__name__)
                return "retry"
        if classify_llm_failure(exc).kind == FailureKind.OUTPUT_LIMIT:
            try:
                outcome = await _split_output_limited_chunk(job_id, chunk_id, token, queue)
            except Exception as split_exc:
                logger.warning("training adaptive split unavailable job_id=%s chunk_id=%s error_type=%s",
                               job_id, chunk_id, type(split_exc).__name__)
                # Never repeat the same output-capped provider call because a
                # token-count/split preparation step had a transient failure.
                failure = (split_exc if classify_llm_failure(split_exc).kind == FailureKind.BUDGET else exc)
                return _handle_chunk_failure(job_id, chunk_id, token, failure)
            if outcome is not None:
                return outcome
        return _handle_chunk_failure(job_id, chunk_id, token, exc)


async def process_finalize(job_id: str) -> str:
    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        if not job or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        if JobStatus(job.status) != JobStatus.SYNTHESIZING:
            return "busy"
        if job.lease_expires_at and job.lease_expires_at > _now():
            return "busy"
        total = db.query(TrainingJobChunk).filter(
            TrainingJobChunk.job_id == job_id, TrainingJobChunk.status != ChunkStatus.SPLIT.value,
        ).count()
        completed = db.query(TrainingJobChunk).filter(
            TrainingJobChunk.job_id == job_id, TrainingJobChunk.status == ChunkStatus.COMPLETED.value,
        ).count()
        if total == 0 or total != job.total_chunks or completed != total:
            return "busy"
        if job.attempt_count >= MAX_STAGE_ATTEMPTS:
            fail_job(job, "synthesis_failed")
            job.completed_at = _now()
            db.commit()
            return "done"
        token = str(uuid4())
        job.lease_token = token
        job.lease_expires_at = _lease_deadline()
        job.attempt_count += 1
        db.commit()
    try:
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            if not job or job.source_text is None or not _target_exists(db, job):
                raise RuntimeError("Training target/source unavailable")
            chunks = db.query(TrainingJobChunk).filter(
                TrainingJobChunk.job_id == job_id, TrainingJobChunk.status != ChunkStatus.SPLIT.value,
            ).order_by(
                TrainingJobChunk.core_start, TrainingJobChunk.chunk_index,
            ).all()
            if len(chunks) != job.total_chunks or any(c.extraction_result is None for c in chunks):
                raise RuntimeError("Incomplete extraction evidence")
            with llm_operation(LLMOperation(job.id, "finalize", job.attempt_count)):
                if job.target_type == "character":
                    source = TrainingSource(
                        id=job.id, character_id=job.target_id, source_type=SourceType(job.training_source_type),
                        raw_text=job.source_text, char_count=job.source_char_count, status=IngestStatus.PENDING,
                    )
                    existing = get_profile(db, job.target_id)
                    canonical = CharacterProfileData.model_validate(existing.data) if existing else None
                    evidence = [ChunkEvidence(_chunk_as_value(job, c), CharacterProfileData.model_validate(c.extraction_result))
                                for c in chunks]
                    run = ExtractionRun(
                        "direct" if job.direct_mode else "chunked", job.source_tokens,
                        [] if job.direct_mode else evidence,
                        evidence[0].extracted if job.direct_mode else None,
                    )
                    finalize_character_run(db, job.user_id, source, canonical, run, commit=False)
                else:
                    source = WorldSource(
                        id=job.id, world_id=job.target_id, source_type=WorldSourceType(job.training_source_type),
                        series_name=job.series_name, episode_number=job.episode_number,
                        raw_text=job.source_text, char_count=job.source_char_count, status=IngestStatus.PENDING,
                    )
                    existing = get_world_profile(db, job.target_id)
                    canonical = WorldProfileData.model_validate(existing.data) if existing else None
                    evidence = [ChunkEvidence(_chunk_as_value(job, c), WorldProfileData.model_validate(c.extraction_result))
                                for c in chunks]
                    run = ExtractionRun(
                        "direct" if job.direct_mode else "chunked", job.source_tokens,
                        [] if job.direct_mode else evidence,
                        evidence[0].extracted if job.direct_mode else None,
                    )
                    finalize_world_run(db, job.user_id, source, canonical, run, commit=False)
            # Fence the final write against cancellation and expired workers.
            locked = _lock_job(db, job_id)
            if not locked or locked.lease_token != token or JobStatus(locked.status) != JobStatus.SYNTHESIZING:
                db.rollback()
                return "done"
            if not _target_exists(db, locked):
                db.rollback()
                return "done"
            db.add(source)
            transition(locked, JobStatus.COMPLETED)
            locked.progress = 100
            locked.completed_at = _now()
            locked.source_text = None
            for chunk in chunks:
                chunk.extraction_result = None
            db.commit()
        logger.info("training finalized job_id=%s status=completed", job_id)
        return "done"
    except Exception as exc:
        return _handle_stage_failure(job_id, token, exc)
