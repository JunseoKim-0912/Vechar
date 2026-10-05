"""Durable training orchestration. Queue payloads contain identifiers only."""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from ..database import SessionLocal
from ..models import (
    Character, IngestStatus, SourceType, TrainingJob, TrainingJobChunk,
    TrainingSource, World, WorldSource, WorldSourceType,
)
from ..schemas import CharacterProfileData, WorldProfileData
from .character_profile_service import get_profile
from .extraction_service import extract_profile_from_text
from .training_aggregation import ChunkEvidence
from .training_chunker import TrainingChunk, normalize_training_text
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


def _terminal_error(exc: Exception) -> str | None:
    if isinstance(exc, HTTPException):
        code = exc.detail.get("code") if isinstance(exc.detail, dict) else None
        if code in {"daily_limit_reached", "monthly_limit_reached"}:
            return code
        if 400 <= exc.status_code < 500 and exc.status_code != 429:
            return code or "training_input_rejected"
    provider_status = getattr(exc, "status_code", None)
    if isinstance(provider_status, int) and 400 <= provider_status < 500 and provider_status != 429:
        return "llm_request_rejected"
    return None


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
        code = _terminal_error(exc)
        if code or job.attempt_count >= MAX_STAGE_ATTEMPTS:
            fail_job(job, code or "training_failed")
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
    logger.warning("training stage failed job_id=%s attempt=%s status=%s error_type=%s",
                   job_id, attempt, status, type(exc).__name__)
    return outcome


async def process_plan(job_id: str, queue: TrainingQueue) -> str:
    """Plan once, then publish chunk IDs. Redelivery republishes checkpointed chunks."""
    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        if not job or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        if JobStatus(job.status) == JobStatus.EXTRACTING:
            chunks = db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job_id).all()
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
                8000 if kind == "character" else 2000,
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
        index=chunk.chunk_index, total=job.total_chunks, source_start=chunk.source_start,
        core_start=chunk.core_start, core_end=chunk.core_end,
        token_start=chunk.token_start, token_end=chunk.token_end,
        overlap_tokens=chunk.overlap_tokens,
        text=(job.source_text or "")[chunk.source_start:chunk.core_end],
    )


def _handle_chunk_failure(job_id: str, chunk_id: str, token: str, exc: Exception) -> str:
    with SessionLocal() as db:
        job = _lock_job(db, job_id)
        chunk = db.query(TrainingJobChunk).filter(TrainingJobChunk.id == chunk_id,
                                                   TrainingJobChunk.job_id == job_id).first()
        if not job or not chunk or chunk.lease_token != token or JobStatus(job.status) in TERMINAL_JOBS:
            return "done"
        code = _terminal_error(exc)
        if code or chunk.attempt_count >= MAX_CHUNK_ATTEMPTS:
            chunk.status = ChunkStatus.FAILED.value
            chunk.error_code = code or "chunk_failed"
            fail_job(job, code or "chunk_failed")
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
    logger.warning("training chunk failed job_id=%s chunk_id=%s attempt=%d status=%s error_type=%s",
                   job_id, chunk_id, attempt, status, type(exc).__name__)
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
        total = db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job_id).count()
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
            chunks = db.query(TrainingJobChunk).filter(TrainingJobChunk.job_id == job_id).order_by(
                TrainingJobChunk.chunk_index,
            ).all()
            if len(chunks) != job.total_chunks or any(c.extraction_result is None for c in chunks):
                raise RuntimeError("Incomplete extraction evidence")
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
