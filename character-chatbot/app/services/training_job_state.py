"""Central job state transitions and bounded retry/retention policy."""

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SPLIT = "split"


ACTIVE_JOBS = frozenset({JobStatus.QUEUED, JobStatus.CHUNKING, JobStatus.EXTRACTING, JobStatus.SYNTHESIZING})
TERMINAL_JOBS = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})
MAX_CHUNK_ATTEMPTS = 3
MAX_STAGE_ATTEMPTS = 3
# Longer than Vercel's standard maximum function invocation; stale claims can
# be reclaimed after a terminated function, without concurrent live workers.
WORK_LEASE_SECONDS = 1200
RETRY_AFTER_SECONDS = 60
# Enough queue deliveries to wait through one expired 20-minute lease; the DB
# attempt caps below remain the authority for actual model invocations.
QUEUE_DELIVERY_MAX_ATTEMPTS = 30

_TRANSITIONS = {
    JobStatus.QUEUED: {JobStatus.CHUNKING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.CHUNKING: {JobStatus.EXTRACTING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.EXTRACTING: {JobStatus.SYNTHESIZING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.SYNTHESIZING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


def transition(job, status: JobStatus) -> None:
    current = JobStatus(job.status)
    if status not in _TRANSITIONS[current]:
        raise ValueError(f"Invalid training job transition: {current} -> {status}")
    job.status = status.value
    job.stage = status.value


def fail_job(job, code: str, message: str = "Training failed. Please try again.") -> None:
    if JobStatus(job.status) in TERMINAL_JOBS:
        return
    transition(job, JobStatus.FAILED)
    job.error_code = code
    job.error_message_safe = message
    job.source_text = None
