"""Vercel Python Queue subscribers (discovered from pyproject.toml)."""

import logging

from vercel.queue import RetryAfter, subscribe

from app.services.training_jobs import process_chunk, process_finalize, process_plan
from app.services.training_job_state import QUEUE_DELIVERY_MAX_ATTEMPTS, RETRY_AFTER_SECONDS
from app.services.training_queue import (
    CHUNK_TOPIC, FINALIZE_TOPIC, MESSAGE_VERSION, PLAN_TOPIC, get_training_queue,
)


logger = logging.getLogger(__name__)


def _ids(payload: object, *, chunk: bool = False) -> tuple[str, str | None] | None:
    if not isinstance(payload, dict) or payload.get("v") != MESSAGE_VERSION:
        logger.warning("training queue ignored unsupported payload version")
        return None
    job_id = payload.get("job_id")
    chunk_id = payload.get("chunk_id")
    if not isinstance(job_id, str) or (chunk and not isinstance(chunk_id, str)):
        logger.warning("training queue ignored malformed identifier payload")
        return None
    return job_id, chunk_id


def _retry_if_needed(result: str) -> None:
    if result in {"busy", "retry"}:
        raise RetryAfter(RETRY_AFTER_SECONDS)


@subscribe(topic=PLAN_TOPIC, retry_after=RETRY_AFTER_SECONDS, max_attempts=QUEUE_DELIVERY_MAX_ATTEMPTS)
async def handle_plan(payload) -> None:
    ids = _ids(payload)
    if ids:
        _retry_if_needed(await process_plan(ids[0], get_training_queue()))


@subscribe(topic=CHUNK_TOPIC, retry_after=RETRY_AFTER_SECONDS, max_attempts=QUEUE_DELIVERY_MAX_ATTEMPTS)
async def handle_chunk(payload) -> None:
    ids = _ids(payload, chunk=True)
    if ids:
        _retry_if_needed(await process_chunk(ids[0], ids[1], get_training_queue()))


@subscribe(topic=FINALIZE_TOPIC, retry_after=RETRY_AFTER_SECONDS, max_attempts=QUEUE_DELIVERY_MAX_ATTEMPTS)
async def handle_finalize(payload) -> None:
    ids = _ids(payload)
    if ids:
        _retry_if_needed(await process_finalize(ids[0]))
