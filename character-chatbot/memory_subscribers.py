"""Vercel Queue subscribers for durable memory operations."""

import logging

from vercel.queue import RetryAfter, subscribe

from app.services.memory_jobs import (
    MEMORY_QUEUE_DELIVERY_ATTEMPTS, MEMORY_QUEUE_RETRY_SECONDS,
    process_deletion, process_ingestion,
)
from app.services.training_queue import MEMORY_DELETE_TOPIC, MEMORY_INGEST_TOPIC, MESSAGE_VERSION

logger = logging.getLogger(__name__)


def _operation_id(payload: object) -> str | None:
    if not isinstance(payload, dict) or payload.get("v") != MESSAGE_VERSION:
        logger.warning("memory queue ignored unsupported payload")
        return None
    operation_id = payload.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        logger.warning("memory queue ignored malformed identifier")
        return None
    return operation_id


@subscribe(topic=MEMORY_INGEST_TOPIC, retry_after=MEMORY_QUEUE_RETRY_SECONDS,
           max_attempts=MEMORY_QUEUE_DELIVERY_ATTEMPTS)
async def handle_ingest(payload) -> None:
    operation_id = _operation_id(payload)
    if operation_id and process_ingestion(operation_id) in {"busy", "retry"}:
        raise RetryAfter(MEMORY_QUEUE_RETRY_SECONDS)


@subscribe(topic=MEMORY_DELETE_TOPIC, retry_after=MEMORY_QUEUE_RETRY_SECONDS,
           max_attempts=MEMORY_QUEUE_DELIVERY_ATTEMPTS)
async def handle_delete(payload) -> None:
    operation_id = _operation_id(payload)
    if operation_id and process_deletion(operation_id) in {"busy", "retry"}:
        raise RetryAfter(MEMORY_QUEUE_RETRY_SECONDS)
