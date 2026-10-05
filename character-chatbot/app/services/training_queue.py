"""Small queue adapter; persisted DB rows, not messages, are authoritative."""

import os
from typing import Protocol
from uuid import uuid4


PLAN_TOPIC = "training-plan"
CHUNK_TOPIC = "training-chunk"
FINALIZE_TOPIC = "training-finalize"
MESSAGE_VERSION = 1
MEMORY_INGEST_TOPIC = "memory-ingest"
MEMORY_DELETE_TOPIC = "memory-delete"


def publish_memory_operation(topic: str, operation_id: str) -> None:
    """Use the same Queue infrastructure; only IDs cross the transport."""
    if topic not in {MEMORY_INGEST_TOPIC, MEMORY_DELETE_TOPIC}:
        raise ValueError("Unsupported memory queue topic")
    payload = {"v": MESSAGE_VERSION, "operation_id": operation_id}
    if os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV"):
        from vercel.queue.sync import send
        # Reconciliation must not be suppressed by a previously accepted send.
        # The DB claim, not transport deduplication, provides idempotency.
        send(topic, payload, idempotency_key=f"{topic}-{operation_id}-{uuid4()}")
    else:
        _local_queue.messages.append((topic, payload))


class TrainingQueue(Protocol):
    async def publish_plan(self, job_id: str) -> None: ...
    async def publish_chunk(self, job_id: str, chunk_id: str) -> None: ...
    async def publish_finalize(self, job_id: str) -> None: ...


class VercelTrainingQueue:
    async def _send(self, topic: str, payload: dict, key: str) -> None:
        from vercel.queue import send
        await send(topic, payload, idempotency_key=key)

    async def publish_plan(self, job_id: str) -> None:
        await self._send(PLAN_TOPIC, {"v": MESSAGE_VERSION, "job_id": job_id}, f"plan-{job_id}")

    async def publish_chunk(self, job_id: str, chunk_id: str) -> None:
        await self._send(CHUNK_TOPIC, {"v": MESSAGE_VERSION, "job_id": job_id, "chunk_id": chunk_id},
                         f"chunk-{chunk_id}")

    async def publish_finalize(self, job_id: str) -> None:
        await self._send(FINALIZE_TOPIC, {"v": MESSAGE_VERSION, "job_id": job_id}, f"finalize-{job_id}")


class InMemoryTrainingQueue:
    """Explicit local/test fake; callers can drain messages into worker functions."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict]] = []

    async def publish_plan(self, job_id: str) -> None:
        self.messages.append((PLAN_TOPIC, {"v": MESSAGE_VERSION, "job_id": job_id}))

    async def publish_chunk(self, job_id: str, chunk_id: str) -> None:
        self.messages.append((CHUNK_TOPIC, {"v": MESSAGE_VERSION, "job_id": job_id, "chunk_id": chunk_id}))

    async def publish_finalize(self, job_id: str) -> None:
        self.messages.append((FINALIZE_TOPIC, {"v": MESSAGE_VERSION, "job_id": job_id}))


_local_queue = InMemoryTrainingQueue()


def get_training_queue() -> TrainingQueue:
    return VercelTrainingQueue() if os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV") else _local_queue
