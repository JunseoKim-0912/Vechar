"""Small queue adapter; persisted DB rows, not messages, are authoritative."""

import os
from typing import Protocol


PLAN_TOPIC = "training-plan"
CHUNK_TOPIC = "training-chunk"
FINALIZE_TOPIC = "training-finalize"
MESSAGE_VERSION = 1


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
