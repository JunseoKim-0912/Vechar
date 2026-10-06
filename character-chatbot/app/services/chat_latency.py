"""Per-request, content-free interaction timing; TTFT needs future streaming."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatLatency:
    started: float = field(default_factory=perf_counter)
    operation: str = "character_chat"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    stages_ms: dict[str, float] = field(default_factory=dict)
    provider_start_ms: float | None = None
    provider_completion_ms: float | None = None

    @contextmanager
    def stage(self, name: str):
        begun = perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = round(
                self.stages_ms.get(name, 0.0) + (perf_counter() - begun) * 1000, 2
            )

    def finish(self, status_code: int) -> dict:
        record = {
            "operation": self.operation, "status_code": status_code,
            "total_ms": round((perf_counter() - self.started) * 1000, 2),
            "stages_ms": dict(self.stages_ms),
            **self.metadata,
        }
        if self.provider_start_ms is not None:
            record["provider_start_ms"] = self.provider_start_ms
            record["pre_llm_preparation_ms"] = self.provider_start_ms
        if self.provider_completion_ms is not None:
            record["provider_completion_ms"] = self.provider_completion_ms
        logger.info("chat_latency %s", record)
        return record


_active: ContextVar[ChatLatency | None] = ContextVar("chat_latency", default=None)


def start_chat_latency():
    tracker = ChatLatency()
    return tracker, _active.set(tracker)


def end_chat_latency(token):
    _active.reset(token)


@contextmanager
def stage(name: str):
    tracker = _active.get()
    if tracker is None:
        yield
    else:
        with tracker.stage(name):
            yield


def provider_started():
    tracker = _active.get()
    if tracker is not None:
        tracker.provider_start_ms = round((perf_counter() - tracker.started) * 1000, 2)


def provider_completed():
    tracker = _active.get()
    if tracker is not None:
        tracker.provider_completion_ms = round((perf_counter() - tracker.started) * 1000, 2)


def set_safe_metadata(*, operation: str | None = None, **fields: str | int | float | bool):
    tracker = _active.get()
    if tracker is not None:
        if operation is not None:
            tracker.operation = operation
        tracker.metadata.update(fields)
