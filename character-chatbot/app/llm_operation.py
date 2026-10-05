"""Per-worker logical operation identity without passing IDs through profile services."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class LLMOperation:
    job_id: str
    stage: str
    attempt: int
    chunk_index: int | None = None

    def key(self, request_type: str) -> str:
        part = f"chunk:{self.chunk_index}" if self.chunk_index is not None else "finalize"
        return f"training:{self.job_id}:{part}:{self.stage}:{self.attempt}:{request_type}"


_current: ContextVar[LLMOperation | None] = ContextVar("llm_operation", default=None)


def current_operation() -> LLMOperation | None:
    return _current.get()


@contextmanager
def llm_operation(operation: LLMOperation) -> Iterator[None]:
    token = _current.set(operation)
    try:
        yield
    finally:
        _current.reset(token)
