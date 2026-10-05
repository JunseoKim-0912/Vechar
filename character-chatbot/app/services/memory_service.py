"""Provider-neutral episodic memory boundary; no-op unless explicitly configured.

The memory scope is (user_id, character_id). A conversation identifies a
source session, not a separate memory namespace. Provider adapters must return
candidates in descending relevance order and translate their own transient
failures to MemoryProviderError (or TimeoutError/ConnectionError).
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from time import perf_counter
from typing import Literal

from ..memory_config import load_memory_config
from ..memory_config import MemoryConfigurationError

logger = logging.getLogger(__name__)
MEMORY_CONFIG = load_memory_config()
PROVIDER_NAME = MEMORY_CONFIG.provider


class MemoryProviderError(RuntimeError):
    """A memory-only provider failure that may degrade to no memories."""


class MemoryInvalidResult(MemoryProviderError):
    """The provider returned malformed or unusable memory candidates."""


class MemoryAuthenticationError(MemoryProviderError):
    """The external provider rejected its configured credentials."""


class MemoryRateLimitError(MemoryProviderError):
    """The external provider is rate-limiting requests."""


class MemoryPartialIngestionError(MemoryProviderError):
    """One episode of a completed turn may have been stored without the other."""


class MemoryScopeError(PermissionError):
    """A provider candidate crosses the authenticated user/character scope."""


class MemoryDeletionNotFound(MemoryProviderError):
    """Already absent at the provider; deletion is complete and idempotent."""


class MemoryDeletionError(MemoryProviderError):
    """Known external deletion failure, including partial deletion."""

    def __init__(self, message: str, *, retryable: bool, partial: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.partial = partial


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    content: str
    user_id: str
    character_id: str
    source_conversation_id: str | None = None
    source_user_message_id: str | None = None
    source_assistant_message_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    relevance_score: float | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class MemoryRetrievalResult:
    candidates: tuple[MemoryCandidate, ...]
    provider: str
    latency_ms: float
    success: bool
    timed_out: bool = False
    error_type: str | None = None

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


DeletionScope = Literal["conversation", "character", "user"]


@dataclass(frozen=True)
class MemoryDeletionResult:
    provider: str
    scope_type: DeletionScope
    scope_id: str
    attempted: bool
    success: bool
    not_found: bool
    latency_ms: float
    retryable: bool = False
    partial: bool = False
    error_type: str | None = None


def _provider_retrieve(*, user_id: str, character_id: str, conversation_id: str,
                       current_message: str) -> Sequence[MemoryCandidate]:
    """The default is no-op; the optional SDK is loaded only when selected."""
    if PROVIDER_NAME == "noop":
        return ()
    return _memmachine_adapter().retrieve(
        user_id=user_id, character_id=character_id,
        conversation_id=conversation_id, current_message=current_message,
    )


@lru_cache(maxsize=1)
def _memmachine_adapter():
    from .memory_providers.memmachine import MemMachineAdapter

    return MemMachineAdapter(MEMORY_CONFIG)


def _validate_candidates(candidates: Sequence[MemoryCandidate], user_id: str,
                         character_id: str) -> tuple[MemoryCandidate, ...]:
    try:
        items = tuple(candidates)
    except TypeError as exc:
        raise MemoryInvalidResult("Memory candidates are not a sequence.") from exc

    for item in items:
        if not isinstance(item, MemoryCandidate):
            raise MemoryInvalidResult("Memory candidate has no valid identity.")
        if item.user_id != user_id or item.character_id != character_id:
            raise MemoryScopeError("Memory candidate scope does not match the authenticated resource.")
        if not isinstance(item.memory_id, str) or not item.memory_id:
            raise MemoryInvalidResult("Memory candidate has no valid identity.")
        if not isinstance(item.content, str) or not item.content.strip():
            raise MemoryInvalidResult("Memory candidate has no usable content.")
        if item.metadata is not None:
            if not isinstance(item.metadata, Mapping):
                raise MemoryInvalidResult("Memory candidate metadata is invalid.")
            if ("user_id" in item.metadata and item.metadata["user_id"] != user_id) or (
                "character_id" in item.metadata and item.metadata["character_id"] != character_id
            ):
                raise MemoryScopeError("Memory candidate metadata contradicts the authenticated scope.")
    return items


def retrieve_for_turn(*, user_id: str, character_id: str, conversation_id: str,
                      current_message: str) -> MemoryRetrievalResult:
    """Retrieve scoped candidates; degrade only recognized memory failures."""
    started = perf_counter()
    try:
        candidates = _validate_candidates(
            _provider_retrieve(
                user_id=user_id, character_id=character_id,
                conversation_id=conversation_id, current_message=current_message,
            ),
            user_id, character_id,
        )
    except (MemoryProviderError, MemoryConfigurationError, TimeoutError, ConnectionError) as exc:
        timed_out = isinstance(exc, TimeoutError)
        logger.warning("Memory retrieval failed: provider=%s error_type=%s timeout=%s",
                       PROVIDER_NAME, type(exc).__name__, timed_out)
        return MemoryRetrievalResult(
            candidates=(), provider=PROVIDER_NAME,
            latency_ms=(perf_counter() - started) * 1000,
            success=False, timed_out=timed_out, error_type=type(exc).__name__,
        )
    return MemoryRetrievalResult(
        candidates=candidates, provider=PROVIDER_NAME,
        latency_ms=(perf_counter() - started) * 1000, success=True,
    )


def _provider_record_completed_turn(*, user_id: str, character_id: str,
                                    conversation_id: str, user_message_id: str,
                                    assistant_message_id: str, user_message: str,
                                    assistant_message: str) -> None:
    """Source IDs preserve provenance; the SDK has no idempotency key."""
    if PROVIDER_NAME == "noop":
        return None
    return _memmachine_adapter().record_completed_turn(
        user_id=user_id, character_id=character_id, conversation_id=conversation_id,
        user_message_id=user_message_id, assistant_message_id=assistant_message_id,
        user_message=user_message, assistant_message=assistant_message,
    )


def record_completed_turn(*, user_id: str, character_id: str, conversation_id: str,
                          user_message_id: str, assistant_message_id: str,
                          user_message: str, assistant_message: str) -> bool:
    """Submit only a completed turn; recognized memory failures do not fail chat."""
    try:
        _provider_record_completed_turn(
            user_id=user_id, character_id=character_id, conversation_id=conversation_id,
            user_message_id=user_message_id, assistant_message_id=assistant_message_id,
            user_message=user_message, assistant_message=assistant_message,
        )
    except (MemoryProviderError, TimeoutError, ConnectionError) as exc:
        logger.warning("Memory recording failed: provider=%s error_type=%s",
                       PROVIDER_NAME, type(exc).__name__)
        return False
    return True


def _provider_delete_conversation(*, user_id: str, character_id: str, conversation_id: str) -> None:
    """Delete only this source conversation; unsupported scopes fail closed."""
    if PROVIDER_NAME == "noop":
        return None
    return _memmachine_adapter().delete_conversation(
        user_id=user_id, character_id=character_id, conversation_id=conversation_id,
    )


def _provider_delete_character(*, user_id: str, character_id: str) -> None:
    """Delete every memory in this character scope; unsupported scopes fail closed."""
    if PROVIDER_NAME == "noop":
        return None
    return _memmachine_adapter().delete_character(user_id=user_id, character_id=character_id)


def _provider_delete_user(*, user_id: str) -> None:
    """Delete all user memories; unsupported scopes fail closed."""
    if PROVIDER_NAME == "noop":
        return None
    return _memmachine_adapter().delete_user(user_id=user_id)


def _delete_memories(scope_type: DeletionScope, scope_id: str,
                     operation: Callable[[], None]) -> MemoryDeletionResult:
    """Not-found is success; known provider failures are explicit and retryable.

    No DB resource is deleted here. Character deletion commits a durable
    tombstone first; its worker uses this result to reconcile external state.
    """
    started = perf_counter()
    attempted = PROVIDER_NAME != "noop"
    try:
        # A provider adapter must return None only after confirmed completion.
        # False or an ambiguous receipt must never permit the DB delete.
        if operation() is not None:
            raise MemoryDeletionError("Ambiguous provider deletion result.", retryable=True)
    except MemoryDeletionNotFound:
        return MemoryDeletionResult(
            PROVIDER_NAME, scope_type, scope_id, attempted, True, True,
            (perf_counter() - started) * 1000,
        )
    except (MemoryDeletionError, MemoryProviderError, TimeoutError, ConnectionError) as exc:
        retryable = exc.retryable if isinstance(exc, MemoryDeletionError) else True
        partial = exc.partial if isinstance(exc, MemoryDeletionError) else False
        logger.warning(
            "Memory deletion failed: provider=%s scope=%s scope_id=%s error_type=%s retryable=%s partial=%s",
            PROVIDER_NAME, scope_type, scope_id, type(exc).__name__, retryable, partial,
        )
        return MemoryDeletionResult(
            PROVIDER_NAME, scope_type, scope_id, attempted, False, False,
            (perf_counter() - started) * 1000,
            retryable=retryable, partial=partial, error_type=type(exc).__name__,
        )
    return MemoryDeletionResult(
        PROVIDER_NAME, scope_type, scope_id, attempted, True, False,
        (perf_counter() - started) * 1000,
    )


def delete_conversation_memories(*, user_id: str, character_id: str,
                                 conversation_id: str) -> MemoryDeletionResult:
    """Delete only this source session, preserving other conversations' memories.

    The caller must authorize the conversation and its character first.
    """
    return _delete_memories(
        "conversation", conversation_id,
        lambda: _provider_delete_conversation(
            user_id=user_id, character_id=character_id, conversation_id=conversation_id,
        ),
    )


def delete_character_memories(*, user_id: str, character_id: str) -> MemoryDeletionResult:
    """Delete the entire (user, character) memory scope; caller must authorize."""
    return _delete_memories(
        "character", character_id,
        lambda: _provider_delete_character(user_id=user_id, character_id=character_id),
    )


def delete_user_memories(*, user_id: str) -> MemoryDeletionResult:
    """Future account cleanup across all characters owned by this user."""
    return _delete_memories(
        "user", user_id, lambda: _provider_delete_user(user_id=user_id),
    )
