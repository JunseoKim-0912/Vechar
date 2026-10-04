"""Select optional chat layers without truncating canonical state or the current turn."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fastapi import HTTPException

from ..chat_context_config import CHAT_CONTEXT_LIMITS, ChatContextLimits
from ..llm import input_size_bytes
from ..models import MessageRole
from .chat_prompt_builder import build_chat_input

TokenCounter = Callable[[str, list[dict]], int]


@dataclass(frozen=True)
class ChatBudgetMetadata:
    # Layer counts are incremental estimates from the same exact gateway counter.
    # The final total is counted over the fully serialized input.
    canonical_tokens: int
    memory_tokens: int
    memory_count: int
    history_tokens: int
    history_count: int
    current_message_tokens: int
    total_estimated_input_tokens: int
    dropped_memory_count: int
    dropped_history_message_count: int


@dataclass(frozen=True)
class ChatBudgetResult:
    input_messages: list[dict[str, str]]
    metadata: ChatBudgetMetadata


def _too_large(code: str) -> HTTPException:
    messages = {
        "chat_canonical_context_too_large": "Character/world profile and chat rules exceed the input limit.",
        "chat_current_message_too_large": "Current user message exceeds the input limit.",
        "chat_required_context_too_large": "Required chat context exceeds the input limit.",
    }
    return HTTPException(
        status_code=413,
        detail={"code": code, "message": messages[code]},
    )


def select_chat_context(
    instructions: str,
    recent_messages: Sequence[tuple[MessageRole, str]],
    current_user_message: str,
    *,
    count_input_tokens: TokenCounter,
    memories: Sequence[str] = (),
    limits: ChatContextLimits = CHAT_CONTEXT_LIMITS,
) -> ChatBudgetResult:
    """Keep the newest history suffix and highest-ranked whole memories.

    Memories must arrive in descending relevance order. Only the first
    ``memory_items`` candidates are inspected, bounding count calls even if a
    future retriever returns hundreds of items. Oversized items are excluded,
    never silently cut mid-sentence. The current turn is never in history.
    """
    # The count endpoint is documented for normal, nonempty messages. A small
    # probe lets us estimate layer deltas without submitting an empty user turn.
    probe_turn = build_chat_input([], ".")
    required_input = build_chat_input([], current_user_message)
    canonical_bytes = input_size_bytes(instructions, probe_turn)
    current_bytes = input_size_bytes("", required_input)
    required_bytes = input_size_bytes(instructions, required_input)
    if canonical_bytes > limits.input_bytes:
        raise _too_large("chat_canonical_context_too_large")
    if current_bytes > limits.input_bytes:
        raise _too_large("chat_current_message_too_large")
    if required_bytes > limits.input_bytes:
        raise _too_large("chat_required_context_too_large")

    canonical_tokens = count_input_tokens(instructions, probe_turn)
    if canonical_tokens > limits.input_tokens:
        raise _too_large("chat_canonical_context_too_large")
    required_tokens = count_input_tokens(instructions, required_input)
    # The probe contributes one content token; layer figures are estimates,
    # while required_tokens and the final request count are exact API counts.
    current_tokens = max(0, required_tokens - canonical_tokens + 1)
    if current_tokens > limits.input_tokens:
        raise _too_large("chat_current_message_too_large")
    if required_tokens > limits.input_tokens:
        raise _too_large("chat_required_context_too_large")

    included_memories: list[str] = []
    memory_total = required_tokens
    for memory in memories[:limits.memory_items]:
        item_input = build_chat_input([], current_user_message, memories=[memory])
        if input_size_bytes(instructions, item_input) > limits.input_bytes:
            continue
        item_total = count_input_tokens(instructions, item_input)
        item_tokens = item_total - required_tokens
        if item_tokens > limits.memory_item_tokens:
            continue

        candidate = [*included_memories, memory]
        candidate_input = build_chat_input([], current_user_message, memories=candidate)
        if input_size_bytes(instructions, candidate_input) > limits.input_bytes:
            continue
        candidate_total = item_total if not included_memories else count_input_tokens(instructions, candidate_input)
        if candidate_total - required_tokens > limits.memory_tokens or candidate_total > limits.input_tokens:
            continue
        included_memories = candidate
        memory_total = candidate_total

    # The DB supplies at most 30 chronological messages. A retained suffix
    # preserves the newest messages without assuming complete turn pairs.
    # Binary search avoids up to 30 provider count round-trips when a long
    # history must be trimmed. Adding complete messages increases the count.
    def measure_suffix(start: int) -> tuple[bool, list[dict[str, str]], int, int]:
        candidate_input = build_chat_input(
            recent_messages[start:], current_user_message, memories=included_memories
        )
        if input_size_bytes(instructions, candidate_input) > limits.input_bytes:
            return False, candidate_input, 0, 0
        candidate_tokens = count_input_tokens(instructions, candidate_input)
        candidate_history_tokens = max(0, candidate_tokens - memory_total)
        fits = candidate_tokens <= limits.input_tokens and candidate_history_tokens <= limits.history_tokens
        return fits, candidate_input, candidate_tokens, candidate_history_tokens

    start = 0
    if not recent_messages:
        final_input = build_chat_input([], current_user_message, memories=included_memories)
        total_tokens, history_tokens = memory_total, 0
    else:
        fits, final_input, total_tokens, history_tokens = measure_suffix(0)
        if not fits:
            low, high = 1, len(recent_messages)
            while low < high:
                middle = (low + high) // 2
                if measure_suffix(middle)[0]:
                    high = middle
                else:
                    low = middle + 1
            start = low
            fits, final_input, total_tokens, history_tokens = measure_suffix(start)
            if not fits:
                # Required layers and memory were checked above, so this is only
                # reachable for a broken/inconsistent counter.
                raise _too_large("chat_required_context_too_large")

    return ChatBudgetResult(
        input_messages=final_input,
        metadata=ChatBudgetMetadata(
            canonical_tokens=canonical_tokens,
            memory_tokens=max(0, memory_total - required_tokens),
            memory_count=len(included_memories),
            history_tokens=history_tokens,
            history_count=len(recent_messages) - start,
            current_message_tokens=current_tokens,
            total_estimated_input_tokens=total_tokens,
            dropped_memory_count=len(memories) - len(included_memories),
            dropped_history_message_count=start,
        ),
    )
