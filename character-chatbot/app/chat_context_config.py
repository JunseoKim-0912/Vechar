"""Initial ordinary-chat context policy; tune against real conversations later."""

from dataclasses import dataclass

from .llm_config import MAX_LLM_INPUT_BYTES


@dataclass(frozen=True)
class ChatContextLimits:
    input_tokens: int = 16000
    input_bytes: int = MAX_LLM_INPUT_BYTES  # The gateway remains the final guard.
    memory_tokens: int = 1500
    memory_items: int = 5
    memory_item_tokens: int = 500
    history_tokens: int = 6000


CHAT_CONTEXT_LIMITS = ChatContextLimits()
# Keep the existing fixed output cap: shorter user turns can still need a
# long, in-character reply, and billing follows tokens actually generated.
ROLEPLAY_MAX_TOKENS = 2000
