"""Bounded conversational evidence with authority separate from character canon."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..models import Message, MessageRole


class Provenance(StrEnum):
    CANONICAL = "canonical"
    USER_PROVIDED = "user_provided"
    PEER_CLAIM = "peer_claim"
    PRIOR_SELF_OUTPUT = "prior_self_output"
    MEMORY = "memory"
    INFERRED = "inferred"


@dataclass(frozen=True)
class KnowledgeProvenance:
    """Only user teaching can authorize contextual learning; none changes canon."""

    user_provided: tuple[str, ...] = ()
    peer_claims: tuple[str, ...] = ()
    prior_self_outputs: tuple[str, ...] = ()
    memory_count: int = 0

    @classmethod
    def for_user_chat(cls, history: Sequence[Message], current_user_message: str,
                      memory_count: int = 0) -> "KnowledgeProvenance":
        return cls(
            user_provided=tuple(m.content for m in history if m.role == MessageRole.USER)[-6:]
                          + (current_user_message,),
            prior_self_outputs=tuple(m.content for m in history
                                     if m.role == MessageRole.CHARACTER)[-6:],
            memory_count=min(max(memory_count, 0), 6),
        )

    @classmethod
    def for_room(cls, history: Sequence[Message], speaker_id: str) -> "KnowledgeProvenance":
        return cls(
            peer_claims=tuple(m.content for m in history
                              if m.role == MessageRole.CHARACTER
                              and m.speaker_character_id != speaker_id)[-6:],
            prior_self_outputs=tuple(m.content for m in history
                                     if m.role == MessageRole.CHARACTER
                                     and m.speaker_character_id == speaker_id)[-6:],
        )

    def categories(self) -> tuple[str, ...]:
        present = [Provenance.CANONICAL.value, Provenance.INFERRED.value]
        if self.user_provided:
            present.append(Provenance.USER_PROVIDED.value)
        if self.peer_claims:
            present.append(Provenance.PEER_CLAIM.value)
        if self.prior_self_outputs:
            present.append(Provenance.PRIOR_SELF_OUTPUT.value)
        if self.memory_count:
            present.append(Provenance.MEMORY.value)
        return tuple(present)

    def safe_counts(self) -> dict[str, int]:
        return {
            "user_provided": min(len(self.user_provided), 7),
            "peer_claim": len(self.peer_claims),
            "prior_self_output": len(self.prior_self_outputs),
            "memory": self.memory_count,
        }

    def prompt_block(self) -> str:
        return ("\n[KNOWLEDGE PROVENANCE — below canon and fidelity]\n"
                "User explanations may be used in this conversation, never as pre-existing expertise. "
                "A peer's claim may be discussed but is not your canon. Your own earlier replies are "
                "PRIOR_SELF_OUTPUT: continuity only, not evidence that you know a subject or can prove it. "
                "Retrieved memory, if any, is unverified context. Preserve uncertainty and your role.\n")
