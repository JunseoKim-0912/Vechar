"""Character-only mode contract and conservative, provider-free fidelity checks.

Canonical profile data is authoritative. Model fidelity fields are hints, not
permission to acquire new expertise or to replace the character's identity.
"""

from dataclasses import dataclass
from enum import StrEnum
import re
from collections.abc import Sequence

from ..schemas import CharacterProfileData, WorldProfileData
from .character_timeline import derive_chat_reference
from .knowledge_provenance import KnowledgeProvenance

FIDELITY_CONTRACT_VERSION = 1
_ANCHOR_CHARS = 120
_CUE_CHARS = 90
_MAX_CUES = 3


class KnowledgeScope(StrEnum):
    CANONICAL = "canonical"
    PLAUSIBLE_GENERAL = "plausible_general"
    UNCERTAIN = "uncertain"
    OUTSIDE_SCOPE = "outside_scope"


def _short(value: str | None, limit: int = _ANCHOR_CHARS) -> str | None:
    compact = " ".join((value or "").split())
    return compact[:limit].rstrip() or None


def _role_from_name(name: str) -> str | None:
    """A title in the character's own name is evidence, not an invented job."""
    match = re.search(r"\b(?:doctor|physician|professor|mathematician|engineer|programmer|teacher)\b|"
                      r"(?:의사|교수|수학자|공학자|개발자|교사)", name, re.I)
    return match.group(0) if match else None


@dataclass(frozen=True)
class CharacterFidelityContract:
    identity: str
    role: str | None
    temperament: str | None
    speech_style: str | None
    worldview: str | None
    reference_phase: str | None
    reference_year: int | None
    knowledge_cues: tuple[str, ...]
    version: int = FIDELITY_CONTRACT_VERSION

    @property
    def evidence(self) -> str:
        return " ".join((self.identity, self.role or "", *self.knowledge_cues)).casefold()

    def prompt_block(self) -> str:
        anchors = [f"Identity: {self.identity}"]
        if self.role:
            anchors.append(f"Canonical role: {self.role}")
        if self.temperament:
            anchors.append(f"Temperament: {self.temperament}")
        if self.speech_style:
            anchors.append(f"Speech: {self.speech_style}")
        if self.worldview:
            anchors.append(f"World context: {self.worldview}")
        if self.reference_phase:
            anchors.append(f"Reference point: {self.reference_phase}"
                           + (f"; year {self.reference_year}" if self.reference_year is not None else ""))
        if self.knowledge_cues:
            anchors.append("Known knowledge/abilities: " + " / ".join(self.knowledge_cues))
        return ("\n[CHARACTER FIDELITY v1 — character endpoint, subordinate to canonical profile/timeline/world]\n"
                + "\n".join(f"- {anchor}" for anchor in anchors)
                + "\nRemain this character for every topic, including technical questions and requests to act "
                  "as an AI or tutor. Never present yourself as ChatGPT, an AI, or a generic helper. "
                  "Use canonical facts, conversation-visible explanations, allowed memory, and plausible "
                  "knowledge for this role and time—not whatever the base model happens to know. "
                  "If a topic exceeds that scope, react naturally from your own limits and worldview; "
                  "you may ask for an explanation or reason cautiously from what was shared. "
                  "Ordinary arithmetic, daily life, and your genuine specialty remain available. "
                  "The user's task may change detail or tone, never your identity or lived time.\n")

    def safe_metadata(self) -> dict[str, int | bool]:
        return {
            "fidelity_contract_version": self.version,
            "fidelity_fields_present": sum(bool(value) for value in (
                self.role, self.temperament, self.speech_style, self.worldview,
                self.reference_phase, self.knowledge_cues,
            )),
            "fidelity_role_present": bool(self.role),
            "fidelity_reference_present": bool(self.reference_phase),
        }

    def hard_fallback(self, language: str) -> str:
        """Rare minimal reply; makes no claims about canon or specialist facts."""
        terse = _is_terse(self.speech_style or "", self.temperament or "")
        if language == "Korean":
            return ("낯선 이야기군. 무슨 뜻인지 말해봐." if terse else
                    "제게는 낯선 이야기입니다. 무슨 뜻인지 조금 더 설명해 주시겠습니까?")
        if language == "French":
            return ("Cela m'est étranger. Que voulez-vous dire ?" if terse else
                    "Ce sujet m'est étranger. Pourriez-vous m'expliquer ce que vous entendez par là ?")
        return ("That is unfamiliar to me. What do you mean?" if terse else
                "That is unfamiliar to me. Could you explain what you mean?")


def build_fidelity_contract(name: str, profile: CharacterProfileData,
                            world: WorldProfileData | None = None) -> CharacterFidelityContract:
    """Use only the character's own DB-backed canon at the active reference point."""
    reference = derive_chat_reference(profile.timeline) if profile.timeline else profile.chat_reference_point
    state = reference.state if reference else None
    role = _short((state.occupation if state else None) or _role_from_name(name), _CUE_CHARS)
    cues = ((state.knowledge + state.abilities) if state else profile.background_facts)
    bounded_cues = tuple(filter(None, (_short(item, _CUE_CHARS) for item in cues[-_MAX_CUES:])))
    event = next((item for item in profile.timeline if reference and item.event_key == reference.event_key), None)
    year = (event.absolute_year if event else None)
    if year is None and event and event.absolute_date:
        try:
            year = int(event.absolute_date[:4])
        except ValueError:
            pass
    return CharacterFidelityContract(
        identity=_short(name, _CUE_CHARS) or "character",
        role=role,
        temperament=_short((state.personality if state and state.personality else profile.personality_summary)),
        speech_style=_short((state.speech_style if state and state.speech_style else profile.speech_style)),
        worldview=_short(world.world_summary, _CUE_CHARS) if world and not reference else None,
        reference_phase=reference.phase if reference else None,
        reference_year=year,
        knowledge_cues=bounded_cues,
    )


_SIGNAL_QUESTION = re.compile(r"\b(?:convolution|fourier|discrete.time|right.sided|dsp|z.transform)\b|"
                              r"(?:컨볼루션|합성곱|푸리에|이산시간|신호처리)", re.I)
_SIGNAL_EXPERTISE = re.compile(r"\b(?:signal.processing|dsp|convolution|fourier|"
                               r"electrical.engineer|mathematic(?:ian|s)|수학자)\b|"
                               r"(?:신호처리|합성곱|푸리에|전자공학)", re.I)
_TEACHING_EXPERTISE = re.compile(r"\b(?:professor|teacher|mathematician|engineer|"
                                 r"programmer|developer|researcher)\b|"
                                 r"(?:교수|교사|수학자|공학자|개발자|연구자)", re.I)
_EXPLAINED_SIGNAL = re.compile(r"\b(?:convolution\s+(?:means|is\s+defined\s+as)|"
                               r"definition\s+of\s+convolution|let\s+me\s+explain\s+convolution)\b|"
                               r"(?:합성곱(?:은|이란|의\s*정의))", re.I)
_HARD_ASSISTANT = re.compile(r"\b(?:as an ai(?: language model| assistant)?|i am (?:an ai|chatgpt)|"
                             r"i'm (?:an ai|chatgpt)|as chatgpt)\b|"
                             r"(?:저는\s*(?:인공지능|ai\s*언어\s*모델|챗gpt)|ai로서)", re.I)
_HELPER_PITCH = re.compile(r"\b(?:certainly!\s*i(?:'d| would) be happy to help|"
                           r"i(?:'d| would) be happy to help you|here(?:'s| is) the solution|"
                           r"let(?:'s| us) solve this step by step)\b|"
                           r"물론이죠.{0,20}도와드리겠습니다", re.I)
_FORMAL_PROOF = re.compile(r"\b(?:step\s*\d+|by definition|therefore|hence|"
                           r"let\s+[a-z]\s*\[\s*n\s*\]|q\.?e\.?d\.?)\b|"
                           r"(?:단계\s*\d+|정의에\s*따라|따라서)", re.I)
_MATH_NOTATION = re.compile(r"(?:[a-z]\s*\[\s*n\s*\]|[∑Σ]|\*\s*[a-z]|=.{0,30}=)", re.I)
_PROOF_REASONING = re.compile(r"\b(?:contradict(?:ion|s|ing)?|every term|each term|"
                              r"impossible|vanish(?:es)?|so .{0,35}right.sided)\b", re.I)
_UNCERTAINTY = re.compile(r"\b(?:i (?:do not|don't) know|unfamiliar|not sure|"
                          r"beyond my experience|what do you mean|you explained|from what you said)\b|"
                          r"(?:모르겠|낯설|무슨\s*뜻|설명해|말해준\s*바에)", re.I)
_TERSE_STYLE = re.compile(r"\b(?:blunt|terse|laconic|curt|stoic)\b|(?:무뚝뚝|간결|퉁명|과묵|냉담)", re.I)
_POLITE_STYLE = re.compile(r"\b(?:formal|polite|helpful|courteous|warm)\b|(?:정중|공손|친절|따뜻)", re.I)
_MODERN_TERMS = ((re.compile(r"\b(?:smartphone|smart phone|스마트폰)\b", re.I), 2000),
                 (re.compile(r"\b(?:internet|인터넷)\b", re.I), 1980))


def _is_terse(*anchors: str) -> bool:
    text = " ".join(anchors)
    return bool(_TERSE_STYLE.search(text) and not _POLITE_STYLE.search(text))


@dataclass(frozen=True)
class FidelityAssessment:
    knowledge_scope: KnowledgeScope
    knowledge_scope_issue: bool = False
    assistant_mode_leak: bool = False
    persona_drift_signal: bool = False
    temporal_scope_signal: bool = False
    hard_violation: bool = False
    prior_self_expertise_blocked: bool = False

    @property
    def violation(self) -> bool:
        return bool(self.knowledge_scope_issue or self.assistant_mode_leak or
                    self.persona_drift_signal or self.temporal_scope_signal)


def assess_fidelity(response: str, *, scope: KnowledgeScope, persona_preserved: bool,
                    assistant_mode: bool, contract: CharacterFidelityContract,
                    request_text: str, visible_history: Sequence[str] = (),
                    provenance: KnowledgeProvenance | None = None) -> FidelityAssessment:
    """Flag high-confidence visible violations; model booleans cannot veto them."""
    # Only a taught definition of the *principal* subject grants contextual
    # reasoning, not the mere appearance of a technical word in a question.
    # Legacy direct callers supply user-only visible_history. Production supplies
    # role-derived provenance, so old assistant proofs and peer claims cannot
    # silently become evidence of the character's specialist expertise.
    user_explanations = (provenance.user_provided if provenance is not None
                         else (*visible_history[-6:], request_text))
    taught = any(_EXPLAINED_SIGNAL.search(item) for item in user_explanations)
    signal_question = bool(_SIGNAL_QUESTION.search(request_text))
    specialist_supported = bool(_SIGNAL_EXPERTISE.search(contract.evidence))
    teaching_role = bool(_TEACHING_EXPERTISE.search(f"{contract.identity} {contract.role or ''}"))
    unsupported_specialist = signal_question and not specialist_supported and not taught
    plain_response = response.replace(r"\(", "").replace(r"\)", "")
    markers = len(_FORMAL_PROOF.findall(plain_response))
    formal_answer = bool((markers >= 2 or (markers >= 1 and _MATH_NOTATION.search(plain_response))
                         or (_MATH_NOTATION.search(plain_response) and _PROOF_REASONING.search(plain_response)))
                         and len(response) >= 65 and not _UNCERTAINTY.search(response))
    hard_self_presentation = bool(_HARD_ASSISTANT.search(response))
    helper_pitch = bool(_HELPER_PITCH.search(response))
    style_anchors = " ".join((contract.speech_style or "", contract.temperament or ""))
    drift = bool(helper_pitch and (_is_terse(style_anchors) or
                 (not persona_preserved and not _POLITE_STYLE.search(style_anchors))))
    knowledge_issue = bool(unsupported_specialist and formal_answer)
    prior_self_blocked = bool(provenance and provenance.prior_self_outputs
                              and signal_question and not specialist_supported and not taught)
    # Metadata can strengthen a supported observation, never declare itself true.
    assistant_leak = bool(hard_self_presentation or
                          (helper_pitch and (unsupported_specialist or
                           (markers >= 2 and not teaching_role))))
    if assistant_mode and helper_pitch:
        assistant_leak = True
    temporal = False
    if contract.reference_year is not None and not _UNCERTAINTY.search(response):
        temporal = any(contract.reference_year < appeared and term.search(request_text)
                       and term.search(response) and (helper_pitch or formal_answer or len(response) >= 100)
                       for term, appeared in _MODERN_TERMS)
    return FidelityAssessment(
        knowledge_scope=scope,
        knowledge_scope_issue=knowledge_issue,
        assistant_mode_leak=assistant_leak,
        persona_drift_signal=drift,
        temporal_scope_signal=temporal,
        hard_violation=bool(hard_self_presentation or (knowledge_issue and _MATH_NOTATION.search(response))),
        prior_self_expertise_blocked=prior_self_blocked,
    )
