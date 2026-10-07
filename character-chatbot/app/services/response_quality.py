"""One shared, bounded post-generation quality pass for both chat modes."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

from .chat_actions import canonicalize_assistant_message, parse_assistant_actions
from .character_fidelity import (
    CharacterFidelityContract, KnowledgeScope, assess_fidelity,
)
from .conversation_loop_guard import is_obvious_loop, is_semantic_loop, semantic_signature
from .conversation_runtime import (
    RUNTIME_VERSION, ChatTurnResult, ConversationRuntimeState, empty_turn, phrase_families,
    phrase_family_hash, style_categories, validated_delta,
)
from .knowledge_provenance import KnowledgeProvenance

_URL = re.compile(r"https?://\S+", re.I)
_SHORT_QUOTE = re.compile(r"[\"“‘«][^\"”’»]{1,50}[\"”’»]")
_LATIN_CLAUSE = re.compile(r"\b[A-Za-zÀ-ÿ]+(?:[\s,;:—-]+[A-Za-zÀ-ÿ]+){5,}\b")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_HANGUL = re.compile(r"[가-힣]+")
_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_WORDS = re.compile(r"[\w가-힣]+", re.U)
_ENGLISH_FUNCTION = {"the", "and", "that", "this", "with", "from", "you", "your", "what", "where",
                     "have", "will", "would", "should", "could", "there", "because", "about", "into", "when"}
_FRENCH_FUNCTION = {"dans", "avec", "pour", "vous", "nous", "elle", "mais", "parce", "que", "des", "les",
                    "une", "est", "suis", "pas", "plus", "comme", "cette", "leur"}
_COMMON_STYLE = {"그래", "알았어", "한다", "해", "네", "응", "그렇다", "okay", "yes", "well", "right", "then"}
_REPEAT_REQUEST = re.compile(r"다시\s*(?:말|해|설명)|반복해|방금\s*한\s*말|정확히\s*뭐라고|repeat|say\s+that\s+again|quote\s+yourself", re.I)


class InvalidConversationTransitionError(RuntimeError):
    """Two bounded drafts failed a hard room transition; persist neither."""


def explicit_repetition_request(message: str) -> bool:
    return bool(_REPEAT_REQUEST.search(message))


def language_mismatch(content: str, language: str) -> bool:
    """Flag only substantial off-language clauses; names/loanwords are ignored."""
    if language not in {"Korean", "English", "French"}:
        return False
    plain = " ".join(segment["text"] for segment in parse_assistant_actions(content).segments)
    plain = _SHORT_QUOTE.sub(" ", _URL.sub(" ", plain))
    if _KANA.search(plain):
        return True
    if language in {"Korean", "English"} and len(_HAN.findall(plain)) >= 10:
        return True
    if language == "English":
        hangul = _HANGUL.findall(plain)
        if len(hangul) >= 3 and sum(map(len, hangul)) >= 9:
            return True
    if language == "Korean":
        for match in _LATIN_CLAUSE.finditer(plain):
            words = [word.casefold() for word in re.findall(r"[A-Za-zÀ-ÿ]+", match.group())]
            if sum(word in _ENGLISH_FUNCTION | _FRENCH_FUNCTION for word in words) >= 2:
                return True
    if language in {"English", "French"}:
        for match in _LATIN_CLAUSE.finditer(plain):
            words = [word.casefold() for word in re.findall(r"[A-Za-zÀ-ÿ]+", match.group())]
            english = sum(word in _ENGLISH_FUNCTION for word in words)
            french = sum(word in _FRENCH_FUNCTION for word in words)
            if language == "English" and french >= 3 and french > english * 2:
                return True
            if language == "French" and english >= 3 and english > french * 2:
                return True
        if language == "French":
            hangul = _HANGUL.findall(plain)
            if len(hangul) >= 3 and sum(map(len, hangul)) >= 9:
                return True
    return False


def _distinctive_phrases(content: str) -> set[str]:
    original = _WORDS.findall(content)
    words = [word.casefold() for word in original]
    phrases = set()
    for width in (2, 3, 4):
        for index in range(len(words) - width + 1):
            part = words[index:index + width]
            phrase = " ".join(part)
            # Names are continuity, not a verbal tic. An initial capital alone
            # may just begin a sentence, so only later title-case tokens count.
            if any(token.istitle() for token in original[index + 1:index + width]):
                continue
            if len(phrase) >= 5 and any(word not in _COMMON_STYLE for word in part):
                phrases.add(phrase)
    return phrases


def style_repetition(content: str, same_speaker_recent: Sequence[str]) -> bool:
    """Three-use cooldown for distinctive phrases or repeated openings/templates."""
    previous = same_speaker_recent[-4:]
    if len(previous) < 2:
        return False
    candidate = " ".join(segment["text"] for segment in parse_assistant_actions(content).segments
                         if segment["type"] == "dialogue")
    recent_dialogue = [" ".join(segment["text"] for segment in parse_assistant_actions(item).segments
                               if segment["type"] == "dialogue") for item in previous]
    phrases = _distinctive_phrases(candidate)
    for phrase in phrases:
        if sum(phrase in _distinctive_phrases(item) for item in recent_dialogue) >= 2:
            return True
    opening = _WORDS.findall(candidate.casefold())[:1]
    if opening and opening[0] in {"그래", "well", "listen"} and sum(
        _WORDS.findall(item.casefold())[:1] == opening for item in recent_dialogue
    ) >= 2:
        return True
    question = re.search(r"(?:^|[.!?]\s+)(왜|why)\b[^?]*\?\s*$", candidate, re.I)
    if question and sum(re.search(r"(?:^|[.!?]\s+)(왜|why)\b[^?]*\?\s*$", item, re.I) is not None
                        for item in recent_dialogue) >= 2:
        return True
    actions = [segment["text"].casefold() for segment in parse_assistant_actions(content).segments
               if segment["type"] == "action"]
    if actions and any(sum(action in [segment["text"].casefold()
                                   for segment in parse_assistant_actions(item).segments
                                   if segment["type"] == "action"] for item in previous) >= 2
                       for action in actions):
        return True
    return False


@dataclass(frozen=True)
class QualitySignals:
    semantic_loop_detected: bool
    style_repetition_detected: bool
    language_mismatch_detected: bool
    action_format_repaired: bool
    action_subject_normalized: bool
    progression_failure: bool = False
    catchphrase_cooldown_triggered: bool = False
    unicode_script_mismatch: bool = False
    action_first_person_violation: bool = False
    progression_detected: bool = False
    phrase_family_hash: str = ""
    knowledge_scope: str = KnowledgeScope.UNCERTAIN.value
    knowledge_scope_issue: bool = False
    assistant_mode_leak: bool = False
    persona_drift_signal: bool = False
    temporal_scope_signal: bool = False
    hard_fidelity_violation: bool = False
    invalid_thread_transition: bool = False
    resolved_thread_reuse: bool = False
    exhausted_thread_reuse: bool = False
    topic_budget_exhausted: bool = False
    active_thread_present: bool = False
    thread_transition_type: str = "hold"
    thread_reopened: bool = False
    style_category_cooldown: bool = False
    prior_self_expertise_blocked: bool = False
    provenance_categories: tuple[str, ...] = ()

    @property
    def fidelity_violation(self) -> bool:
        return bool(self.knowledge_scope_issue or self.assistant_mode_leak or
                    self.persona_drift_signal or self.temporal_scope_signal)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(name for name, active in (
            ("semantic_loop", self.semantic_loop_detected),
            ("style_repetition", self.style_repetition_detected),
            ("language_mismatch", self.language_mismatch_detected),
            ("progression_failure", self.progression_failure),
            ("invalid_thread_transition", self.invalid_thread_transition),
            ("resolved_thread_reuse", self.resolved_thread_reuse),
            ("exhausted_thread_reuse", self.exhausted_thread_reuse),
            ("catchphrase_cooldown", self.catchphrase_cooldown_triggered),
            ("style_category_cooldown", self.style_category_cooldown),
            ("action_first_person", self.action_first_person_violation),
            ("knowledge_scope", self.knowledge_scope_issue),
            ("assistant_mode_leak", self.assistant_mode_leak),
            ("persona_drift", self.persona_drift_signal),
            ("temporal_scope", self.temporal_scope_signal),
        ) if active)


@dataclass(frozen=True)
class QualityOutcome:
    text: str
    initial: QualitySignals
    final: QualitySignals
    retry_count: int
    retry_fallback: bool
    turn: ChatTurnResult
    hard_fidelity_fallback: bool = False

    def safe_metadata(self, conversation_type: str, language: str) -> dict:
        return {
            "conversation_type": conversation_type,
            "response_language": language,
            "semantic_loop_detected": self.initial.semantic_loop_detected,
            "style_repetition_detected": self.initial.style_repetition_detected,
            "language_mismatch_detected": self.initial.language_mismatch_detected,
            "action_format_repaired": self.initial.action_format_repaired or self.final.action_format_repaired,
            "action_subject_normalized": self.initial.action_subject_normalized or self.final.action_subject_normalized,
            "corrective_retry": bool(self.retry_count),
            "retry_count": self.retry_count,
            "retry_reason_categories": ",".join(self.initial.reasons) or "none",
            "retry_success": bool(self.retry_count and not self.final.reasons and not self.retry_fallback),
            "retry_fallback": self.retry_fallback,
            "retry_quality_unresolved": bool(self.retry_count and self.final.reasons),
            "conversation_runtime_version": RUNTIME_VERSION,
            "progression_detected": self.final.progression_detected,
            "progression_failure": self.initial.progression_failure,
            "catchphrase_cooldown_triggered": self.initial.catchphrase_cooldown_triggered,
            "phrase_family_hash": self.initial.phrase_family_hash,
            "unicode_script_mismatch": self.initial.unicode_script_mismatch,
            "action_first_person_violation": self.initial.action_first_person_violation,
            "knowledge_scope": self.initial.knowledge_scope,
            "fidelity_violation": self.initial.fidelity_violation,
            "assistant_mode_leak": self.initial.assistant_mode_leak,
            "persona_drift_signal": self.initial.persona_drift_signal,
            "temporal_scope_signal": self.initial.temporal_scope_signal,
            "fidelity_retry_reason": ",".join(reason for reason in self.initial.reasons if reason in {
                "knowledge_scope", "assistant_mode_leak", "persona_drift", "temporal_scope",
            }) or "none",
            "hard_fidelity_fallback": self.hard_fidelity_fallback,
            "active_thread_present": self.final.active_thread_present,
            "thread_transition_type": self.final.thread_transition_type,
            "resolved_thread_reuse_detected": self.initial.resolved_thread_reuse,
            "exhausted_thread_reuse_detected": self.initial.exhausted_thread_reuse,
            "thread_reopened": self.final.thread_reopened,
            "topic_budget_exhausted": self.final.topic_budget_exhausted,
            "provenance_categories_used": ",".join(self.final.provenance_categories),
            "prior_self_expertise_blocked": self.initial.prior_self_expertise_blocked,
            "style_category_cooldown": self.initial.style_category_cooldown,
            "final_quality_status": "fallback" if self.retry_fallback else
                                    "accepted_after_retry" if self.retry_count else "accepted",
        }


def _as_turn(value: ChatTurnResult | str) -> ChatTurnResult:
    return value if isinstance(value, ChatTurnResult) else empty_turn(value)


def _assess(value: ChatTurnResult | str, same_speaker_recent: Sequence[str],
            recent_visible: Sequence[str], language: str, *, allow_repetition: bool,
            runtime_state: ConversationRuntimeState | None, speaker: str,
            room: bool, fidelity_contract: CharacterFidelityContract | None,
            request_text: str, visible_history: Sequence[str],
            provenance: KnowledgeProvenance | None) -> tuple[str, QualitySignals, ChatTurnResult]:
    turn = _as_turn(value)
    canonical, parsed = canonicalize_assistant_message(turn.response)
    turn = turn.model_copy(update={"response": canonical})
    semantic = False if allow_repetition else (
        is_obvious_loop(canonical, recent_visible[-4:])
        or is_semantic_loop(canonical, same_speaker_recent[-5:])
    )
    delta = validated_delta(runtime_state, turn) if runtime_state else {"progression": False}
    signature = semantic_signature(canonical)
    settled = bool(runtime_state and signature and signature[0] in runtime_state.established_points)
    resolved_advice = bool(delta.get("resolved_reuse"))
    progression_failure = bool(room and runtime_state and not allow_repetition and
                               not delta["progression"] and
                               (turn.progression.repeated_point or settled or resolved_advice
                                or (runtime_state.threads and not runtime_state.select_active_thread())))
    families = phrase_families(canonical)
    cooled = families & runtime_state.active_families(speaker) if runtime_state and not allow_repetition else set()
    categories = style_categories(canonical)
    cooled_categories = (categories & runtime_state.active_style_categories(speaker)
                         if runtime_state and not allow_repetition else set())
    action_first_person = language == "English" and any(
        re.match(r"^I\s+", part["text"], re.I) for part in parsed.segments if part["type"] == "action"
    )
    fidelity = assess_fidelity(
        canonical, scope=turn.fidelity.knowledge_scope,
        persona_preserved=turn.fidelity.persona_preserved,
        assistant_mode=turn.fidelity.assistant_mode,
        contract=fidelity_contract, request_text=request_text, visible_history=visible_history,
        provenance=provenance,
    ) if fidelity_contract else None
    signals = QualitySignals(
        semantic_loop_detected=semantic,
        style_repetition_detected=False if allow_repetition else style_repetition(canonical, same_speaker_recent),
        language_mismatch_detected=language_mismatch(canonical, language),
        action_format_repaired=parsed.format_repaired,
        action_subject_normalized=parsed.subject_normalized,
        progression_failure=progression_failure,
        catchphrase_cooldown_triggered=bool(cooled),
        unicode_script_mismatch=bool(_KANA.search(canonical)),
        action_first_person_violation=bool(action_first_person),
        progression_detected=bool(delta["progression"]),
        phrase_family_hash=phrase_family_hash(sorted(cooled)[0]) if cooled else "",
        knowledge_scope=fidelity.knowledge_scope.value if fidelity else turn.fidelity.knowledge_scope.value,
        knowledge_scope_issue=fidelity.knowledge_scope_issue if fidelity else False,
        assistant_mode_leak=fidelity.assistant_mode_leak if fidelity else False,
        persona_drift_signal=fidelity.persona_drift_signal if fidelity else False,
        temporal_scope_signal=fidelity.temporal_scope_signal if fidelity else False,
        hard_fidelity_violation=fidelity.hard_violation if fidelity else False,
        invalid_thread_transition=bool(room and delta.get("invalid_transition")),
        resolved_thread_reuse=bool(room and delta.get("resolved_reuse")),
        exhausted_thread_reuse=bool(room and delta.get("exhausted_reuse")),
        topic_budget_exhausted=bool(room and delta.get("topic_budget_exhausted")),
        active_thread_present=bool(runtime_state and runtime_state.select_active_thread()),
        thread_transition_type=str(delta.get("transition", "hold")),
        thread_reopened=bool(delta.get("reopened")),
        style_category_cooldown=bool(cooled_categories),
        prior_self_expertise_blocked=fidelity.prior_self_expertise_blocked if fidelity else False,
        provenance_categories=provenance.categories() if provenance else (),
    )
    return canonical, signals, turn


def _correction(reasons: tuple[str, ...], language: str,
                runtime_state: ConversationRuntimeState | None, speaker: str) -> str:
    clauses = []
    if "semantic_loop" in reasons:
        clauses.append("Advance an unresolved thread or add a new canon-consistent consequence; do not restate settled desires, advice, or motifs")
    if "style_repetition" in reasons:
        clauses.append("Vary distinctive phrases, opening/closing templates, and action wording while preserving character voice")
    if "language_mismatch" in reasons:
        clauses.append(f"Respond entirely in {language}, including dialogue and actions; no unexpected Japanese kana; proper nouns may remain unchanged")
    if "progression_failure" in reasons:
        clauses.append("Resolve or open a genuinely different thread or show a consequence; do not re-offer settled advice")
    if any(reason in reasons for reason in ("invalid_thread_transition", "resolved_thread_reuse",
                                            "exhausted_thread_reuse")):
        clauses.append("Follow the single ACTIVE thread or a canon-related new consequence; do not reactivate a resolved or exhausted thread without a visible new event")
    if "catchphrase_cooldown" in reasons:
        clauses.append("Omit the cooled catchphrase family entirely; express the same voice with different wording")
    if "style_category_cooldown" in reasons:
        clauses.append("Avoid the cooled style category, including synonym insults; preserve the same character's playful or blunt voice without that device")
    if "action_first_person" in reasons:
        clauses.append("Write English actions as grammatical third-person stage directions, e.g. 'Lowers his gaze.', not 'I lower my gaze.'")
    if "knowledge_scope" in reasons:
        clauses.append("The draft claimed specialist knowledge unsupported by this character's role, era, or canon. Respond from the character's actual limits; use only conversation-provided explanations for cautious reasoning, not a textbook solution")
    if "assistant_mode_leak" in reasons or "persona_drift" in reasons:
        clauses.append("Stay fully in the canonical character's own voice; no AI self-presentation, generic tutor, or customer-support framing")
    if "temporal_scope" in reasons:
        clauses.append("Do not claim knowledge of inventions or events beyond the canonical reference point; respond naturally to what is unfamiliar")
    state = runtime_state.prompt_block(speaker, room=True) if runtime_state else ""
    return "\n[ONE CORRECTIVE RETRY]\n" + state + ". ".join(clauses) + ". Do not restart the scene."


def select_quality_response(
    first_response: ChatTurnResult | str,
    *,
    same_speaker_recent: Sequence[str],
    recent_visible: Sequence[str] = (),
    language: str,
    allow_repetition: bool,
    retry: Callable[[str], ChatTurnResult | str],
    runtime_state: ConversationRuntimeState | None = None,
    speaker: str = "character",
    room: bool = False,
    retry_allowed: bool = True,
    fidelity_contract: CharacterFidelityContract | None = None,
    request_text: str = "",
    visible_history: Sequence[str] = (),
    provenance: KnowledgeProvenance | None = None,
) -> QualityOutcome:
    """Never call the provider more than once after the initial response."""
    first, initial, first_turn = _assess(
        first_response, same_speaker_recent, recent_visible or same_speaker_recent, language,
        allow_repetition=allow_repetition, runtime_state=runtime_state, speaker=speaker, room=room,
        fidelity_contract=fidelity_contract, request_text=request_text, visible_history=visible_history,
        provenance=provenance,
    )
    def hard_fallback() -> QualityOutcome:
        safe = fidelity_contract.hard_fallback(language) if fidelity_contract else first
        canonical, safe_signals, safe_turn = _assess(
            empty_turn(safe), same_speaker_recent, recent_visible or same_speaker_recent, language,
            allow_repetition=allow_repetition, runtime_state=runtime_state, speaker=speaker, room=room,
            fidelity_contract=fidelity_contract, request_text=request_text, visible_history=visible_history,
            provenance=provenance,
        )
        if room and safe_signals.invalid_thread_transition:
            raise InvalidConversationTransitionError("No valid room transition")
        return QualityOutcome(canonical, initial, safe_signals, 0 if not retry_allowed else 1,
                              True, safe_turn, True)

    if not initial.reasons or not retry_allowed:
        if room and initial.invalid_thread_transition:
            raise InvalidConversationTransitionError("No valid room transition")
        if initial.hard_fidelity_violation:
            return hard_fallback()
        return QualityOutcome(first, initial, initial, 0, False, first_turn)
    try:
        second_raw = retry(_correction(initial.reasons, language, runtime_state, speaker))
        second, final, second_turn = _assess(
            second_raw, same_speaker_recent, recent_visible or same_speaker_recent, language,
            allow_repetition=allow_repetition, runtime_state=runtime_state, speaker=speaker, room=room,
            fidelity_contract=fidelity_contract, request_text=request_text, visible_history=visible_history,
            provenance=provenance,
        )
        if room and final.invalid_thread_transition:
            raise InvalidConversationTransitionError("No valid room transition after retry")
        if final.hard_fidelity_violation:
            return hard_fallback() if initial.hard_fidelity_violation else QualityOutcome(
                first, initial, initial, 1, True, first_turn,
            )
        if initial.hard_fidelity_violation:
            if final.fidelity_violation or final.language_mismatch_detected:
                return hard_fallback()
            return QualityOutcome(second, initial, final, 1, False, second_turn)
        # Bounded fallback: never accept a retry that worsens deterministic safety.
        if len(final.reasons) > len(initial.reasons) or set(final.reasons) - set(initial.reasons):
            if room and initial.invalid_thread_transition:
                raise InvalidConversationTransitionError("No safe room transition after retry")
            return QualityOutcome(first, initial, initial, 1, True, first_turn)
        return QualityOutcome(second, initial, final, 1, False, second_turn)
    except InvalidConversationTransitionError:
        raise
    except Exception:
        if room and initial.invalid_thread_transition:
            raise InvalidConversationTransitionError("Room transition retry failed") from None
        if initial.hard_fidelity_violation:
            return hard_fallback()
        return QualityOutcome(first, initial, initial, 1, True, first_turn)
