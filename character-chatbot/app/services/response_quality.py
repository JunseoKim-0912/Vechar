"""One shared, bounded post-generation quality pass for both chat modes."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

from .chat_actions import canonicalize_assistant_message, parse_assistant_actions
from .conversation_loop_guard import is_obvious_loop, is_semantic_loop

_URL = re.compile(r"https?://\S+", re.I)
_SHORT_QUOTE = re.compile(r"[\"“‘«][^\"”’»]{1,50}[\"”’»]")
_LATIN_CLAUSE = re.compile(r"\b[A-Za-zÀ-ÿ]+(?:[\s,;:—-]+[A-Za-zÀ-ÿ]+){5,}\b")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_HANGUL = re.compile(r"[가-힣]+")
_WORDS = re.compile(r"[\w가-힣]+", re.U)
_ENGLISH_FUNCTION = {"the", "and", "that", "this", "with", "from", "you", "your", "what", "where",
                     "have", "will", "would", "should", "could", "there", "because", "about", "into", "when"}
_FRENCH_FUNCTION = {"dans", "avec", "pour", "vous", "nous", "elle", "mais", "parce", "que", "des", "les",
                    "une", "est", "suis", "pas", "plus", "comme", "cette", "leur"}
_COMMON_STYLE = {"그래", "알았어", "한다", "해", "네", "응", "그렇다", "okay", "yes", "well", "right", "then"}
_REPEAT_REQUEST = re.compile(r"다시\s*(?:말|해|설명)|반복해|방금\s*한\s*말|정확히\s*뭐라고|repeat|say\s+that\s+again|quote\s+yourself", re.I)


def explicit_repetition_request(message: str) -> bool:
    return bool(_REPEAT_REQUEST.search(message))


def language_mismatch(content: str, language: str) -> bool:
    """Flag only substantial off-language clauses; names/loanwords are ignored."""
    if language not in {"Korean", "English", "French"}:
        return False
    plain = " ".join(segment["text"] for segment in parse_assistant_actions(content).segments)
    plain = _SHORT_QUOTE.sub(" ", _URL.sub(" ", plain))
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

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(name for name, active in (
            ("semantic_loop", self.semantic_loop_detected),
            ("style_repetition", self.style_repetition_detected),
            ("language_mismatch", self.language_mismatch_detected),
        ) if active)


@dataclass(frozen=True)
class QualityOutcome:
    text: str
    initial: QualitySignals
    final: QualitySignals
    retry_count: int
    retry_fallback: bool

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
        }


def _assess(content: str, same_speaker_recent: Sequence[str], recent_visible: Sequence[str], language: str,
            *, allow_repetition: bool) -> tuple[str, QualitySignals]:
    canonical, parsed = canonicalize_assistant_message(content)
    semantic = False if allow_repetition else (
        is_obvious_loop(canonical, recent_visible[-4:])
        or is_semantic_loop(canonical, same_speaker_recent[-5:])
    )
    signals = QualitySignals(
        semantic_loop_detected=semantic,
        style_repetition_detected=False if allow_repetition else style_repetition(canonical, same_speaker_recent),
        language_mismatch_detected=language_mismatch(canonical, language),
        action_format_repaired=parsed.format_repaired,
        action_subject_normalized=parsed.subject_normalized,
    )
    return canonical, signals


def _correction(reasons: tuple[str, ...], language: str) -> str:
    clauses = []
    if "semantic_loop" in reasons:
        clauses.append("Advance an unresolved thread or add a new canon-consistent consequence; do not restate settled desires, advice, or motifs")
    if "style_repetition" in reasons:
        clauses.append("Vary distinctive phrases, opening/closing templates, and action wording while preserving character voice")
    if "language_mismatch" in reasons:
        clauses.append(f"Respond entirely in {language}, including dialogue and actions; proper nouns may remain unchanged")
    return "\n[ONE CORRECTIVE RETRY]\n" + ". ".join(clauses) + ". Do not restart the scene."


def select_quality_response(
    first_response: str,
    *,
    same_speaker_recent: Sequence[str],
    recent_visible: Sequence[str] = (),
    language: str,
    allow_repetition: bool,
    retry: Callable[[str], str],
) -> QualityOutcome:
    """Never call the provider more than once after the initial response."""
    first, initial = _assess(first_response, same_speaker_recent, recent_visible or same_speaker_recent, language,
                             allow_repetition=allow_repetition)
    if not initial.reasons:
        return QualityOutcome(first, initial, initial, 0, False)
    try:
        second_raw = retry(_correction(initial.reasons, language))
        second, final = _assess(second_raw, same_speaker_recent, recent_visible or same_speaker_recent, language,
                                allow_repetition=allow_repetition)
        return QualityOutcome(second, initial, final, 1, False)
    except Exception:
        return QualityOutcome(first, initial, initial, 1, True)
