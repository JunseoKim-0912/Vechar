"""Conservative lexical and proposition-level guards for short-window loops."""

from difflib import SequenceMatcher
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .chat_actions import normalize_assistant_actions, parse_assistant_actions

_FILLER = re.compile(
    r"\b(?:can you|could you|would you|please|again|tell me|tell us|about|as i said|"
    r"what about|how about|do you|did you|je voudrais|est-ce que)\b", re.I,
)
_STOP = frozenset(
    "a an and are as at be but by for from had has have i in is it me my of on or our "
    "she that the their them there they this to was were what when where who why will with you your "
    "we he his her do did can could would about tell again please et le la les de des du je tu il elle nous vous".split()
)


def _normalized(text: str) -> str:
    text = normalize_assistant_actions(text).casefold()
    text = _FILLER.sub(" ", text)
    return " ".join(re.findall(r"[\w가-힣]+", text))


def _content_words(text: str) -> set[str]:
    words = set()
    for word in text.split():
        if word in _STOP or len(word) < 3:
            continue
        if word.endswith("ing") and len(word) > 6:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 5:
            word = word[:-2]
        words.add(word)
    return words


def is_obvious_loop(candidate: str, recent_messages: Sequence[str]) -> bool:
    """Only the last four turns matter; older natural callbacks stay possible."""
    current = _normalized(candidate)
    if len(current) < 8 or len(candidate.strip()) < 16:
        return False
    current_words = _content_words(current)
    for previous in recent_messages[-4:]:
        earlier = _normalized(previous)
        if not earlier:
            continue
        if current == earlier:
            return True
        if min(len(current), len(earlier)) >= 20 and SequenceMatcher(
            None, current, earlier, autojunk=False,
        ).ratio() >= 0.88:
            return True
        earlier_words = _content_words(earlier)
        if (len(current_words) >= 3 and len(earlier_words) >= 3
                and max(len(current), len(earlier)) <= 2 * min(len(current), len(earlier))
                and len(current_words & earlier_words) / len(current_words | earlier_words) >= 0.85):
            return True
    return False


_DESIRE = re.compile(r"\b(?:want|wish|long|desire|hope|yearn)\b|원하|바라|싶", re.I)
_NEAR = re.compile(r"\b(?:near|nearer|close|closer|proximity)\b|가까", re.I)
_DISCLOSE = re.compile(r"\b(?:tell|say|speak|confess|voice|express)\b|말하|말해|전하|고백", re.I)
_HIDE = re.compile(r"\b(?:hide|conceal|hold back)\b|숨기|감추", re.I)
_SPATIAL = re.compile(r"\b(?:away|far|distance|snow)\b|멀리|거리|눈", re.I)
_OBSTACLE = re.compile(r"\b(?:keeps? me|makes? me|cannot|can't|unable|helpless|away|far)\b|막|못하|무력|멀리", re.I)
_PROGRESSED = re.compile(r"\b(?:called|told|answered|heard|replied|decided|went|arrived|returned|changed)\b|불렀|대답했|들었|결정했|도착했", re.I)
_MOTIFS = ("snow", "distance", "helpless", "family", "music", "violin", "heat", "death",
           "눈", "거리", "가족", "음악", "바이올린", "더위", "죽음")
_NAME = re.compile(r"\b[A-Z][a-z]{3,}\b")
_NAME_STOP = {"Tell", "Just", "Then", "What", "When", "Where", "There", "Your", "This", "That", "The"}


def _anchors(text: str) -> frozenset[str]:
    return frozenset(word.casefold() for word in _NAME.findall(text)
                     if word not in _NAME_STOP and word.casefold() not in _MOTIFS)


def semantic_signature(text: str) -> tuple[str, frozenset[str]] | None:
    """Only clear recurrent conversational points receive a signature.

    The handful of generic patterns target desires, repeated disclosure advice,
    and inability-to-reach motifs. Unknown content is deliberately left alone.
    """
    plain = normalize_assistant_actions(text)
    if _PROGRESSED.search(plain) and not _DESIRE.search(plain):
        return None
    anchors = _anchors(plain)
    if _DESIRE.search(plain) and _NEAR.search(plain):
        return "desire:proximity", anchors
    if (_HIDE.search(plain) or _DISCLOSE.search(plain) and re.search(
        r"\b(?:what|feel|want|truth|secret|heart)\b|마음|원하|진실", plain, re.I,
    )):
        return "advice:disclosure", anchors
    if _SPATIAL.search(plain) and _OBSTACLE.search(plain):
        return "constraint:access", anchors
    return None


def is_semantic_loop(candidate: str, same_speaker_recent: Sequence[str]) -> bool:
    current = semantic_signature(candidate)
    if current is None:
        return False
    point, anchors = current
    for previous in same_speaker_recent[-5:]:
        earlier = semantic_signature(previous)
        if earlier is None or earlier[0] != point:
            continue
        prior_anchors = earlier[1]
        if anchors and prior_anchors and not (anchors & prior_anchors):
            continue
        return True
    return False


@dataclass(frozen=True)
class ConversationState:
    established_points: tuple[str, ...]
    resolved_points: tuple[str, ...]
    open_threads: tuple[str, ...]
    recent_actions: tuple[str, ...]
    recent_questions: tuple[str, ...]
    recent_advice: tuple[str, ...]
    recent_character_motifs: tuple[str, ...]

    def prompt_block(self) -> str:
        fields = (
            ("Established point tags", self.established_points),
            ("Recently acted-on/resolved signals", self.resolved_points),
            ("Potentially open threads", self.open_threads),
            ("Recent action signals", self.recent_actions),
            ("Recent question signals", self.recent_questions),
            ("Recent advice tags", self.recent_advice),
            ("Recent motifs", self.recent_character_motifs),
        )
        lines = [f"- {name}: {', '.join(values) if values else 'none detected'}" for name, values in fields]
        return ("\n[RECENT CONVERSATION STATE — approximate tags, not new canon]\n"
                + "\n".join(lines)
                + "\nUse the actual recent messages to resolve these tags; do not invent facts from them.\n")


def derive_conversation_state(recent_messages: Sequence[str]) -> ConversationState:
    """Bounded, content-free-in-logs tags computed from visible recent turns."""
    recent = recent_messages[-12:]
    signatures = [semantic_signature(message) for message in recent]
    points = tuple(dict.fromkeys(sig[0] for sig in signatures if sig))[-4:]
    resolved = tuple("follow-up action taken" for message in recent[-4:]
                     if _PROGRESSED.search(normalize_assistant_actions(message)))[:2]
    open_threads = ("latest turn asks a question",) if recent and "?" in recent[-1] else ()
    actions = tuple("physical action present" for message in recent[-4:]
                    if any(seg["type"] == "action" for seg in parse_assistant_actions(message).segments))[:2]
    questions = tuple("recent question" for message in recent[-4:] if "?" in message)[:3]
    advice = tuple(dict.fromkeys(sig[0] for sig in signatures[-6:]
                                  if sig and sig[0].startswith("advice:")))
    motifs = tuple(motif for motif in _MOTIFS if any(
        motif in normalize_assistant_actions(message).casefold() for message in recent[-6:]
    ))[:5]
    return ConversationState(points, resolved, open_threads, actions, questions, advice, motifs)
