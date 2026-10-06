"""Conservative, bounded lexical guard for obvious recent conversational loops."""

from difflib import SequenceMatcher
import re
from collections.abc import Sequence

from .chat_actions import normalize_assistant_actions

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
