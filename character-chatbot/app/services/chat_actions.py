"""Tolerant action parsing for completed assistant messages.

The DB/API retains its historical string shape. New output is canonicalized to
<action> blocks; typed segments form the internal parsing boundary. Partial
streaming buffers can be parsed without crashing or discarding the message.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionParseResult:
    segments: tuple[dict[str, str], ...]
    format_repaired: bool = False
    subject_normalized: bool = False


_OPEN = re.compile(r"<\s*action\s*>", re.I)
_CLOSE = re.compile(r"<\s*/\s*action\s*>", re.I)
_PAIR = re.compile(r"<action>(.*?)</action>", re.I | re.S)
_INTERNAL = re.compile(r"\[\s*Character\s+action\s*:\s*(.*?)\]", re.I | re.S)
_INTERNAL_FINAL = re.compile(r"\[\s*Character\s+action\s*:\s*([^\]]+)$", re.I | re.S)
_NESTED = re.compile(r"<action>\s*<action>(.*?)</action>\s*</action>", re.I | re.S)
_KOREAN_SAFE = re.compile(
    r"^(?:나는|내가|난)\s+((?:고개를 숙인다|창밖을 바라본다|손등으로 눈물을 훔친다|"
    r"한숨을 쉰다|시선을 돌린다|어깨를 으쓱한다)(?:[.!?…]*)?)$"
)
_ENGLISH_SAFE = re.compile(r"^I\s+(look|turn|nod|smile|glance|raise|bow|pause|step)\b(.*)$", re.I | re.S)
_ENGLISH_VERBS = {
    "look": "Looks", "turn": "Turns", "nod": "Nods", "smile": "Smiles",
    "glance": "Glances", "raise": "Raises", "bow": "Bows", "pause": "Pauses", "step": "Steps",
}


def _subjectless(action: str) -> tuple[str, bool]:
    korean = _KOREAN_SAFE.fullmatch(action)
    if korean:
        return korean.group(1), True
    english = _ENGLISH_SAFE.fullmatch(action)
    if english and not re.search(r"\b(?:my|mine|myself)\b", english.group(2), re.I):
        return _ENGLISH_VERBS[english.group(1).lower()] + english.group(2), True
    return action, False


def parse_assistant_actions(content: str) -> ActionParseResult:
    """Repair only known wrappers; unknown HTML stays escaped dialogue in the UI."""
    text = content or ""
    repaired = False
    subject_changed = False

    converted = _INTERNAL.sub(lambda m: f"<action>{m.group(1)}</action>", text)
    if converted != text:
        repaired = True
    text = converted
    converted = _INTERNAL_FINAL.sub(lambda m: f"<action>{m.group(1)}</action>", text)
    if converted != text:
        repaired = True
    text = converted
    converted = _OPEN.sub("<action>", text)
    converted = _CLOSE.sub("</action>", converted)
    if converted != text:
        repaired = True
    text = converted
    for _ in range(3):
        converted = _NESTED.sub(lambda m: f"<action>{m.group(1)}</action>", text)
        if converted == text:
            break
        text, repaired = converted, True
    if re.search(r"</\s*action(?![\w>])", text, re.I):
        text = re.sub(r"</\s*action(?![\w>])", "</action>", text, flags=re.I)
        repaired = True
    if (text.count("<action>") > text.count("</action>")
            and re.search(r"[.!?…]\s*$", text)):
        text += "</action>"
        repaired = True

    segments: list[dict[str, str]] = []
    cursor = 0
    for match in _PAIR.finditer(text):
        if match.start() > cursor:
            dialogue = text[cursor:match.start()]
            if dialogue:
                segments.append({"type": "dialogue", "text": dialogue})
        action = match.group(1).strip()
        if action:
            if action != match.group(1):
                repaired = True
            action, changed = _subjectless(action)
            subject_changed |= changed
            segments.append({"type": "action", "text": action})
        cursor = match.end()
    if cursor < len(text) or not segments:
        dialogue = text[cursor:]
        cleaned = re.sub(r"</?\s*action\s*>?", "", dialogue, flags=re.I)
        if cleaned != dialogue:
            repaired = True
        if cleaned:
            segments.append({"type": "dialogue", "text": cleaned})
    return ActionParseResult(tuple(segments), repaired, subject_changed)


def canonicalize_assistant_message(content: str) -> tuple[str, ActionParseResult]:
    parsed = parse_assistant_actions(content)
    canonical = "".join(
        f"<action>{segment['text']}</action>" if segment["type"] == "action" else segment["text"]
        for segment in parsed.segments
    )
    return canonical, parsed


def normalize_assistant_actions(content: str) -> str:
    """History/memory-only semantic text, never the frontend display format."""
    parsed = parse_assistant_actions(content)
    return "".join(
        f"[Character action: {segment['text']}]" if segment["type"] == "action" else segment["text"]
        for segment in parsed.segments
    )
