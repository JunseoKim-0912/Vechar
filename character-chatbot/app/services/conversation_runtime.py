"""Bounded, lower-priority conversation state; never a source of canonical truth."""

from dataclasses import dataclass, field
import hashlib
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from ..models import Message, MessageRole
from .chat_actions import canonicalize_assistant_message, parse_assistant_actions
from .conversation_loop_guard import semantic_signature

RUNTIME_VERSION = 1
COOLDOWN_REPLIES = 3
MAX_ESTABLISHED = 8
MAX_RESOLVED = 6
MAX_OPEN = 5
MAX_ACTIONS = 5
MAX_ADVICE = 5
MAX_STYLE = 10
MAX_LABEL = 64

_WORDS = re.compile(r"[\w가-힣]+", re.U)
_DEVELOPMENT = re.compile(
    r"\b(?:calls?|called|answers?|answered|responds?|responded|reacts?|reacted|"
    r"arrives?|arrived|leaves?|left|changes?|changed|decides?|decided|enters?|entered)\b|"
    r"불렀|부른다|대답|반응|도착|떠났|결정|변했", re.I,
)
_CALL = re.compile(r"\b(?:calls?|called|cries? out to)\b|부르|불렀", re.I)
_ADVICE = re.compile(r"\b(?:should|must|try to|tell her|tell him|call her|call him)\b|"
                     r"(?:말해|말하|불러|불러봐|해야|해봐)", re.I)
_NEGATED_DEVELOPMENT = re.compile(r"\b(?:nothing|not|never)\b.{0,18}\b(?:changed|happened|answered|reacted)\b|"
                                   r"아무 일도.{0,12}않", re.I)
_NEGATED_CALL = re.compile(r"\b(?:cannot|can't|couldn't|did not|didn't|never|unable to)\b.{0,24}\bcall\b|"
                           r"(?:부르|불렀).{0,8}(?:못|않)", re.I)
_CATCHPHRASE = re.compile(r"(?<![가-힣])(?:이\s*|아이고[, ]*\s*|참\s*)?"
                          r"(맹추야|바보야|멍청이야|얼간이야)(?![가-힣])")
_GENERIC = {"그래", "응", "아니", "yes", "no", "okay", "well"}


class TurnProgression(BaseModel):
    """Untrusted, compact hints emitted in the same chat generation call."""

    model_config = ConfigDict(extra="forbid")
    topic: str
    new_development: str | None
    resolved_thread: str | None
    opened_thread: str | None
    action_taken: str | None
    advice_given: str | None
    repeated_point: bool


class ChatTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: str
    progression: TurnProgression

    @field_validator("response")
    @classmethod
    def visible_response_only(cls, value: str) -> str:
        visible, _ = canonicalize_assistant_message(value)
        if not visible.strip() or (value.lstrip().startswith("{") and '"progression"' in value):
            raise ValueError("The visible response must contain dialogue, not turn metadata")
        return value


def empty_turn(response: str) -> ChatTurnResult:
    """Compatibility for provider-free legacy fixtures, never the production gateway."""
    return ChatTurnResult(response=response, progression=TurnProgression(
        topic="", new_development=None, resolved_thread=None, opened_thread=None,
        action_taken=None, advice_given=None, repeated_point=False,
    ))


def turn_output_instructions() -> str:
    return """
[STRUCTURED TURN RESULT]
Fill the response field with only the character's visible dialogue and <action> blocks.
Progression fields are short snake_case hints, not prose: topic, new_development,
resolved_thread, opened_thread, action_taken, advice_given, repeated_point.
Use null for absent changes. A resolved_thread must name an actually open thread;
do not claim a decision or action occurred when it was only suggested or imagined.
Never place JSON keys or progression metadata inside response. Normal output needs
one provider call; the server validates these hints against the visible response.
"""


def compact_label(value: str | None) -> str | None:
    if not value or len(value) > 100 or len(value.split()) > 8:
        return None
    label = re.sub(r"[^\w가-힣-]+", "_", value.casefold()).strip("_")
    return label[:MAX_LABEL] if label and label not in _GENERIC else None


def _bounded(items: Sequence[str], cap: int) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))[-cap:]


def phrase_families(content: str) -> set[str]:
    dialogue = " ".join(part["text"] for part in parse_assistant_actions(content).segments
                        if part["type"] == "dialogue")
    return {match.group(1) for match in _CATCHPHRASE.finditer(dialogue)}


def phrase_family_hash(family: str) -> str:
    return hashlib.sha256(family.encode("utf-8")).hexdigest()[:12]


@dataclass
class ConversationRuntimeState:
    """A versioned snapshot committed atomically with the visible assistant turn."""

    version: int = RUNTIME_VERSION
    last_message_id: str | None = None
    established_points: list[str] = field(default_factory=list)
    resolved_threads: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
    recent_advice: list[str] = field(default_factory=list)
    style_uses: list[dict] = field(default_factory=list)
    speaker_turns: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_storage(cls, payload: dict | None, messages: Sequence[Message]) -> "ConversationRuntimeState":
        latest = next((m.id for m in reversed(messages) if m.role == MessageRole.CHARACTER), None)
        if isinstance(payload, dict) and payload.get("version") == RUNTIME_VERSION and payload.get("last_message_id") == latest:
            try:
                return cls(
                    last_message_id=latest,
                    established_points=_bounded(payload.get("established_points", []), MAX_ESTABLISHED),
                    resolved_threads=_bounded(payload.get("resolved_threads", []), MAX_RESOLVED),
                    open_threads=_bounded(payload.get("open_threads", []), MAX_OPEN),
                    recent_actions=_bounded(payload.get("recent_actions", []), MAX_ACTIONS),
                    recent_advice=_bounded(payload.get("recent_advice", []), MAX_ADVICE),
                    style_uses=[item for item in payload.get("style_uses", [])[-MAX_STYLE:]
                                if isinstance(item, dict) and isinstance(item.get("family"), str)],
                    speaker_turns={str(k): int(v) for k, v in payload.get("speaker_turns", {}).items()
                                   if isinstance(v, int) and 0 <= v < 1000000},
                )
            except (TypeError, ValueError, AttributeError):
                pass
        return cls.rebuild(messages)

    @classmethod
    def rebuild(cls, messages: Sequence[Message]) -> "ConversationRuntimeState":
        """Old rooms/corrupt snapshots fall back to persisted visible history only.

        We cannot invent model-only thread metadata for old turns; known actions,
        advice and style usage are conservatively reconstructed instead.
        """
        state = cls()
        for message in messages[-30:]:
            if message.role != MessageRole.CHARACTER:
                continue
            state._record_visible(message.content, message.speaker_character_id or "character")
            state.last_message_id = message.id
        return state

    def _record_visible(self, response: str, speaker: str) -> None:
        ordinal = self.speaker_turns.get(speaker, 0) + 1
        self.speaker_turns[speaker] = ordinal
        signature = semantic_signature(response)
        if signature:
            self.established_points = _bounded([*self.established_points, signature[0]], MAX_ESTABLISHED)
            if signature[0].startswith("advice:"):
                self.recent_advice = _bounded([*self.recent_advice, signature[0]], MAX_ADVICE)
        actions = [part["text"] for part in parse_assistant_actions(response).segments
                   if part["type"] == "action"]
        self.recent_actions = _bounded([*self.recent_actions, *(compact_label(a) or "" for a in actions)], MAX_ACTIONS)
        for family in phrase_families(response):
            self.style_uses.append({"speaker": speaker, "family": family, "at": ordinal})
        self.style_uses = self.style_uses[-MAX_STYLE:]

    def active_families(self, speaker: str) -> set[str]:
        ordinal = self.speaker_turns.get(speaker, 0)
        return {item["family"] for item in self.style_uses
                if item.get("speaker") == speaker and 0 <= ordinal - item.get("at", -100) < COOLDOWN_REPLIES}

    def prompt_block(self, speaker: str) -> str:
        def row(name: str, values: Sequence[str]) -> str:
            return f"- {name}: {', '.join(values) if values else 'none'}"
        return "\n[CONVERSATION RUNTIME v1 — lower priority than canon; hints, not facts]\n" + "\n".join((
            row("Already established", self.established_points),
            row("Resolved threads", self.resolved_threads),
            row("Still open", self.open_threads),
            row("Recent actions", self.recent_actions),
            row("Recent advice", self.recent_advice),
            row("Catchphrase cooldown for this speaker", sorted(self.active_families(speaker))),
        )) + ("\nDo not repeat a resolved suggestion or cooled phrase. Advance an open thread when natural. "
               "These derived/model hints never override the canonical profile, timeline, or world.\n")

    def to_storage(self) -> dict:
        return {
            "version": RUNTIME_VERSION, "last_message_id": self.last_message_id,
            "established_points": self.established_points[-MAX_ESTABLISHED:],
            "resolved_threads": self.resolved_threads[-MAX_RESOLVED:],
            "open_threads": self.open_threads[-MAX_OPEN:],
            "recent_actions": self.recent_actions[-MAX_ACTIONS:],
            "recent_advice": self.recent_advice[-MAX_ADVICE:],
            "style_uses": self.style_uses[-MAX_STYLE:],
            "speaker_turns": self.speaker_turns,
        }


def evidenced_development(response: str, label: str | None) -> bool:
    normalized = compact_label(label)
    if not normalized:
        return False
    words = {word.casefold() for word in _WORDS.findall(response) if len(word) >= 3}
    label_words = {word for word in normalized.split("_") if len(word) >= 3}
    return bool(_DEVELOPMENT.search(response) and not _NEGATED_DEVELOPMENT.search(response)
                and words.intersection(label_words))


def _thread_evidenced(response: str, label: str, *, resolving: bool = False) -> bool:
    if "call" in label:
        return bool(_CALL.search(response) and not _ADVICE.search(response)
                    and not _NEGATED_CALL.search(response))
    if "respond" in label or "answer" in label or "reply" in label:
        if resolving:
            return bool(re.search(r"\b(?:answered|responded|replied)\b|대답했|응답했", response, re.I)
                        and not _NEGATED_DEVELOPMENT.search(response))
        return bool(re.search(r"\b(?:responds?|answered?|replies?|reply|answer)\b|대답|응답", response, re.I))
    words = {word.casefold() for word in _WORDS.findall(response) if len(word) >= 4}
    label_words = {word for word in label.split("_") if len(word) >= 4 and word not in
                   {"should", "whether", "does", "that", "with"}}
    return bool(_DEVELOPMENT.search(response) and not _NEGATED_DEVELOPMENT.search(response)
                and words.intersection(label_words))


def validated_delta(state: ConversationRuntimeState, turn: ChatTurnResult) -> dict[str, str | None | bool]:
    """Never promote a model claim without visible evidence and current-state checks."""
    p, response = turn.progression, turn.response
    resolved = compact_label(p.resolved_thread)
    if resolved not in state.open_threads or not _thread_evidenced(response, resolved, resolving=True):
        resolved = None
    opened = compact_label(p.opened_thread)
    if opened in state.open_threads or opened in state.resolved_threads or opened == resolved:
        opened = None
    if opened and not _thread_evidenced(response, opened):
        opened = None
    action = compact_label(p.action_taken)
    action_words = {word for word in action.split("_") if len(word) >= 4 and word not in
                    {"gregor", "grete", "doctor", "meursault"}} if action else set()
    visible_words = {word.casefold() for word in _WORDS.findall(response) if len(word) >= 4}
    if action and not (action_words & visible_words and
                       (any(part["type"] == "action" for part in parse_assistant_actions(response).segments)
                        or _DEVELOPMENT.search(response))):
        action = None
    development = compact_label(p.new_development) if evidenced_development(response, p.new_development) else None
    advice = compact_label(p.advice_given) if _ADVICE.search(response) else None
    return {"resolved": resolved, "opened": opened, "action": action,
            "development": development, "advice": advice,
            "progression": bool(resolved or opened or action or development)}


def resolved_advice_repeated(state: ConversationRuntimeState, response: str) -> bool:
    if not state.resolved_threads or not _ADVICE.search(response):
        return False
    text = response.casefold()
    for thread in state.resolved_threads:
        key_words = [word for word in thread.split("_") if len(word) >= 4 and word not in
                     {"should", "whether", "does", "that", "with", "from", "gregor"}]
        if key_words and any(word in text for word in key_words):
            return True
        if "call" in thread and _CALL.search(response):
            return True
    return False


def advance_runtime(state: ConversationRuntimeState, turn: ChatTurnResult, response: str,
                    speaker: str, message_id: str) -> ConversationRuntimeState:
    """The accepted visible message, not metadata alone, determines stored state."""
    delta = validated_delta(state, turn.model_copy(update={"response": response}))
    if delta["resolved"]:
        state.open_threads = [item for item in state.open_threads if item != delta["resolved"]]
        state.resolved_threads = _bounded([*state.resolved_threads, delta["resolved"]], MAX_RESOLVED)
    if delta["opened"]:
        state.open_threads = _bounded([*state.open_threads, delta["opened"]], MAX_OPEN)
    if delta["development"]:
        state.established_points = _bounded([*state.established_points, delta["development"]], MAX_ESTABLISHED)
    if delta["advice"]:
        state.recent_advice = _bounded([*state.recent_advice, delta["advice"]], MAX_ADVICE)
    state._record_visible(response, speaker)
    state.last_message_id = message_id
    return state
