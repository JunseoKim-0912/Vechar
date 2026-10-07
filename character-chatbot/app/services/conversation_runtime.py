"""Bounded, lower-priority conversation state; never a source of canonical truth."""

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import Message, MessageRole
from .chat_actions import canonicalize_assistant_message, parse_assistant_actions
from .character_fidelity import KnowledgeScope
from .conversation_loop_guard import semantic_signature

RUNTIME_VERSION = 3
COOLDOWN_REPLIES = 3
STYLE_COOLDOWN_REPLIES = 2
THREAD_TURN_BUDGET = 4
MAX_THREADS = 10
MAX_ESTABLISHED = 8
MAX_RESOLVED = 6
MAX_OPEN = 5
MAX_ACTIONS = 5
MAX_ADVICE = 5
MAX_STYLE = 10
MAX_STYLE_CATEGORIES = 12
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
_THREAD_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_STYLE_PATTERNS = {
    "teasing_insult": re.compile(r"(?:맹추야|바보야|이\s*녀석아|멍청이야|얼간이야)(?![가-힣])", re.I),
    "rhetorical_challenge": re.compile(r"(?:꼭.{0,28}(?:해야|필요)|must i|must you|do you really have to)[^.!?]*[?？]", re.I),
    "dismissive_address": re.compile(r"(?:네가\s*알\s*게\s*뭐|what's it to you|none of your business)", re.I),
    "repeated_self_pity": re.compile(r"(?:아무도\s*나를|나만\s*늘|no one ever cares about me)", re.I),
}


class ThreadStatus(StrEnum):
    OPEN = "open"
    ACTIVE = "active"
    RESOLVED = "resolved"
    EXHAUSTED = "exhausted"
    REOPENED = "reopened"


class ThreadTransition(StrEnum):
    HOLD = "hold"
    ADVANCE = "advance"
    RESOLVE = "resolve"
    BRANCH = "branch"
    EXHAUST = "exhaust"
    REOPEN = "reopen"


@dataclass
class ThreadRecord:
    id: str
    status: ThreadStatus = ThreadStatus.OPEN
    turns_spent: int = 0
    budget: int = THREAD_TURN_BUDGET
    parent_id: str | None = None
    last_development: str | None = None

    @classmethod
    def from_storage(cls, value: dict) -> "ThreadRecord":
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError("Invalid thread record")
        thread_id = value["id"]
        if not _THREAD_ID.fullmatch(thread_id):
            raise ValueError("Invalid thread id")
        spent, budget = value.get("turns_spent", 0), value.get("budget", THREAD_TURN_BUDGET)
        if not isinstance(spent, int) or not isinstance(budget, int) or not 0 <= spent <= 20 or not 1 <= budget <= 8:
            raise ValueError("Invalid thread budget")
        parent = value.get("parent_id")
        development = value.get("last_development")
        return cls(thread_id, ThreadStatus(value.get("status", "open")), spent, budget,
                   parent if isinstance(parent, str) and _THREAD_ID.fullmatch(parent) else None,
                   compact_label(development))

    def to_storage(self) -> dict:
        return {"id": self.id, "status": self.status.value, "turns_spent": self.turns_spent,
                "budget": self.budget, "parent_id": self.parent_id,
                "last_development": self.last_development}


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
    active_thread_id: str | None = None
    transition: ThreadTransition = ThreadTransition.HOLD
    new_thread_id: str | None = None
    resolved_thread_ids: list[str] = Field(default_factory=list, max_length=2)
    reopened_thread_ids: list[str] = Field(default_factory=list, max_length=2)


class TurnFidelity(BaseModel):
    """Untrusted same-call self-report, checked against canon and visible text."""

    model_config = ConfigDict(extra="forbid")
    knowledge_scope: KnowledgeScope
    persona_preserved: bool
    assistant_mode: bool


class ChatTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: str
    progression: TurnProgression
    fidelity: TurnFidelity

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
    ), fidelity=TurnFidelity(knowledge_scope=KnowledgeScope.UNCERTAIN,
                            persona_preserved=True, assistant_mode=False))


def turn_output_instructions() -> str:
    return """
[STRUCTURED TURN RESULT]
Fill the response field with only the character's visible dialogue and <action> blocks.
Progression fields are short snake_case hints, not prose: topic, new_development,
resolved_thread, opened_thread, action_taken, advice_given, repeated_point.
Also supply active_thread_id, transition (hold/advance/resolve/branch/exhaust/reopen),
new_thread_id, resolved_thread_ids, reopened_thread_ids. Use the ACTIVE thread selected
in the runtime block. A transition needs a visible action, consequence, answer, or new
event; repeating a desire or suggestion is not a transition. Reopen a closed thread
only when a new event is visible. Keep IDs compact snake_case, not sentences.
Use null for absent changes. A resolved_thread must name an actually open thread;
do not claim a decision or action occurred when it was only suggested or imagined.
Fidelity fields are compact: knowledge_scope is canonical, plausible_general,
uncertain, or outside_scope; persona_preserved and assistant_mode describe the
visible answer. Do not claim canonical knowledge without canonical evidence.
Never place JSON keys or metadata inside response. Normal output needs one
provider call; the server validates these untrusted hints against visible text.
"""


def compact_label(value: str | None) -> str | None:
    if not value or len(value) > 100 or len(value.split()) > 8:
        return None
    label = re.sub(r"[^\w가-힣-]+", "_", value.casefold()).strip("_")
    return label[:MAX_LABEL] if label and label not in _GENERIC else None


def compact_thread_id(value: str | None) -> str | None:
    """Thread keys must survive JSON reload as stable short ASCII identifiers."""
    label = compact_label(value)
    return label if label and _THREAD_ID.fullmatch(label) else None


def _bounded(items: Sequence[str], cap: int) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))[-cap:]


def phrase_families(content: str) -> set[str]:
    dialogue = " ".join(part["text"] for part in parse_assistant_actions(content).segments
                        if part["type"] == "dialogue")
    return {match.group(1) for match in _CATCHPHRASE.finditer(dialogue)}


def phrase_family_hash(family: str) -> str:
    return hashlib.sha256(family.encode("utf-8")).hexdigest()[:12]


def style_categories(content: str) -> set[str]:
    """A small voice-level taxonomy, applied only to visible dialogue."""
    dialogue = " ".join(part["text"] for part in parse_assistant_actions(content).segments
                        if part["type"] == "dialogue")
    return {name for name, pattern in _STYLE_PATTERNS.items() if pattern.search(dialogue)}


@dataclass
class ConversationRuntimeState:
    """A versioned snapshot committed atomically with the visible assistant turn."""

    version: int = RUNTIME_VERSION
    last_message_id: str | None = None
    established_points: list[str] = field(default_factory=list)
    resolved_threads: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    threads: list[ThreadRecord] = field(default_factory=list)
    active_thread_id: str | None = None
    recent_actions: list[str] = field(default_factory=list)
    recent_advice: list[str] = field(default_factory=list)
    style_uses: list[dict] = field(default_factory=list)
    style_category_uses: list[dict] = field(default_factory=list)
    knowledge_provenance_summary: dict[str, int] = field(default_factory=dict)
    speaker_turns: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_storage(cls, payload: dict | None, messages: Sequence[Message]) -> "ConversationRuntimeState":
        latest = next((m.id for m in reversed(messages) if m.role == MessageRole.CHARACTER), None)
        if isinstance(payload, dict) and payload.get("version") == RUNTIME_VERSION and payload.get("last_message_id") == latest:
            try:
                stored_threads = payload["threads"]
                if not isinstance(stored_threads, list) or len(stored_threads) > MAX_THREADS:
                    raise ValueError("Invalid thread state")
                state = cls(
                    last_message_id=latest,
                    established_points=_bounded(payload.get("established_points", []), MAX_ESTABLISHED),
                    threads=[ThreadRecord.from_storage(item) for item in stored_threads],
                    active_thread_id=payload.get("active_thread_id"),
                    recent_actions=_bounded(payload.get("recent_actions", []), MAX_ACTIONS),
                    recent_advice=_bounded(payload.get("recent_advice", []), MAX_ADVICE),
                    style_uses=[item for item in payload.get("style_uses", [])[-MAX_STYLE:]
                                if isinstance(item, dict) and isinstance(item.get("family"), str)
                                and isinstance(item.get("speaker"), str)
                                and isinstance(item.get("at"), int)],
                    style_category_uses=[item for item in payload.get("style_category_uses", [])[-MAX_STYLE_CATEGORIES:]
                                         if isinstance(item, dict) and item.get("category") in _STYLE_PATTERNS
                                         and isinstance(item.get("speaker"), str)
                                         and isinstance(item.get("at"), int)],
                    knowledge_provenance_summary={str(k): min(max(v, 0), 30)
                                                  for k, v in payload.get("knowledge_provenance_summary", {}).items()
                                                  if k in {"user_provided", "peer_claim", "prior_self_output", "memory"}
                                                  and isinstance(v, int)},
                    speaker_turns={str(k): int(v) for k, v in list(payload.get("speaker_turns", {}).items())[:4]
                                   if isinstance(v, int) and 0 <= v < 1000000},
                )
                if len({item.id for item in state.threads}) != len(state.threads):
                    raise ValueError("Duplicate thread ids")
                if state.active_thread_id is not None and state.active_thread_id not in {item.id for item in state.threads}:
                    raise ValueError("Unknown active thread")
                state._sync_legacy()
                return state
            except (KeyError, TypeError, ValueError, AttributeError):
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
        for category in style_categories(response):
            self.style_category_uses.append({"speaker": speaker, "category": category, "at": ordinal})
        self.style_category_uses = self.style_category_uses[-MAX_STYLE_CATEGORIES:]

    def _ensure_threads(self) -> None:
        """Support provider-free legacy fixtures; persisted v1/v2 are rebuilt instead."""
        if self.threads:
            return
        for thread_id in self.open_threads[-MAX_OPEN:]:
            if _THREAD_ID.fullmatch(thread_id):
                self.threads.append(ThreadRecord(thread_id))
        for thread_id in self.resolved_threads[-MAX_RESOLVED:]:
            if _THREAD_ID.fullmatch(thread_id) and thread_id not in {t.id for t in self.threads}:
                self.threads.append(ThreadRecord(thread_id, ThreadStatus.RESOLVED))
        self.threads = self.threads[-MAX_THREADS:]

    def _sync_legacy(self) -> None:
        self.open_threads = [t.id for t in self.threads
                             if t.status in {ThreadStatus.OPEN, ThreadStatus.ACTIVE, ThreadStatus.REOPENED}][-MAX_OPEN:]
        self.resolved_threads = [t.id for t in self.threads if t.status == ThreadStatus.RESOLVED][-MAX_RESOLVED:]

    def thread(self, thread_id: str | None) -> ThreadRecord | None:
        self._ensure_threads()
        return next((item for item in self.threads if item.id == thread_id), None)

    def select_active_thread(self) -> ThreadRecord | None:
        """Deterministic single target; related open branches win over random novelty."""
        self._ensure_threads()
        eligible = [item for item in self.threads if item.status in {
            ThreadStatus.OPEN, ThreadStatus.ACTIVE, ThreadStatus.REOPENED,
        } and item.turns_spent < item.budget]
        if not eligible:
            return None
        current = next((item for item in eligible if item.id == self.active_thread_id), None)
        if current:
            return current
        previous = set((self.active_thread_id or "").split("_"))
        return min(eligible, key=lambda item: (
            -len(previous & set(item.id.split("_"))) - int(item.parent_id == self.active_thread_id),
            item.turns_spent, self.threads.index(item),
        ))

    def active_families(self, speaker: str) -> set[str]:
        ordinal = self.speaker_turns.get(speaker, 0)
        return {item["family"] for item in self.style_uses
                if item.get("speaker") == speaker and 0 <= ordinal - item.get("at", -100) < COOLDOWN_REPLIES}

    def active_style_categories(self, speaker: str) -> set[str]:
        ordinal = self.speaker_turns.get(speaker, 0)
        return {item["category"] for item in self.style_category_uses
                if item.get("speaker") == speaker
                and isinstance(item.get("at"), int)
                and 0 <= ordinal - item["at"] < STYLE_COOLDOWN_REPLIES}

    def prompt_block(self, speaker: str, *, room: bool = False) -> str:
        def row(name: str, values: Sequence[str]) -> str:
            return f"- {name}: {', '.join(values) if values else 'none'}"
        target = self.select_active_thread() if room else None
        exhausted = [t.id for t in self.threads if t.status == ThreadStatus.EXHAUSTED]
        mode = ("Advance only the ACTIVE thread or a canon-related consequence. "
                "Do not return to RESOLVED/EXHAUSTED advice without a visible new event. "
                "If no thread is active, open a related consequence rather than restating the old topic. "
                if room else "The user controls callbacks; prior topics may be revisited when requested. ")
        return f"\n[CONVERSATION RUNTIME v{RUNTIME_VERSION} — below canon and fidelity]\n" + "\n".join((
            row("Already established", self.established_points),
            f"- ACTIVE thread: {target.id if target else 'none'}"
            + (f" ({target.turns_spent}/{target.budget} stagnant turns)" if target else ""),
            row("Resolved threads", self.resolved_threads),
            row("Exhausted threads", exhausted),
            row("Other open threads", [t.id for t in self.threads if t.id != (target.id if target else None)
                                        and t.status in {ThreadStatus.OPEN, ThreadStatus.REOPENED}]),
            row("Recent actions", self.recent_actions),
            row("Recent advice", self.recent_advice),
            row("Catchphrase cooldown for this speaker", sorted(self.active_families(speaker))),
            row("Style-category cooldown", sorted(self.active_style_categories(speaker))),
        )) + ("\n" + mode + "Preserve character voice with a different expression. "
               "These derived/model hints never override canon.\n")

    def to_storage(self) -> dict:
        self._ensure_threads()
        self._sync_legacy()
        return {
            "version": RUNTIME_VERSION, "last_message_id": self.last_message_id,
            "established_points": self.established_points[-MAX_ESTABLISHED:],
            "threads": [item.to_storage() for item in self.threads[-MAX_THREADS:]],
            "active_thread_id": self.active_thread_id,
            "resolved_threads": self.resolved_threads[-MAX_RESOLVED:],
            "open_threads": self.open_threads[-MAX_OPEN:],
            "recent_actions": self.recent_actions[-MAX_ACTIONS:],
            "recent_advice": self.recent_advice[-MAX_ADVICE:],
            "style_uses": self.style_uses[-MAX_STYLE:],
            "style_category_uses": self.style_category_uses[-MAX_STYLE_CATEGORIES:],
            "knowledge_provenance_summary": self.knowledge_provenance_summary,
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


def _topic_overlap(response: str, thread_id: str) -> bool:
    words = {word.casefold() for word in _WORDS.findall(response)}
    meaningful = {word for word in thread_id.split("_") if len(word) >= 4 and word not in {
        "should", "whether", "does", "that", "with", "from", "gregor", "doctor",
    }}
    return bool(meaningful & words) or ("call" in thread_id and bool(_CALL.search(response)))


def validated_delta(state: ConversationRuntimeState, turn: ChatTurnResult) -> dict[str, str | None | bool]:
    """Never promote a model claim without visible evidence and current-state checks."""
    p, response = turn.progression, turn.response
    state._ensure_threads()
    target = state.select_active_thread()
    requested_resolved = [*p.resolved_thread_ids, p.resolved_thread]
    resolved = next((candidate for candidate in (compact_thread_id(item) for item in requested_resolved)
                     if candidate and state.thread(candidate)
                     and state.thread(candidate).status in {
                         ThreadStatus.OPEN, ThreadStatus.ACTIVE, ThreadStatus.REOPENED,
                     } and _thread_evidenced(response, candidate, resolving=True)), None)
    if resolved is None:
        resolved = None
    opened = compact_thread_id(p.new_thread_id or p.opened_thread)
    if opened and (state.thread(opened) or opened == resolved):
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
    if development in state.established_points:
        development = None
    # A fresh label is not a new branch when the visible turn only rephrases
    # the same unresolved topic. The first topic has no predecessor to link.
    if opened and target and not (resolved or action or development):
        opened = None
    advice = compact_label(p.advice_given) if _ADVICE.search(response) else None
    reopened = next((candidate for candidate in (compact_thread_id(item) for item in p.reopened_thread_ids)
                     if candidate and state.thread(candidate)
                     and state.thread(candidate).status in {ThreadStatus.RESOLVED, ThreadStatus.EXHAUSTED}
                     and development and _topic_overlap(response, candidate)), None)
    progression = bool(resolved or opened or action or development or reopened)
    claimed = compact_thread_id(p.active_thread_id)
    wrong_target = bool(target and p.active_thread_id and claimed != target.id and not reopened)
    false_claim = bool((any(requested_resolved) and not resolved)
                       or (p.transition == ThreadTransition.RESOLVE and not resolved)
                       or (p.transition == ThreadTransition.BRANCH and not opened)
                       or (p.transition == ThreadTransition.REOPEN and not reopened))
    resolved_reuse = resolved_advice_repeated(state, response) and not reopened
    exhausted_reuse = exhausted_topic_repeated(state, response, progression=progression)
    return {"resolved": resolved, "opened": opened, "action": action,
            "development": development, "advice": advice,
            "reopened": reopened, "progression": progression,
            "transition": p.transition.value,
            "wrong_target": wrong_target, "false_claim": false_claim,
            "resolved_reuse": resolved_reuse, "exhausted_reuse": exhausted_reuse,
            "invalid_transition": bool(wrong_target or false_claim or resolved_reuse or exhausted_reuse),
            "topic_budget_exhausted": bool(target and target.turns_spent + 1 >= target.budget and not progression)}


def resolved_advice_repeated(state: ConversationRuntimeState, response: str) -> bool:
    if not state.resolved_threads or not _ADVICE.search(response):
        return False
    return any(_topic_overlap(response, thread) for thread in state.resolved_threads)


def exhausted_topic_repeated(state: ConversationRuntimeState, response: str, *, progression: bool) -> bool:
    if progression:
        return False
    return any(item.status == ThreadStatus.EXHAUSTED and _topic_overlap(response, item.id)
               for item in state.threads)


def advance_runtime(state: ConversationRuntimeState, turn: ChatTurnResult, response: str,
                    speaker: str, message_id: str, *, strict_room: bool = False) -> ConversationRuntimeState:
    """The accepted visible message, not metadata alone, determines stored state."""
    delta = validated_delta(state, turn.model_copy(update={"response": response}))
    if strict_room and delta["invalid_transition"]:
        raise ValueError("Invalid room thread transition")
    target = state.select_active_thread()
    if delta["resolved"]:
        state.thread(delta["resolved"]).status = ThreadStatus.RESOLVED
    if delta["reopened"]:
        reopened = state.thread(delta["reopened"])
        reopened.status = ThreadStatus.REOPENED
        reopened.turns_spent = 0
        reopened.last_development = delta["development"]
    if delta["opened"]:
        state.threads.append(ThreadRecord(delta["opened"], parent_id=target.id if target else None))
        state.threads = state.threads[-MAX_THREADS:]
    if target and target.status not in {ThreadStatus.RESOLVED, ThreadStatus.EXHAUSTED}:
        if delta["progression"]:
            target.turns_spent = 0
            target.last_development = delta["development"]
        else:
            target.turns_spent += 1
            if target.turns_spent >= target.budget:
                target.status = ThreadStatus.EXHAUSTED
        if target.status == ThreadStatus.OPEN:
            target.status = ThreadStatus.ACTIVE
    state.active_thread_id = (delta["opened"] or delta["reopened"] or
                              (target.id if target and target.status in {
                                  ThreadStatus.OPEN, ThreadStatus.ACTIVE, ThreadStatus.REOPENED,
                              } else None))
    state._sync_legacy()
    if delta["development"]:
        state.established_points = _bounded([*state.established_points, delta["development"]], MAX_ESTABLISHED)
    if delta["advice"]:
        state.recent_advice = _bounded([*state.recent_advice, delta["advice"]], MAX_ADVICE)
    state._record_visible(response, speaker)
    state.last_message_id = message_id
    return state
