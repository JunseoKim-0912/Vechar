"""Deterministic reconciliation of typed character events and their living chat state."""

from collections import deque
from datetime import date
from itertools import chain
import re

from ..schemas import (
    CanonicalCharacterState, ChatReferencePoint, RelationshipState,
    TimelineEvent, TimelineStateChanges,
)


def _key(value: str) -> str:
    return re.sub(r"[^\w]+", "-", value.casefold()).strip("-")


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _merge_changes(old: TimelineStateChanges, new: TimelineStateChanges, *, prefer_new: bool,
                   conflicts: list[str]) -> TimelineStateChanges:
    result = old.model_copy(deep=True)
    for field in (
        "occupation", "affiliations", "location", "physical_condition", "mental_condition", "abilities",
        "possessions", "personality", "speech_style", "goals", "loyalties",
    ):
        incoming = getattr(new, field)
        previous = getattr(result, field)
        if incoming is None:
            continue
        if previous is not None and previous != incoming and not prefer_new:
            conflicts.append(f"Conflicting {field}: {previous!r} / {incoming!r}")
        else:
            setattr(result, field, incoming)
    relationships = {item.name.casefold(): item for item in result.relationships}
    for item in new.relationships:
        previous = relationships.get(item.name.casefold())
        if previous is not None and previous.status != item.status and not prefer_new:
            conflicts.append(f"Conflicting relationship with {item.name}: {previous.status} / {item.status}")
        else:
            relationships[item.name.casefold()] = item
    result.relationships = list(relationships.values())
    result.knowledge = _unique(result.knowledge + new.knowledge)
    return result


def _merge_event(old: TimelineEvent, new: TimelineEvent, *, prefer_new: bool) -> TimelineEvent:
    if prefer_new:
        corrected = new.model_copy(deep=True)
        corrected.source_ids = _unique(old.source_ids + new.source_ids)
        corrected.chunk_indices = list(dict.fromkeys(old.chunk_indices + new.chunk_indices))
        return corrected
    result = old.model_copy(deep=True)
    temporal_conflicts = []
    state_conflicts = []
    for field in ("age", "absolute_year", "absolute_date", "relative_to", "relative_offset_months"):
        incoming = getattr(new, field)
        previous = getattr(result, field)
        if incoming is None:
            continue
        if previous is not None and previous != incoming and not prefer_new:
            setattr(result, field, None)
            temporal_conflicts.append(f"Conflicting {field}: {previous!r} / {incoming!r}")
        else:
            setattr(result, field, incoming)
    for field in ("time_label", "relative_label"):
        if getattr(new, field) and (prefer_new or not getattr(result, field)):
            setattr(result, field, getattr(new, field))
    if new.summary and new.summary not in result.summary:
        result.summary = f"{result.summary} / {new.summary}" if result.summary else new.summary
    result.state_changes = _merge_changes(result.state_changes, new.state_changes,
                                          prefer_new=prefer_new, conflicts=state_conflicts)
    if new.relative_order != "unknown" and result.relative_order != new.relative_order:
        if result.relative_order != "unknown":
            temporal_conflicts.append("Conflicting relative order")
            result.relative_order = "unknown"
        else:
            result.relative_order = new.relative_order
    result.source_ids = _unique(result.source_ids + new.source_ids)
    result.chunk_indices = list(dict.fromkeys(result.chunk_indices + new.chunk_indices))
    result.is_death = new.is_death if prefer_new else result.is_death or new.is_death
    if result.canonicality != new.canonicality:
        result.canonicality = new.canonicality if prefer_new else "uncertain"
    if temporal_conflicts:
        result.precision = "conflicting"
    elif prefer_new or result.precision == "unknown":
        result.precision = new.precision
    result.temporal_uncertainty = " / ".join(
        _unique([result.temporal_uncertainty, new.temporal_uncertainty] + temporal_conflicts)
    )
    result.evidence_conflicts = _unique(result.evidence_conflicts + new.evidence_conflicts + state_conflicts)
    return result


def _positions(events: list[TimelineEvent]) -> dict[str, tuple[str, int] | None]:
    """Resolve relative relations with ordinal coordinates; never persist invented dates."""
    by_key = {_key(event.event_key): event for event in events}
    links: dict[str, list[tuple[str, int]]] = {key: [] for key in by_key}
    relations: list[tuple[str, str, int]] = []
    for key, event in by_key.items():
        anchor = _key(event.relative_to or "")
        if anchor in by_key and anchor != key:
            offset = (event.relative_offset_months * 32 if event.relative_offset_months is not None
                      else 1 if event.relative_order == "after"
                      else -1 if event.relative_order == "before"
                      else 0 if event.relative_order == "same" else None)
            if offset is not None:
                relations.append((anchor, key, offset))
                links[anchor].append((key, offset))
                links[key].append((anchor, -offset))

    positions: dict[str, tuple[str, int] | None] = {key: None for key in by_key}
    # An event with both a stated age and year bridges the two coordinate systems.
    age_year_offset = next(
        ((event.age - event.absolute_year) * 12 * 32 for event in events
         if event.age is not None and event.absolute_year is not None), None
    )
    for key, event in by_key.items():
        if event.precision == "conflicting":
            continue
        if event.absolute_date:
            try:
                exact = date.fromisoformat(event.absolute_date)
            except ValueError:
                exact = None
            if exact is not None:
                date_position = (exact.year * 12 + exact.month - 1) * 32 + exact.day
                positions[key] = ("age", date_position + age_year_offset) if age_year_offset is not None else (
                    "year", date_position)
                continue
        if event.age is not None:
            positions[key] = ("age", event.age * 12 * 32)
        elif event.absolute_year is not None:
            positions[key] = (
                "age", event.absolute_year * 12 * 32 + age_year_offset
            ) if age_year_offset is not None else ("year", event.absolute_year * 12 * 32)

    # Age and year are coarse buckets. A stated within-bucket relation (not an
    # exact-date conflict) resolves their local order without inventing a date.
    for anchor, key, offset in relations:
        anchored, dependent = positions[anchor], positions[key]
        if (anchored is not None and dependent is not None and anchored == dependent
                and 0 < abs(offset) < 12 * 32
                and not by_key[anchor].absolute_date and not by_key[key].absolute_date):
            positions[key] = (anchored[0], anchored[1] + offset)

    def spread(seed: str) -> None:
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            family, month = positions[current]
            for neighbour, offset in links[current]:
                proposed = (family, month + offset)
                if positions[neighbour] is None:
                    positions[neighbour] = proposed
                    queue.append(neighbour)
                # Contradictory relation evidence remains unresolved, not guessed.
                elif positions[neighbour] != proposed:
                    by_key[neighbour].temporal_uncertainty = " / ".join(_unique([
                        by_key[neighbour].temporal_uncertainty, "Conflicting relative chronology",
                    ]))

    for key, position in list(positions.items()):
        if position is not None:
            spread(key)
    for key in list(positions):
        if positions[key] is None and links[key]:
            positions[key] = (f"relative:{key}", 0)
            spread(key)
    return positions


def lived_events_before_reference(
    timeline: list[TimelineEvent], reference: ChatReferencePoint,
) -> list[TimelineEvent]:
    """Only chronology comparable to the boundary is safe as lived prompt context."""
    actual = [event.model_copy(deep=True) for event in timeline if event.canonicality == "canonical"]
    positions = _positions(actual)
    boundary = positions.get(_key(reference.event_key or ""))
    if boundary is None:
        return []
    lived = [event for event in actual if not event.is_death
            and (position := positions.get(_key(event.event_key))) is not None
            and position[0] == boundary[0] and position[1] <= boundary[1]]
    return sorted(lived, key=lambda event: positions[_key(event.event_key)][1])


def _terminal_evidence(event: TimelineEvent) -> bool:
    """A narrow, explicit end-of-story cue; source/chunk order is never evidence."""
    if event.narrative_role != "main":
        return False
    label = " ".join((event.event_key, event.time_label, event.relative_label or "")).casefold()
    return bool(re.search(
        r"(?:final|last)[-_ ](?:night|day|moment|scene|chapter|state)|"
        r"end[-_ ]of[-_ ](?:story|novel|life)|마지막[-_ ]?(?:밤|날|순간|장면)|최후",
        label,
    ))


def reconcile_timeline(
    existing: list[TimelineEvent], incoming: list[TimelineEvent], *,
    source_id: str | None = None, prefer_new: bool = False,
) -> list[TimelineEvent]:
    """Dedupe by semantic event key, retain conflicts, then order by time evidence."""
    by_key: dict[str, TimelineEvent] = {}
    original_order: dict[str, int] = {}
    for original, is_incoming in chain(
        ((event, False) for event in existing), ((event, True) for event in incoming),
    ):
        event = original.model_copy(deep=True)
        event.event_key = _key(event.event_key) or _key(event.summary)
        if not event.event_key:
            continue
        if source_id and is_incoming:
            event.source_ids = _unique(event.source_ids + [source_id])
        key = event.event_key
        if key in by_key:
            by_key[key] = _merge_event(by_key[key], event, prefer_new=prefer_new)
        else:
            original_order[key] = len(original_order)
            by_key[key] = event
    events = list(by_key.values())
    positions = _positions(events)

    def sort_key(event: TimelineEvent):
        position = positions.get(event.event_key)
        if position is None:
            return (3, original_order[event.event_key], 0, event.is_death)
        family, month = position
        family_rank = 0 if family == "age" else 1 if family == "year" else 2
        return (family_rank, family if family_rank == 2 else "", month, event.is_death)

    events.sort(key=sort_key)
    for index, event in enumerate(events, 1):
        event.sequence_index = index * 10
    return events


def _apply(state: CanonicalCharacterState, event: TimelineEvent) -> None:
    if event.age is not None:
        state.age = event.age
    changes = event.state_changes
    for field in (
        "occupation", "affiliations", "location", "physical_condition", "mental_condition", "abilities",
        "possessions", "personality", "speech_style", "goals", "loyalties",
    ):
        value = getattr(changes, field)
        if value is not None:
            setattr(state, field, value)
    relationships = {item.name.casefold(): item for item in state.relationships}
    for item in changes.relationships:
        relationships[item.name.casefold()] = item
    state.relationships = list(relationships.values())
    state.knowledge = _unique(state.knowledge + changes.knowledge)


def derive_chat_reference(
    timeline: list[TimelineEvent], previous: ChatReferencePoint | None = None,
) -> ChatReferencePoint | None:
    """Never use ingestion order or a post-death event as the default roleplay point."""
    actual = [event.model_copy(deep=True) for event in timeline if event.canonicality == "canonical"]
    if not actual:
        return ChatReferencePoint(
            status="unknown", summary="No canonical event established",
            reason="Chronology cannot be inferred from noncanonical or uncertain events",
        ) if timeline else None
    positions = _positions(actual)
    deaths = [event for event in actual if event.is_death]
    if deaths:
        death_families = {positions[_key(event.event_key)][0] for event in deaths
                          if positions.get(_key(event.event_key)) is not None}
        if len(death_families) > 1:
            return ChatReferencePoint(status="unknown", summary="Canonical death chronology unresolved",
                                      reason="Incomparable death boundaries")
        # An explicit death is the cutoff, regardless of its source chunk.
        boundary = max(deaths, key=lambda event: (positions.get(_key(event.event_key)) or ("", -1))[1])
        phase = "immediately_before_death"
        status = "deceased_in_canon"
    else:
        evidenced = [event for event in actual if positions.get(event.event_key) is not None]
        if not evidenced:
            return ChatReferencePoint(
                status="unknown", summary="Canonical chronology unresolved",
                reason="No dated or relative evidence establishes the latest living state",
            )
        families = {positions[_key(event.event_key)][0] for event in evidenced}
        terminal = [event for event in evidenced if _terminal_evidence(event)]
        if len(families) > 1 and terminal:
            terminal_families = {positions[_key(event.event_key)][0] for event in terminal}
            if len(terminal_families) == 1:
                family = next(iter(terminal_families))
                comparable = [event for event in evidenced if positions[_key(event.event_key)][0] == family]
            else:
                comparable = []
        elif len(families) == 1:
            comparable = evidenced
        else:
            # Incomparable undated components have no defensible latest event.
            return ChatReferencePoint(status="unknown", summary="Canonical chronology unresolved",
                                      reason="Disconnected event chronologies have no unique final state")
        if not comparable:
            return ChatReferencePoint(status="unknown", summary="Canonical chronology unresolved",
                                      reason="Conflicting final-state evidence")
        boundary = max(comparable, key=lambda event: positions[_key(event.event_key)][1])
        phase = "at_event"
        status = "alive" if evidenced else "unknown"

    boundary_position = positions.get(boundary.event_key)
    state = CanonicalCharacterState()
    comparable_events = sorted(
        (event for event in actual if positions.get(_key(event.event_key)) is not None
         and boundary_position is not None
         and positions[_key(event.event_key)][0] == boundary_position[0]),
        key=lambda event: positions[_key(event.event_key)][1],
    )
    for event in comparable_events:
        if event.event_key == boundary.event_key and phase == "immediately_before_death":
            break
        position = positions.get(event.event_key)
        if boundary_position is None and event.event_key != boundary.event_key:
            continue
        if boundary_position is not None and (position is None or position[0] != boundary_position[0]
                                              or position[1] > boundary_position[1]):
            continue
        _apply(state, event)
        if event.event_key == boundary.event_key:
            break
    if phase == "immediately_before_death" and boundary.age is not None:
        state.age = boundary.age
    summary = f"Immediately before: {boundary.summary}" if phase == "immediately_before_death" else boundary.summary
    return ChatReferencePoint(
        event_key=boundary.event_key, phase=phase, age=state.age, status=status,
        summary=summary,
        reason="Final living canonical state before death" if deaths else "Latest living canonical state supported by chronology",
        state=state,
    )
