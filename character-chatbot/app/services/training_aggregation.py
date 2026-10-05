"""Typed intermediate evidence aggregation; no canonical profile is written here."""

from dataclasses import dataclass
import re
from typing import Generic, TypeVar

from ..schemas import (
    CharacterProfileData, MentionedCharacter, TimelineEvent, WorldProfileData,
)
from .character_timeline import derive_chat_reference, reconcile_timeline
from .training_chunker import TrainingChunk


ProfileT = TypeVar("ProfileT", CharacterProfileData, WorldProfileData)


@dataclass(frozen=True)
class ChunkEvidence(Generic[ProfileT]):
    chunk: TrainingChunk
    extracted: ProfileT


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        marker = _key(item)
        if marker and marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def aggregate_character_evidence(
    pieces: list[ChunkEvidence[CharacterProfileData]], source_id: str,
) -> CharacterProfileData:
    """Merge overlapping evidence without treating chunk order as event chronology."""
    if not pieces:
        raise ValueError("No character chunk evidence")
    events: list[TimelineEvent] = []
    signatures: dict[tuple[str, int | None, int | None, str | None], list[tuple[str, int]]] = {}
    for piece in pieces:
        for original in piece.extracted.timeline:
            event = original.model_copy(deep=True)
            event.source_ids = _dedupe(event.source_ids + [source_id])
            event.chunk_indices = list(dict.fromkeys(event.chunk_indices + [piece.chunk.index]))
            # Only adjacent overlapping chunks get a fallback description match;
            # distant identical wording may describe distinct repeated events.
            signature = (_key(event.summary), event.age, event.absolute_year, event.absolute_date)
            matching = next((key for key, index in signatures.get(signature, [])
                             if piece.chunk.has_overlap and index == piece.chunk.index - 1), None)
            if signature[0] and matching:
                event.event_key = matching
            signatures.setdefault(signature, []).append((event.event_key, piece.chunk.index))
            events.append(event)
    timeline = reconcile_timeline([], events)
    return CharacterProfileData(
        personality_summary=next((piece.extracted.personality_summary for piece in pieces
                                  if piece.extracted.personality_summary), ""),
        speech_style=next((piece.extracted.speech_style for piece in pieces
                           if piece.extracted.speech_style), ""),
        background_facts=_dedupe([fact for piece in pieces for fact in piece.extracted.background_facts]),
        relationships=_dedupe([relation for piece in pieces for relation in piece.extracted.relationships]),
        sample_dialogues=_dedupe([dialogue for piece in pieces for dialogue in piece.extracted.sample_dialogues]),
        do_not_do=_dedupe([rule for piece in pieces for rule in piece.extracted.do_not_do]),
        timeline=timeline,
        chat_reference_point=derive_chat_reference(timeline),
    )


def _merge_characters(characters: list[MentionedCharacter]) -> list[MentionedCharacter]:
    merged: list[MentionedCharacter] = []
    for character in characters:
        aliases = _dedupe([character.name] + character.aliases)
        matching = next((item for item in merged if {_key(item.name), *map(_key, item.aliases)}
                         & set(map(_key, aliases))), None)
        if matching is None:
            merged.append(character.model_copy(deep=True))
        else:
            matching.aliases = [name for name in _dedupe(matching.aliases + aliases)
                                if _key(name) != _key(matching.name)]
    return merged


def aggregate_world_evidence(pieces: list[ChunkEvidence[WorldProfileData]]) -> WorldProfileData:
    if not pieces:
        raise ValueError("No world chunk evidence")
    return WorldProfileData(
        world_summary=next((piece.extracted.world_summary for piece in pieces
                            if piece.extracted.world_summary), ""),
        key_facts=_dedupe([fact for piece in pieces for fact in piece.extracted.key_facts]),
        timeline_notes=_dedupe([note for piece in pieces for note in piece.extracted.timeline_notes]),
        mentioned_characters=_merge_characters([
            character for piece in pieces for character in piece.extracted.mentioned_characters
        ]),
    )
