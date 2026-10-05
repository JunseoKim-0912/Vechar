"""Token-bounded, source-order-preserving chunks with semantic cut points."""

from bisect import bisect_left
from dataclasses import dataclass
import re
import unicodedata
from collections.abc import Callable

from ..llm_config import (
    TRAINING_CHUNK_HARD_MAX_TOKENS, TRAINING_CHUNK_OVERLAP_TOKENS,
    TRAINING_CHUNK_TARGET_TOKENS,
)


@dataclass(frozen=True)
class TrainingChunk:
    index: int
    total: int
    source_start: int
    core_start: int
    core_end: int
    token_start: int
    token_end: int
    overlap_tokens: int
    text: str

    @property
    def has_overlap(self) -> bool:
        return self.overlap_tokens > 0


def normalize_training_text(source: str) -> str:
    """Normalize transport-level line endings without changing story content."""
    return source.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _boundaries(source: str) -> list[tuple[int, int]]:
    """A lower rank is a stronger semantic boundary."""
    choices: dict[int, int] = {}

    def record(position: int, rank: int) -> None:
        if 0 < position < len(source):
            choices[position] = min(rank, choices.get(position, rank))

    for match in re.finditer(r"(?im)^(?:[ \t]{0,3}#{1,6}[ \t]+|[ \t]*chapter[ \t]+\d+|[ \t]*제[ \t]*\d+[ \t]*장)", source):
        record(match.start(), 0)
    for match in re.finditer(r"\n[ \t]*\n", source):
        record(match.end(), 1)
    for match in re.finditer(r"\n", source):
        record(match.end(), 2)
    for match in re.finditer(r"[.!?。！？][\"'”’)?\]]*[ \t]+", source):
        record(match.end(), 3)
    return sorted(choices.items())


def _safe_cut(source: str, position: int, minimum: int) -> int:
    cut = min(max(position, minimum + 1), len(source))
    # Python indexes code points, not UTF-8 bytes. Avoid splitting a combining
    # sequence or an emoji joiner when a hard split is unavoidable.
    while cut < len(source) and cut > minimum + 1 and (
        unicodedata.category(source[cut]) in {"Mn", "Mc", "Me"}
        or source[cut] == "\u200d" or source[cut - 1] == "\u200d"
    ):
        cut -= 1
    return cut


def _semantic_cut(source: str, boundaries: list[tuple[int, int]], start: int, desired: int) -> int:
    if desired >= len(source):
        return len(source)
    minimum = start + max(1, int((desired - start) * 0.6))
    candidates = [(position, rank) for position, rank in boundaries
                  if minimum <= position <= desired]
    if candidates:
        position, _ = min(candidates, key=lambda item: (item[1], -item[0]))
        return position
    return _safe_cut(source, desired, start)


def _overlap_start(
    source: str, boundaries: list[tuple[int, int]], core_start: int, core_end: int,
    chars_per_token: float, count_tokens: Callable[[str], int],
) -> tuple[int, int]:
    approximate = max(core_start, core_end - max(1, int(TRAINING_CHUNK_OVERLAP_TOKENS * chars_per_token)))
    positions = [position for position, _ in boundaries]
    boundary_index = bisect_left(positions, approximate)
    minimum_overlap_chars = max(1, int(TRAINING_CHUNK_OVERLAP_TOKENS * chars_per_token / 2))
    if (boundary_index < len(positions)
            and positions[boundary_index] <= core_end - minimum_overlap_chars):
        approximate = positions[boundary_index]
    approximate = _safe_cut(source, approximate, core_start - 1) if approximate > core_start else core_start
    overlap_tokens = count_tokens(source[approximate:core_end]) if approximate < core_end else 0
    while overlap_tokens > TRAINING_CHUNK_OVERLAP_TOKENS and approximate < core_end:
        reduction = max(1, int((core_end - approximate) *
                               (overlap_tokens - TRAINING_CHUNK_OVERLAP_TOKENS) / overlap_tokens))
        approximate = _safe_cut(source, approximate + reduction, approximate)
        overlap_tokens = count_tokens(source[approximate:core_end]) if approximate < core_end else 0
    return approximate, overlap_tokens


def build_training_chunks(
    source: str, count_tokens: Callable[[str], int], total_tokens: int,
) -> list[TrainingChunk]:
    """Core spans cover the source exactly once; only bounded leading context overlaps."""
    if not source or total_tokens <= 0:
        raise ValueError("Training source and token count must be nonempty")
    boundaries = _boundaries(source)
    chars_per_token = len(source) / total_tokens
    drafts: list[tuple[int, int, int, int, int, int, str]] = []
    core_start = text_start = token_cursor = overlap_tokens = 0

    while core_start < len(source):
        # The full source was counted once. Avoid repeatedly sending a large
        # shrinking suffix to the token-count endpoint; count it exactly near
        # the last chunk, where the direct remainder decision matters.
        estimated_remaining = total_tokens - token_cursor + overlap_tokens
        if core_start == 0:
            remainder_tokens = total_tokens
        elif estimated_remaining <= TRAINING_CHUNK_HARD_MAX_TOKENS * 3 // 2:
            remainder_tokens = count_tokens(source[text_start:])
        else:
            remainder_tokens = TRAINING_CHUNK_HARD_MAX_TOKENS + 1
        if remainder_tokens <= TRAINING_CHUNK_HARD_MAX_TOKENS:
            end = len(source)
            chunk_tokens = remainder_tokens
        else:
            desired = min(len(source), max(core_start + 1,
                                           text_start + int(TRAINING_CHUNK_TARGET_TOKENS * chars_per_token)))
            end = _semantic_cut(source, boundaries, core_start, desired)
            chunk_tokens = count_tokens(source[text_start:end])
            # Local token density can differ substantially from the whole source.
            # The provider count, not the character estimate, enforces the hard cap.
            while chunk_tokens > TRAINING_CHUNK_HARD_MAX_TOKENS:
                span = end - text_start
                desired = text_start + max(1, int(span * TRAINING_CHUNK_TARGET_TOKENS / chunk_tokens))
                end = _semantic_cut(source, boundaries, core_start, desired)
                if end <= core_start:
                    end = _safe_cut(source, core_start + 1, core_start)
                chunk_tokens = count_tokens(source[text_start:end])
            if end <= core_start:
                raise ValueError("Cannot advance a token-bounded training chunk")

        core_tokens = max(1, chunk_tokens - overlap_tokens)
        next_cursor = min(total_tokens, token_cursor + core_tokens)
        if end == len(source):
            next_cursor = total_tokens
        drafts.append((text_start, core_start, end, token_cursor, next_cursor,
                       overlap_tokens, source[text_start:end]))
        token_cursor = next_cursor
        if end == len(source):
            break
        previous_core_start, core_start = core_start, end
        text_start, overlap_tokens = _overlap_start(
            source, boundaries, previous_core_start, core_start, chars_per_token, count_tokens,
        )

    total = len(drafts)
    return [TrainingChunk(index=index, total=total, source_start=start,
                          core_start=core_start, core_end=end, token_start=token_start,
                          token_end=token_end, overlap_tokens=overlap, text=text)
            for index, (start, core_start, end, token_start, token_end, overlap, text)
            in enumerate(drafts, 1)]
