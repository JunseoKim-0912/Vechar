"""Fallback boundary and size invariants; token counting is local and deterministic."""

import unittest

from app.llm_config import ADAPTIVE_SPLIT_OVERLAP_TOKENS, MIN_ADAPTIVE_CHUNK_TOKENS
from app.services.training_chunker import TrainingChunk, split_training_chunk


class AdaptiveChunkerTests(unittest.TestCase):
    def test_prefers_chapter_boundary_and_keeps_core_coverage_with_small_overlap(self):
        source = "p" * 100 + "A" * 5900 + "\n# Chapter 2\n" + "B" * 6100
        parent = TrainingChunk(2, 2, 0, 100, len(source), 0, len(source) - 100, 100,
                               source)
        children = split_training_chunk(source, parent, len)
        self.assertIsNotNone(children)
        left, right = children
        self.assertEqual(source[left.core_start:left.core_end] + source[right.core_start:right.core_end],
                         source[parent.core_start:parent.core_end])
        self.assertEqual(left.source_start, parent.source_start)
        self.assertEqual(right.core_start, left.core_end)
        self.assertGreaterEqual(len(source[left.core_start:left.core_end]), MIN_ADAPTIVE_CHUNK_TOKENS)
        self.assertGreaterEqual(len(source[right.core_start:right.core_end]), MIN_ADAPTIVE_CHUNK_TOKENS)
        self.assertLessEqual(right.overlap_tokens, ADAPTIVE_SPLIT_OVERLAP_TOKENS)
        self.assertGreater(right.source_start, parent.core_start)
        self.assertTrue(source[left.core_end:left.core_end + 2] in {"\n#", "# "})

    def test_refuses_parent_that_cannot_make_two_minimum_children(self):
        source = "a" * (2 * MIN_ADAPTIVE_CHUNK_TOKENS - 1)
        parent = TrainingChunk(1, 1, 0, 0, len(source), 0, len(source), 0, source)
        self.assertIsNone(split_training_chunk(source, parent, len))
