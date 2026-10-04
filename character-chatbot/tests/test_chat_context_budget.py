import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from app import llm
from app.chat_context_config import ChatContextLimits
from app.models import MessageRole
from app.services.chat_context_budget import select_chat_context
from app.services.chat_prompt_builder import build_chat_input


def fake_count(instructions, messages):
    """Deterministic test counter; production uses the gateway's Responses count."""
    return len(instructions) + sum(len(message["content"]) + 2 for message in messages)


def limits(**overrides):
    values = {
        "input_tokens": 10000,
        "input_bytes": 60000,
        "memory_tokens": 1500,
        "memory_items": 5,
        "memory_item_tokens": 500,
        "history_tokens": 6000,
    }
    values.update(overrides)
    return ChatContextLimits(**values)


class ChatContextBudgetTests(unittest.TestCase):
    def select(self, history=(), current="현재 질문", memories=(), instructions="정체성 / 세계 / 규칙", **overrides):
        return select_chat_context(
            instructions, history, current, memories=memories,
            count_input_tokens=fake_count, limits=limits(**overrides),
        )

    def test_history_below_budget_is_unchanged_and_current_is_last(self):
        history = [(MessageRole.USER, f"질문 {index}") for index in range(30)]
        result = self.select(history)
        self.assertEqual(result.input_messages, build_chat_input(history, "현재 질문"))
        self.assertEqual(result.metadata.history_count, 30)
        self.assertEqual(result.metadata.dropped_history_message_count, 0)
        self.assertEqual(result.input_messages[-1], {"role": "user", "content": "현재 질문"})

    def test_history_drops_oldest_messages_not_turn_pairs(self):
        history = [
            (MessageRole.USER, "old" * 4),
            (MessageRole.CHARACTER, "middle" * 2),
            (MessageRole.USER, "new" * 4),
        ]
        result = self.select(history, history_tokens=28)
        self.assertEqual(result.input_messages, build_chat_input(history[1:], "현재 질문"))
        self.assertEqual((result.metadata.history_count, result.metadata.dropped_history_message_count), (2, 1))
        self.assertLessEqual(result.metadata.history_tokens, 28)

    def test_incomplete_user_only_history_keeps_newest_and_current(self):
        history = [(MessageRole.USER, "오래된 질문"), (MessageRole.USER, "최근 질문")]
        newest_cost = fake_count("", build_chat_input(history[-1:], "현재 질문")) - fake_count(
            "", build_chat_input([], "현재 질문")
        )
        result = self.select(history, history_tokens=newest_cost)
        self.assertEqual(result.input_messages, build_chat_input(history[-1:], "현재 질문"))
        self.assertEqual(result.metadata.history_count, 1)

    def test_long_history_needs_only_logarithmic_count_probes(self):
        counter = Mock(side_effect=fake_count)
        history = [(MessageRole.USER, "H" * 300) for _ in range(30)]
        result = select_chat_context(
            "canonical", history, "current", count_input_tokens=counter,
            limits=limits(history_tokens=600),
        )
        self.assertEqual(result.metadata.history_count, 1)
        self.assertLessEqual(counter.call_count, 9)

    def test_no_memory_matches_existing_builder_and_metadata_counts_role_overhead(self):
        history = [(MessageRole.CHARACTER, "이전 답변")]
        result = self.select(history)
        self.assertEqual(result.input_messages, build_chat_input(history, "현재 질문"))
        self.assertEqual(result.metadata.memory_count, 0)
        self.assertEqual(result.metadata.memory_tokens, 0)
        self.assertEqual(result.metadata.current_message_tokens, len("현재 질문"))
        self.assertEqual(result.metadata.history_tokens, len("이전 답변") + 2)
        self.assertEqual(result.metadata.total_estimated_input_tokens, fake_count("정체성 / 세계 / 규칙", result.input_messages))

    def test_up_to_five_small_memories_keep_relevance_order(self):
        memories = [f"기억 {index}" for index in range(5)]
        result = self.select(memories=memories)
        self.assertEqual(result.input_messages, build_chat_input([], "현재 질문", memories=memories))
        self.assertEqual((result.metadata.memory_count, result.metadata.dropped_memory_count), (5, 0))

    def test_memory_count_cap_ignores_later_retrieval_items(self):
        memories = [f"기억 {index}" for index in range(100)]
        result = self.select(memories=memories)
        self.assertEqual(result.input_messages, build_chat_input([], "현재 질문", memories=memories[:5]))
        self.assertEqual((result.metadata.memory_count, result.metadata.dropped_memory_count), (5, 95))

    def test_memory_budget_drops_lower_ranked_items(self):
        memories = ["high", "middle", "low"]
        base = fake_count("", build_chat_input([], "현재 질문"))
        first_two_cost = fake_count("", build_chat_input([], "현재 질문", memories=memories[:2])) - base
        result = self.select(memories=memories, memory_tokens=first_two_cost)
        self.assertEqual(result.input_messages, build_chat_input([], "현재 질문", memories=memories[:2]))
        self.assertEqual((result.metadata.memory_count, result.metadata.dropped_memory_count), (2, 1))

    def test_oversized_memory_is_excluded_whole_and_does_not_monopolize_budget(self):
        memories = ["X" * 1000, "유용한 기억"]
        result = self.select(memories=memories, memory_item_tokens=200)
        self.assertEqual(result.input_messages, build_chat_input([], "현재 질문", memories=["유용한 기억"]))
        self.assertEqual((result.metadata.memory_count, result.metadata.dropped_memory_count), (1, 1))

    def test_optional_layers_cannot_displace_canonical_or_current_message(self):
        instructions = "C" * 40
        current = "U" * 20
        required = fake_count(instructions, build_chat_input([], current))
        history = [(MessageRole.USER, "H" * 100)]
        result = self.select(
            history, current, ["M" * 100], instructions,
            input_tokens=required, memory_item_tokens=1000,
        )
        self.assertEqual(result.input_messages, build_chat_input([], current))
        self.assertEqual(result.metadata.dropped_memory_count, 1)
        self.assertEqual(result.metadata.dropped_history_message_count, 1)
        self.assertEqual(result.metadata.canonical_tokens, fake_count(instructions, build_chat_input([], ".")))

    def test_excessive_canonical_current_and_combination_fail_explicitly(self):
        cases = [
            ("C" * 60, "u", "chat_canonical_context_too_large"),
            ("C", "U" * 61, "chat_current_message_too_large"),
            ("C" * 35, "U" * 35, "chat_required_context_too_large"),
        ]
        for instructions, current, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(HTTPException) as caught:
                    self.select(current=current, instructions=instructions, input_tokens=60)
                self.assertEqual(caught.exception.status_code, 413)
                self.assertEqual(caught.exception.detail["code"], expected_code)

    def test_final_byte_guard_also_trims_optional_layers(self):
        required_bytes = llm.input_size_bytes("canon", build_chat_input([], "current"))
        result = self.select(
            [(MessageRole.USER, "H" * 200)], "current", ["M" * 200], "canon",
            input_bytes=required_bytes, memory_item_tokens=1000,
        )
        self.assertEqual(result.input_messages, build_chat_input([], "current"))

    def test_gateway_counter_uses_same_model_count_endpoint_and_capacity_check(self):
        client = Mock()
        client.responses.input_tokens.count.return_value.input_tokens = 42
        db = Mock()
        with patch.object(llm, "model_for_task", return_value="chat-model") as model:
            with patch.object(llm, "check_capacity") as capacity:
                with patch.object(llm, "_get_client", return_value=client):
                    counter = llm.make_chat_input_counter(db, "user-id", 2000)
                    result = counter("canonical", [{"role": "user", "content": "hello"}])
        self.assertEqual(result, 42)
        model.assert_called_once_with("chat")
        capacity.assert_called_once_with(db.get_bind.return_value, "user-id", "chat-model", 2000)
        client.responses.input_tokens.count.assert_called_once_with(
            model="chat-model", instructions="canonical", input=[{"role": "user", "content": "hello"}],
        )
        client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
