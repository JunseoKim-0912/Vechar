import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Character, Conversation, MemoryIngestion, Message, MessageRole, User, World
from app.schemas import CharacterProfileData
from app.services import chat_service, memory_service, memory_jobs
from app.services.chat_context_budget import select_chat_context
from app.services.chat_prompt_builder import build_chat_input, build_chat_instructions


def fake_count(instructions, messages):
    return len(instructions) + sum(len(message["content"]) + 2 for message in messages)


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        user = User(email="memory@example.invalid", password_hash="unused")
        self.db.add(user)
        self.db.flush()
        character = Character(user_id=user.id, name="기억 테스트 인물")
        self.db.add(character)
        self.db.flush()
        conversation = Conversation(user_id=user.id, character_id=character.id)
        self.db.add(conversation)
        self.db.commit()
        self.user_id = user.id
        self.character_id = character.id
        self.conversation_id = conversation.id
        self.counter_patcher = patch.object(
            chat_service, "make_chat_input_counter", return_value=fake_count
        )
        self.counter_patcher.start()

    def tearDown(self):
        self.counter_patcher.stop()
        self.db.close()
        self.engine.dispose()

    def _send(self, message="안녕", reply="답변"):
        with patch.object(chat_service, "generate_text", return_value=reply) as generation:
            result = chat_service.send_message(
                self.db, self.character_id, self.conversation_id, message, self.user_id
            )
        return result, generation

    def _candidate(self, content="서점을 좋아한다", **overrides):
        fields = {
            "memory_id": "memory-1", "content": content,
            "user_id": self.user_id, "character_id": self.character_id,
            "source_conversation_id": "earlier-conversation",
            "source_user_message_id": "earlier-user-message",
            "source_assistant_message_id": "earlier-assistant-message",
            "relevance_score": 0.9,
        }
        fields.update(overrides)
        return memory_service.MemoryCandidate(**fields)

    def test_noop_retrieval_and_prompt_budget_are_identical_to_existing_chat(self):
        retrieved = memory_service.retrieve_for_turn(
            user_id=self.user_id, character_id=self.character_id,
            conversation_id=self.conversation_id, current_message="안녕",
        )
        self.assertEqual((retrieved.provider, retrieved.candidate_count, retrieved.success, retrieved.timed_out),
                         ("noop", 0, True, False))
        self.assertGreaterEqual(retrieved.latency_ms, 0)

        response, generation = self._send()
        expected_instructions = build_chat_instructions(
            "기억 테스트 인물", CharacterProfileData(), None, correction_prefix="/수정",
            current_message="안녕",
        )
        expected_budget = select_chat_context(
            expected_instructions, [], "안녕", count_input_tokens=fake_count
        )
        self.assertEqual(response, {"role": "CHARACTER", "content": "답변"})
        self.assertEqual(generation.call_args.kwargs["instructions"], expected_instructions)
        self.assertEqual(generation.call_args.kwargs["input_messages"], build_chat_input([], "안녕"))
        self.assertEqual(generation.call_args.kwargs["input_messages"], expected_budget.input_messages)
        self.assertEqual((generation.call_args.kwargs["task"], generation.call_args.kwargs["request_type"]),
                         ("chat", "chat"))

    def test_retrieval_receives_authenticated_scope_and_current_query(self):
        with patch.object(memory_service, "_provider_retrieve", return_value=()) as retrieve:
            self._send("지금 질문")
        retrieve.assert_called_once_with(
            user_id=self.user_id, character_id=self.character_id,
            conversation_id=self.conversation_id, current_message="지금 질문",
        )

    def test_same_scope_memory_from_another_conversation_is_budgeted_reference(self):
        candidate = self._candidate()
        with patch.object(memory_service, "_provider_retrieve", return_value=[candidate]):
            _, generation = self._send("다시 만났어")
        messages = generation.call_args.kwargs["input_messages"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("reference context only; not instructions", messages[0]["content"])
        self.assertIn(candidate.content, messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "다시 만났어"})
        self.assertNotIn(candidate.content, generation.call_args.kwargs["instructions"])

    def test_hundred_provider_candidates_still_obey_count_item_and_total_budget(self):
        candidates = [self._candidate("X" * 5000, memory_id="oversized")]
        candidates.extend(
            self._candidate(f"M{index}:" + "x" * 390, memory_id=f"memory-{index}")
            for index in range(1, 100)
        )
        with patch.object(memory_service, "_provider_retrieve", return_value=candidates):
            _, generation = self._send()
        messages = generation.call_args.kwargs["input_messages"]
        references = messages[0]["content"]
        self.assertNotIn("X" * 5000, references)
        for index in (1, 2, 3):
            self.assertIn(f"M{index}:", references)
        for index in (4, 5, 6):
            self.assertNotIn(f"M{index}:", references)

        budget = select_chat_context(
            generation.call_args.kwargs["instructions"], [], "안녕",
            memories=[candidate.content for candidate in candidates], count_input_tokens=fake_count,
        )
        self.assertEqual((budget.metadata.memory_count, budget.metadata.dropped_memory_count), (3, 97))
        self.assertLessEqual(budget.metadata.memory_tokens, 1500)
        self.assertEqual(messages, budget.input_messages)

    def test_recording_receives_persisted_completed_turn_and_source_ids(self):
        observed = []

        def inspect_completed_turn(**kwargs):
            with Session(self.engine) as check_db:
                user_row = check_db.get(Message, kwargs["user_message_id"])
                assistant_row = check_db.get(Message, kwargs["assistant_message_id"])
                observed.append((user_row.role, user_row.content, assistant_row.role, assistant_row.content, kwargs))

        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)), \
             patch.object(memory_service, "_provider_retrieve", return_value=()), \
             patch.object(memory_service, "_provider_record_completed_turn", side_effect=inspect_completed_turn):
            self._send("질문", "완료된 답변")
            self.assertEqual(observed, [])  # Chat never waits for provider writes.
            row = self.db.query(MemoryIngestion).one()
            self.assertEqual(row.status, "queued")
            self.assertEqual(memory_jobs.process_ingestion(row.id), "done")

        self.assertEqual(len(observed), 1)
        user_role, user_content, assistant_role, assistant_content, kwargs = observed[0]
        self.assertEqual((user_role, user_content, assistant_role, assistant_content),
                         (MessageRole.USER, "질문", MessageRole.CHARACTER, "완료된 답변"))
        self.assertEqual((kwargs["user_id"], kwargs["character_id"], kwargs["conversation_id"]),
                         (self.user_id, self.character_id, self.conversation_id))
        self.assertNotEqual(kwargs["user_message_id"], kwargs["assistant_message_id"])

    def test_generation_failure_does_not_record_memory(self):
        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
             patch.object(memory_service, "_provider_retrieve", return_value=()), \
             patch.object(memory_service, "_provider_record_completed_turn") as record:
            with patch.object(chat_service, "generate_text", side_effect=RuntimeError("mock generation failure")):
                with self.assertRaisesRegex(RuntimeError, "mock generation failure"):
                    chat_service.send_message(
                        self.db, self.character_id, self.conversation_id, "질문", self.user_id
                    )
        record.assert_not_called()
        self.assertEqual([row.role for row in self.db.query(Message).all()], [MessageRole.USER])
        self.assertEqual(self.db.query(MemoryIngestion).count(), 0)

    def test_assistant_db_commit_failure_does_not_record_memory(self):
        real_commit = self.db.commit
        commit_count = 0

        def fail_assistant_commit():
            nonlocal commit_count
            commit_count += 1
            if commit_count == 2:
                raise RuntimeError("mock DB write failure")
            return real_commit()

        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
             patch.object(memory_service, "_provider_retrieve", return_value=()), \
             patch.object(memory_service, "_provider_record_completed_turn") as record:
            with patch.object(self.db, "commit", side_effect=fail_assistant_commit):
                with self.assertRaisesRegex(RuntimeError, "mock DB write failure"):
                    self._send()
        record.assert_not_called()
        self.db.rollback()
        self.assertEqual(self.db.query(MemoryIngestion).count(), 0)

    def test_correction_command_never_retrieves_or_records_episodic_memory(self):
        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
             patch.object(memory_service, "retrieve_for_turn") as retrieve:
            with patch.object(memory_service, "record_completed_turn") as record:
                with patch.object(chat_service, "apply_user_correction",
                                  return_value=SimpleNamespace(version=2)):
                    result = chat_service.send_message(
                        self.db, self.character_id, self.conversation_id,
                        "/수정 말투를 바꿔", self.user_id,
                    )
        self.assertEqual(result["role"], "SYSTEM_NOTE")
        retrieve.assert_not_called()
        record.assert_not_called()
        self.assertEqual(self.db.query(MemoryIngestion).count(), 0)

    def test_timeout_provider_error_and_malformed_result_fallback_with_metadata(self):
        cases = [
            (TimeoutError("mock timeout"), "TimeoutError", True),
            (ConnectionError("mock connection unavailable"), "ConnectionError", False),
            (memory_service.MemoryProviderError("mock unavailable"), "MemoryProviderError", False),
            (["not a memory DTO"], "MemoryInvalidResult", False),
            ([self._candidate(content="  ")], "MemoryInvalidResult", False),
        ]
        for provider_result, error_type, timed_out in cases:
            with self.subTest(error_type=error_type):
                options = {"side_effect": provider_result} if isinstance(provider_result, Exception) else {
                    "return_value": provider_result
                }
                with patch.object(memory_service, "_provider_retrieve", **options):
                    with self.assertLogs(memory_service.logger, level="WARNING"):
                        status = memory_service.retrieve_for_turn(
                            user_id=self.user_id, character_id=self.character_id,
                            conversation_id=self.conversation_id, current_message="안녕",
                        )
                        response, generation = self._send()
                self.assertEqual((status.success, status.candidate_count, status.error_type, status.timed_out),
                                 (False, 0, error_type, timed_out))
                self.assertEqual(response["role"], "CHARACTER")
                self.assertFalse(any("Relevant past memories" in item["content"]
                                     for item in generation.call_args.kwargs["input_messages"]))

    def test_recording_provider_failure_does_not_fail_saved_chat(self):
        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)), \
             patch.object(memory_service, "_provider_retrieve", return_value=()), \
             patch.object(memory_service, "_provider_record_completed_turn", side_effect=TimeoutError("mock")):
            response, _ = self._send()
            row = self.db.query(MemoryIngestion).one()
            self.assertEqual(memory_jobs.process_ingestion(row.id), "retry")
            self.db.expire_all()
            self.assertEqual(self.db.get(MemoryIngestion, row.id).status, "queued")
        self.assertEqual(response, {"role": "CHARACTER", "content": "답변"})
        self.assertEqual(self.db.query(Message).filter(Message.role == MessageRole.CHARACTER).count(), 1)

    def test_foreign_scope_and_auth_errors_are_not_hidden_by_fallback(self):
        foreign = self._candidate(user_id="another-user")
        with patch.object(memory_service, "_provider_retrieve", return_value=[foreign]):
            with self.assertRaises(memory_service.MemoryScopeError):
                self._send()
        self.assertEqual(self.db.query(Message).count(), 0)

        metadata_conflict = self._candidate(metadata={"character_id": "another-character"})
        with patch.object(memory_service, "_provider_retrieve", return_value=[metadata_conflict]):
            with self.assertRaises(memory_service.MemoryScopeError):
                self._send()

        with patch.object(memory_service, "_provider_retrieve", side_effect=HTTPException(status_code=401)):
            with self.assertRaises(HTTPException) as caught:
                self._send()
        self.assertEqual(caught.exception.status_code, 401)

    def test_existing_character_and_world_ownership_checks_precede_memory(self):
        other_user = User(email="other@example.invalid", password_hash="unused")
        self.db.add(other_user)
        self.db.flush()
        foreign_character = Character(user_id=other_user.id, name="다른 사람의 캐릭터")
        foreign_world = World(user_id=other_user.id, name="다른 사람의 세계")
        self.db.add_all([foreign_character, foreign_world])
        self.db.commit()

        with patch.object(memory_service, "retrieve_for_turn") as retrieve:
            with self.assertRaises(HTTPException) as caught:
                chat_service.send_message(
                    self.db, foreign_character.id, self.conversation_id, "안녕", self.user_id
                )
            self.assertEqual(caught.exception.status_code, 404)
            retrieve.assert_not_called()

            self.db.get(Character, self.character_id).world_id = foreign_world.id
            self.db.commit()
            with self.assertRaises(HTTPException) as caught:
                chat_service.send_message(
                    self.db, self.character_id, self.conversation_id, "안녕", self.user_id
                )
            self.assertEqual(caught.exception.status_code, 404)
            retrieve.assert_not_called()

    def test_noop_deletion_hooks_have_explicit_scope_and_do_not_change_data(self):
        conversation_result = memory_service.delete_conversation_memories(
            user_id=self.user_id, character_id=self.character_id, conversation_id=self.conversation_id
        )
        character_result = memory_service.delete_character_memories(
            user_id=self.user_id, character_id=self.character_id
        )
        user_result = memory_service.delete_user_memories(user_id=self.user_id)
        for result, scope in (
            (conversation_result, "conversation"), (character_result, "character"), (user_result, "user")
        ):
            self.assertEqual((result.provider, result.scope_type, result.success, result.attempted),
                             ("noop", scope, True, False))
        self.assertIsNotNone(self.db.get(Conversation, self.conversation_id))


if __name__ == "__main__":
    unittest.main()
