import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import llm, main
from app.auth import issue_token
from app.database import Base, get_db
from app.models import Character, Conversation, MemoryDeletion, Message, MessageRole, User, World
from app.services import memory_service, memory_jobs


class MemoryDeletionTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            alice = User(email="delete-alice@example.invalid", password_hash="unused")
            bob = User(email="delete-bob@example.invalid", password_hash="unused")
            db.add_all([alice, bob])
            db.flush()
            world = World(user_id=alice.id, name="삭제 테스트 세계")
            db.add(world)
            db.flush()
            character = Character(user_id=alice.id, world_id=world.id, name="삭제 테스트 인물")
            foreign = Character(user_id=bob.id, name="다른 사람의 인물")
            db.add_all([character, foreign])
            db.flush()
            first = Conversation(user_id=alice.id, character_id=character.id)
            second = Conversation(user_id=alice.id, character_id=character.id)
            db.add_all([first, second])
            db.flush()
            db.add_all([
                Message(conversation_id=first.id, role=MessageRole.USER, content="첫 대화"),
                Message(conversation_id=second.id, role=MessageRole.USER, content="둘째 대화"),
            ])
            db.commit()
            self.alice_id, self.bob_id = alice.id, bob.id
            self.world_id, self.character_id, self.foreign_id = world.id, character.id, foreign.id
            self.first_id, self.second_id = first.id, second.id

        def test_db():
            with Session(self.engine) as db:
                yield db

        main.app.dependency_overrides[get_db] = test_db
        self.engine_patch = patch.object(main, "engine", self.engine)
        self.engine_patch.start()
        self.no_network = patch.object(llm, "_get_client", side_effect=AssertionError("Unexpected OpenAI call"))
        self.no_network.start()
        self.client = TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.no_network.stop()
        self.engine_patch.stop()
        main.app.dependency_overrides.clear()
        self.engine.dispose()

    def _headers(self, user_id=None):
        return {"Authorization": f"Bearer {issue_token(user_id or self.alice_id)}"}

    def _delete_character(self, user_id=None):
        return self.client.delete(f"/characters/{self.character_id}", headers=self._headers(user_id))

    def test_character_deletion_persists_tombstone_before_async_cleanup(self):
        observed = []

        def inspect_delete(**kwargs):
            with Session(self.engine) as db:
                observed.append((kwargs, db.get(Character, self.character_id) is not None,
                                 db.query(Conversation).filter(Conversation.character_id == self.character_id).count()))

        with patch.object(memory_service, "PROVIDER_NAME", "mock"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)), \
             patch.object(memory_service, "_provider_delete_character", side_effect=inspect_delete) as hook:
            response = self._delete_character()
            with Session(self.engine) as db:
                deletion = db.query(MemoryDeletion).one()
                self.assertEqual((deletion.status, deletion.user_id, deletion.character_id),
                                 ("queued", self.alice_id, self.character_id))
            hook.assert_not_called()
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")

        self.assertEqual(response.status_code, 204)
        hook.assert_called_once_with(user_id=self.alice_id, character_id=self.character_id)
        self.assertEqual(observed[0][1:], (False, 0))
        with Session(self.engine) as db:
            self.assertIsNone(db.get(Character, self.character_id))
            self.assertEqual(db.query(Conversation).filter(Conversation.character_id == self.character_id).count(), 0)
            self.assertEqual(db.query(Message).filter(
                Message.conversation_id.in_([self.first_id, self.second_id])
            ).count(), 0)
            self.assertIsNotNone(db.get(World, self.world_id))

    def test_foreign_character_cannot_trigger_memory_deletion(self):
        with patch.object(memory_service, "_provider_delete_character") as hook:
            response = self._delete_character(self.bob_id)
        self.assertEqual(response.status_code, 404)
        hook.assert_not_called()
        with Session(self.engine) as db:
            self.assertIsNotNone(db.get(Character, self.character_id))

    def test_retryable_partial_provider_failure_keeps_tombstone_for_retry(self):
        error = memory_service.MemoryDeletionError("mock partial delete", retryable=True, partial=True)
        with patch.object(memory_service, "PROVIDER_NAME", "mock"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)), \
             patch.object(memory_service, "_provider_delete_character", side_effect=error):
            response = self._delete_character()
            with Session(self.engine) as db:
                deletion = db.query(MemoryDeletion).one()
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "retry")
        self.assertEqual(response.status_code, 204)
        with Session(self.engine) as db:
            self.assertIsNone(db.get(Character, self.character_id))
            self.assertEqual(db.get(MemoryDeletion, deletion.id).status, "queued")

    def test_nonretryable_failure_retains_failed_tombstone(self):
        error = memory_service.MemoryDeletionError("mock permanent failure", retryable=False)
        with patch.object(memory_service, "PROVIDER_NAME", "mock"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)), \
             patch.object(memory_service, "_provider_delete_character", side_effect=error):
            response = self._delete_character()
            with Session(self.engine) as db:
                deletion = db.query(MemoryDeletion).one()
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")
        self.assertEqual(response.status_code, 204)
        with Session(self.engine) as db:
            self.assertIsNone(db.get(Character, self.character_id))
            self.assertEqual(db.get(MemoryDeletion, deletion.id).status, "failed")

    def test_ambiguous_provider_result_retains_retryable_tombstone(self):
        with patch.object(memory_service, "PROVIDER_NAME", "mock"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)), \
             patch.object(memory_service, "_provider_delete_character", return_value=False):
            response = self._delete_character()
            with Session(self.engine) as db:
                deletion = db.query(MemoryDeletion).one()
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "retry")
        self.assertEqual(response.status_code, 204)
        with Session(self.engine) as db:
            self.assertIsNone(db.get(Character, self.character_id))

    def test_already_absent_provider_memory_counts_as_success(self):
        with patch.object(memory_service, "PROVIDER_NAME", "mock"), \
             patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)):
            with patch.object(memory_service, "_provider_delete_character",
                              side_effect=memory_service.MemoryDeletionNotFound()):
                response = self._delete_character()
                with Session(self.engine) as db:
                    deletion = db.query(MemoryDeletion).one()
                self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")
        self.assertEqual(response.status_code, 204)
        with Session(self.engine) as db:
            self.assertIsNone(db.get(Character, self.character_id))

    def test_conversation_contract_targets_only_source_session_and_is_idempotent(self):
        external = {self.first_id, self.second_id}
        calls = []

        def delete_one(**kwargs):
            calls.append(kwargs)
            target = kwargs["conversation_id"]
            if target not in external:
                raise memory_service.MemoryDeletionNotFound()
            external.remove(target)

        with patch.object(memory_service, "PROVIDER_NAME", "mock"):
            with patch.object(memory_service, "_provider_delete_conversation", side_effect=delete_one):
                first = memory_service.delete_conversation_memories(
                    user_id=self.alice_id, character_id=self.character_id, conversation_id=self.first_id
                )
                repeated = memory_service.delete_conversation_memories(
                    user_id=self.alice_id, character_id=self.character_id, conversation_id=self.first_id
                )
        self.assertEqual(external, {self.second_id})
        self.assertEqual(calls, [{
            "user_id": self.alice_id, "character_id": self.character_id, "conversation_id": self.first_id,
        }] * 2)
        self.assertEqual((first.success, first.not_found, first.scope_type, first.scope_id),
                         (True, False, "conversation", self.first_id))
        self.assertEqual((repeated.success, repeated.not_found), (True, True))
        with Session(self.engine) as db:
            self.assertIsNotNone(db.get(Conversation, self.first_id))  # No conversation DELETE endpoint yet.

    def test_user_scope_deletion_contract_is_repeatable(self):
        deleted_users = set()

        def delete_user(**kwargs):
            user_id = kwargs["user_id"]
            if user_id in deleted_users:
                raise memory_service.MemoryDeletionNotFound()
            deleted_users.add(user_id)

        with patch.object(memory_service, "PROVIDER_NAME", "mock"):
            with patch.object(memory_service, "_provider_delete_user", side_effect=delete_user) as hook:
                first = memory_service.delete_user_memories(user_id=self.alice_id)
                repeated = memory_service.delete_user_memories(user_id=self.alice_id)
        self.assertEqual((first.success, first.not_found, first.scope_type, first.scope_id),
                         (True, False, "user", self.alice_id))
        self.assertEqual((repeated.success, repeated.not_found), (True, True))
        self.assertEqual(hook.call_count, 2)

    def test_timeout_result_marks_retryable_without_deleting_db(self):
        with patch.object(memory_service, "PROVIDER_NAME", "mock"):
            with patch.object(memory_service, "_provider_delete_character", side_effect=TimeoutError("mock timeout")):
                with self.assertLogs(memory_service.logger, level="WARNING"):
                    result = memory_service.delete_character_memories(
                        user_id=self.alice_id, character_id=self.character_id
                    )
        self.assertEqual((result.success, result.retryable, result.error_type, result.attempted),
                         (False, True, "TimeoutError", True))
        with Session(self.engine) as db:
            self.assertIsNotNone(db.get(Character, self.character_id))

    def test_world_delete_does_not_delete_character_or_invoke_memory_hook(self):
        with patch.object(memory_service, "_provider_delete_character") as character_hook:
            with patch.object(memory_service, "_provider_delete_conversation") as conversation_hook:
                response = self.client.delete(f"/worlds/{self.world_id}", headers=self._headers())
        self.assertEqual(response.status_code, 204)
        character_hook.assert_not_called()
        conversation_hook.assert_not_called()
        with Session(self.engine) as db:
            character = db.get(Character, self.character_id)
            self.assertIsNotNone(character)
            self.assertIsNone(character.world_id)
            self.assertEqual(db.query(Conversation).filter(Conversation.character_id == self.character_id).count(), 2)


if __name__ == "__main__":
    unittest.main()
