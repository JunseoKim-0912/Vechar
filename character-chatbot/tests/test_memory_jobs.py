"""Durable memory outbox tests; no real MemMachine or OpenAI requests."""

import unittest
from datetime import timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.main import app
from app.models import (
    Character, Conversation, MemoryDeletion, MemoryIngestion, Message, MessageRole, User,
)
from app.services import memory_jobs, memory_service
from app.services.training_queue import MEMORY_DELETE_TOPIC, MEMORY_INGEST_TOPIC


class MemoryJobTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        user = User(email="jobs@example.invalid", password_hash="unused")
        self.db.add(user)
        self.db.flush()
        character = Character(user_id=user.id, name="A")
        self.db.add(character)
        self.db.flush()
        conversation = Conversation(user_id=user.id, character_id=character.id)
        self.db.add(conversation)
        self.db.flush()
        user_turn = Message(conversation_id=conversation.id, role=MessageRole.USER, content="We met at 18")
        assistant_turn = Message(conversation_id=conversation.id, role=MessageRole.CHARACTER, content="I remember")
        self.db.add_all([user_turn, assistant_turn])
        self.db.commit()
        self.scope = dict(user_id=user.id, character_id=character.id,
                          conversation_id=conversation.id, user_message_id=user_turn.id,
                          assistant_message_id=assistant_turn.id)
        self.patches = [
            patch.object(memory_service, "PROVIDER_NAME", "memmachine"),
            patch.object(memory_jobs, "SessionLocal", lambda: Session(self.engine)),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.db.close()
        self.engine.dispose()

    def ingestion(self):
        row = memory_jobs.schedule_ingestion(self.db, **self.scope)
        self.db.commit()
        return row.id

    def test_same_completed_turn_has_one_ledger_record_and_one_write(self):
        first = self.ingestion()
        second = memory_jobs.schedule_ingestion(self.db, **self.scope)
        self.assertEqual(first, second.id)
        self.assertEqual(self.db.query(MemoryIngestion).count(), 1)
        with patch.object(memory_service, "_provider_record_completed_turn") as provider:
            self.assertEqual(memory_jobs.process_ingestion(first), "done")
            self.assertEqual(memory_jobs.process_ingestion(first), "done")
        provider.assert_called_once()
        kwargs = provider.call_args.kwargs
        self.assertEqual(kwargs["user_message"], "We met at 18")
        self.assertEqual(kwargs["assistant_message"], "I remember")
        self.db.expire_all()
        self.assertEqual(self.db.get(MemoryIngestion, first).status, "completed")

    def test_stale_lease_reclaim_and_live_lease_exclusion(self):
        operation_id = self.ingestion()
        row = self.db.get(MemoryIngestion, operation_id)
        row.status = "processing"
        row.lease_token = "old"
        row.lease_expires_at = memory_jobs._now() + timedelta(minutes=5)
        self.db.commit()
        with patch.object(memory_service, "_provider_record_completed_turn") as provider:
            self.assertEqual(memory_jobs.process_ingestion(operation_id), "busy")
            provider.assert_not_called()
            row.lease_expires_at = memory_jobs._now() - timedelta(seconds=1)
            self.db.commit()
            self.assertEqual(memory_jobs.process_ingestion(operation_id), "done")
            provider.assert_called_once()

    def test_transient_failure_retries_then_retains_failed_row(self):
        operation_id = self.ingestion()
        with patch.object(memory_service, "_provider_record_completed_turn", side_effect=TimeoutError("secret")):
            for _ in range(memory_service.MEMORY_CONFIG.max_ingestion_attempts - 1):
                self.assertEqual(memory_jobs.process_ingestion(operation_id), "retry")
            self.assertEqual(memory_jobs.process_ingestion(operation_id), "done")
        self.db.expire_all()
        row = self.db.get(MemoryIngestion, operation_id)
        self.assertEqual((row.status, row.attempt_count, row.last_error_code),
                         ("failed", memory_service.MEMORY_CONFIG.max_ingestion_attempts, "TimeoutError"))
        self.assertEqual(memory_jobs.process_ingestion(operation_id), "done")

    def test_invalid_turn_and_correction_do_not_reach_provider(self):
        operation_id = self.ingestion()
        self.db.get(Message, self.scope["user_message_id"]).is_correction_cmd = True
        self.db.commit()
        with patch.object(memory_service, "_provider_record_completed_turn") as provider:
            self.assertEqual(memory_jobs.process_ingestion(operation_id), "done")
            provider.assert_not_called()
        self.db.expire_all()
        self.assertEqual(self.db.get(MemoryIngestion, operation_id).last_error_code, "invalid_turn")

    def test_deletion_tombstone_cancels_delayed_ingestion_and_is_idempotent(self):
        operation_id = self.ingestion()
        deletion = memory_jobs.schedule_character_deletion(
            self.db, user_id=self.scope["user_id"], character_id=self.scope["character_id"])
        self.db.delete(self.db.get(Character, self.scope["character_id"]))
        self.db.commit()
        with patch.object(memory_service, "_provider_record_completed_turn") as write, \
             patch.object(memory_service, "delete_character_memories",
                          return_value=memory_service.MemoryDeletionResult(
                              "memmachine", "character", self.scope["character_id"], True, True, True, 0,
                          )) as delete:
            self.assertEqual(memory_jobs.process_ingestion(operation_id), "done")
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")
            write.assert_not_called()
            delete.assert_called_once()
        self.db.expire_all()
        self.assertEqual(self.db.get(MemoryDeletion, deletion.id).status, "completed")

    def test_deletion_waits_for_active_ingestion_before_provider_purge(self):
        operation_id = self.ingestion()
        row = self.db.get(MemoryIngestion, operation_id)
        row.status = "processing"
        row.lease_token = "worker"
        row.lease_expires_at = memory_jobs._now() + timedelta(minutes=5)
        deletion = memory_jobs.schedule_character_deletion(
            self.db, user_id=self.scope["user_id"], character_id=self.scope["character_id"])
        self.db.delete(self.db.get(Character, self.scope["character_id"]))
        self.db.commit()
        with patch.object(memory_service, "delete_character_memories") as delete:
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "retry")
            delete.assert_not_called()
        row.status = "cancelled"
        self.db.commit()
        with patch.object(memory_service, "delete_character_memories",
                          return_value=memory_service.MemoryDeletionResult(
                              "memmachine", "character", self.scope["character_id"], True, True, False, 0,
                          )):
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")

    def test_deletion_failure_retries_then_keeps_failed_tombstone(self):
        deletion = memory_jobs.schedule_character_deletion(
            self.db, user_id=self.scope["user_id"], character_id=self.scope["character_id"])
        self.db.commit()
        failure = memory_service.MemoryDeletionResult(
            "memmachine", "character", self.scope["character_id"], True, False, False, 0,
            retryable=True, error_type="TimeoutError",
        )
        with patch.object(memory_service, "delete_character_memories", return_value=failure):
            for _ in range(memory_service.MEMORY_CONFIG.max_deletion_attempts - 1):
                self.assertEqual(memory_jobs.process_deletion(deletion.id), "retry")
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "done")
        self.db.expire_all()
        self.assertEqual(self.db.get(MemoryDeletion, deletion.id).status, "failed")

    def test_temporary_noop_switch_does_not_erase_previous_cleanup_duty(self):
        operation_id = self.ingestion()
        with patch.object(memory_service, "PROVIDER_NAME", "noop"):
            deletion = memory_jobs.schedule_character_deletion(
                self.db, user_id=self.scope["user_id"], character_id=self.scope["character_id"])
            self.db.delete(self.db.get(Character, self.scope["character_id"]))
            self.db.commit()
            self.assertEqual(deletion.provider, "memmachine")
            self.assertEqual(memory_jobs.process_deletion(deletion.id), "busy")
            self.assertEqual(memory_jobs.process_ingestion(operation_id), "done")
        self.db.expire_all()
        self.assertEqual(self.db.get(MemoryDeletion, deletion.id).status, "queued")

    def test_reconcile_republishes_pending_identifiers_without_content(self):
        operation_id = self.ingestion()
        deletion = memory_jobs.schedule_character_deletion(
            self.db, user_id=self.scope["user_id"], character_id=self.scope["character_id"])
        self.db.commit()
        messages = []
        with patch.object(memory_jobs, "publish_memory_operation", side_effect=lambda *args: messages.append(args)):
            self.assertEqual(memory_jobs.reconcile_pending(), {"ingest": 0, "delete": 1})
        self.assertEqual(messages, [(MEMORY_DELETE_TOPIC, deletion.id)])
        self.assertNotIn("We met at 18", str(messages))

    def test_publish_failure_keeps_committed_intent_for_reconciliation(self):
        operation_id = self.ingestion()
        with patch.object(memory_jobs, "publish_memory_operation", side_effect=ConnectionError("offline")):
            with self.assertLogs(memory_jobs.logger, level="WARNING"):
                memory_jobs.publish_safe(MEMORY_INGEST_TOPIC, operation_id)
        self.assertEqual(self.db.get(MemoryIngestion, operation_id).status, "queued")
        with patch.object(memory_jobs, "publish_memory_operation") as publish:
            self.assertEqual(memory_jobs.reconcile_pending(), {"ingest": 1, "delete": 0})
        publish.assert_called_once_with(MEMORY_INGEST_TOPIC, operation_id)

    def test_cron_reconciler_requires_configured_bearer_secret(self):
        with TestClient(app) as client, patch("app.routers.memory_ops_router.reconcile_pending",
                                             return_value={"ingest": 1, "delete": 0}) as reconcile:
            with patch.dict("os.environ", {"CRON_SECRET": "test-secret-123456789"}):
                self.assertEqual(client.get("/internal/memory/reconcile").status_code, 401)
                self.assertEqual(client.get("/internal/memory/reconcile", headers={
                    "Authorization": "Bearer wrong",
                }).status_code, 401)
                response = client.get("/internal/memory/reconcile", headers={
                    "Authorization": "Bearer test-secret-123456789",
                })
            self.assertEqual(response.status_code, 200)
            reconcile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
