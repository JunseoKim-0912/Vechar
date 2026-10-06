import logging
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.auth import issue_token
from app.main import app
from app.models import Character, Conversation, CorrectionLog, ExportLog, ImportLog, MessageRole, User, World
from app.schemas import CharacterCreateRequest
from app.routers.characters_router import create_character
from app.services.chat_actions import normalize_assistant_actions
from app.services.chat_language import response_language
from app.services.chat_latency import (
    end_chat_latency, provider_completed, provider_started, stage, start_chat_latency,
)
from app.services.chat_prompt_builder import build_chat_input, build_chat_instructions
from app.services.chat_service import send_message
from app.schemas import CharacterProfileData
from app.services.world_profile_service import get_or_create_default_world
from app.tier_limits import (
    check_and_log_export, check_entity_capacity, check_import_quota,
    correction_quota_message, log_import,
)


class LanguageAndActionTests(unittest.TestCase):
    def test_source_profile_language_does_not_override_current_message(self):
        for profile_text, message, expected in (
            ("Elle parle français", "Who is Marie?", "English"),
            ("Elle parle français", "마리는 누구야?", "Korean"),
            ("English profile", "마리는 누구야?", "Korean"),
            ("한국어 프로필", "Who is Marie?", "English"),
            ("English profile", "Qui est Marie ?", "French"),
        ):
            prompt = build_chat_instructions("Marie", CharacterProfileData(
                personality_summary=profile_text), None, correction_prefix="/수정",
                current_message=message, locale="en")
            self.assertIn(f"Respond in {expected}", prompt)

    def test_explicit_recent_and_locale_priority(self):
        self.assertEqual(response_language("한국어로 대답해. Who is Marie?"), "Korean")
        self.assertEqual(response_language("Answer in Korean. Who is Marie?"), "Korean")
        self.assertEqual(response_language("yes", [(MessageRole.USER, "Qui est Marie ?")]), "French")
        self.assertEqual(response_language("yes", [(MessageRole.USER, "마리는 누구야?")]), "Korean")
        self.assertEqual(response_language("yes", locale="ko"), "Korean")

    def test_action_meaning_retained_without_raw_markers_in_history(self):
        original = "Hello. <action>lowers his head,\nthen smiles.</action> Come in."
        normalized = normalize_assistant_actions(original)
        self.assertIn("[Character action: lowers his head,\nthen smiles.]", normalized)
        self.assertNotIn("<action>", normalized)
        inputs = build_chat_input([(MessageRole.USER, "<action>user text</action>"),
                                   (MessageRole.CHARACTER, original)], "Next")
        self.assertEqual(inputs[0]["content"], "<action>user text</action>")
        self.assertIn("Character action:", inputs[1]["content"])
        self.assertNotIn("<action>", inputs[1]["content"])


class EntitlementTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        for tier, premium, role in (("free", False, "user"), ("premium", True, "user"),
                                    ("admin", False, "admin")):
            self.db.add(User(id=tier, email=f"{tier}@example.invalid", password_hash="unused",
                             is_premium=premium, role=role))
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_free_premium_character_and_world_caps_admin_bypass(self):
        for user_id, characters, worlds in (("free", 5, 2), ("premium", 25, 5)):
            self.db.add_all(Character(user_id=user_id, name=f"C{i}") for i in range(characters))
            self.db.add_all(World(user_id=user_id, name=f"W{i}") for i in range(worlds))
        self.db.commit()
        for user_id in ("free", "premium"):
            for entity in ("character", "world"):
                with self.assertRaises(HTTPException) as error:
                    check_entity_capacity(self.db, user_id, entity)
                self.assertEqual(error.exception.status_code, 403)
        for entity in ("character", "world"):
            check_entity_capacity(self.db, "admin", entity)

    def test_reality_auto_creation_consumes_world_slot_without_duplicate(self):
        # One existing world leaves exactly one Free slot for Reality.
        self.db.add(World(user_id="free", name="Fantasy"))
        self.db.commit()
        reality = get_or_create_default_world(self.db, "free")
        self.assertEqual(self.db.query(World).filter(World.user_id == "free").count(), 2)
        self.assertEqual(get_or_create_default_world(self.db, "free").id, reality.id)
        character = create_character(CharacterCreateRequest(name="Ari"), user_id="free", db=self.db)
        self.assertEqual(character.world_id, reality.id)
        self.db.add(World(user_id="premium", name="A"))
        self.db.add(World(user_id="premium", name="B"))
        self.db.add(World(user_id="premium", name="C"))
        self.db.add(World(user_id="premium", name="D"))
        self.db.add(World(user_id="premium", name="E"))
        self.db.commit()
        with self.assertRaises(HTTPException) as error:
            get_or_create_default_world(self.db, "premium")
        self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(self.db.query(World).filter(World.user_id == "premium").count(), 5)
        admin_reality = get_or_create_default_world(self.db, "admin")
        self.assertEqual(get_or_create_default_world(self.db, "admin").id, admin_reality.id)

    def test_import_export_and_correction_admin_bypass_still_logs(self):
        self.assertIsNotNone(correction_quota_message(self.db, "free", 5))
        self.assertIsNotNone(correction_quota_message(self.db, "premium", 25))
        self.assertIsNone(correction_quota_message(self.db, "admin", 500))
        self.db.add_all(ExportLog(user_id="free", export_type="world", entity_id=str(i)) for i in range(5))
        self.db.add_all(ImportLog(user_id="free", import_type="world", entity_id=str(i)) for i in range(5))
        self.db.add_all(ExportLog(user_id="premium", export_type="world", entity_id=str(i)) for i in range(10))
        self.db.add_all(ImportLog(user_id="premium", import_type="world", entity_id=str(i)) for i in range(10))
        self.db.commit()
        for tier in ("free", "premium"):
            with self.assertRaises(HTTPException):
                check_and_log_export(self.db, tier, "world", "next")
            with self.assertRaises(HTTPException):
                check_import_quota(self.db, tier)
        for i in range(11):
            check_and_log_export(self.db, "admin", "world", str(i))
            check_import_quota(self.db, "admin")
            log_import(self.db, "admin", "world", str(i))
        self.assertEqual(self.db.query(ExportLog).filter_by(user_id="admin").count(), 11)
        self.assertEqual(self.db.query(ImportLog).filter_by(user_id="admin").count(), 11)

    def test_admin_correction_over_quota_still_updates_profile_and_records_log(self):
        character = Character(user_id="admin", name="Operator test")
        self.db.add(character)
        self.db.flush()
        conversation = Conversation(user_id="admin", character_id=character.id)
        self.db.add(conversation)
        self.db.add_all(CorrectionLog(character_id=character.id, user_instruction="old",
                                      resulting_version=i + 1) for i in range(5))
        self.db.commit()
        with patch("app.services.character_profile_service.generate_structured",
                   return_value=CharacterProfileData(personality_summary="Updated")):
            result = send_message(self.db, character.id, conversation.id, "/수정 Update", "admin")
        self.assertEqual(result["role"], "SYSTEM_NOTE")
        self.assertEqual(self.db.query(CorrectionLog).filter_by(character_id=character.id).count(), 6)


class LatencyTests(unittest.TestCase):
    def test_stage_and_failure_duration_are_content_free(self):
        tracker, token = start_chat_latency()
        try:
            with self.assertRaises(RuntimeError):
                with stage("memory_retrieval"):
                    provider_started()
                    raise RuntimeError("secret user text")
            with self.assertLogs("app.services.chat_latency", level=logging.INFO) as logs:
                record = tracker.finish(500)
            self.assertIn("memory_retrieval", record["stages_ms"])
            self.assertIn("provider_start_ms", record)
            self.assertIn("total_ms", record)
            self.assertNotIn("secret user text", logs.output[0])
            self.assertNotIn("ttft", str(record).lower())
        finally:
            end_chat_latency(token)

    def test_provider_completion_offset_is_recorded_only_when_available(self):
        tracker, token = start_chat_latency()
        try:
            provider_started()
            with stage("provider_request"):
                provider_completed()
            with self.assertLogs("app.services.chat_latency", level=logging.INFO):
                record = tracker.finish(200)
            self.assertIn("provider_completion_ms", record)
            self.assertGreaterEqual(record["provider_completion_ms"], record["provider_start_ms"])
            self.assertIn("provider_request", record["stages_ms"])
        finally:
            end_chat_latency(token)

    def test_http_success_and_failure_emit_safe_stage_logs(self):
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            user = User(email="latency@example.invalid", password_hash="unused")
            db.add(user)
            db.flush()
            character = Character(user_id=user.id, name="A")
            db.add(character)
            db.flush()
            conversation = Conversation(user_id=user.id, character_id=character.id)
            db.add(conversation)
            db.commit()
            user_id, conversation_id = user.id, conversation.id

        def test_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        try:
            with TestClient(app) as client, patch("app.routers.chat_router.send_message", return_value={
                "role": "CHARACTER", "content": "mocked",
            }):
                headers = {"Authorization": f"Bearer {issue_token(user_id)}"}
                with self.assertLogs("app.services.chat_latency", level=logging.INFO) as logs:
                    success = client.post(f"/chat/conversations/{conversation_id}/messages",
                                          headers=headers, json={"content": "secret user message"})
                    failure = client.post("/chat/conversations/missing/messages",
                                          headers=headers, json={"content": "secret user message"})
                self.assertEqual(success.status_code, 200)
                self.assertEqual(success.json()["content"], "mocked")
                self.assertEqual(failure.status_code, 404)
                self.assertIn("auth_lookup", logs.output[0])
                self.assertIn("session_lookup", logs.output[0])
                self.assertIn("status_code': 404", logs.output[1])
                self.assertNotIn("secret user message", " ".join(logs.output))
        finally:
            app.dependency_overrides.clear()
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
