import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import llm
from app.database import Base
from app.models import (
    Character,
    Conversation,
    CorrectionLog,
    Message,
    MessageRole,
    SourceType,
    User,
    WorldSourceType,
)
from app.schemas import CharacterProfileData
from app.services import chat_service, character_profile_service, extraction_service, world_extraction_service


class OpenAIAdapterTests(unittest.TestCase):
    def test_responses_output_text_is_returned_as_string(self):
        client = Mock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=5)
        client.responses.create.return_value = SimpleNamespace(
            output_text="응답 문자열", status="completed", model="gpt-5.5",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        messages = [{"role": "user", "content": "안녕"}]
        db = Mock()

        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.5", "OPENAI_API_KEY": ""}):
            with patch.object(llm, "_get_client", return_value=client):
                with patch.object(llm, "check_capacity"):
                    with patch.object(llm, "reserve_usage", return_value="usage-id") as reserve:
                        with patch.object(llm, "record_response"):
                            result = llm.generate_text(db, "user-id", "chat", "기존 시스템 지침", messages, 2000)

        self.assertEqual(result, "응답 문자열")
        reserve.assert_called_once_with(db.get_bind.return_value, "user-id", "chat", "gpt-5.5", 5, 2000)
        client.responses.create.assert_called_once_with(
            model="gpt-5.5",
            instructions="기존 시스템 지침",
            input=messages,
            max_output_tokens=2000,
        )

    def test_missing_key_does_not_create_client_or_call_network(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with patch.object(llm, "OpenAI") as client_type:
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    llm._get_client()
        client_type.assert_not_called()

    def test_empty_output_is_rejected(self):
        client = Mock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=5)
        client.responses.create.return_value = SimpleNamespace(
            output_text="", status="completed", model="gpt-5.5",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.5"}):
            with patch.object(llm, "_get_client", return_value=client):
                with patch.object(llm, "check_capacity"):
                    with patch.object(llm, "reserve_usage", return_value="usage-id"):
                        with patch.object(llm, "record_response"):
                            with self.assertRaisesRegex(RuntimeError, "no completed text output"):
                                llm.generate_text(Mock(), "user-id", "chat", "지침", [{"role": "user", "content": "질문"}], 100)


class ServiceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _character_and_conversation(self):
        user = User(email="test@example.invalid", password_hash="unused")
        self.db.add(user)
        self.db.flush()
        character = Character(user_id=user.id, name="가상 인물")
        self.db.add(character)
        self.db.flush()
        conversation = Conversation(character_id=character.id, user_id=user.id)
        self.db.add(conversation)
        self.db.commit()
        return user, character, conversation

    def test_character_extraction_keeps_json_validation(self):
        output = CharacterProfileData(personality_summary="차분함", speech_style="존댓말").model_dump_json()
        with patch.object(extraction_service, "generate_text", return_value=f"```json\n{output}\n```") as call:
            result = extraction_service.extract_profile_from_text(
                self.db, "user-id", "가상 인물은 차분하다.", SourceType.MANUAL_DESCRIPTION, "가상 인물"
            )
        self.assertEqual(result.personality_summary, "차분함")
        self.assertIn("가상 인물", call.call_args.kwargs["instructions"])
        self.assertEqual(call.call_args.kwargs["max_output_tokens"], 2000)

    def test_world_extraction_keeps_json_validation(self):
        output = json.dumps(
            {"world_summary": "도시 세계", "key_facts": ["마법 없음"], "timeline_notes": [], "mentioned_characters": []},
            ensure_ascii=False,
        )
        with patch.object(world_extraction_service, "generate_text", return_value=output):
            result = world_extraction_service.extract_world_profile_from_text(
                self.db, "user-id", "마법이 없는 도시", WorldSourceType.DESCRIPTION, None, None
            )
        self.assertEqual(result.key_facts, ["마법 없음"])

    def test_profile_correction_keeps_validation_and_write_path(self):
        user, character, _ = self._character_and_conversation()
        updated = CharacterProfileData(personality_summary="신중함", do_not_do=["소리치기"])
        with patch.object(character_profile_service, "generate_text", return_value=updated.model_dump_json()):
            profile = character_profile_service.apply_user_correction(
                self.db, user.id, character.id, "소리치지 않게 수정"
            )
        self.assertEqual(CharacterProfileData.model_validate(profile.data), updated)
        self.assertEqual(self.db.query(CorrectionLog).count(), 1)

    def test_chat_passes_prompt_and_history_in_order_and_uses_text_reply(self):
        user, character, conversation = self._character_and_conversation()
        character_profile_service.set_initial_profile(
            self.db, character.id, CharacterProfileData(personality_summary="차분함")
        )
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.db.add_all(
            [
                Message(conversation_id=conversation.id, role=MessageRole.USER, content="첫 질문", created_at=start),
                Message(
                    conversation_id=conversation.id,
                    role=MessageRole.CHARACTER,
                    content="첫 답변",
                    created_at=start + timedelta(seconds=1),
                ),
            ]
        )
        self.db.commit()

        with patch.object(chat_service, "generate_text", return_value="새 답변") as call:
            result = chat_service.send_message(self.db, character.id, conversation.id, "새 질문", user.id)

        self.assertEqual(result, {"role": "CHARACTER", "content": "새 답변"})
        self.assertIn("차분함", call.call_args.kwargs["instructions"])
        self.assertEqual(
            call.call_args.kwargs["input_messages"],
            [
                {"role": "user", "content": "첫 질문"},
                {"role": "assistant", "content": "첫 답변"},
                {"role": "user", "content": "새 질문"},
            ],
        )
        self.assertEqual(self.db.query(Message).order_by(Message.created_at.desc()).first().content, "새 답변")


if __name__ == "__main__":
    unittest.main()
