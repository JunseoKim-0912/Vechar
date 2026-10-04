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
    World,
    WorldSourceType,
)
from app.schemas import (
    CharacterProfileData, CharacterSynthesisResult, MentionedCharacter,
    WorldCharacterRankingResult, WorldProfileData, WorldSynthesisResult,
)
from app.services import (
    character_extraction_service, chat_service, character_profile_service,
    extraction_service, world_extraction_service, world_profile_service,
)


class OpenAIAdapterTests(unittest.TestCase):
    def test_responses_output_text_is_returned_as_string(self):
        client = Mock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=5)
        client.responses.create.return_value = SimpleNamespace(
            output_text="응답 문자열", status="completed", model="gpt-6-luna",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        messages = [{"role": "user", "content": "안녕"}]
        db = Mock()

        with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna", "OPENAI_API_KEY": ""}):
            with patch.object(llm, "_get_client", return_value=client):
                with patch.object(llm, "check_capacity"):
                    with patch.object(llm, "reserve_usage", return_value="usage-id") as reserve:
                        with patch.object(llm, "record_response"):
                            result = llm.generate_text(db, "user-id", "chat", "기존 시스템 지침", messages, 2000, task="chat")

        self.assertEqual(result, "응답 문자열")
        reserve.assert_called_once_with(db.get_bind.return_value, "user-id", "chat", "gpt-6-luna", 5, 2000)
        client.responses.create.assert_called_once_with(
            model="gpt-6-luna",
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
            output_text="", status="completed", model="gpt-6-luna",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna"}):
            with patch.object(llm, "_get_client", return_value=client):
                with patch.object(llm, "check_capacity"):
                    with patch.object(llm, "reserve_usage", return_value="usage-id"):
                        with patch.object(llm, "record_response"):
                            with self.assertRaisesRegex(RuntimeError, "no completed text output"):
                                llm.generate_text(Mock(), "user-id", "chat", "지침", [{"role": "user", "content": "질문"}], 100, task="chat")


class ServiceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.counter_patcher = patch.object(
            chat_service, "make_chat_input_counter",
            return_value=lambda instructions, messages: len(instructions) + sum(
                len(message["content"]) + 2 for message in messages
            ),
        )
        self.counter_patcher.start()
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)

    def tearDown(self):
        self.counter_patcher.stop()
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

    def test_character_extraction_uses_structured_profile(self):
        output = CharacterProfileData(personality_summary="차분함", speech_style="존댓말")
        with patch.object(extraction_service, "generate_structured", return_value=output) as call:
            result = extraction_service.extract_profile_from_text(
                self.db, "user-id", "가상 인물은 차분하다.", SourceType.MANUAL_DESCRIPTION, "가상 인물"
            )
        self.assertEqual(result.personality_summary, "차분함")
        self.assertIn("가상 인물", call.call_args.kwargs["instructions"])
        self.assertEqual(call.call_args.kwargs["max_output_tokens"], 2000)
        self.assertEqual(call.call_args.kwargs["task"], "analysis")
        self.assertIs(call.call_args.kwargs["response_model"], CharacterProfileData)

    def test_world_extraction_uses_structured_profile(self):
        output = WorldProfileData(world_summary="도시 세계", key_facts=["마법 없음"])
        with patch.object(world_extraction_service, "generate_structured", return_value=output) as call:
            result = world_extraction_service.extract_world_profile_from_text(
                self.db, "user-id", "마법이 없는 도시", WorldSourceType.DESCRIPTION, None, None
            )
        self.assertEqual(result.key_facts, ["마법 없음"])
        self.assertEqual(call.call_args.kwargs["task"], "analysis")
        self.assertIs(call.call_args.kwargs["response_model"], WorldProfileData)

    def test_profile_correction_keeps_validation_and_write_path(self):
        user, character, _ = self._character_and_conversation()
        updated = CharacterProfileData(personality_summary="신중함", do_not_do=["소리치기"])
        with patch.object(character_profile_service, "generate_structured", return_value=updated) as call:
            profile = character_profile_service.apply_user_correction(
                self.db, user.id, character.id, "소리치지 않게 수정"
            )
        self.assertEqual(CharacterProfileData.model_validate(profile.data), updated)
        self.assertEqual(self.db.query(CorrectionLog).count(), 1)
        self.assertEqual(call.call_args.kwargs["task"], "analysis")
        self.assertIs(call.call_args.kwargs["response_model"], CharacterProfileData)

    def test_character_profile_merge_uses_analysis(self):
        user, character, _ = self._character_and_conversation()
        character_profile_service.set_initial_profile(
            self.db, character.id, CharacterProfileData(personality_summary="기존 성격")
        )
        output = CharacterSynthesisResult(personality_summary="새 성격", speech_style="존댓말")
        with patch.object(character_profile_service, "generate_structured", return_value=output) as call:
            character_profile_service.merge_training_source(
                self.db, user.id, character.id, CharacterProfileData(personality_summary="새 성격")
            )
        self.assertEqual(call.call_args.kwargs["task"], "analysis")
        self.assertEqual(call.call_args.kwargs["request_type"], "character_synthesis")
        self.assertIs(call.call_args.kwargs["response_model"], CharacterSynthesisResult)

    def test_world_character_extraction_uses_analysis(self):
        output = CharacterProfileData(personality_summary="차분함")
        with patch.object(character_extraction_service, "generate_structured", return_value=output) as call:
            character_extraction_service.extract_character_from_world_text(
                self.db, "user-id", "인물", ["소설 본문"]
            )
        self.assertEqual(call.call_args.kwargs["task"], "analysis")
        self.assertIs(call.call_args.kwargs["response_model"], CharacterProfileData)

    def test_world_profile_operations_use_analysis(self):
        user = User(email="world@example.invalid", password_hash="unused")
        self.db.add(user)
        self.db.flush()
        world = World(user_id=user.id, name="가상 세계")
        self.db.add(world)
        self.db.commit()
        world_profile_service.set_initial_world_profile(
            self.db, world.id, WorldProfileData(world_summary="기존 세계")
        )

        with patch.object(world_profile_service, "generate_structured",
                          return_value=WorldCharacterRankingResult(mentioned_characters=[])) as call:
            world_profile_service._rank_and_dedupe_characters(
                self.db, user.id, [], [MentionedCharacter(name="인물")]
            )
        self.assertEqual((call.call_args.kwargs["request_type"], call.call_args.kwargs["task"]),
                         ("world_character_ranking", "analysis"))
        self.assertIs(call.call_args.kwargs["response_model"], WorldCharacterRankingResult)

        with patch.object(world_profile_service, "generate_structured",
                          return_value=WorldSynthesisResult(world_summary="새 세계")) as call:
            world_profile_service.merge_world_source(
                self.db, user.id, world.id, WorldProfileData(world_summary="새 세계")
            )
        self.assertEqual((call.call_args.kwargs["request_type"], call.call_args.kwargs["task"]),
                         ("world_synthesis", "analysis"))
        self.assertIs(call.call_args.kwargs["response_model"], WorldSynthesisResult)

        profile = WorldProfileData(world_summary="편집된 세계")
        with patch.object(world_profile_service, "generate_structured", return_value=profile) as call:
            world_profile_service.apply_world_edit(self.db, user.id, world.id, "modify", "세계 변경")
        self.assertEqual((call.call_args.kwargs["request_type"], call.call_args.kwargs["task"]),
                         ("world_edit", "analysis"))
        self.assertIs(call.call_args.kwargs["response_model"], WorldProfileData)

        with patch.object(world_profile_service, "generate_text", return_value="세계 요약") as call:
            world_profile_service.summarize_world(self.db, user.id, world.id)
        self.assertEqual((call.call_args.kwargs["request_type"], call.call_args.kwargs["task"]),
                         ("world_summary", "analysis"))

        with patch.object(world_profile_service, "generate_structured", return_value=profile) as call:
            world_profile_service.compact_world_profile(self.db, user.id, world.id)
        self.assertEqual((call.call_args.kwargs["request_type"], call.call_args.kwargs["task"]),
                         ("world_compaction", "analysis"))
        self.assertIs(call.call_args.kwargs["response_model"], WorldProfileData)

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
        self.assertEqual(call.call_args.kwargs["task"], "chat")
        self.assertEqual(
            call.call_args.kwargs["input_messages"],
            [
                {"role": "user", "content": "첫 질문"},
                {"role": "assistant", "content": "첫 답변"},
                {"role": "user", "content": "새 질문"},
            ],
        )
        self.assertEqual(self.db.query(Message).order_by(Message.created_at.desc()).first().content, "새 답변")

    def test_current_user_message_is_sent_once_after_saved_history(self):
        user, character, conversation = self._character_and_conversation()
        self.db.add(Message(conversation_id=conversation.id, role=MessageRole.USER, content="이전 질문"))
        self.db.commit()

        with patch.object(chat_service, "generate_text", return_value="새 답변") as call:
            chat_service.send_message(self.db, character.id, conversation.id, "현재 질문", user.id)

        sent = call.call_args.kwargs["input_messages"]
        self.assertEqual(sent, [
            {"role": "user", "content": "이전 질문"},
            {"role": "user", "content": "현재 질문"},
        ])
        self.assertEqual(sum(message["content"] == "현재 질문" for message in sent), 1)
        self.assertEqual(
            self.db.query(Message).filter(
                Message.conversation_id == conversation.id,
                Message.role == MessageRole.USER,
                Message.content == "현재 질문",
            ).count(), 1,
        )

    def test_chat_keeps_db_latest_thirty_then_trims_oldest_messages(self):
        user, character, conversation = self._character_and_conversation()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.db.add_all([
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=f"{index:02d}" + "x" * 248,
                created_at=start + timedelta(seconds=index),
            ) for index in range(35)
        ])
        self.db.commit()

        with patch.object(chat_service, "generate_text", return_value="답변") as call:
            chat_service.send_message(self.db, character.id, conversation.id, "현재 질문", user.id)

        sent = call.call_args.kwargs["input_messages"]
        self.assertEqual([message["content"][:2] for message in sent[:-1]],
                         [f"{index:02d}" for index in range(12, 35)])
        self.assertEqual(sent[-1]["content"], "현재 질문")

    def test_prior_correction_message_remains_in_ordinary_history_for_now(self):
        user, character, conversation = self._character_and_conversation()
        self.db.add(Message(
            conversation_id=conversation.id, role=MessageRole.USER,
            content="/수정 과거 명령", is_correction_cmd=True,
        ))
        self.db.commit()

        with patch.object(chat_service, "generate_text", return_value="답변") as call:
            chat_service.send_message(self.db, character.id, conversation.id, "안녕", user.id)
        self.assertEqual(call.call_args.kwargs["input_messages"][0]["content"], "/수정 과거 명령")

    def test_chat_with_world_uses_canonical_world_prompt(self):
        user, character, conversation = self._character_and_conversation()
        world = World(user_id=user.id, name="도시")
        self.db.add(world)
        self.db.flush()
        character.world_id = world.id
        self.db.commit()
        world_profile_service.set_initial_world_profile(
            self.db, world.id,
            WorldProfileData(world_summary="마법 없는 도시", key_facts=["기차 운행"], timeline_notes=["미사용"]),
        )

        with patch.object(chat_service, "generate_text", return_value="답변") as call:
            chat_service.send_message(self.db, character.id, conversation.id, "질문", user.id)

        instructions = call.call_args.kwargs["instructions"]
        self.assertIn("세계관 개요: 마법 없는 도시", instructions)
        self.assertIn("세계관 사실: 기차 운행", instructions)
        self.assertNotIn("미사용", instructions)
        self.assertEqual(call.call_args.kwargs["task"], "chat")

    def test_correction_bypasses_roleplay_prompt_builder(self):
        user, character, conversation = self._character_and_conversation()
        with patch.object(chat_service, "generate_text") as chat_call:
            with patch.object(chat_service, "build_chat_instructions") as instructions_call:
                with patch.object(chat_service, "select_chat_context") as budget_call:
                    with patch.object(chat_service, "apply_user_correction",
                                      return_value=SimpleNamespace(version=2)) as correction_call:
                        result = chat_service.send_message(
                            self.db, character.id, conversation.id, "/수정 차분하게", user.id
                        )

        self.assertEqual(result["role"], "SYSTEM_NOTE")
        correction_call.assert_called_once_with(self.db, user.id, character.id, "차분하게")
        chat_call.assert_not_called()
        instructions_call.assert_not_called()
        budget_call.assert_not_called()
        saved = self.db.query(Message).filter(Message.conversation_id == conversation.id).one()
        self.assertEqual((saved.content, saved.is_correction_cmd), ("/수정 차분하게", True))


if __name__ == "__main__":
    unittest.main()
