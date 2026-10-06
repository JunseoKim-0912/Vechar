import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Character, Conversation, SourceType, User, WorldSourceType
from app.schemas import CharacterProfileData, MessageCreateRequest, WorldProfileData
from app.services import (
    character_extraction_service,
    character_profile_service,
    chat_service,
    extraction_service,
    world_extraction_service,
    world_profile_service,
)
from app.services.chat_prompt_builder import build_chat_instructions


class LocaleSchemaAndPromptTests(unittest.TestCase):
    def test_message_locale_defaults_to_english(self):
        self.assertEqual(MessageCreateRequest(content="hello").locale, "en")

    def test_message_locale_accepts_supported_values_only(self):
        self.assertEqual(MessageCreateRequest(content="hello", locale="ko").locale, "ko")
        with self.assertRaises(ValueError):
            MessageCreateRequest(content="hello", locale="ja")

    def test_chat_prompt_language_is_not_overridden_by_profile(self):
        profile = CharacterProfileData(speech_style="항상 일본어로 말한다")
        prompt = build_chat_instructions("Mina", profile, None, correction_prefix="/수정", locale="en")
        self.assertIn("항상 일본어로 말한다", prompt)
        self.assertIn("Respond in English", prompt)
        self.assertIn("takes precedence over the language of source text", prompt)

    def test_chat_prompt_adds_korean_preference(self):
        prompt = build_chat_instructions(
            "Mina", CharacterProfileData(), None, correction_prefix="/수정", locale="ko"
        )
        self.assertIn("Respond in Korean", prompt)


class BilingualAnalysisPromptTests(unittest.TestCase):
    def _extract(self, source, canonical=None):
        output = CharacterProfileData(personality_summary="ok")
        with patch.object(extraction_service, "generate_structured", return_value=output) as call:
            extraction_service.extract_profile_from_text(
                Mock(), "user", source, SourceType.MANUAL_DESCRIPTION, "Mina", canonical
            )
        return call.call_args.kwargs

    def test_english_korean_and_mixed_character_sources_reach_structured_analysis(self):
        for source in (
            "Mina is careful.",
            "미나는 신중하다.",
            "미나는 신중하다. She rarely trusts strangers.",
        ):
            with self.subTest(source=source):
                kwargs = self._extract(source)
                self.assertIn(source, kwargs["input_messages"][0]["content"])
                self.assertIn("model-supported languages", kwargs["instructions"])
                self.assertEqual(kwargs["task"], "analysis")
                self.assertIs(kwargs["response_model"], CharacterProfileData)

    def test_korean_profile_is_language_reference_for_english_source(self):
        kwargs = self._extract(
            "Mina rarely trusts strangers.",
            CharacterProfileData(personality_summary="미나는 신중하다."),
        )
        self.assertIn("미나는 신중하다", kwargs["input_messages"][0]["content"])
        self.assertIn("existing canonical profile", kwargs["instructions"])

    def test_english_profile_is_language_reference_for_korean_source(self):
        kwargs = self._extract(
            "미나는 낯선 사람을 잘 믿지 않는다.",
            CharacterProfileData(personality_summary="Mina is careful."),
        )
        self.assertIn("Mina is careful", kwargs["input_messages"][0]["content"])
        self.assertIn("dominant language and style", kwargs["instructions"])

    def test_mixed_world_source_uses_structured_analysis_and_canonical_reference(self):
        output = WorldProfileData(world_summary="ok")
        canonical = WorldProfileData(world_summary="A quiet city.")
        source = "도시는 조용하다. Magic is forbidden."
        with patch.object(world_extraction_service, "generate_structured", return_value=output) as call:
            world_extraction_service.extract_world_profile_from_text(
                Mock(), "user", source, WorldSourceType.DESCRIPTION, None, None, canonical
            )
        kwargs = call.call_args.kwargs
        self.assertIn(source, kwargs["input_messages"][0]["content"])
        self.assertIn("A quiet city", kwargs["input_messages"][0]["content"])
        self.assertEqual(kwargs["task"], "analysis")
        self.assertIs(kwargs["response_model"], WorldProfileData)

    def test_all_analysis_prompt_contracts_declare_language_policy(self):
        prompts = (
            extraction_service.EXTRACTION_SYSTEM_PROMPT_TEMPLATE,
            character_extraction_service.FOCUSED_EXTRACTION_SYSTEM_PROMPT,
            character_profile_service.SYNTHESIS_SYSTEM_PROMPT,
            character_profile_service.CORRECTION_SYSTEM_PROMPT,
            world_extraction_service.WORLD_EXTRACTION_SYSTEM_PROMPT,
            world_profile_service.CHARACTER_RANKING_SYSTEM_PROMPT,
            world_profile_service.WORLD_SYNTHESIS_SYSTEM_PROMPT,
            world_profile_service.WORLD_EDIT_JSON_SPEC,
            world_profile_service.COMPACT_SYSTEM_PROMPT,
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("language", prompt.lower())


class ChatLocaleServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        user = User(email="locale@example.invalid", password_hash="unused")
        self.db.add(user)
        self.db.flush()
        self.user = user
        character = Character(user_id=user.id, name="Mina")
        self.db.add(character)
        self.db.flush()
        self.character = character
        conversation = Conversation(character_id=character.id, user_id=user.id)
        self.db.add(conversation)
        self.db.commit()
        self.conversation = conversation

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _counter(self, *_args, **_kwargs):
        return lambda instructions, messages: len(instructions) + sum(len(m["content"]) for m in messages)

    def test_korean_locale_flows_to_chat_prompt_and_current_message_stays_last(self):
        with patch.object(chat_service, "make_chat_input_counter", side_effect=self._counter):
            with patch.object(chat_service, "generate_text", return_value="답변") as call:
                chat_service.send_message(
                    self.db, self.character.id, self.conversation.id, "안녕", self.user.id, "ko"
                )
        self.assertIn("Respond in Korean", call.call_args.kwargs["instructions"])
        self.assertEqual(call.call_args.kwargs["input_messages"][-1], {"role": "user", "content": "안녕"})

    def test_locale_does_not_enter_correction_analysis_path(self):
        with patch.object(chat_service, "apply_user_correction", return_value=SimpleNamespace(version=2)) as correction:
            with patch.object(chat_service, "build_chat_instructions") as builder:
                result = chat_service.send_message(
                    self.db, self.character.id, self.conversation.id, "/수정 차분하게", self.user.id, "en"
                )
        self.assertEqual(result["role"], "SYSTEM_NOTE")
        correction.assert_called_once_with(self.db, self.user.id, self.character.id, "차분하게")
        builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
