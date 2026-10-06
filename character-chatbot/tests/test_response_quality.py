"""Synthetic, provider-free checks for the shared conversation quality boundary."""

import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Character, Conversation, Message, MessageRole, User
from app.services import chat_service, memory_jobs, memory_service
from app.services.chat_actions import (
    canonicalize_assistant_message, normalize_assistant_actions, parse_assistant_actions,
)
from app.services.conversation_loop_guard import derive_conversation_state, is_semantic_loop
from app.services.response_quality import (
    explicit_repetition_request, language_mismatch, select_quality_response, style_repetition,
)


class SemanticProgressionTests(unittest.TestCase):
    def test_grete_desire_paraphrases_are_same_state(self):
        history = ["I want Grete to play nearer.", "I want to be closer to Grete."]
        self.assertTrue(is_semantic_loop("I want Grete's music closer to me.", history))

    def test_repeated_advice_and_doctor_motif_are_same_point(self):
        self.assertTrue(is_semantic_loop("Don't hide it.", ["Tell her what you want."]))
        self.assertTrue(is_semantic_loop("Just say what you feel.", ["Don't hide it."]))
        self.assertTrue(is_semantic_loop("Distance makes me helpless.", ["Snow keeps me away."]))
        self.assertTrue(is_semantic_loop("I am helpless in the snow.", ["Distance makes me helpless."]))
        self.assertFalse(is_semantic_loop("Tell him to leave the room.",
                                          ["Tell her what you want."]))

    def test_progression_and_natural_late_callback_are_allowed(self):
        self.assertFalse(is_semantic_loop("I called Grete and she answered.",
                                          ["I want Grete to play nearer."]))
        self.assertFalse(is_semantic_loop("I fear returning home after this.",
                                          ["Snow keeps me away."]))
        self.assertFalse(is_semantic_loop("Professional duty now falls to me.",
                                          ["Distance makes me helpless."]))
        self.assertFalse(is_semantic_loop("I want Grete's music closer to me.", [
            "I want Grete to play nearer.", "I called to her.", "She answered me.",
            "The family may object.", "I considered their response.", "We chose another room.",
        ]))

    def test_state_is_bounded_tags_not_raw_dialogue(self):
        state = derive_conversation_state([
            "I want Grete to play nearer.", "Tell her what you want.",
            "<action>Raises a hand.</action> Will she answer?",
        ])
        block = state.prompt_block()
        self.assertIn("desire:proximity", state.established_points)
        self.assertIn("advice:disclosure", state.recent_advice)
        self.assertTrue(state.open_threads)
        self.assertTrue(state.recent_actions)
        self.assertNotIn("Grete", block)


class StyleAndLanguageTests(unittest.TestCase):
    def test_catchphrase_cooldown_and_common_endings(self):
        self.assertTrue(style_repetition("또 그러네, 이 바보야.", [
            "참 답답하구나, 이 바보야.", "정말 그럴 거니, 이 바보야.",
        ]))
        self.assertFalse(style_repetition("그래. 오늘은 간다.", ["해.", "한다.", "알았어."]))
        self.assertTrue(style_repetition("By the crooked moon, this is strange.", [
            "By the crooked moon, I was surprised.", "By the crooked moon, the road changed.",
        ]))
        self.assertTrue(style_repetition("그래, 새로운 길이 있네.", [
            "그래, 먼저 들어가자.", "그래, 생각해보자.",
        ]))
        self.assertFalse(style_repetition("Gregor Samsa leaves the room.", [
            "Gregor Samsa looks at the clock.", "Gregor Samsa considers his family.",
        ]))

    def test_language_guard_is_conservative_and_checks_actions(self):
        self.assertFalse(language_mismatch("그레고르 Samsa가 창밖을 본다.", "Korean"))
        self.assertFalse(language_mismatch("Gregor Samsa와 이야기를 나눴어. 커피도 마셨어.", "Korean"))
        self.assertFalse(language_mismatch("<action>고개를 숙인다.</action> 이제 가자.", "Korean"))
        self.assertTrue(language_mismatch("우리는 간다. This is a complete English sentence about the road ahead.", "Korean"))
        self.assertTrue(language_mismatch("<action>This is a complete English sentence about the gesture.</action> 그래.", "Korean"))
        self.assertTrue(language_mismatch("We will leave now. 나는 여기서 계속 기다리고 있다.", "English"))
        self.assertTrue(language_mismatch("I know the answer, and you should listen to what I say now.", "French"))
        self.assertFalse(language_mismatch("I met Gregor Samsa near the station.", "English"))

    def test_single_aggregated_retry_budget_and_fallback(self):
        calls = []

        def retry(instruction):
            calls.append(instruction)
            return "<action>고개를 숙인다.</action> 이제 다른 길을 택하자."

        outcome = select_quality_response(
            "이 바보야. This is a complete English sentence about the road ahead.",
            same_speaker_recent=["저리 가, 이 바보야.", "뭘 하는 거야, 이 바보야."],
            language="Korean", allow_repetition=False, retry=retry,
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("style", outcome.initial.reasons[0])
        self.assertIn("language_mismatch", outcome.initial.reasons)
        self.assertEqual(outcome.retry_count, 1)
        self.assertFalse(outcome.retry_fallback)
        self.assertIn("<action>고개를 숙인다.</action>", outcome.text)
        safe = str(outcome.safe_metadata("user_character", "Korean"))
        self.assertNotIn("이 바보야", safe)
        self.assertNotIn("road ahead", safe)
        failed = select_quality_response(
            "Tell her what you want.", same_speaker_recent=["Don't hide it."],
            language="English", allow_repetition=False,
            retry=lambda _: (_ for _ in ()).throw(RuntimeError("mock retry failure")),
        )
        self.assertEqual(failed.retry_count, 1)
        self.assertTrue(failed.retry_fallback)

    def test_explicit_repeat_request_suppresses_semantic_guard(self):
        self.assertTrue(explicit_repetition_request("방금 한 말 반복해줘"))
        self.assertTrue(explicit_repetition_request("Please say that again"))
        outcome = select_quality_response(
            "Tell her what you want.", same_speaker_recent=["Tell her what you want.",
                                                              "Tell her what you want."],
            language="English", allow_repetition=True,
            retry=lambda _: self.fail("explicit repetition should not retry"),
        )
        self.assertEqual(outcome.retry_count, 0)


class ActionParserTests(unittest.TestCase):
    def test_known_variants_canonicalize_to_typed_segments(self):
        for raw in (
            "<action>고개를 숙인다.</action>",
            "[Character action: 고개를 숙인다.]",
            "<action> 고개를 숙인다. </action>",
            "<action><action>고개를 숙인다.</action></action>",
            "<action>고개를 숙인다.</action",
            "<action>고개를 숙인다.",
        ):
            canonical, parsed = canonicalize_assistant_message(raw)
            self.assertEqual(canonical, "<action>고개를 숙인다.</action>", raw)
            self.assertEqual(parsed.segments, ({"type": "action", "text": "고개를 숙인다."},), raw)

    def test_multiple_actions_order_and_history_marker(self):
        raw = "안녕. <action>고개를 숙인다.</action> 이제 가자. [Character action: 손을 흔든다.]"
        canonical, parsed = canonicalize_assistant_message(raw)
        self.assertEqual([item["type"] for item in parsed.segments],
                         ["dialogue", "action", "dialogue", "action"])
        self.assertNotIn("[Character action:", canonical)
        self.assertIn("[Character action: 고개를 숙인다.]", normalize_assistant_actions(canonical))

    def test_safe_subjectless_actions_and_ambiguous_text(self):
        for raw, expected in (
            ("<action>나는 고개를 숙인다.</action>", "<action>고개를 숙인다.</action>"),
            ("<action>내가 창밖을 바라본다.</action>", "<action>창밖을 바라본다.</action>"),
        ):
            canonical, parsed = canonicalize_assistant_message(raw)
            self.assertEqual(canonical, expected)
            self.assertTrue(parsed.subject_normalized)
        for raw in (
            "<action>나는 그가 고개를 숙이는 것을 본다.</action>",
            "<action>Gregor lowers his gaze.</action>",
            "<action>\"I look toward the window.\"</action>",
            "<action>I look toward the window.</action>",
        ):
            canonical, parsed = canonicalize_assistant_message(raw)
            self.assertEqual(canonical, raw)
            self.assertFalse(parsed.subject_normalized)

    def test_unsafe_html_and_partial_buffer_are_never_executed_or_lost(self):
        raw = "Hello <script>alert(1)</script> <act"
        canonical, parsed = canonicalize_assistant_message(raw)
        self.assertEqual(canonical, raw)
        self.assertEqual(parsed.segments[0]["type"], "dialogue")
        self.assertEqual(parse_assistant_actions("<action>" ).segments, ())


class UserChatIntegrationTests(unittest.TestCase):
    def test_user_chat_repairs_action_and_uses_one_language_retry(self):
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                               poolclass=StaticPool)
        Base.metadata.create_all(engine)
        try:
            with Session(engine) as db:
                user = User(email="quality@example.invalid", password_hash="unused")
                db.add(user)
                db.flush()
                character = Character(user_id=user.id, name="Ari")
                db.add(character)
                db.flush()
                conversation = Conversation(user_id=user.id, character_id=character.id)
                db.add(conversation)
                db.commit()
                user_id, character_id, conversation_id = user.id, character.id, conversation.id
            with Session(engine) as db, \
                 patch.object(memory_service, "retrieve_for_turn", return_value=memory_service.MemoryRetrievalResult(
                     candidates=(), provider="noop", latency_ms=0, success=True,
                 )), \
                 patch.object(memory_jobs, "schedule_ingestion", return_value=None), \
                 patch.object(memory_jobs, "publish_safe"), \
                 patch.object(chat_service, "make_chat_input_counter", return_value=lambda *_: 50), \
                 patch.object(chat_service, "generate_text", side_effect=[
                     "This is a complete English sentence about the road ahead.",
                     "그래. [Character action: 나는 고개를 숙인다.]",
                 ]) as provider:
                result = chat_service.send_message(db, character_id, conversation_id,
                                                   "지금 어디에 있어?", user_id, locale="ko")
                self.assertEqual(provider.call_count, 2)
                self.assertEqual(result["content"], "그래. <action>고개를 숙인다.</action>")
                self.assertEqual(db.query(Message).filter_by(conversation_id=conversation_id).count(), 2)
                self.assertEqual(db.query(Message).filter_by(conversation_id=conversation_id,
                                                             role=MessageRole.CHARACTER).one().content,
                                 result["content"])
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
