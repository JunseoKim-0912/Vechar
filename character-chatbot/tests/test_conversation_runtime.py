"""Provider-free stateful conversation contract and recovery tests."""

import unittest
from types import SimpleNamespace

from openai.lib._parsing._responses import type_to_text_format_param

from app.models import MessageRole
from app.services.conversation_runtime import (
    COOLDOWN_REPLIES, RUNTIME_VERSION, ChatTurnResult, ConversationRuntimeState,
    TurnFidelity, TurnProgression, advance_runtime, empty_turn, phrase_families, validated_delta,
)
from app.services.character_fidelity import KnowledgeScope
from app.services.response_quality import language_mismatch, select_quality_response


def turn(response, **changes):
    fields = dict(topic="grete_music", new_development=None, resolved_thread=None,
                  opened_thread=None, action_taken=None, advice_given=None,
                  repeated_point=False)
    fields.update(changes)
    return ChatTurnResult(response=response, progression=TurnProgression(**fields),
                          fidelity=TurnFidelity(knowledge_scope=KnowledgeScope.UNCERTAIN,
                                                persona_preserved=True, assistant_mode=False))


class RuntimeStateTests(unittest.TestCase):
    def test_structured_schema_is_compact_required_and_same_call(self):
        fmt = type_to_text_format_param(ChatTurnResult)
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["strict"])
        self.assertIn("response", fmt["schema"]["required"])
        self.assertIn("progression", fmt["schema"]["required"])
        self.assertIn("fidelity", fmt["schema"]["required"])
        with self.assertRaises(ValueError):
            empty_turn("  ")
        with self.assertRaises(ValueError):
            empty_turn("<action></action>")
        with self.assertRaises(ValueError):
            empty_turn('{"response":"hi","progression":{}}')

    def test_call_resolves_thread_and_opens_response_thread(self):
        state = ConversationRuntimeState(open_threads=["should_gregor_call_grete"])
        result = turn("Gregor calls Grete. He waits for an answer.",
                      new_development="gregor_calls_grete",
                      resolved_thread="should_gregor_call_grete",
                      opened_thread="does_grete_respond")
        delta = validated_delta(state, result)
        self.assertTrue(delta["progression"])
        advance_runtime(state, result, result.response, "gregor", "message-1")
        self.assertEqual(state.open_threads, ["does_grete_respond"])
        self.assertEqual(state.resolved_threads, ["should_gregor_call_grete"])

    def test_resolved_advice_repetition_and_established_desire_fail(self):
        state = ConversationRuntimeState(
            established_points=["desire:proximity"],
            resolved_threads=["should_gregor_call_grete"], open_threads=["does_grete_respond"],
        )
        calls = []

        def retry(correction):
            calls.append(correction)
            return turn("Grete answers from the doorway.", new_development="grete_answers")

        result = select_quality_response(
            turn("You should call Grete.", repeated_point=False),
            same_speaker_recent=[], language="English", allow_repetition=False,
            retry=retry, runtime_state=state, speaker="doctor", room=True,
        )
        self.assertTrue(result.initial.progression_failure)
        self.assertEqual(result.retry_count, 1)
        self.assertIn("Resolved threads", calls[0])
        self.assertIn("does_grete_respond", calls[0])
        desire = select_quality_response(
            turn("I want to hear the music closer.", new_development="unsupported_label"),
            same_speaker_recent=[], language="English", allow_repetition=False,
            retry=lambda _: turn("The family reacts when Gregor calls.",
                                  new_development="family_reacts"),
            runtime_state=state, speaker="gregor", room=True,
        )
        self.assertTrue(desire.initial.progression_failure)
        self.assertFalse(desire.final.progression_failure)

    def test_new_consequence_accepted_and_false_metadata_not_trusted(self):
        state = ConversationRuntimeState(open_threads=["does_grete_respond"])
        good = select_quality_response(
            turn("The family reacts when Gregor calls Grete.",
                 new_development="family_reacts"),
            same_speaker_recent=["Gregor calls Grete."], language="English",
            allow_repetition=False, retry=lambda _: self.fail("unnecessary retry"),
            runtime_state=state, speaker="doctor", room=True,
        )
        self.assertEqual(good.retry_count, 0)
        self.assertTrue(good.final.progression_detected)
        bogus = validated_delta(state, turn("Nothing has changed.",
                                            new_development="family_reacts",
                                            resolved_thread="does_grete_respond"))
        self.assertFalse(bogus["progression"])
        self.assertFalse(validated_delta(state, turn("Grete might answer someday.",
                                                    resolved_thread="does_grete_respond"))["resolved"])
        self.assertFalse(validated_delta(state, turn("<action>Raises a hand.</action>",
                                                    action_taken="walks_to_window"))["action"])
        call_state = ConversationRuntimeState(open_threads=["should_gregor_call_grete"])
        self.assertFalse(validated_delta(call_state, turn("Gregor cannot call Grete.",
                                                         resolved_thread="should_gregor_call_grete"))["resolved"])

    def test_snapshot_survives_reentry_and_invalid_version_rebuilds(self):
        state = ConversationRuntimeState(open_threads=["should_gregor_call_grete"])
        response = turn("Gregor calls Grete. Will she answer?",
                        resolved_thread="should_gregor_call_grete",
                        opened_thread="does_grete_respond")
        advance_runtime(state, response, response.response, "gregor", "m1")
        messages = [SimpleNamespace(id="m1", role=MessageRole.CHARACTER,
                                    speaker_character_id="gregor", content=response.response)]
        restored = ConversationRuntimeState.from_storage(state.to_storage(), messages)
        self.assertEqual(restored.version, RUNTIME_VERSION)
        self.assertEqual(restored.open_threads, ["does_grete_respond"])
        self.assertEqual(restored.resolved_threads, ["should_gregor_call_grete"])
        stale = state.to_storage() | {"version": RUNTIME_VERSION + 1}
        rebuilt = ConversationRuntimeState.from_storage(stale, messages)
        self.assertEqual(rebuilt.last_message_id, "m1")
        self.assertFalse(rebuilt.open_threads)  # Old plain text cannot prove model-only hints.

    def test_catchphrase_family_cooldown_and_expiry(self):
        state = ConversationRuntimeState()
        advance_runtime(state, empty_turn("참 맹추야."), "참 맹추야.", "jeomsuni", "m1")
        self.assertEqual(phrase_families("아이고, 이 맹추야!"), {"맹추야"})
        self.assertIn("맹추야", state.active_families("jeomsuni"))
        result = select_quality_response(
            turn("다시 그래, 이 맹추야."), same_speaker_recent=[], language="Korean",
            allow_repetition=False, retry=lambda _: turn("다른 길로 가자."),
            runtime_state=state, speaker="jeomsuni", room=False,
        )
        self.assertTrue(result.initial.catchphrase_cooldown_triggered)
        self.assertFalse(result.final.catchphrase_cooldown_triggered)
        self.assertNotIn("맹추야", str(result.safe_metadata("user_character", "Korean")))
        for index in range(COOLDOWN_REPLIES):
            advance_runtime(state, empty_turn("다른 길로 가자."), "다른 길로 가자.",
                            "jeomsuni", f"m{index + 2}")
        self.assertNotIn("맹추야", state.active_families("jeomsuni"))
        self.assertFalse(phrase_families("그래. 응. 아니. yes. no."))

    def test_kana_and_action_style_share_one_corrective_retry(self):
        state = ConversationRuntimeState(resolved_threads=["should_gregor_call_grete"])
        advance_runtime(state, empty_turn("이 맹추야."), "이 맹추야.", "doctor", "m1")
        calls = []

        def retry(correction):
            calls.append(correction)
            return turn("Grete answers. <action>Lowers his gaze.</action>",
                        new_development="grete_answers")

        result = select_quality_response(
            turn("You should call Grete, 이 맹추야. づ <action>I lower my gaze.</action>"),
            same_speaker_recent=[], language="English", allow_repetition=False,
            retry=retry, runtime_state=state, speaker="doctor", room=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.retry_count, 1)
        self.assertTrue(result.initial.progression_failure)
        self.assertTrue(result.initial.catchphrase_cooldown_triggered)
        self.assertTrue(result.initial.unicode_script_mismatch)
        self.assertTrue(result.initial.action_first_person_violation)
        self.assertFalse(result.final.reasons)
        self.assertTrue(language_mismatch("가까づ고 싶어.", "Korean"))
        self.assertTrue(language_mismatch("カタカナ", "Korean"))
        self.assertFalse(language_mismatch("안녕, Gregor.", "Korean"))
        self.assertFalse(language_mismatch("Gregor Samsa와 이야기한다.", "Korean"))

    def test_explicit_repetition_keeps_safety_checks(self):
        state = ConversationRuntimeState()
        advance_runtime(state, empty_turn("이 맹추야."), "이 맹추야.", "jeomsuni", "m1")
        result = select_quality_response(
            turn("다시 말할게, 이 맹추야. づ"), same_speaker_recent=["다시 말할게, 이 맹추야."],
            language="Korean", allow_repetition=True,
            retry=lambda _: turn("다시 말할게. "), runtime_state=state,
            speaker="jeomsuni", room=False,
        )
        self.assertFalse(result.initial.semantic_loop_detected)
        self.assertFalse(result.initial.catchphrase_cooldown_triggered)
        self.assertTrue(result.initial.language_mismatch_detected)
        self.assertEqual(result.retry_count, 1)

    def test_retry_falls_back_if_it_introduces_new_safety_violation(self):
        result = select_quality_response(
            turn("You should call Grete."), same_speaker_recent=[], language="English",
            allow_repetition=False, retry=lambda _: turn("Grete answered. づ", new_development="grete_answered"),
            runtime_state=ConversationRuntimeState(resolved_threads=["should_gregor_call_grete"]),
            speaker="doctor", room=True,
        )
        self.assertTrue(result.retry_fallback)
        self.assertEqual(result.text, "You should call Grete.")
        self.assertEqual(result.retry_count, 1)


if __name__ == "__main__":
    unittest.main()
