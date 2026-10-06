"""Durable room and existing-chat regression tests; all provider calls are mocked."""

import unittest
from datetime import timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.auth import issue_token
from app.database import Base, get_db
from app.main import app
from app.models import (
    Character, CharacterProfile, Conversation, MemoryIngestion, Message, MessageRole, User, World, utcnow,
)
from app.llm_failures import LLMStructuredOutputError
from app.schemas import CharacterProfileData, TimelineEvent, TimelineStateChanges
from app.services import character_conversations as rooms, chat_service, memory_jobs, memory_service
from app.services.conversation_loop_guard import is_obvious_loop
from app.services.conversation_runtime import ChatTurnResult, TurnProgression, empty_turn


def fake_counter(*_args, **_kwargs):
    return lambda instructions, messages: (len(instructions) + sum(len(m["content"]) for m in messages)) // 5 + 1


class CharacterConversationAPITests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            owner = User(email="owner@example.invalid", password_hash="unused")
            outsider = User(email="outsider@example.invalid", password_hash="unused")
            db.add_all([owner, outsider])
            db.flush()
            world_a = World(user_id=owner.id, name="World A")
            world_b = World(user_id=owner.id, name="World B")
            db.add_all([world_a, world_b])
            db.flush()
            first = Character(user_id=owner.id, name="Ari", world_id=world_a.id)
            second = Character(user_id=owner.id, name="Bex", world_id=world_b.id)
            foreign = Character(user_id=outsider.id, name="Other")
            db.add_all([first, second, foreign])
            db.flush()
            first_profile = CharacterProfileData(
                personality_summary="French source character",
                timeline=[TimelineEvent(event_key="final-night", age=21, summary="Final night",
                                        state_changes=TimelineStateChanges(location="prison"))],
            )
            second_profile = CharacterProfileData(personality_summary="B-only secret persona")
            db.add_all([
                CharacterProfile(character_id=first.id, data=first_profile.model_dump(), version=1),
                CharacterProfile(character_id=second.id, data=second_profile.model_dump(), version=1),
            ])
            db.commit()
            self.owner_id, self.outsider_id = owner.id, outsider.id
            self.first_id, self.second_id, self.foreign_id = first.id, second.id, foreign.id

        def test_db():
            with Session(self.engine) as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        self.client = TestClient(app, raise_server_exceptions=False)
        self.client.__enter__()
        self.headers = {"Authorization": f"Bearer {issue_token(self.owner_id)}"}
        self.other_headers = {"Authorization": f"Bearer {issue_token(self.outsider_id)}"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        app.dependency_overrides.clear()
        self.engine.dispose()

    def create_room(self, language="en"):
        response = self.client.post("/character-conversations", headers=self.headers, json={
            "character_a_id": self.first_id, "character_b_id": self.second_id,
            "name": "  A Strange Meeting  ", "language": language,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_create_list_rename_delete_and_owner_isolation(self):
        room = self.create_room()
        self.assertEqual(room["name"], "A Strange Meeting")
        self.assertEqual(room["turn_index"], 0)
        self.assertEqual([p["position"] for p in room["participants"]], [0, 1])
        self.assertEqual(room["next_speaker_character_id"], self.first_id)
        self.assertEqual(len(self.client.get("/character-conversations", headers=self.headers).json()), 1)
        self.assertEqual(self.client.get("/character-conversations", headers=self.other_headers).json(), [])
        for method, path in (("get", f"/character-conversations/{room['id']}"),
                             ("get", f"/character-conversations/{room['id']}/messages"),
                             ("delete", f"/character-conversations/{room['id']}")):
            self.assertEqual(getattr(self.client, method)(path, headers=self.other_headers).status_code, 404)
        self.assertEqual(self.client.post(f"/character-conversations/{room['id']}/next",
                                          headers=self.other_headers, json={"expected_turn_index": 0}).status_code, 404)
        renamed = self.client.patch(f"/character-conversations/{room['id']}", headers=self.headers,
                                    json={"name": "Renamed"})
        self.assertEqual(renamed.json()["name"], "Renamed")
        self.assertEqual(self.client.delete(f"/character-conversations/{room['id']}",
                                            headers=self.headers).status_code, 204)
        self.assertEqual(self.client.get(f"/character-conversations/{room['id']}",
                                         headers=self.headers).status_code, 404)

    def test_invalid_pair_and_room_name(self):
        self.assertEqual(self.client.get("/character-conversations").status_code, 401)
        for second_id, expected in ((self.first_id, 422), (self.foreign_id, 404)):
            response = self.client.post("/character-conversations", headers=self.headers, json={
                "character_a_id": self.first_id, "character_b_id": second_id,
                "name": "Meeting", "language": "en",
            })
            self.assertEqual(response.status_code, expected)
        blank = self.client.post("/character-conversations", headers=self.headers, json={
            "character_a_id": self.first_id, "character_b_id": self.second_id,
            "name": "  ", "language": "en",
        })
        self.assertEqual(blank.status_code, 422)

    def test_next_is_exactly_one_turn_and_alternates_with_isolated_canon(self):
        room = self.create_room(language="en")
        responses = [
            "The weather seems different tonight.",
            "Your observation makes me consider the road.",
            "The road may lead us somewhere unfamiliar.",
            "Then I will ask what you expect to find.",
        ]
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=responses) as generation, \
             patch.object(memory_jobs, "schedule_ingestion") as memory_write:
            for expected in range(4):
                result = self.client.post(f"/character-conversations/{room['id']}/next",
                                          headers=self.headers, json={"expected_turn_index": expected})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["room"]["turn_index"], expected + 1)
                self.assertEqual(result.json()["message"]["turn_index"], expected + 1)
        self.assertEqual(generation.call_count, 4)
        memory_write.assert_not_called()
        instructions_a = generation.call_args_list[0].kwargs["instructions"]
        instructions_b = generation.call_args_list[1].kwargs["instructions"]
        self.assertIn("Current location: prison", instructions_a)
        self.assertNotIn("B-only secret persona", instructions_a)
        self.assertIn("B-only secret persona", instructions_b)
        self.assertNotIn("French source character", instructions_b)
        self.assertIn("Respond in English", instructions_a)
        self.assertEqual(generation.call_args_list[0].kwargs["task"], "chat")
        self.assertEqual(generation.call_args_list[0].kwargs["request_type"], "character_conversation")
        history = self.client.get(f"/character-conversations/{room['id']}/messages", headers=self.headers)
        self.assertEqual([m["speaker_character_id"] for m in history.json()],
                         [self.first_id, self.second_id, self.first_id, self.second_id])
        self.assertEqual([m["content"] for m in history.json()], responses)
        with Session(self.engine) as db:
            self.assertEqual(db.query(MemoryIngestion).count(), 0)

    def test_korean_room_ignores_french_source_language(self):
        room = self.create_room(language="ko")
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", return_value="안녕하세요.") as generation:
            response = self.client.post(f"/character-conversations/{room['id']}/next",
                                        headers=self.headers, json={"expected_turn_index": 0})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Respond in Korean", generation.call_args.kwargs["instructions"])

    def test_obvious_loop_gets_one_bounded_retry(self):
        room = self.create_room()
        with Session(self.engine) as db:
            db.add(Message(conversation_id=room["id"], role=MessageRole.CHARACTER,
                           speaker_character_id=self.first_id, turn_index=1,
                           content="Tell me about your family."))
            db.query(Conversation).filter_by(id=room["id"]).update({"turn_index": 1})
            db.commit()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=[
                 "Tell me about your family.", "The road ahead seems newly uncertain.",
             ]) as generation:
            result = self.client.post(f"/character-conversations/{room['id']}/next",
                                      headers=self.headers, json={"expected_turn_index": 1})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(generation.call_count, 2)
        self.assertEqual(result.json()["message"]["content"], "The road ahead seems newly uncertain.")
        self.assertEqual(result.json()["room"]["turn_index"], 2)

    def test_second_loop_result_is_used_without_more_retries(self):
        room = self.create_room()
        with Session(self.engine) as db:
            db.add(Message(conversation_id=room["id"], role=MessageRole.CHARACTER,
                           speaker_character_id=self.first_id, turn_index=1,
                           content="Tell me about your family."))
            db.query(Conversation).filter_by(id=room["id"]).update({"turn_index": 1})
            db.commit()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", return_value="Tell me about your family.") as generation:
            result = self.client.post(f"/character-conversations/{room['id']}/next",
                                      headers=self.headers, json={"expected_turn_index": 1})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(generation.call_count, 2)

    def test_room_combines_style_and_language_repair_in_one_retry(self):
        room = self.create_room(language="ko")
        prior = [
            (self.first_id, "참 답답하구나, 이 바보야."),
            (self.second_id, "그렇다면 다른 길은 어때?"),
            (self.first_id, "왜 그러니, 이 바보야."),
            (self.second_id, "이제 직접 결정해 봐."),
        ]
        with Session(self.engine) as db:
            for index, (speaker, content) in enumerate(prior, 1):
                db.add(Message(conversation_id=room["id"], role=MessageRole.CHARACTER,
                               speaker_character_id=speaker, turn_index=index, content=content))
            db.query(Conversation).filter_by(id=room["id"]).update({"turn_index": 4})
            db.commit()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=[
                 "이 바보야. This is a complete English sentence about the road ahead.",
                 "[Character action: 나는 고개를 숙인다.] 이제 다른 길을 찾자.",
             ]) as generation:
            response = self.client.post(f"/character-conversations/{room['id']}/next",
                                        headers=self.headers, json={"expected_turn_index": 4})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(generation.call_count, 2)
        self.assertEqual(response.json()["room"]["turn_index"], 5)
        self.assertEqual(response.json()["message"]["content"],
                         "<action>고개를 숙인다.</action> 이제 다른 길을 찾자.")

    def test_loop_guard_conservative_progression_and_late_callback(self):
        self.assertTrue(is_obvious_loop("Tell me about your family?",
                                        ["Tell me about your family."]))
        self.assertTrue(is_obvious_loop("My family waited for me at home.",
                                        ["My family was waiting for me at home."]))
        self.assertFalse(is_obvious_loop("The river has flooded the road.",
                                         ["Tell me about your family."]))
        self.assertFalse(is_obvious_loop("Tell me about your family?", [
            "Tell me about your family.", "We should visit the harbor.",
            "The harbor is closed now.", "Perhaps the mountain route is open.",
            "We can reach the mountain by dawn.",
        ]))

    def test_failure_does_not_advance_and_stale_next_does_not_generate(self):
        room = self.create_room()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=RuntimeError("mock failure")) as generation:
            failed = self.client.post(f"/character-conversations/{room['id']}/next",
                                      headers=self.headers, json={"expected_turn_index": 0})
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(generation.call_count, 1)
        with Session(self.engine) as db:
            stored = db.get(Conversation, room["id"])
            self.assertEqual(stored.turn_index, 0)
            self.assertIsNone(stored.generation_token)
            self.assertEqual(db.query(Message).filter_by(conversation_id=room["id"]).count(), 0)
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", return_value="A fresh reply that moves forward.") as generation:
            retried = self.client.post(f"/character-conversations/{room['id']}/next",
                                       headers=self.headers, json={"expected_turn_index": 0})
            self.assertEqual(retried.status_code, 200, retried.text)
            stale = self.client.post(f"/character-conversations/{room['id']}/next",
                                     headers=self.headers, json={"expected_turn_index": 0})
            self.assertEqual(stale.status_code, 409)
            self.assertEqual(generation.call_count, 1)

    def test_structured_metadata_is_persisted_but_not_exposed_after_reentry(self):
        room = self.create_room()
        with Session(self.engine) as db:
            db.query(Conversation).filter_by(id=room["id"]).update({"runtime_state": {
                "version": 1, "last_message_id": None, "open_threads": ["should_gregor_call_grete"],
            }})
            db.commit()
        first = ChatTurnResult(response="Gregor calls Grete. Will she answer?", progression=TurnProgression(
            topic="grete", new_development="gregor_calls_grete",
            resolved_thread="should_gregor_call_grete", opened_thread="does_grete_respond",
            action_taken=None, advice_given=None, repeated_point=False,
        ))
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=[first, empty_turn("She waits by the doorway.")]) as generation:
            created = self.client.post(f"/character-conversations/{room['id']}/next",
                                       headers=self.headers, json={"expected_turn_index": 0})
            self.assertEqual(created.status_code, 200, created.text)
            self.assertEqual(created.json()["message"]["content"], first.response)
            self.assertNotIn("progression", created.text)
            with Session(self.engine) as db:
                stored = db.get(Conversation, room["id"]).runtime_state
                self.assertEqual(stored["open_threads"], ["does_grete_respond"])
                self.assertEqual(stored["resolved_threads"], ["should_gregor_call_grete"])
            resumed = self.client.post(f"/character-conversations/{room['id']}/next",
                                       headers=self.headers, json={"expected_turn_index": 1})
            self.assertEqual(resumed.status_code, 200, resumed.text)
            self.assertIn("does_grete_respond", generation.call_args_list[1].kwargs["instructions"])

    def test_schema_failure_consumes_only_one_retry_and_releases_claim(self):
        room = self.create_room()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=[
                 LLMStructuredOutputError("invalid"), empty_turn("A different path opens."),
             ]) as generation:
            recovered = self.client.post(f"/character-conversations/{room['id']}/next",
                                         headers=self.headers, json={"expected_turn_index": 0})
        self.assertEqual(recovered.status_code, 200, recovered.text)
        self.assertEqual(generation.call_count, 2)
        failed_room = self.create_room()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", side_effect=LLMStructuredOutputError("invalid")) as generation:
            failed = self.client.post(f"/character-conversations/{failed_room['id']}/next",
                                      headers=self.headers, json={"expected_turn_index": 0})
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(failed.json()["detail"]["code"], "room_generation_failed")
        self.assertEqual(generation.call_count, 2)
        with Session(self.engine) as db:
            stored = db.get(Conversation, failed_room["id"])
            self.assertEqual(stored.turn_index, 0)
            self.assertIsNone(stored.generation_token)
            self.assertEqual(db.query(Message).filter_by(conversation_id=failed_room["id"]).count(), 0)

    def test_expired_claim_recovers_without_advance_or_phantom_message(self):
        room = self.create_room()
        with Session(self.engine) as db:
            db.query(Conversation).filter_by(id=room["id"]).update({
                "generation_token": "old-token", "generation_lease_expires_at": utcnow() - timedelta(seconds=1),
            })
            db.commit()
        with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(rooms, "generate_text", return_value=empty_turn("A fresh turn.")) as generation:
            result = self.client.post(f"/character-conversations/{room['id']}/next",
                                      headers=self.headers, json={"expected_turn_index": 0})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(generation.call_count, 1)
        with Session(self.engine) as db:
            stored = db.get(Conversation, room["id"])
            self.assertEqual(stored.turn_index, 1)
            self.assertIsNone(stored.generation_token)
            self.assertEqual(db.query(Message).filter_by(conversation_id=room["id"]).count(), 1)

    def test_character_deletion_cascades_room_and_messages(self):
        room = self.create_room()
        with Session(self.engine) as db:
            db.add(Message(conversation_id=room["id"], role=MessageRole.CHARACTER,
                           speaker_character_id=self.first_id, turn_index=1, content="A turn"))
            db.commit()
        with patch.object(memory_service, "PROVIDER_NAME", "noop"):
            deleted = self.client.delete(f"/characters/{self.second_id}", headers=self.headers)
        self.assertEqual(deleted.status_code, 204, deleted.text)
        with Session(self.engine) as db:
            self.assertIsNone(db.get(Conversation, room["id"]))
            self.assertEqual(db.query(Message).filter_by(conversation_id=room["id"]).count(), 0)

    def test_existing_user_chat_resumes_from_db_and_keeps_memory_intent(self):
        with patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
             patch.object(memory_service, "retrieve_for_turn", return_value=memory_service.MemoryRetrievalResult(
                 candidates=(), provider="memmachine", latency_ms=0, success=True,
             )), \
             patch.object(memory_jobs, "publish_safe"), \
             patch.object(chat_service, "make_chat_input_counter", side_effect=fake_counter), \
             patch.object(chat_service, "generate_text", return_value="I remember the bridge."):
            created = self.client.post("/chat/conversations", headers=self.headers,
                                       json={"character_id": self.first_id, "reuse_existing": True})
            self.assertEqual(created.status_code, 201)
            conversation_id = created.json()["id"]
            sent = self.client.post(f"/chat/conversations/{conversation_id}/messages",
                                    headers=self.headers, json={"content": "Do you remember the bridge?"})
            self.assertEqual(sent.status_code, 200, sent.text)
            resumed = self.client.post("/chat/conversations", headers=self.headers,
                                       json={"character_id": self.first_id, "reuse_existing": True})
            self.assertEqual(resumed.json()["id"], conversation_id)
            history = self.client.get(f"/chat/conversations/{conversation_id}/messages", headers=self.headers)
        self.assertEqual([m["content"] for m in history.json()],
                         ["Do you remember the bridge?", "I remember the bridge."])
        with Session(self.engine) as db:
            intent = db.query(MemoryIngestion).one()
            self.assertEqual(intent.user_message_id, history.json()[0]["id"])
            self.assertEqual(intent.assistant_message_id, history.json()[1]["id"])


if __name__ == "__main__":
    unittest.main()
