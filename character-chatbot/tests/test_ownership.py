import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import llm, main
from app.auth import issue_token
from app.database import Base, get_db
from app.models import Character, Conversation, Message, User, World, WorldProfile
from app.schemas import WorldProfileData
from app.services import chat_service


class OwnershipRouteTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            alice = User(email="alice@example.invalid", password_hash="unused")
            bob = User(email="bob@example.invalid", password_hash="unused")
            db.add_all([alice, bob])
            db.flush()
            bob_world = World(user_id=bob.id, name="Bob world")
            db.add(bob_world)
            db.flush()
            bob_character = Character(user_id=bob.id, world_id=bob_world.id, name="Bob character")
            db.add(bob_character)
            db.flush()
            bob_conversation = Conversation(user_id=bob.id, character_id=bob_character.id)
            db.add(bob_conversation)
            db.commit()
            self.alice_id = alice.id
            self.bob_id = bob.id
            self.bob_world_id = bob_world.id
            self.bob_character_id = bob_character.id
            self.bob_conversation_id = bob_conversation.id

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

    def _headers(self, user_id):
        return {"Authorization": f"Bearer {issue_token(user_id)}"}

    def _post_as_alice(self, path, payload):
        return self.client.post(path, json=payload, headers=self._headers(self.alice_id))

    def test_own_world_character_and_conversation_still_work(self):
        world_response = self._post_as_alice("/worlds/", {"name": "Alice world"})
        self.assertEqual(world_response.status_code, 201)
        world_id = world_response.json()["id"]
        with Session(self.engine) as db:
            db.add(
                WorldProfile(
                    world_id=world_id,
                    data=WorldProfileData(world_summary="Alice setting").model_dump(),
                    version=1,
                )
            )
            db.commit()

        character_response = self._post_as_alice(
            "/characters/", {"name": "Alice character", "world_id": world_id}
        )
        self.assertEqual(character_response.status_code, 201)
        character_id = character_response.json()["id"]
        self.assertEqual(character_response.json()["world_id"], world_id)
        self.assertEqual(self.client.get(f"/characters/{character_id}", headers=self._headers(self.alice_id)).status_code, 200)
        self.assertEqual(self.client.get(f"/worlds/{world_id}", headers=self._headers(self.alice_id)).status_code, 200)
        self.assertEqual(
            self.client.get(f"/characters/{character_id}/export", headers=self._headers(self.alice_id)).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(f"/worlds/{world_id}/export", headers=self._headers(self.alice_id)).status_code,
            200,
        )

        conversation_response = self._post_as_alice("/chat/conversations", {"character_id": character_id})
        self.assertEqual(conversation_response.status_code, 201)
        conversation_id = conversation_response.json()["id"]
        with patch.object(chat_service, "generate_text", return_value="Mock reply") as generate:
            message_response = self._post_as_alice(
                f"/chat/conversations/{conversation_id}/messages", {"content": "Hello"}
            )
        self.assertEqual(message_response.status_code, 200)
        self.assertEqual(message_response.json()["content"], "Mock reply")
        self.assertIn("Alice setting", generate.call_args.kwargs["instructions"])
        messages = self.client.get(
            f"/chat/conversations/{conversation_id}/messages", headers=self._headers(self.alice_id)
        )
        self.assertEqual(messages.status_code, 200)
        self.assertEqual([m["content"] for m in messages.json()], ["Hello", "Mock reply"])

        default_character = self._post_as_alice("/characters/", {"name": "Default world character"})
        self.assertEqual(default_character.status_code, 201)
        with Session(self.engine) as db:
            world = db.get(World, default_character.json()["world_id"])
            self.assertEqual((world.user_id, world.name), (self.alice_id, "현실"))

    def test_character_creation_rejects_foreign_or_missing_world(self):
        for world_id in (self.bob_world_id, "missing-world", ""):
            with self.subTest(world_id=world_id):
                response = self._post_as_alice("/characters/", {"name": "Blocked", "world_id": world_id})
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["detail"], "World not found")
        with Session(self.engine) as db:
            self.assertEqual(db.query(Character).filter(Character.user_id == self.alice_id).count(), 0)

    def test_foreign_character_routes_reject_access(self):
        path = f"/characters/{self.bob_character_id}"
        headers = self._headers(self.alice_id)
        responses = [
            self.client.get(path, headers=headers),
            self.client.get(f"{path}/export", headers=headers),
            self.client.delete(path, headers=headers),
            self.client.post(
                f"{path}/training-sources", data={"source_type": "STORY", "text": "Private"}, headers=headers
            ),
            self.client.post(f"{path}/profile/rollback", json={}, headers=headers),
            self._post_as_alice("/chat/conversations", {"character_id": self.bob_character_id}),
        ]
        self.assertTrue(all(response.status_code == 404 for response in responses))
        with Session(self.engine) as db:
            self.assertIsNotNone(db.get(Character, self.bob_character_id))

    def test_foreign_world_routes_reject_access(self):
        path = f"/worlds/{self.bob_world_id}"
        headers = self._headers(self.alice_id)
        responses = [
            self.client.get(path, headers=headers),
            self.client.get(f"{path}/export", headers=headers),
            self.client.delete(path, headers=headers),
            self.client.post(
                f"{path}/sources", data={"source_type": "DESCRIPTION", "text": "Private"}, headers=headers
            ),
            self.client.get(f"{path}/summary", headers=headers),
            self.client.post(f"{path}/edit", json={"operation": "add", "instruction": "Private"}, headers=headers),
            self.client.post(f"{path}/compact", headers=headers),
            self.client.post(f"{path}/extract-character", json={"name": "Private"}, headers=headers),
        ]
        self.assertTrue(all(response.status_code == 404 for response in responses))
        with Session(self.engine) as db:
            self.assertIsNotNone(db.get(World, self.bob_world_id))

    def test_foreign_conversation_rejects_read_and_write(self):
        path = f"/chat/conversations/{self.bob_conversation_id}/messages"
        self.assertEqual(self.client.get(path, headers=self._headers(self.alice_id)).status_code, 404)
        self.assertEqual(self._post_as_alice(path, {"content": "Blocked"}).status_code, 404)
        with Session(self.engine) as db:
            self.assertEqual(db.query(Message).filter(Message.conversation_id == self.bob_conversation_id).count(), 0)

    def test_legacy_cross_user_world_link_is_blocked_before_chat(self):
        with Session(self.engine) as db:
            character = Character(user_id=self.alice_id, world_id=self.bob_world_id, name="Legacy link")
            db.add(character)
            db.flush()
            conversation = Conversation(user_id=self.alice_id, character_id=character.id)
            db.add(conversation)
            db.commit()
            character_id, conversation_id = character.id, conversation.id

        response = self._post_as_alice("/chat/conversations", {"character_id": character_id})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "World not found")
        response = self._post_as_alice(
            f"/chat/conversations/{conversation_id}/messages", {"content": "Blocked"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "World not found")
        with Session(self.engine) as db:
            self.assertEqual(db.query(Message).filter(Message.conversation_id == conversation_id).count(), 0)

    def test_legacy_conversation_with_foreign_character_is_blocked(self):
        with Session(self.engine) as db:
            conversation = Conversation(user_id=self.alice_id, character_id=self.bob_character_id)
            db.add(conversation)
            db.commit()
            conversation_id = conversation.id

        path = f"/chat/conversations/{conversation_id}/messages"
        self.assertEqual(self.client.get(path, headers=self._headers(self.alice_id)).status_code, 404)
        self.assertEqual(self._post_as_alice(path, {"content": "Blocked"}).status_code, 404)

    def test_imported_entities_belong_to_importing_user(self):
        world = self._post_as_alice(
            "/worlds/import",
            {"export_type": "world", "schema_version": 1, "name": "Imported world", "profile_data": {}},
        )
        character = self._post_as_alice(
            "/characters/import",
            {"export_type": "character", "schema_version": 1, "name": "Imported character", "profile_data": {}},
        )
        self.assertEqual(world.status_code, 201)
        self.assertEqual(character.status_code, 201)
        self.assertIsNone(character.json()["world_id"])
        with Session(self.engine) as db:
            self.assertEqual(db.get(World, world.json()["id"]).user_id, self.alice_id)
            self.assertEqual(db.get(Character, character.json()["id"]).user_id, self.alice_id)


if __name__ == "__main__":
    unittest.main()
