"""Opt-in real PostgreSQL migration, persistence, and turn-fencing tests."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.database import Base
from app.models import Character, Conversation, Message, User
from app.services import character_conversations as rooms


def fake_counter(*_args, **_kwargs):
    return lambda instructions, messages: (len(instructions) + sum(len(m["content"]) for m in messages)) // 5 + 1


class LocalPostgresCharacterConversationTests(unittest.TestCase):
    def test_upgrade_downgrade_restart_and_concurrent_next(self):
        url = os.getenv("LOCAL_POSTGRES_ROOM_TEST_URL")
        if not url:
            self.skipTest("Disposable localhost PostgreSQL URL not configured")
        parsed = make_url(url)
        if (parsed.get_backend_name() != "postgresql"
                or parsed.host not in {"localhost", "127.0.0.1"}
                or parsed.database != "vechar_rooms_test"):
            self.fail("Room integration test accepts only localhost/vechar_rooms_test")
        engine = create_engine(url)
        config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
        try:
            with engine.connect() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, "0006_adaptive_training_chunks")
                with Session(bind=connection) as db:
                    owner = User(email="room-pg@example.invalid", password_hash="unused")
                    db.add(owner)
                    db.flush()
                    first = Character(user_id=owner.id, name="First")
                    second = Character(user_id=owner.id, name="Second")
                    db.add_all([first, second])
                    db.flush()
                    legacy_id = str(uuid4())
                    db.execute(text(
                        "INSERT INTO conversations (id, user_id, character_id, created_at) "
                        "VALUES (:id, :user_id, :character_id, now())"
                    ), {"id": legacy_id, "user_id": owner.id, "character_id": first.id})
                    db.execute(text(
                        "INSERT INTO messages (id, conversation_id, role, content, created_at) "
                        "VALUES (:id, :conversation_id, 'USER', :content, now())"
                    ), {"id": str(uuid4()), "conversation_id": legacy_id, "content": "Prior history"})
                    db.commit()
                    owner_id, first_id, second_id = owner.id, first.id, second.id
                command.upgrade(config, "0007_character_conversations")
                self.assertEqual(compare_metadata(MigrationContext.configure(connection), Base.metadata), [])
                self.assertEqual(connection.execute(text(
                    "SELECT kind FROM conversations WHERE id=:id"
                ), {"id": legacy_id}).scalar_one(), "user_character")
                self.assertEqual(connection.execute(text(
                    "SELECT count(*) FROM messages WHERE conversation_id=:id"
                ), {"id": legacy_id}).scalar_one(), 1)
                command.downgrade(config, "0006_adaptive_training_chunks")
                self.assertEqual(connection.execute(text(
                    "SELECT count(*) FROM messages WHERE conversation_id=:id"
                ), {"id": legacy_id}).scalar_one(), 1)
                command.upgrade(config, "0007_character_conversations")
                self.assertEqual(compare_metadata(MigrationContext.configure(connection), Base.metadata), [])
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one(), "0007_character_conversations")
            with Session(engine) as db:
                room = rooms.create_room(db, owner_id, first_id, second_id, "Persistent Pair", "en")
                room_id = room["id"]
            entered, release = Event(), Event()
            calls = []

            def generate(**_kwargs):
                calls.append(1)
                entered.set()
                self.assertTrue(release.wait(10))
                return "A new thought about the road ahead."

            def advance():
                with Session(engine) as db:
                    return rooms.next_turn(db, room_id, owner_id, 0)

            with patch.object(rooms, "make_chat_input_counter", side_effect=fake_counter), \
                 patch.object(rooms, "generate_text", side_effect=generate):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(advance)
                    self.assertTrue(entered.wait(10))
                    second = pool.submit(advance)
                    with self.assertRaisesRegex(Exception, "409"):
                        second.result(timeout=10)
                    release.set()
                    self.assertEqual(first.result(timeout=10)["room"]["turn_index"], 1)
            self.assertEqual(len(calls), 1)
            # A new engine/session simulates a backend process restart.
            engine.dispose()
            fresh_engine = create_engine(url)
            try:
                with Session(fresh_engine) as db:
                    self.assertEqual(rooms.get_room(db, room_id, owner_id)["turn_index"], 1)
                    history = rooms.list_messages(db, room_id, owner_id)
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0].speaker_character_id, first_id)
                    legacy_history = db.query(Message).filter_by(conversation_id=legacy_id).all()
                    self.assertEqual([message.content for message in legacy_history], ["Prior history"])
            finally:
                fresh_engine.dispose()
        finally:
            engine.dispose()
