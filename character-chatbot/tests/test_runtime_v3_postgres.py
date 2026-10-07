"""Opt-in v3 JSON, rebuild, and concurrent-turn checks on disposable localhost PostgreSQL."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.models import Character, Conversation, Message, User
from app.services import character_conversations as rooms
from app.services.conversation_runtime import (
    ChatTurnResult, ConversationRuntimeState, TurnFidelity, TurnProgression,
)
from app.services.character_fidelity import KnowledgeScope


def _fake_counter(*_args, **_kwargs):
    return lambda instructions, messages: (len(instructions) + sum(len(m["content"]) for m in messages)) // 5 + 1


def test_runtime_v3_persists_rebuilds_and_fences_concurrent_next():
    url = os.getenv("LOCAL_POSTGRES_RUNTIME_V3_TEST_URL")
    if not url:
        pytest.skip("Disposable localhost v3 PostgreSQL URL not configured")
    parsed = make_url(url)
    if (parsed.get_backend_name() != "postgresql"
            or parsed.host not in {"localhost", "127.0.0.1"}
            or parsed.database != "vechar_runtime_v3_test"):
        pytest.fail("v3 integration accepts only localhost/vechar_runtime_v3_test")
    engine = create_engine(url)
    try:
        config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
        with engine.connect() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        with Session(engine) as db:
            owner = User(email="runtime-v3@example.invalid", password_hash="unused")
            db.add(owner)
            db.flush()
            db.add_all([Character(user_id=owner.id, name="First"),
                        Character(user_id=owner.id, name="Second")])
            db.commit()
            owner_id = owner.id
            characters = db.query(Character).filter_by(user_id=owner_id).order_by(Character.name).all()
            room = rooms.create_room(db, owner_id, characters[0].id, characters[1].id, "v3", "en")
            room_id = room["id"]
            old = rooms.create_room(db, owner_id, characters[0].id, characters[1].id, "v2", "en")
            db.query(Conversation).filter_by(id=old["id"]).update({"runtime_state": {
                "version": 2, "last_message_id": None, "open_threads": ["false_thread"],
            }})
            db.commit()
            recovered = ConversationRuntimeState.from_storage(
                db.get(Conversation, old["id"]).runtime_state, [],
            )
            assert recovered.version == 3 and not recovered.threads

        entered, release = Event(), Event()
        result = ChatTurnResult(
            response="First calls Second and waits for an answer.",
            progression=TurnProgression(
                topic="second_answer", new_development="first_calls_second",
                resolved_thread=None, opened_thread="does_second_answer",
                action_taken=None, advice_given=None, repeated_point=False,
                new_thread_id="does_second_answer", transition="branch",
            ),
            fidelity=TurnFidelity(knowledge_scope=KnowledgeScope.UNCERTAIN,
                                  persona_preserved=True, assistant_mode=False),
        )
        calls = []

        def generate(**_kwargs):
            calls.append(1)
            entered.set()
            assert release.wait(10)
            return result

        def advance():
            with Session(engine) as db:
                return rooms.next_turn(db, room_id, owner_id, 0)

        with patch.object(rooms, "make_chat_input_counter", side_effect=_fake_counter), \
             patch.object(rooms, "generate_text", side_effect=generate):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(advance)
                assert entered.wait(10)
                second = pool.submit(advance)
                with pytest.raises(Exception, match="409"):
                    second.result(timeout=10)
                release.set()
                assert first.result(timeout=10)["room"]["turn_index"] == 1
        assert len(calls) == 1
        engine.dispose()
        fresh = create_engine(url)
        try:
            with Session(fresh) as db:
                stored = db.get(Conversation, room_id)
                messages = db.query(Message).filter_by(conversation_id=room_id).all()
                assert stored.turn_index == 1 and len(messages) == 1
                state = ConversationRuntimeState.from_storage(stored.runtime_state, messages)
                assert state.version == 3
                assert state.active_thread_id == "does_second_answer"
                assert state.knowledge_provenance_summary["peer_claim"] == 0
        finally:
            fresh.dispose()
    finally:
        engine.dispose()
