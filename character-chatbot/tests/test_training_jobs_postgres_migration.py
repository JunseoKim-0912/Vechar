"""Opt-in migration integration against a disposable localhost PostgreSQL database only."""

import os
import asyncio
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
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.models import Character, Conversation, MemoryIngestion, Message, MessageRole, TrainingJob, TrainingJobChunk, User
from app.database import Base
from app.schemas import CharacterProfileData
from app.services import memory_jobs, memory_service, training_jobs
from app.services.training_queue import InMemoryTrainingQueue


class LocalPostgresTrainingMigrationTests(unittest.TestCase):
    def test_baseline_upgrade_downgrade_retains_existing_user(self):
        url = os.getenv("LOCAL_POSTGRES_MIGRATION_TEST_URL")
        if not url:
            self.skipTest("Disposable localhost PostgreSQL URL not configured")
        parsed = make_url(url)
        if parsed.get_backend_name() != "postgresql" or parsed.host not in {"localhost", "127.0.0.1"} or parsed.database != "vechar_jobs_test":
            self.fail("Migration test accepts only localhost/vechar_jobs_test, never production")
        config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
        engine = create_engine(url)
        try:
            with engine.connect() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, "0001_initial_schema")
                with Session(bind=connection) as db:
                    user = User(email=f"migration-{uuid4()}@example.invalid", password_hash="unused")
                    db.add(user)
                    db.commit()
                    user_id = user.id
                command.upgrade(config, "head")
                self.assertEqual(connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one(), "0003_memory_ingestion")
                self.assertTrue({"training_jobs", "training_job_chunks", "memory_ingestions",
                                 "memory_deletions"} <= set(inspect(connection).get_table_names()))
                self.assertEqual(compare_metadata(MigrationContext.configure(connection), Base.metadata), [])
                self.assertEqual(connection.exec_driver_sql("SELECT count(*) FROM users WHERE id = %s", (user_id,)).scalar_one(), 1)
                # Two independent PostgreSQL connections race for one chunk.
                with Session(engine) as db:
                    target = Character(user_id=user_id, name="Concurrency fixture")
                    db.add(target)
                    db.flush()
                    job = TrainingJob(user_id=user_id, target_type="character", target_id=target.id,
                                      source_type="text", training_source_type="STORY", source_text="hello",
                                      source_hash="0" * 64, source_char_count=5, status="extracting",
                                      stage="extracting", total_chunks=1, source_tokens=5, direct_mode=True)
                    db.add(job)
                    db.flush()
                    chunk = TrainingJobChunk(job_id=job.id, chunk_index=1, status="queued",
                                             source_start=0, core_start=0, core_end=5,
                                             token_start=0, token_end=5, token_count=5, overlap_tokens=0)
                    db.add(chunk)
                    db.commit()
                    job_id, chunk_id, target_id = job.id, chunk.id, target.id
                entered, release = Event(), Event()
                calls = []

                def extract(*args, **kwargs):
                    calls.append(1)
                    entered.set()
                    self.assertTrue(release.wait(10))
                    return CharacterProfileData(personality_summary="once")

                queue = InMemoryTrainingQueue()
                with patch.object(training_jobs, "SessionLocal", lambda: Session(engine)), \
                     patch.object(training_jobs, "extract_profile_from_text", side_effect=extract):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        first = pool.submit(lambda: asyncio.run(training_jobs.process_chunk(job_id, chunk_id, queue)))
                        self.assertTrue(entered.wait(10))
                        second = pool.submit(lambda: asyncio.run(training_jobs.process_chunk(job_id, chunk_id, queue)))
                        self.assertEqual(second.result(timeout=10), "busy")
                        release.set()
                        self.assertEqual(first.result(timeout=10), "done")
                self.assertEqual(len(calls), 1)
                with Session(engine) as db:
                    conversation = Conversation(user_id=user_id, character_id=target_id)
                    db.add(conversation)
                    db.flush()
                    user_turn = Message(conversation_id=conversation.id, role=MessageRole.USER, content="hello")
                    assistant_turn = Message(conversation_id=conversation.id, role=MessageRole.CHARACTER, content="hi")
                    db.add_all([user_turn, assistant_turn])
                    db.flush()
                    ingestion = MemoryIngestion(
                        user_id=user_id, character_id=target_id, conversation_id=conversation.id,
                        user_message_id=user_turn.id, assistant_message_id=assistant_turn.id,
                        provider="memmachine",
                    )
                    db.add(ingestion)
                    db.commit()
                    ingestion_id = ingestion.id
                memory_entered, memory_release = Event(), Event()
                memory_calls = []

                def record_memory(**kwargs):
                    memory_calls.append(kwargs)
                    memory_entered.set()
                    self.assertTrue(memory_release.wait(10))

                with patch.object(memory_jobs, "SessionLocal", lambda: Session(engine)), \
                     patch.object(memory_service, "PROVIDER_NAME", "memmachine"), \
                     patch.object(memory_service, "_provider_record_completed_turn", side_effect=record_memory):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        first_memory = pool.submit(memory_jobs.process_ingestion, ingestion_id)
                        self.assertTrue(memory_entered.wait(10))
                        second_memory = pool.submit(memory_jobs.process_ingestion, ingestion_id)
                        self.assertEqual(second_memory.result(timeout=10), "busy")
                        memory_release.set()
                        self.assertEqual(first_memory.result(timeout=10), "done")
                self.assertEqual(len(memory_calls), 1)
                command.downgrade(config, "0001_initial_schema")
                self.assertEqual(connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one(), "0001_initial_schema")
                self.assertFalse({"training_jobs", "training_job_chunks", "memory_ingestions",
                                  "memory_deletions"} & set(inspect(connection).get_table_names()))
                self.assertEqual(connection.exec_driver_sql("SELECT count(*) FROM users WHERE id = %s", (user_id,)).scalar_one(), 1)
        finally:
            engine.dispose()
