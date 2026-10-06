"""Opt-in migration integration against a disposable localhost PostgreSQL database only."""

import os
import asyncio
import hashlib
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
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.models import Character, Conversation, MemoryIngestion, Message, MessageRole, TrainingJob, TrainingJobChunk, User
from app.database import Base
from app.llm_failures import DuplicateLLMOperation, FailureKind, LLMResponseError
from app import llm_usage
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
                user_id = str(uuid4())
                premium_id = str(uuid4())
                for account_id, premium in ((user_id, False), (premium_id, True)):
                    connection.execute(text(
                        "INSERT INTO users (id, email, password_hash, is_premium) "
                        "VALUES (:id, :email, :password_hash, :premium)"
                    ), {"id": account_id, "email": f"migration-{account_id}@example.invalid",
                        "password_hash": "unused", "premium": premium})
                connection.commit()
                command.upgrade(config, "head")
                self.assertEqual(connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one(), "0007_character_conversations")
                self.assertTrue({"training_jobs", "training_job_chunks", "memory_ingestions",
                                 "memory_deletions"} <= set(inspect(connection).get_table_names()))
                self.assertEqual(compare_metadata(MigrationContext.configure(connection), Base.metadata), [])
                self.assertEqual(connection.exec_driver_sql("SELECT count(*) FROM users WHERE id = %s", (user_id,)).scalar_one(), 1)
                self.assertEqual(connection.execute(text("SELECT role FROM users WHERE id = :id"),
                                                    {"id": user_id}).scalar_one(), "user")
                self.assertEqual(connection.execute(text("SELECT is_premium, role FROM users WHERE id = :id"),
                                                    {"id": premium_id}).one(), (True, "user"))
                operation_key = f"training:{user_id}:chunk:1:extract:1:character_extraction"
                def reserve_once():
                    try:
                        return llm_usage.reserve_usage(engine, user_id, "character_extraction",
                                                       "gpt-6.1-sol", 10, 20,
                                                       operation_key=operation_key)
                    except DuplicateLLMOperation:
                        return "duplicate"

                with ThreadPoolExecutor(max_workers=2) as pool:
                    reservations = list(pool.map(lambda _: reserve_once(), range(2)))
                self.assertEqual(reservations.count("duplicate"), 1)
                self.assertEqual(connection.execute(text(
                    "SELECT count(*) FROM llm_usage WHERE operation_key = :key"
                ), {"key": operation_key}).scalar_one(), 1)
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
                # Persist an adaptive split under real PostgreSQL row locks. A
                # competing delivery must not create a second child pair.
                split_source = "A" * 11_790
                with Session(engine) as db:
                    split_target = Character(user_id=user_id, name="Split fixture")
                    db.add(split_target)
                    db.flush()
                    split_job = TrainingJob(
                        user_id=user_id, target_type="character", target_id=split_target.id,
                        source_type="text", training_source_type="STORY", source_text=split_source,
                        source_hash=hashlib.sha256(split_source.encode("utf-8")).hexdigest(),
                        source_char_count=len(split_source), source_tokens=len(split_source),
                        status="extracting", stage="extracting", total_chunks=1, direct_mode=True,
                    )
                    db.add(split_job)
                    db.flush()
                    split_chunk = TrainingJobChunk(
                        job_id=split_job.id, chunk_index=1, status="queued",
                        source_start=0, core_start=0, core_end=len(split_source),
                        token_start=0, token_end=len(split_source), token_count=len(split_source), overlap_tokens=0,
                    )
                    db.add(split_chunk)
                    db.commit()
                    split_job_id, split_chunk_id, split_target_id = split_job.id, split_chunk.id, split_target.id
                split_entered, split_release = Event(), Event()
                split_calls = []

                def overflow(*args, **kwargs):
                    split_calls.append(1)
                    split_entered.set()
                    self.assertTrue(split_release.wait(10))
                    raise LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")

                split_queue = InMemoryTrainingQueue()
                with patch.object(training_jobs, "SessionLocal", lambda: Session(engine)), \
                     patch.object(training_jobs, "make_training_text_counter", return_value=len), \
                     patch.object(training_jobs, "extract_profile_from_text", side_effect=overflow):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        first_split = pool.submit(lambda: asyncio.run(training_jobs.process_chunk(
                            split_job_id, split_chunk_id, split_queue)))
                        self.assertTrue(split_entered.wait(10))
                        competing = pool.submit(lambda: asyncio.run(training_jobs.process_chunk(
                            split_job_id, split_chunk_id, split_queue)))
                        self.assertEqual(competing.result(timeout=10), "busy")
                        split_release.set()
                        self.assertEqual(first_split.result(timeout=10), "done")
                    self.assertEqual(asyncio.run(training_jobs.process_chunk(
                        split_job_id, split_chunk_id, split_queue)), "done")
                self.assertEqual(len(split_calls), 1)
                with Session(engine) as db:
                    self.assertEqual(db.get(TrainingJob, split_job_id).total_chunks, 2)
                    self.assertEqual(db.get(TrainingJobChunk, split_chunk_id).status, "split")
                    children = db.query(TrainingJobChunk).filter_by(
                        job_id=split_job_id, parent_chunk_id=split_chunk_id,
                    ).order_by(TrainingJobChunk.child_order).all()
                    self.assertEqual(len(children), 2)
                    self.assertEqual([child.status for child in children], ["queued", "queued"])
                    db.query(TrainingJobChunk).filter_by(job_id=split_job_id).delete()
                    db.query(TrainingJob).filter_by(id=split_job_id).delete()
                    db.query(Character).filter_by(id=split_target_id).delete()
                    db.commit()
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
                self.assertEqual(connection.execute(text("SELECT is_premium FROM users WHERE id = :id"),
                                                    {"id": premium_id}).scalar_one(), True)
        finally:
            engine.dispose()
