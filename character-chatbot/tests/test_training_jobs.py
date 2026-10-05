"""Queue orchestration tests; every LLM call is a local mock."""

import asyncio
import unittest
from datetime import timedelta
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Character, CharacterProfile, TrainingJob, TrainingJobChunk, TrainingSource, User, World, WorldProfile, WorldSource
from app.schemas import CharacterProfileData, CharacterSynthesisResult, WorldProfileData
from app.services import training_jobs, training_pipeline
from app.services.training_job_state import MAX_CHUNK_ATTEMPTS
from app.services.training_queue import InMemoryTrainingQueue


class TrainingJobWorkerTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(self.engine)
        self.session_patch = patch.object(training_jobs, "SessionLocal", self.sessions)
        self.session_patch.start()
        self.counter_patch = patch.object(training_pipeline, "make_training_text_counter", return_value=len)
        self.counter_patch.start()
        self.queue = InMemoryTrainingQueue()
        with self.sessions() as db:
            user = User(email="worker@example.invalid", password_hash="unused")
            db.add(user)
            db.flush()
            character = Character(user_id=user.id, name="Worker character")
            world = World(user_id=user.id, name="Worker world")
            db.add_all([character, world])
            db.commit()
            self.user_id, self.character_id, self.world_id = user.id, character.id, world.id

    def tearDown(self):
        self.counter_patch.stop()
        self.session_patch.stop()
        self.engine.dispose()

    def run_async(self, awaitable):
        return asyncio.run(awaitable)

    def submit(self, kind="character", text="A story"):
        with self.sessions() as db:
            job = self.run_async(training_jobs.submit_job(
                db, self.queue, user_id=self.user_id, target_type=kind,
                target_id=self.character_id if kind == "character" else self.world_id,
                source_type="text", training_source_type="STORY" if kind == "character" else "DESCRIPTION",
                raw_text=text,
            ))
            return job.id

    def plan(self, job_id):
        self.assertEqual(self.run_async(training_jobs.process_plan(job_id, self.queue)), "done")
        with self.sessions() as db:
            chunks = db.query(TrainingJobChunk).filter_by(job_id=job_id).order_by(TrainingJobChunk.chunk_index).all()
            return [c.id for c in chunks]

    def test_direct_character_and_world_complete_and_clear_private_artifacts(self):
        for kind in ("character", "world"):
            with self.subTest(kind=kind):
                job_id = self.submit(kind)
                chunks = self.plan(job_id)
                self.assertEqual(len(chunks), 1)
                result = (CharacterProfileData(personality_summary="New persona") if kind == "character"
                          else WorldProfileData(world_summary="New world"))
                extractor = "extract_profile_from_text" if kind == "character" else "extract_world_profile_from_text"
                with patch.object(training_jobs, extractor, return_value=result) as mock:
                    self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunks[0], self.queue)), "done")
                    self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunks[0], self.queue)), "done")
                    mock.assert_called_once()
                self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
                self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
                with self.sessions() as db:
                    job = db.get(TrainingJob, job_id)
                    self.assertEqual((job.status, job.progress), ("completed", 100))
                    self.assertIsNone(job.source_text)
                    self.assertIsNone(db.get(TrainingJobChunk, chunks[0]).extraction_result)
                    source_model = TrainingSource if kind == "character" else WorldSource
                    profile_model = CharacterProfile if kind == "character" else WorldProfile
                    self.assertEqual(db.query(source_model).count(), 1)
                    self.assertEqual(db.query(source_model).one().raw_text, "A story")
                    self.assertEqual(db.query(profile_model).count(), 1)

    def test_claim_busy_and_stale_lease_reclaimed(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with self.sessions() as db:
            chunk = db.get(TrainingJobChunk, chunk_id)
            chunk.status = "processing"
            chunk.lease_token = "old"
            chunk.lease_expires_at = training_jobs._now() + timedelta(minutes=2)
            db.commit()
        with patch.object(training_jobs, "extract_profile_from_text", return_value=CharacterProfileData()) as mock:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "busy")
            mock.assert_not_called()
            with self.sessions() as db:
                db.get(TrainingJobChunk, chunk_id).lease_expires_at = training_jobs._now() - timedelta(seconds=1)
                db.commit()
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            mock.assert_called_once()

    def test_transient_failure_retries_then_terminal_limit(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=RuntimeError("temporary")) as mock:
            for attempt in range(MAX_CHUNK_ATTEMPTS):
                result = self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue))
                self.assertEqual(result, "retry" if attempt < MAX_CHUNK_ATTEMPTS - 1 else "done")
            self.assertEqual(mock.call_count, MAX_CHUNK_ATTEMPTS)
        with self.sessions() as db:
            job = db.get(TrainingJob, job_id)
            self.assertEqual(job.status, "failed")
            self.assertIsNone(job.source_text)
            self.assertEqual(db.get(TrainingJobChunk, chunk_id).status, "failed")
            self.assertEqual(db.query(CharacterProfile).count(), 0)

    def test_transient_chunk_failure_then_success_calls_provider_twice_only(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=[
            RuntimeError("temporary"), CharacterProfileData(personality_summary="recovered"),
        ]) as mock:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "retry")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            self.assertEqual(mock.call_count, 2)
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJobChunk, chunk_id).attempt_count, 2)

    def test_budget_exhaustion_is_terminal_without_second_call(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        budget = HTTPException(429, detail={"code": "daily_limit_reached"})
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=budget) as mock:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            mock.assert_called_once()
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).error_code, "daily_limit_reached")

    def test_delete_cancels_and_late_messages_do_not_recreate_target(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with self.sessions() as db:
            training_jobs.cancel_target_jobs(db, user_id=self.user_id, target_type="character", target_id=self.character_id)
            db.delete(db.get(Character, self.character_id))
            db.commit()
        self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
        self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).status, "cancelled")
            self.assertEqual(db.query(CharacterProfile).count(), 0)

    def test_large_source_reuses_semantic_chunker_and_waits_for_all_chunks(self):
        job_id = self.submit(text=("First paragraph.\n\n" * 2500))
        chunks = self.plan(job_id)
        self.assertGreater(len(chunks), 1)
        with self.sessions() as db:
            rows = db.query(TrainingJobChunk).filter_by(job_id=job_id).order_by(TrainingJobChunk.chunk_index).all()
            self.assertEqual(rows[0].core_start, 0)
            self.assertEqual(rows[-1].core_end, len(db.get(TrainingJob, job_id).source_text))
            self.assertEqual([row.chunk_index for row in rows], list(range(1, len(rows) + 1)))
        self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "busy")
        with patch.object(training_jobs, "extract_profile_from_text", return_value=CharacterProfileData(personality_summary="from chunks")):
            for chunk_id in reversed(chunks):
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
        with patch.object(training_pipeline, "generate_structured", return_value=CharacterSynthesisResult(
            personality_summary="synthesized", speech_style="steady",
        )) as synthesis:
            self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
            self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
            synthesis.assert_called_once()
        with self.sessions() as db:
            profile = db.query(CharacterProfile).one()
            self.assertEqual(profile.data["personality_summary"], "synthesized")
            self.assertEqual(db.get(TrainingJob, job_id).completed_chunks, len(chunks))

    def test_final_synthesis_failure_preserves_existing_profile(self):
        with self.sessions() as db:
            db.add(CharacterProfile(character_id=self.character_id,
                                    data=CharacterProfileData(personality_summary="old").model_dump(), version=1))
            db.commit()
        job_id = self.submit(text="Chapter one.\n\n" * 2500)
        chunks = self.plan(job_id)
        with patch.object(training_jobs, "extract_profile_from_text", return_value=CharacterProfileData(personality_summary="new")):
            for chunk_id in chunks:
                self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue))
        with patch.object(training_pipeline, "generate_structured", side_effect=RuntimeError("provider down")) as mock:
            self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "retry")
            mock.assert_called_once()
        with self.sessions() as db:
            self.assertEqual(db.query(CharacterProfile).one().data["personality_summary"], "old")
            self.assertEqual(db.query(TrainingSource).count(), 0)
            self.assertEqual(db.get(TrainingJob, job_id).status, "synthesizing")

    def test_publish_failure_replays_checkpoint_without_reextracting(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text", return_value=CharacterProfileData()) as extract, \
             patch.object(self.queue, "publish_finalize", side_effect=[RuntimeError("queue unavailable"), None]) as publish:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "retry")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            extract.assert_called_once()
            self.assertEqual(publish.call_count, 2)
