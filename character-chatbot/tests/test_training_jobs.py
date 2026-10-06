"""Queue orchestration tests; every LLM call is a local mock."""

import asyncio
import hashlib
import os
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2
from fastapi import HTTPException
from openai import InternalServerError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app import llm
from app.llm_failures import FailureKind, LLMResponseError, LLMStructuredOutputError
from app.llm_operation import current_operation
from app.models import Character, CharacterProfile, LLMUsage, TrainingJob, TrainingJobChunk, TrainingSource, User, World, WorldProfile, WorldSource
from app.schemas import CharacterProfileData, CharacterSynthesisResult, TimelineEvent, WorldProfileData
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
        self.split_counter_patch = patch.object(training_jobs, "make_training_text_counter", return_value=len)
        self.split_counter_patch.start()
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
        self.split_counter_patch.stop()
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

    def children(self, job_id, parent_id):
        with self.sessions() as db:
            return [row.id for row in db.query(TrainingJobChunk).filter_by(
                job_id=job_id, parent_chunk_id=parent_id,
            ).order_by(TrainingJobChunk.child_order).all()]

    def test_output_cap_splits_only_failed_chunk_and_settles_parent_and_children(self):
        job_id = self.submit(text="A" * 25_667)
        first, second = self.plan(job_id)
        evidence = CharacterProfileData(personality_summary="salient").model_dump_json()
        synthesis = CharacterSynthesisResult(personality_summary="canonical", speech_style="quiet").model_dump_json()

        def response(status, output, output_tokens, response_id):
            return SimpleNamespace(
                id=response_id, status=status, model="gpt-6.1-sol", output_text=output,
                output=[], incomplete_details=(SimpleNamespace(reason="max_output_tokens")
                                               if status == "incomplete" else None),
                usage=SimpleNamespace(input_tokens=100, output_tokens=output_tokens,
                                      total_tokens=100 + output_tokens,
                                      input_tokens_details=SimpleNamespace(cached_tokens=0),
                                      output_tokens_details=SimpleNamespace(reasoning_tokens=output_tokens // 2)),
            )

        client = Mock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=100)
        client.responses.create.side_effect = [
            response("completed", evidence, 100, "resp-first"),
            response("incomplete", "", 8000, "resp-overflow"),
            response("completed", evidence, 90, "resp-child-one"),
            response("completed", evidence, 95, "resp-child-two"),
            response("completed", synthesis, 80, "resp-final"),
        ]
        with patch.dict(os.environ, {"OPENAI_ANALYSIS_MODEL": "gpt-6.1-sol"}), \
             patch.object(llm, "_get_client", return_value=client):
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, first, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, second, self.queue)), "done")
            children = self.children(job_id, second)
            self.assertEqual(len(children), 2)
            self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "busy")
            for child_id in children:
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, child_id, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, children[0], self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
        self.assertEqual(client.responses.create.call_count, 5)
        with self.sessions() as db:
            job = db.get(TrainingJob, job_id)
            self.assertEqual((job.status, job.total_chunks, job.completed_chunks), ("completed", 3, 3))
            self.assertEqual(db.get(TrainingJobChunk, first).attempt_count, 1)
            self.assertEqual(db.get(TrainingJobChunk, second).status, "split")
            rows = db.query(LLMUsage).filter_by(user_id=self.user_id).all()
            self.assertEqual(len(rows), 5)
            parent_usage = next(row for row in rows if row.provider_response_id == "resp-overflow")
            self.assertEqual((parent_usage.status, parent_usage.output_tokens,
                              str(parent_usage.estimated_cost_usd)),
                             ("failed_response", 8000, "0.08020000"))
            child_keys = [row.operation_key for row in rows if row.provider_response_id in
                          {"resp-child-one", "resp-child-two"}]
            self.assertEqual(len(set(child_keys)), 2)
            self.assertTrue(all(key and "chunk:" in key for key in child_keys))
            self.assertTrue(all(row.reconciled_at is not None for row in rows))
            self.assertEqual(db.query(CharacterProfile).one().data["personality_summary"], "canonical")

    def test_output_cap_fallback_is_bounded_and_replays_parent_without_duplicate_children(self):
        job_id = self.submit(text="B" * 12_000)
        root = self.plan(job_id)[0]
        error = LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=error) as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            children = self.children(job_id, root)
            self.assertEqual(len(children), 2)
            with self.sessions() as db:
                self.assertEqual(db.get(TrainingJobChunk, root).status, "split")
                self.assertEqual([db.get(TrainingJobChunk, child).status for child in children],
                                 ["queued", "queued"])
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            self.assertEqual(self.children(job_id, root), children)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, children[0], self.queue)), "done")
            grandchildren = self.children(job_id, children[0])
            self.assertEqual(len(grandchildren), 2)
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, grandchildren[0], self.queue)), "done")
            self.assertEqual(extract.call_count, 3)
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).error_code, "llm_output_limit")
            self.assertEqual(db.get(TrainingJobChunk, grandchildren[0]).split_depth, 2)
            self.assertEqual(db.query(TrainingJobChunk).filter_by(job_id=job_id).count(), 5)

    def test_adaptive_children_preserve_overlap_dedup_and_flashback_chronology(self):
        job_id = self.submit(text="C" * 12_000)
        root = self.plan(job_id)[0]
        error = LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=error):
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
        children = self.children(job_id, root)
        duplicate = lambda key: TimelineEvent(event_key=key, age=20, summary="Shared boundary event")
        evidence = [
            CharacterProfileData(timeline=[
                duplicate("left"), TimelineEvent(event_key="death", age=30, summary="Death", is_death=True),
            ]),
            CharacterProfileData(timeline=[
                duplicate("right"), TimelineEvent(event_key="youth", age=10, summary="Early childhood",
                                                  narrative_role="flashback"),
            ]),
        ]
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=evidence):
            for child in children:
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, child, self.queue)), "done")
        with patch.object(training_pipeline, "generate_structured", return_value=CharacterSynthesisResult(
            personality_summary="canonical", speech_style="quiet",
        )):
            self.assertEqual(self.run_async(training_jobs.process_finalize(job_id)), "done")
        with self.sessions() as db:
            profile = CharacterProfileData.model_validate(db.query(CharacterProfile).one().data)
            self.assertEqual([event.age for event in profile.timeline], [10, 20, 30])
            self.assertEqual(profile.chat_reference_point.age, 30)
            self.assertEqual(profile.chat_reference_point.phase, "immediately_before_death")

    def test_cancellation_after_split_makes_delayed_child_deliveries_no_ops(self):
        job_id = self.submit(text="D" * 12_000)
        root = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text",
                          side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")):
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
        children = self.children(job_id, root)
        with self.sessions() as db:
            training_jobs.cancel_target_jobs(db, user_id=self.user_id, target_type="character",
                                             target_id=self.character_id)
            db.delete(db.get(Character, self.character_id))
            db.commit()
        with patch.object(training_jobs, "extract_profile_from_text") as extract:
            for child in children:
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, child, self.queue)), "done")
            extract.assert_not_called()

    def test_french_and_english_sources_use_same_output_cap_fallback(self):
        for text in ("Il se souvient de son enfance. " * 400,
                     "He remembers his childhood. " * 450):
            with self.subTest(language=text[:2]):
                job_id = self.submit(text=text)
                root = self.plan(job_id)[0]
                with patch.object(training_jobs, "extract_profile_from_text",
                                  side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")):
                    self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
                self.assertEqual(len(self.children(job_id, root)), 2)

    def test_long_chunk_schema_failure_does_not_split(self):
        job_id = self.submit(text="E" * 12_000)
        root = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text",
                          side_effect=LLMStructuredOutputError("invalid")) as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            extract.assert_called_once()
        self.assertEqual(self.children(job_id, root), [])
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).error_code, "llm_structured_output_invalid")

    def test_split_publish_failure_replays_saved_children_without_reextracting_parent(self):
        job_id = self.submit(text="F" * 12_000)
        root = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text",
                          side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")) as extract, \
             patch.object(self.queue, "publish_chunk", side_effect=[RuntimeError("queue down"), None, None]):
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "retry")
            children = self.children(job_id, root)
            self.assertEqual(len(children), 2)
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            self.assertEqual(self.children(job_id, root), children)
            extract.assert_called_once()
        before_replay = len(self.queue.messages)
        self.assertEqual(self.run_async(training_jobs.process_plan(job_id, self.queue)), "done")
        replayed = [payload["chunk_id"] for _, payload in self.queue.messages[before_replay:]
                    if "chunk_id" in payload]
        self.assertTrue(set(children) <= set(replayed))

    def test_split_preparation_error_retries_without_reextracting_parent(self):
        job_id = self.submit(text="A" * 11_790)
        root = self.plan(job_id)[0]
        error = LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=error) as extract, \
             patch.object(training_jobs, "make_training_text_counter",
                          side_effect=[RuntimeError("counter unavailable"), len]):
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "retry")
            with self.sessions() as db:
                self.assertEqual(db.get(TrainingJob, job_id).status, "extracting")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            self.assertEqual(len(self.children(job_id, root)), 2)
            extract.assert_called_once()

    def test_child_insert_failure_rolls_back_parent_split_and_can_recover(self):
        job_id = self.submit(text="A" * 11_790)
        root = self.plan(job_id)[0]

        def reject_children(session, flush_context, instances):
            if any(isinstance(row, TrainingJobChunk) and row.parent_chunk_id == root
                   for row in session.new):
                raise RuntimeError("child insert unavailable")

        event.listen(Session, "before_flush", reject_children)
        try:
            with patch.object(training_jobs, "extract_profile_from_text",
                              side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")) as extract:
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "retry")
                extract.assert_called_once()
        finally:
            event.remove(Session, "before_flush", reject_children)
        with self.sessions() as db:
            parent = db.get(TrainingJobChunk, root)
            self.assertEqual((parent.status, parent.error_code), ("queued", "adaptive_split_pending"))
            self.assertEqual(db.query(TrainingJobChunk).filter_by(parent_chunk_id=root).count(), 0)
        with patch.object(training_jobs, "extract_profile_from_text") as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            extract.assert_not_called()
        self.assertEqual(len(self.children(job_id, root)), 2)

    def test_repeated_split_preparation_failure_has_distinct_terminal_code(self):
        job_id = self.submit(text="A" * 11_790)
        root = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text",
                          side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")) as extract, \
             patch.object(training_jobs, "make_training_text_counter",
                          side_effect=RuntimeError("counter unavailable")):
            outcomes = [self.run_async(training_jobs.process_chunk(job_id, root, self.queue))
                        for _ in range(MAX_CHUNK_ATTEMPTS)]
            self.assertEqual(outcomes, ["retry", "retry", "done"])
            extract.assert_called_once()
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).error_code, "adaptive_split_failed")
            self.assertEqual(db.get(TrainingJobChunk, root).status, "failed")

    def test_production_sized_roots_split_into_durable_ordered_children(self):
        error = LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")
        for core_tokens in (11_790, 13_877):
            with self.subTest(core_tokens=core_tokens):
                with self.sessions() as db:
                    target = Character(user_id=self.user_id, name=f"Root {core_tokens}")
                    db.add(target)
                    db.commit()
                    self.character_id = target.id
                job_id = self.submit(text=("Chapter one.\n\n" + "A" * (core_tokens - 14)))
                root = self.plan(job_id)[0]
                with patch.object(training_jobs, "extract_profile_from_text", side_effect=error):
                    self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
                with self.sessions() as db:
                    job = db.get(TrainingJob, job_id)
                    parent = db.get(TrainingJobChunk, root)
                    children = db.query(TrainingJobChunk).filter_by(
                        job_id=job_id, parent_chunk_id=root,
                    ).order_by(TrainingJobChunk.child_order).all()
                    self.assertEqual((job.status, job.total_chunks), ("extracting", 2))
                    self.assertEqual(parent.status, "split")
                    self.assertEqual(len(children), 2)
                    self.assertEqual([child.status for child in children], ["queued", "queued"])
                    self.assertEqual([child.split_depth for child in children], [1, 1])
                    self.assertEqual([child.child_order for child in children], [1, 2])
                    self.assertEqual(children[0].core_start, parent.core_start)
                    self.assertEqual(children[0].core_end, children[1].core_start)
                    self.assertEqual(children[1].core_end, parent.core_end)
                    self.assertTrue(all(child.token_end - child.token_start >= 3_000 for child in children))
                    self.assertLessEqual(children[1].overlap_tokens, 128)
                    child_ids = [child.id for child in children]
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
                self.assertEqual(self.children(job_id, root), child_ids)

    def test_source_normalization_and_offset_mismatch_are_detected_before_split(self):
        job_id = self.submit(text="\ufeff" + "A" * 11_790 + "\r\n")
        root = self.plan(job_id)[0]
        with self.sessions() as db:
            job = db.get(TrainingJob, job_id)
            self.assertNotIn("\r", job.source_text)
            chunk = db.get(TrainingJobChunk, root)
            chunk.core_end = len(job.source_text) + 1
            db.commit()
        with patch.object(training_jobs, "extract_profile_from_text",
                          side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")) as extract:
            with self.assertLogs(training_jobs.logger, level="WARNING") as captured:
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "retry")
            self.assertTrue(any("split_stage=source_reconstruction" in row for row in captured.output))
            self.assertEqual(self.children(job_id, root), [])
            with self.sessions() as db:
                chunk = db.get(TrainingJobChunk, root)
                self.assertEqual(chunk.error_code, "adaptive_split_pending")
                chunk.core_end = len(db.get(TrainingJob, job_id).source_text)
                db.commit()
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            extract.assert_called_once()
        self.assertEqual(len(self.children(job_id, root)), 2)

    def test_cleared_source_during_output_limit_is_reported_not_misclassified_exhausted(self):
        job_id = self.submit(text="A" * 11_790)
        root = self.plan(job_id)[0]

        def lose_source(*args, **kwargs):
            with self.sessions() as db:
                db.get(TrainingJob, job_id).source_text = None
                db.commit()
            raise LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")

        with patch.object(training_jobs, "extract_profile_from_text", side_effect=lose_source) as extract:
            with self.assertLogs(training_jobs.logger, level="WARNING") as captured:
                self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "retry")
            self.assertTrue(any("split_stage=source_reconstruction" in row for row in captured.output))
            extract.assert_called_once()
        with self.sessions() as db:
            job = db.get(TrainingJob, job_id)
            self.assertEqual(job.status, "extracting")
            self.assertEqual(db.get(TrainingJobChunk, root).error_code, "adaptive_split_pending")
            job.source_text = "A" * 11_790
            db.commit()
        with patch.object(training_jobs, "extract_profile_from_text") as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
            extract.assert_not_called()
        self.assertEqual(len(self.children(job_id, root)), 2)

    def test_overlapped_production_second_root_reconstructs_and_splits(self):
        source = "A" * 11_790 + "B" * 13_877
        with self.sessions() as db:
            job = TrainingJob(
                user_id=self.user_id, target_type="character", target_id=self.character_id,
                source_type="text", training_source_type="STORY", source_text=source,
                source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                source_char_count=len(source), source_tokens=len(source),
                status="extracting", stage="extracting", total_chunks=2, completed_chunks=1,
                direct_mode=False,
            )
            db.add(job)
            db.flush()
            first = TrainingJobChunk(
                job_id=job.id, chunk_index=1, status="completed",
                source_start=0, core_start=0, core_end=11_790,
                token_start=0, token_end=11_790, token_count=11_790, overlap_tokens=0,
                extraction_result=CharacterProfileData().model_dump(),
            )
            second = TrainingJobChunk(
                job_id=job.id, chunk_index=2, status="queued",
                source_start=11_790 - 698, core_start=11_790, core_end=len(source),
                token_start=11_790, token_end=len(source), token_count=13_877 + 698,
                overlap_tokens=698,
            )
            db.add_all([first, second])
            db.commit()
            job_id, root = job.id, second.id
        with patch.object(training_jobs, "extract_profile_from_text",
                          side_effect=LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete")):
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, root, self.queue)), "done")
        with self.sessions() as db:
            parent = db.get(TrainingJobChunk, root)
            children = db.query(TrainingJobChunk).filter_by(
                job_id=job_id, parent_chunk_id=root,
            ).order_by(TrainingJobChunk.child_order).all()
            self.assertEqual(parent.status, "split")
            self.assertEqual(len(children), 2)
            self.assertEqual(children[0].source_start, 11_790 - 698)
            self.assertEqual(children[0].overlap_tokens, 698)
            self.assertEqual(children[0].core_start, 11_790)
            self.assertEqual(children[0].core_end, children[1].core_start)
            self.assertEqual(children[1].core_end, len(source))
            self.assertTrue(all(child.core_end - child.core_start >= 3_000 for child in children))

    def test_terminal_failure_fences_inflight_sibling_and_delayed_delivery(self):
        job_id = self.submit(text="A" * 25_667)
        first, second = self.plan(job_id)
        with self.sessions() as db:
            root = db.get(TrainingJobChunk, first)
            root.status = "processing"
            root.lease_token = "root-a"
            root.attempt_count = 1
            db.commit()

        def finish_sibling(*args, **kwargs):
            self.assertEqual(training_jobs._handle_chunk_failure(
                job_id, first, "root-a", LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete"),
                chunk_code="adaptive_split_exhausted",
            ), "done")
            return CharacterProfileData(personality_summary="late result")

        with patch.object(training_jobs, "extract_profile_from_text", side_effect=finish_sibling) as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, second, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, second, self.queue)), "done")
            extract.assert_called_once()
        with self.sessions() as db:
            job = db.get(TrainingJob, job_id)
            self.assertEqual((job.status, job.error_code), ("failed", "llm_output_limit"))
            self.assertEqual(db.get(TrainingJobChunk, first).error_code, "adaptive_split_exhausted")
            sibling = db.get(TrainingJobChunk, second)
            self.assertEqual((sibling.status, sibling.error_code), ("failed", "job_terminal"))
            self.assertIsNone(sibling.lease_token)
            self.assertEqual(db.query(TrainingJobChunk).filter_by(job_id=job_id, status="processing").count(), 0)

    def test_delayed_delivery_heals_preexisting_terminal_processing_chunk(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with self.sessions() as db:
            job = db.get(TrainingJob, job_id)
            job.status = "failed"
            job.stage = "failed"
            job.source_text = None
            chunk = db.get(TrainingJobChunk, chunk_id)
            chunk.status = "processing"
            chunk.lease_token = "obsolete"
            db.commit()
        with patch.object(training_jobs, "extract_profile_from_text") as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            extract.assert_not_called()
        with self.sessions() as db:
            chunk = db.get(TrainingJobChunk, chunk_id)
            self.assertEqual((chunk.status, chunk.error_code), ("failed", "job_terminal"))
            self.assertIsNone(chunk.lease_token)

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

    def test_exhausted_stale_claim_is_terminal_without_new_provider_call(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with self.sessions() as db:
            chunk = db.get(TrainingJobChunk, chunk_id)
            chunk.status = "processing"
            chunk.attempt_count = MAX_CHUNK_ATTEMPTS
            chunk.lease_token = "expired"
            chunk.lease_expires_at = training_jobs._now() - timedelta(seconds=1)
            db.commit()
        with patch.object(training_jobs, "extract_profile_from_text") as extract:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            extract.assert_not_called()
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).status, "failed")
            self.assertEqual(db.get(TrainingJobChunk, chunk_id).status, "failed")

    def test_transient_failure_retries_then_terminal_limit(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=TimeoutError("temporary")) as mock:
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
            TimeoutError("temporary"), CharacterProfileData(personality_summary="recovered"),
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

    def test_next_call_exceeds_remaining_budget_is_terminal(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        budget = HTTPException(429, detail={"code": "request_exceeds_remaining_daily_budget"})
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=budget) as mock:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            mock.assert_called_once()
        with self.sessions() as db:
            self.assertEqual(db.get(TrainingJob, job_id).error_code,
                             "request_exceeds_remaining_daily_budget")

    def test_output_cap_and_schema_failures_never_blind_retry_in_any_source_language(self):
        for source, error, expected in (
            ("Bonjour. Il se souvient de son enfance.",
             LLMResponseError(FailureKind.OUTPUT_LIMIT, "incomplete"), "llm_output_limit"),
            ("He recalls the old timeline.",
             LLMStructuredOutputError("invalid schema"), "llm_structured_output_invalid"),
        ):
            with self.subTest(source=source[:8]):
                job_id = self.submit(text=source)
                chunk_id = self.plan(job_id)[0]
                with patch.object(training_jobs, "extract_profile_from_text", side_effect=error) as mock:
                    self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
                    self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
                    mock.assert_called_once()
                with self.sessions() as db:
                    self.assertEqual(db.get(TrainingJob, job_id).error_code, expected)
                    self.assertEqual(db.get(TrainingJobChunk, chunk_id).attempt_count, 1)

    def test_provider_503_retries_with_existing_attempt_cap(self):
        response = httpx2.Response(503, request=httpx2.Request("POST", "https://example.invalid"))
        error = InternalServerError("temporary", response=response, body=None)
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        with patch.object(training_jobs, "extract_profile_from_text", side_effect=error) as mock:
            for attempt in range(MAX_CHUNK_ATTEMPTS):
                result = self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue))
                self.assertEqual(result, "retry" if attempt < MAX_CHUNK_ATTEMPTS - 1 else "done")
            self.assertEqual(mock.call_count, MAX_CHUNK_ATTEMPTS)

    def test_duplicate_delivery_keeps_one_logical_attempt(self):
        job_id = self.submit()
        chunk_id = self.plan(job_id)[0]
        contexts = []

        def extract(*args, **kwargs):
            contexts.append(current_operation())
            return CharacterProfileData(personality_summary="once")

        with patch.object(training_jobs, "extract_profile_from_text", side_effect=extract) as mock:
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            self.assertEqual(self.run_async(training_jobs.process_chunk(job_id, chunk_id, self.queue)), "done")
            mock.assert_called_once()
        self.assertEqual(len(contexts), 1)
        self.assertEqual((contexts[0].job_id, contexts[0].chunk_index, contexts[0].attempt),
                         (job_id, 1, 1))

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
            self.assertEqual(db.query(TrainingJobChunk).filter(
                TrainingJobChunk.job_id == job_id,
                TrainingJobChunk.parent_chunk_id.is_not(None),
            ).count(), 0)

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
        with patch.object(training_pipeline, "generate_structured", side_effect=TimeoutError("provider down")) as mock:
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
