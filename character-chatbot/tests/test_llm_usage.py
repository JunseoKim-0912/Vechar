import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import llm
from app.database import Base
from app.auth import get_current_user_id
from app.database import get_db
from app.models import Character, Conversation, LLMUsage, User
from app.tier_limits import TIER_LIMITS


class LLMUsageTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).parent / f".llm-usage-{uuid4().hex}.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as db:
            user = User(email="free@example.invalid", password_hash="unused")
            premium = User(email="premium@example.invalid", password_hash="unused", is_premium=True)
            db.add_all([user, premium])
            db.commit()
            self.free_id = user.id
            self.premium_id = premium.id

    def tearDown(self):
        self.engine.dispose()
        self.db_path.unlink(missing_ok=True)

    def _client(self, input_tokens=10, output_tokens=4, output_text="OK"):
        client = Mock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=input_tokens)
        client.responses.create.return_value = SimpleNamespace(
            model="gpt-5.5-2026-04-23",
            status="completed",
            output_text=output_text,
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        return client

    def _call(self, user_id, client, output_cap=20, use_outer_patches=False):
        with Session(self.engine) as db:
            if use_outer_patches:
                return llm.generate_text(
                    db, user_id, "chat", "짧은 지침", [{"role": "user", "content": "안녕"}], output_cap
                )
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.5", "OPENAI_API_KEY": ""}):
                with patch.object(llm, "_get_client", return_value=client):
                    return llm.generate_text(
                        db, user_id, "chat", "짧은 지침", [{"role": "user", "content": "안녕"}], output_cap
                    )

    def _prior_usage(self, user_id, budget_tokens):
        with Session(self.engine) as db:
            db.add(
                LLMUsage(
                    user_id=user_id,
                    request_type="chat",
                    model="gpt-5.5",
                    status="completed",
                    input_tokens=budget_tokens,
                    total_tokens=budget_tokens,
                    reserved_total_tokens=budget_tokens,
                    budget_tokens=budget_tokens,
                    created_at=datetime.now(timezone.utc),
                )
            )
            db.commit()

    def test_below_limit_records_actual_usage_and_cost(self):
        client = self._client()
        with patch.dict(TIER_LIMITS["free"], {"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}):
            self.assertEqual(self._call(self.free_id, client), "OK")
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.user_id, row.request_type, row.model),
                             (self.free_id, "chat", "gpt-5.5-2026-04-23"))
            self.assertEqual((row.input_tokens, row.output_tokens, row.total_tokens), (10, 4, 14))
            self.assertEqual((row.reserved_total_tokens, row.budget_tokens, row.status), (30, 14, "completed"))
            self.assertGreater(row.estimated_cost_usd, 0)

    def test_daily_limit_blocks_before_generation(self):
        self._prior_usage(self.free_id, 80)
        client = self._client()
        with patch.dict(TIER_LIMITS["free"], {"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}):
            with self.assertRaises(HTTPException) as caught:
                self._call(self.free_id, client)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.detail["code"], "daily_limit_reached")
        client.responses.create.assert_not_called()

    def test_monthly_limit_blocks_before_generation(self):
        self._prior_usage(self.free_id, 80)
        client = self._client()
        with patch.dict(TIER_LIMITS["free"], {"max_llm_tokens_per_day": 1000, "max_llm_tokens_per_month": 100}):
            with self.assertRaises(HTTPException) as caught:
                self._call(self.free_id, client)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.detail["code"], "monthly_limit_reached")
        client.responses.create.assert_not_called()

    def test_free_and_premium_limits_differ(self):
        with patch.dict(TIER_LIMITS["free"], {"max_llm_tokens_per_day": 20, "max_llm_tokens_per_month": 100}):
            with patch.dict(TIER_LIMITS["premium"],
                            {"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}):
                with self.assertRaises(HTTPException):
                    self._call(self.free_id, self._client())
                self.assertEqual(self._call(self.premium_id, self._client()), "OK")

    def test_unknown_provider_failure_keeps_reserved_budget_but_no_actual_usage(self):
        client = self._client()
        client.responses.create.side_effect = RuntimeError("mock transport failure")
        with self.assertRaisesRegex(RuntimeError, "mock transport failure"):
            self._call(self.free_id, client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.total_tokens, row.budget_tokens), ("failed", 0, 30))
            self.assertEqual(row.error_type, "RuntimeError")

    def test_token_count_failure_is_logged_without_generation(self):
        client = self._client()
        client.responses.input_tokens.count.side_effect = RuntimeError("mock count failure")
        with self.assertRaisesRegex(RuntimeError, "mock count failure"):
            self._call(self.free_id, client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.total_tokens, row.budget_tokens), ("preflight_failed", 0, 0))
        client.responses.create.assert_not_called()

    def test_global_output_cap_blocks_before_openai_request(self):
        from app.llm_config import MAX_LLM_OUTPUT_TOKENS

        client = self._client()
        with self.assertRaises(HTTPException) as caught:
            self._call(self.free_id, client, output_cap=MAX_LLM_OUTPUT_TOKENS + 1)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(caught.exception.detail["code"], "llm_output_too_large")
        client.responses.input_tokens.count.assert_not_called()

    def test_global_input_cap_blocks_before_openai_request(self):
        from app.llm_config import MAX_LLM_INPUT_BYTES

        client = self._client()
        with Session(self.engine) as db:
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.5", "OPENAI_API_KEY": ""}):
                with patch.object(llm, "_get_client", return_value=client):
                    with self.assertRaises(HTTPException) as caught:
                        llm.generate_text(db, self.free_id, "chat", "x" * (MAX_LLM_INPUT_BYTES + 1), [], 20)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(caught.exception.detail["code"], "llm_input_too_large")
        client.responses.input_tokens.count.assert_not_called()

    def test_unusable_response_still_records_provider_usage(self):
        client = self._client(output_text="")
        with self.assertRaisesRegex(RuntimeError, "no completed text output"):
            self._call(self.free_id, client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.total_tokens, row.budget_tokens), ("failed_response", 14, 14))

    def test_concurrent_requests_cannot_both_reserve_same_budget(self):
        client = self._client()
        barrier = Barrier(2)

        def count(**_):
            barrier.wait(timeout=5)
            return SimpleNamespace(input_tokens=10)

        client.responses.input_tokens.count.side_effect = count

        def run():
            try:
                return self._call(self.free_id, client, use_outer_patches=True)
            except HTTPException as exc:
                return exc.detail["code"]

        with patch.dict(TIER_LIMITS["free"], {"max_llm_tokens_per_day": 30, "max_llm_tokens_per_month": 100}):
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.5", "OPENAI_API_KEY": ""}):
                with patch.object(llm, "_get_client", return_value=client):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        results = list(pool.map(lambda _: run(), range(2)))
        self.assertCountEqual(results, ["OK", "daily_limit_reached"])
        self.assertEqual(client.responses.create.call_count, 1)

    def test_http_route_preserves_clear_429_error(self):
        from app import main

        with Session(self.engine) as db:
            character = Character(user_id=self.free_id, name="가상 인물")
            db.add(character)
            db.flush()
            conversation = Conversation(character_id=character.id, user_id=self.free_id)
            db.add(conversation)
            db.commit()
            conversation_id = conversation.id
        self._prior_usage(self.free_id, 80)
        client = self._client()

        def test_db():
            with Session(self.engine) as db:
                yield db

        main.app.dependency_overrides[get_db] = test_db
        main.app.dependency_overrides[get_current_user_id] = lambda: self.free_id
        try:
            with patch.object(main, "engine", self.engine):
                with patch.dict(TIER_LIMITS["free"],
                                {"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}):
                    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.5", "OPENAI_API_KEY": ""}):
                        with patch.object(llm, "_get_client", return_value=client):
                            with TestClient(main.app) as api:
                                response = api.post(
                                    f"/chat/conversations/{conversation_id}/messages",
                                    json={"content": "안녕"},
                                )
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.json()["detail"]["code"], "daily_limit_reached")
            client.responses.create.assert_not_called()
        finally:
            main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
