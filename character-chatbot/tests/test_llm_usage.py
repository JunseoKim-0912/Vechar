import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
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
from app.llm_config import model_for_task
from app.llm_usage import _estimated_cost
from app.database import Base
from app.auth import get_current_user_id
from app.database import get_db
from app.models import Character, Conversation, LLMUsage, User
from app.schemas import CharacterProfileData, WorldProfileData
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

    def _client(self, input_tokens=10, output_tokens=4, output_text="OK", model="gpt-6-luna"):
        client = Mock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=input_tokens)
        client.responses.create.return_value = SimpleNamespace(
            model=model,
            status="completed",
            output_text=output_text,
            output=[],
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )
        return client

    def _structured_call(self, client, request_type="character_extraction", output_cap=20,
                         response_model=CharacterProfileData):
        with Session(self.engine) as db:
            with patch.dict(os.environ, {"OPENAI_ANALYSIS_MODEL": "gpt-6.1-sol", "OPENAI_API_KEY": ""}):
                with patch.object(llm, "_get_client", return_value=client):
                    return llm.generate_structured(
                        db, self.free_id, request_type, "분석 지침",
                        [{"role": "user", "content": "분석 자료"}], output_cap,
                        task="analysis", response_model=response_model,
                    )

    def _call(self, user_id, client, output_cap=20, use_outer_patches=False, task="chat", request_type="chat"):
        with Session(self.engine) as db:
            if use_outer_patches:
                return llm.generate_text(
                    db, user_id, request_type, "짧은 지침", [{"role": "user", "content": "안녕"}], output_cap, task=task
                )
            with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna", "OPENAI_ANALYSIS_MODEL": "gpt-6.1-sol", "OPENAI_API_KEY": ""}):
                with patch.object(llm, "_get_client", return_value=client):
                    return llm.generate_text(
                        db, user_id, request_type, "짧은 지침", [{"role": "user", "content": "안녕"}], output_cap, task=task
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
                             (self.free_id, "chat", "gpt-6-luna"))
            self.assertEqual((row.input_tokens, row.output_tokens, row.total_tokens), (10, 4, 14))
            self.assertEqual((row.reserved_total_tokens, row.budget_tokens, row.status), (30, 14, "completed"))
            self.assertGreater(row.estimated_cost_usd, 0)

    def test_analysis_routes_through_same_gateway_and_records_model_cost(self):
        client = self._client(model="gpt-6.1-sol")
        with patch.dict(TIER_LIMITS["free"], {"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}):
            self.assertEqual(
                self._call(self.free_id, client, task="analysis", request_type="character_extraction"), "OK"
            )
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.request_type, row.model, row.total_tokens),
                             ("character_extraction", "gpt-6.1-sol", 14))
            self.assertEqual(row.estimated_cost_usd, Decimal("0.00006000"))
        self.assertEqual(client.responses.input_tokens.count.call_args.kwargs["model"], "gpt-6.1-sol")
        self.assertEqual(client.responses.create.call_args.kwargs["model"], "gpt-6.1-sol")

    def test_structured_response_records_actual_usage_and_schema_tokens(self):
        output = CharacterProfileData(personality_summary="차분함").model_dump_json()
        client = self._client(output_text=output, model="gpt-6.1-sol")
        client.responses.create.return_value.usage.input_tokens_details.cached_tokens = 2

        result = self._structured_call(client)

        self.assertIsInstance(result, CharacterProfileData)
        self.assertEqual(result.personality_summary, "차분함")
        counted = client.responses.input_tokens.count.call_args.kwargs
        created = client.responses.create.call_args.kwargs
        self.assertEqual(counted["model"], "gpt-6.1-sol")
        self.assertEqual(counted["text"], created["text"])
        self.assertEqual(created["text"]["format"]["type"], "json_schema")
        self.assertTrue(created["text"]["format"]["strict"])
        self.assertEqual(set(created["text"]["format"]["schema"]["required"]),
                         set(CharacterProfileData.model_fields))
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.request_type, row.model, row.input_tokens, row.cached_input_tokens,
                              row.output_tokens, row.total_tokens, row.status),
                             ("character_extraction", "gpt-6.1-sol", 10, 2, 4, 14, "completed"))
            self.assertEqual(row.estimated_cost_usd, Decimal("0.00005620"))

    def test_structured_schema_mismatch_records_usage_as_failed_response(self):
        client = self._client(output_text='{"personality_summary": 123}', model="gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "structured schema validation"):
            self._structured_call(client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.error_type, row.total_tokens, row.budget_tokens),
                             ("failed_response", "ValueError", 14, 14))
            self.assertEqual(row.estimated_cost_usd, Decimal("0.00006000"))

    def test_structured_missing_defaulted_field_is_not_silently_accepted(self):
        client = self._client(output_text='{"personality_summary": "차분함"}', model="gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            self._structured_call(client)
        with Session(self.engine) as db:
            self.assertEqual(db.query(LLMUsage).one().status, "failed_response")

    def test_structured_extra_field_is_not_silently_ignored(self):
        output = CharacterProfileData(personality_summary="차분함").model_dump()
        output["unexpected"] = "not part of the schema"
        client = self._client(output_text=json.dumps(output), model="gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "structured schema validation"):
            self._structured_call(client)
        with Session(self.engine) as db:
            self.assertEqual(db.query(LLMUsage).one().status, "failed_response")

    def test_structured_nested_missing_field_is_rejected(self):
        output = json.dumps({
            "world_summary": "세계", "key_facts": [], "timeline_notes": [],
            "mentioned_characters": [{"name": "인물"}],
        })
        client = self._client(output_text=output, model="gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "missing required fields: aliases"):
            self._structured_call(client, request_type="world_extraction", response_model=WorldProfileData)
        with Session(self.engine) as db:
            self.assertEqual(db.query(LLMUsage).one().status, "failed_response")

    def test_structured_empty_response_is_rejected_and_metered(self):
        client = self._client(output_text="", model="gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "no structured output"):
            self._structured_call(client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.total_tokens), ("failed_response", 14))

    def test_structured_refusal_is_rejected_and_metered(self):
        client = self._client(output_text="", model="gpt-6.1-sol")
        client.responses.create.return_value.output = [SimpleNamespace(
            type="message", content=[SimpleNamespace(type="refusal", refusal="refused")]
        )]
        with self.assertRaises(llm.LLMRefusalError):
            self._structured_call(client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.error_type, row.total_tokens),
                             ("failed_response", "LLMRefusalError", 14))

    def test_structured_api_error_keeps_reservation(self):
        client = self._client(model="gpt-6.1-sol")
        client.responses.create.side_effect = RuntimeError("mock transport failure")
        with self.assertRaisesRegex(RuntimeError, "mock transport failure"):
            self._structured_call(client)
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual((row.status, row.total_tokens, row.budget_tokens), ("failed", 0, 30))

    def test_structured_daily_and_monthly_limits_use_existing_reservation_gate(self):
        self._prior_usage(self.free_id, 80)
        for limits, expected in [
            ({"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}, "daily_limit_reached"),
            ({"max_llm_tokens_per_day": 1000, "max_llm_tokens_per_month": 100}, "monthly_limit_reached"),
        ]:
            with self.subTest(expected=expected):
                client = self._client(model="gpt-6.1-sol")
                with patch.dict(TIER_LIMITS["free"], limits):
                    with self.assertRaises(HTTPException) as caught:
                        self._structured_call(client)
                self.assertEqual(caught.exception.detail["code"], expected)
                client.responses.create.assert_not_called()

    def test_structured_sdk_schemas_keep_required_nested_fields(self):
        from openai.lib._parsing._responses import type_to_text_format_param

        for model in (CharacterProfileData, WorldProfileData):
            with self.subTest(model=model.__name__):
                schema = type_to_text_format_param(model)["schema"]
                self.assertEqual(set(schema["required"]), set(model.model_fields))
                self.assertIs(schema["additionalProperties"], False)
        nested = type_to_text_format_param(WorldProfileData)["schema"]["$defs"]["MentionedCharacter"]
        self.assertEqual(set(nested["required"]), {"name", "aliases"})
        self.assertIs(nested["additionalProperties"], False)

    def test_cost_uses_reported_model_and_distinct_prices(self):
        self.assertEqual(_estimated_cost("gpt-6.1-sol", 10, 0, 4), Decimal("0.000060"))
        self.assertEqual(_estimated_cost("gpt-6-luna", 10, 0, 4), Decimal("0.0000030"))
        self.assertEqual(_estimated_cost("gpt-6.1-sol", 10, 2, 4), Decimal("0.0000562"))
        with self.assertLogs(level="WARNING") as logged:
            self.assertIsNone(_estimated_cost("unpriced-model", 10, 0, 4))
        self.assertIn("unpriced-model", logged.output[0])

    def test_usage_cost_uses_response_model_even_if_requested_model_differs(self):
        client = self._client(model="gpt-6.1-sol")
        self._call(self.free_id, client, task="chat", request_type="chat")
        self.assertEqual(client.responses.create.call_args.kwargs["model"], "gpt-6-luna")
        with Session(self.engine) as db:
            row = db.query(LLMUsage).one()
            self.assertEqual(row.model, "gpt-6.1-sol")
            self.assertEqual(row.estimated_cost_usd, Decimal("0.00006000"))

    def test_analysis_daily_and_monthly_limits_block_before_generation(self):
        self._prior_usage(self.free_id, 80)
        for limits, expected in [
            ({"max_llm_tokens_per_day": 100, "max_llm_tokens_per_month": 1000}, "daily_limit_reached"),
            ({"max_llm_tokens_per_day": 1000, "max_llm_tokens_per_month": 100}, "monthly_limit_reached"),
        ]:
            with self.subTest(expected=expected):
                client = self._client(model="gpt-6.1-sol")
                with patch.dict(TIER_LIMITS["free"], limits):
                    with self.assertRaises(HTTPException) as caught:
                        self._call(self.free_id, client, task="analysis", request_type="world_extraction")
                self.assertEqual(caught.exception.detail["code"], expected)
                client.responses.create.assert_not_called()

    def test_model_configuration_is_explicit_and_legacy_value_is_not_fallback(self):
        with patch.dict(os.environ, {"OPENAI_ANALYSIS_MODEL": "gpt-6.1-sol", "OPENAI_CHAT_MODEL": "gpt-6-luna"}):
            self.assertEqual(model_for_task("analysis"), "gpt-6.1-sol")
            self.assertEqual(model_for_task("chat"), "gpt-6-luna")
        with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "", "OPENAI_MODEL": "gpt-5.5"}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_CHAT_MODEL"):
                model_for_task("chat")
            with patch.object(llm, "_get_client") as get_client:
                with Session(self.engine) as db:
                    with self.assertRaisesRegex(RuntimeError, "OPENAI_CHAT_MODEL"):
                        llm.generate_text(db, self.free_id, "chat", "지침", [], 20, task="chat")
            get_client.assert_not_called()
        with patch.dict(os.environ, {"OPENAI_ANALYSIS_MODEL": "unknown-model"}):
            with self.assertRaisesRegex(RuntimeError, "No pricing configured"):
                model_for_task("analysis")
        with patch.dict(os.environ, {"OPENAI_ANALYSIS_MODEL": " "}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_ANALYSIS_MODEL"):
                model_for_task("analysis")
        with self.assertRaisesRegex(ValueError, "Unknown LLM task"):
            model_for_task("invalid")

    def test_service_files_do_not_embed_model_ids(self):
        services = Path(__file__).parents[1] / "app" / "services"
        for service in services.glob("*.py"):
            with self.subTest(service=service.name):
                self.assertNotIn("gpt-", service.read_text(encoding="utf-8"))
                self.assertNotIn("from openai", service.read_text(encoding="utf-8"))
                self.assertNotIn("import openai", service.read_text(encoding="utf-8"))

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
            with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna", "OPENAI_API_KEY": ""}):
                with patch.object(llm, "_get_client", return_value=client):
                    with self.assertRaises(HTTPException) as caught:
                        llm.generate_text(db, self.free_id, "chat", "x" * (MAX_LLM_INPUT_BYTES + 1), [], 20, task="chat")
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
            with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna", "OPENAI_API_KEY": ""}):
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
                    with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna", "OPENAI_API_KEY": ""}):
                        with patch.object(llm, "_get_client", return_value=client):
                            with TestClient(main.app) as api:
                                response = api.post(
                                    f"/chat/conversations/{conversation_id}/messages",
                                    json={"content": "안녕"},
                                )
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.json()["detail"]["code"], "daily_limit_reached")
            client.responses.input_tokens.count.assert_not_called()
            client.responses.create.assert_not_called()
        finally:
            main.app.dependency_overrides.clear()

    def test_budgeted_chat_route_still_records_actual_usage_through_gateway(self):
        from app import main

        with Session(self.engine) as db:
            character = Character(user_id=self.free_id, name="가상 인물")
            db.add(character)
            db.flush()
            conversation = Conversation(character_id=character.id, user_id=self.free_id)
            db.add(conversation)
            db.commit()
            conversation_id = conversation.id
        client = self._client()

        def test_db():
            with Session(self.engine) as db:
                yield db

        main.app.dependency_overrides[get_db] = test_db
        main.app.dependency_overrides[get_current_user_id] = lambda: self.free_id
        try:
            with patch.object(main, "engine", self.engine):
                with patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-6-luna", "OPENAI_API_KEY": ""}):
                    with patch.object(llm, "_get_client", return_value=client):
                        with TestClient(main.app) as api:
                            response = api.post(
                                f"/chat/conversations/{conversation_id}/messages", json={"content": "안녕"},
                            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["content"], "OK")
            self.assertGreaterEqual(client.responses.input_tokens.count.call_count, 3)
            client.responses.create.assert_called_once()
            with Session(self.engine) as db:
                row = db.query(LLMUsage).one()
                self.assertEqual((row.request_type, row.model, row.input_tokens, row.output_tokens, row.status),
                                 ("chat", "gpt-6-luna", 10, 4, "completed"))
        finally:
            main.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
