import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from sqlalchemy import create_mock_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from app import main
from app.database import Base, engine_kwargs_for_url, normalize_database_url
from app.models import LLMUsage
from app import llm_usage


class DatabaseUrlTests(unittest.TestCase):
    def test_sqlite_gets_thread_connect_arg_only(self):
        self.assertEqual(
            engine_kwargs_for_url("sqlite:///./dev.db"),
            {"connect_args": {"check_same_thread": False}},
        )
        self.assertEqual(
            engine_kwargs_for_url("postgresql://user:password@host/database"),
            {},
        )

    def test_managed_postgres_ssl_query_is_preserved(self):
        url = normalize_database_url(
            "postgresql://user:password@host/database?sslmode=require&channel_binding=require"
        )
        parsed = make_url(url)
        self.assertEqual(parsed.get_backend_name(), "postgresql")
        self.assertEqual(parsed.get_driver_name(), "psycopg2")
        self.assertEqual(parsed.query["sslmode"], "require")
        self.assertEqual(parsed.query["channel_binding"], "require")

    def test_legacy_provider_postgres_scheme_is_normalized(self):
        normalized = normalize_database_url("postgres://user:password@host/database?sslmode=require")
        self.assertEqual(
            normalized,
            "postgresql://user:password@host/database?sslmode=require",
        )

    def test_missing_database_url_is_rejected(self):
        for value in (None, "", "   "):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
                normalize_database_url(value)


class PostgreSQLMetadataTests(unittest.TestCase):
    def test_full_schema_compiles_for_postgresql(self):
        statements = []
        engine = None

        def record(sql, *multiparams, **params):
            statements.append(str(sql.compile(dialect=engine.dialect)))

        engine = create_mock_engine("postgresql+psycopg2://", record)
        Base.metadata.create_all(engine)

        expected_tables = {
            "users", "characters", "character_profiles", "character_profile_history",
            "training_sources", "worlds", "world_profiles", "world_profile_history",
            "world_sources", "conversations", "messages", "correction_logs",
            "export_logs", "import_logs", "llm_usage", "training_jobs", "training_job_chunks",
        }
        expected_tables.update({"memory_ingestions", "memory_deletions"})
        self.assertEqual(set(Base.metadata.tables), expected_tables)
        compiled = "\n".join(statements)
        self.assertIn("CREATE TABLE users", compiled)
        self.assertIn("CREATE TABLE llm_usage", compiled)
        self.assertIn("CREATE TYPE messagerole AS ENUM", compiled)
        self.assertNotIn("DROP TABLE", compiled)


class UsageLockingTests(unittest.TestCase):
    def test_postgresql_reservation_uses_row_lock(self):
        db = Mock()
        db.bind.dialect.name = "postgresql"
        query = db.query.return_value
        filtered = query.filter.return_value
        locked = filtered.with_for_update.return_value
        locked.first.return_value = SimpleNamespace(is_premium=False)

        session_context = Mock()
        session_context.__enter__ = Mock(return_value=db)
        session_context.__exit__ = Mock(return_value=False)

        with patch.object(llm_usage, "Session", return_value=session_context):
            with patch.object(llm_usage, "_period_usage", return_value=0):
                llm_usage.reserve_usage(Mock(), "user-id", "chat", "gpt-6-luna", 10, 20)

        filtered.with_for_update.assert_called_once_with()
        db.execute.assert_not_called()
        db.add.assert_called_once()
        self.assertIsInstance(db.add.call_args.args[0], LLMUsage)
        self.assertEqual(db.add.call_args.args[0].estimated_cost_usd, Decimal("0.000011"))
        db.commit.assert_called_once_with()


class HealthEndpointTests(unittest.TestCase):
    def test_health_is_process_only(self):
        with patch.object(main.engine, "connect") as connect:
            self.assertEqual(main.health(), {"ok": True})
        connect.assert_not_called()

    def test_ready_checks_database(self):
        connection = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=connection)
        context.__exit__ = Mock(return_value=False)
        with patch.object(main.engine, "connect", return_value=context):
            self.assertEqual(main.ready(), {"ready": True})
        self.assertEqual(str(connection.execute.call_args.args[0]), "SELECT 1")

    def test_ready_returns_503_without_leaking_driver_error(self):
        with patch.object(main.engine, "connect", side_effect=SQLAlchemyError("private host details")):
            with self.assertRaises(HTTPException) as raised:
                main.ready()
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail, "Database unavailable")


if __name__ == "__main__":
    unittest.main()
