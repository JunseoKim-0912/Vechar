"""Migration foundation checks without connecting to a deployed database."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from app.database import Base
from app import models  # noqa: F401 - populate Base.metadata
from app.main import app


BACKEND_ROOT = Path(__file__).resolve().parent.parent
INI_PATH = BACKEND_ROOT / "alembic.ini"
INITIAL_REVISION = "0001_initial_schema"
HEAD_REVISION = "0002_training_jobs"


class AlembicFoundationTests(unittest.TestCase):
    def make_config(self):
        return Config(str(INI_PATH))

    def test_config_has_one_initial_head_and_no_credentials(self):
        config = self.make_config()
        script = ScriptDirectory.from_config(config)
        self.assertEqual(script.get_heads(), [HEAD_REVISION])
        self.assertIsNone(script.get_revision(INITIAL_REVISION).down_revision)
        self.assertEqual(script.get_revision(HEAD_REVISION).down_revision, INITIAL_REVISION)
        self.assertIsNone(config.get_main_option("sqlalchemy.url"))
        ini_text = INI_PATH.read_text(encoding="utf-8")
        self.assertNotIn("postgresql://", ini_text)
        self.assertNotIn("postgres://", ini_text)

    def test_application_startup_does_not_create_or_migrate_schema(self):
        self.assertEqual(app.router.on_startup, [])

    def test_fresh_sqlite_upgrade_matches_models_and_downgrades(self):
        engine = create_engine("sqlite://")
        try:
            with engine.connect() as connection:
                config = self.make_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
                self.assertEqual(
                    set(inspect(connection).get_table_names()),
                    set(Base.metadata.tables) | {"alembic_version"},
                )
                self.assertEqual(
                    connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one(),
                    HEAD_REVISION,
                )
                self.assertEqual(compare_metadata(MigrationContext.configure(connection), Base.metadata), [])
                command.downgrade(config, "base")
                self.assertEqual(set(inspect(connection).get_table_names()), {"alembic_version"})
        finally:
            engine.dispose()

    def test_baseline_to_training_jobs_and_back_preserves_baseline(self):
        engine = create_engine("sqlite://")
        try:
            with engine.connect() as connection:
                config = self.make_config()
                config.attributes["connection"] = connection
                command.upgrade(config, INITIAL_REVISION)
                baseline_tables = set(inspect(connection).get_table_names())
                command.upgrade(config, HEAD_REVISION)
                self.assertEqual(connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one(), HEAD_REVISION)
                self.assertIn("training_jobs", inspect(connection).get_table_names())
                self.assertIn("training_job_chunks", inspect(connection).get_table_names())
                command.downgrade(config, INITIAL_REVISION)
                self.assertEqual(set(inspect(connection).get_table_names()), baseline_tables)
        finally:
            engine.dispose()

    def test_postgresql_offline_ddl_creates_shared_enums_once_before_tables(self):
        output = io.StringIO()
        # Offline SQL is compile-only; no PostgreSQL connection is attempted.
        with patch("app.database.DATABASE_URL", "postgresql+psycopg2://localhost/vechar_test?sslmode=require"):
            with contextlib.redirect_stdout(output):
                command.upgrade(self.make_config(), "head", sql=True)
        ddl = output.getvalue()
        for name in ("changereason", "sourcetype", "worldsourcetype", "ingeststatus", "messagerole"):
            self.assertEqual(ddl.count(f"CREATE TYPE {name} AS ENUM"), 1)
            self.assertLess(ddl.index(f"CREATE TYPE {name}"), ddl.index("CREATE TABLE world_sources"))
        for table in Base.metadata.tables:
            self.assertIn(f"CREATE TABLE {table}", ddl)
        self.assertIn("ON DELETE SET NULL", ddl)
        self.assertIn("ON DELETE CASCADE", ddl)
        self.assertIn("NUMERIC(12, 8)", ddl)
        self.assertIn("CREATE UNIQUE INDEX ix_users_email", ddl)
        self.assertIn("CREATE INDEX ix_llm_usage_user_created", ddl)

    def test_postgresql_offline_downgrade_drops_tables_before_enums(self):
        output = io.StringIO()
        with patch("app.database.DATABASE_URL", "postgresql+psycopg2://localhost/vechar_test"):
            with contextlib.redirect_stdout(output):
                command.downgrade(self.make_config(), "0001_initial_schema:base", sql=True)
        ddl = output.getvalue()
        baseline_tables = set(Base.metadata.tables) - {"training_jobs", "training_job_chunks"}
        last_table_drop = max(ddl.index(f"DROP TABLE {table}") for table in baseline_tables)
        for name in ("changereason", "sourcetype", "worldsourcetype", "ingeststatus", "messagerole"):
            self.assertEqual(ddl.count(f"DROP TYPE {name}"), 1)
            self.assertLess(last_table_drop, ddl.index(f"DROP TYPE {name}"))


if __name__ == "__main__":
    unittest.main()
