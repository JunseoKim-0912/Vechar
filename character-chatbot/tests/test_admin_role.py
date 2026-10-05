"""Role authorization and operator promotion without real external services."""

import contextlib
import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import User, UserRole
from app.scripts import set_user_role
from app.tier_limits import TIER_LIMITS, get_limits


class AdminRoleTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)

        def test_db():
            with Session(self.engine) as db:
                yield db

        app.dependency_overrides[get_db] = test_db
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        app.dependency_overrides.clear()
        self.engine.dispose()

    def test_public_signup_ignores_claimed_role_and_premium_flags(self):
        response = self.client.post("/auth/signup", json={
            "email": "signup@example.com", "password": "safe-password-123",
            "role": "admin", "is_premium": True,
        })
        self.assertEqual(response.status_code, 201)
        with Session(self.engine) as db:
            user = db.query(User).filter(User.email == "signup@example.com").one()
            self.assertEqual(user.role, UserRole.USER.value)
            self.assertFalse(user.is_premium)
            self.assertEqual(get_limits(db, user.id), TIER_LIMITS["free"])
        # No public role mutation route exists, even with a valid user token.
        self.assertEqual(self.client.patch("/auth/role", json={"role": "admin"}, headers={
            "Authorization": f"Bearer {response.json()['token']}",
        }).status_code, 404)

    def test_admin_role_does_not_grant_premium_product_quotas(self):
        with Session(self.engine) as db:
            admin = User(email="operator@example.invalid", password_hash="unused",
                         role=UserRole.ADMIN.value, is_premium=False)
            db.add(admin)
            db.commit()
            self.assertEqual(get_limits(db, admin.id), TIER_LIMITS["free"])

    def test_invalid_role_is_rejected_by_database_constraint(self):
        with Session(self.engine) as db:
            db.add(User(email="invalid@example.invalid", password_hash="unused", role="superuser"))
            with self.assertRaises(IntegrityError):
                db.commit()
            db.rollback()

    def test_operator_command_previews_requires_exact_id_then_applies(self):
        with Session(self.engine) as db:
            user = User(email="owner@example.invalid", password_hash="never-print-this",
                        is_premium=True)
            db.add(user)
            db.commit()
            user_id = user.id

        output = io.StringIO()
        with patch.object(set_user_role, "SessionLocal", lambda: Session(self.engine)), \
             contextlib.redirect_stdout(output):
            self.assertEqual(set_user_role.main([
                "--email", "owner@example.invalid", "--role", "admin",
            ]), 0)
            self.assertEqual(set_user_role.main([
                "--email", "owner@example.invalid", "--role", "admin",
                "--apply", "--expect-user-id", "wrong-id",
            ]), 1)
            with Session(self.engine) as db:
                self.assertEqual(db.get(User, user_id).role, UserRole.USER.value)
            self.assertEqual(set_user_role.main([
                "--email", "owner@example.invalid", "--role", "admin",
                "--apply", "--expect-user-id", user_id,
            ]), 0)
        with Session(self.engine) as db:
            updated = db.get(User, user_id)
            self.assertEqual((updated.role, updated.is_premium), (UserRole.ADMIN.value, True))
        self.assertIn(user_id, output.getvalue())
        self.assertNotIn("never-print-this", output.getvalue())

    def test_operator_command_rejects_missing_user_and_invalid_role(self):
        with patch.object(set_user_role, "SessionLocal", lambda: Session(self.engine)), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(set_user_role.main([
                "--email", "missing@example.invalid", "--role", "admin", "--apply",
                "--expect-user-id", "any-id",
            ]), 1)
            with self.assertRaises(SystemExit) as caught:
                set_user_role.main(["--email", "missing@example.invalid", "--role", "superuser"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
