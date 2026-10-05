"""Preview or explicitly change a DB user permission role.

Usage:
  python -m app.scripts.set_user_role --email EMAIL --role admin
  python -m app.scripts.set_user_role --email EMAIL --role admin --apply --expect-user-id ID
"""

import argparse
import sys

from sqlalchemy.exc import SQLAlchemyError

from ..database import SessionLocal
from ..models import User, UserRole


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview or apply an operator-only user role change")
    parser.add_argument("--email", required=True, help="Exact existing account email")
    parser.add_argument("--role", required=True, choices=[role.value for role in UserRole])
    parser.add_argument("--apply", action="store_true", help="Commit the change after preview")
    parser.add_argument("--expect-user-id", help="User ID copied from the preview; required with --apply")
    args = parser.parse_args(argv)

    try:
        with SessionLocal() as db:
            query = db.query(User).filter(User.email == args.email)
            user = (query.with_for_update() if args.apply else query).first()
            if user is None:
                print("Account not found; no change made.", file=sys.stderr)
                return 1

            print(f"Target user ID: {user.id}")
            print(f"Email: {user.email}")
            print(f"Current role: {user.role}; requested role: {args.role}")
            print(f"Premium tier unchanged: {bool(user.is_premium)}")
            if not args.apply:
                print("Preview only. Re-run with --apply --expect-user-id ID to commit.")
                return 0
            if args.expect_user_id != user.id:
                print("Confirmation user ID does not match; no change made.", file=sys.stderr)
                return 1
            user.role = args.role
            db.commit()
            print("Role updated.")
            return 0
    except SQLAlchemyError:
        print("Database operation failed; no credentials or row data are shown.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
