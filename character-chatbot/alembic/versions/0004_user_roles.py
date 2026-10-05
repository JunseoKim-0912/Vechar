"""Add a DB-authoritative permission role, separate from premium tier.

Revision ID: 0004_user_roles
Revises: 0003_memory_ingestion
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_user_roles"
down_revision = "0003_memory_ingestion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The server default backfills every existing Free/Premium row as non-admin
    # and makes future inserts safe even when a client omits the field.
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("role", sa.String(length=16), nullable=False,
                                   server_default=sa.text("'user'")))
        batch.create_check_constraint("ck_users_role", "role IN ('user', 'admin')")


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_constraint("ck_users_role", type_="check")
        batch.drop_column("role")
