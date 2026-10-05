"""Durable conversational memory ingestion and character deletion intents.

Revision ID: 0003_memory_ingestion
Revises: 0002_training_jobs
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_memory_ingestion"
down_revision = "0002_training_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_ingestions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("character_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("user_message_id", sa.String(), nullable=False),
        sa.Column("assistant_message_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_error_code", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("provider", "assistant_message_id", name="uq_memory_ingestion_turn"),
        sa.CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')", name="ck_memory_ingestion_status"),
    )
    op.create_index("ix_memory_ingestions_status_lease", "memory_ingestions", ["status", "lease_expires_at"])
    op.create_index("ix_memory_ingestions_character", "memory_ingestions", ["user_id", "character_id"])
    op.create_table(
        "memory_deletions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("character_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_error_code", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("provider", "character_id", name="uq_memory_deletion_character"),
        sa.CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')", name="ck_memory_deletion_status"),
    )
    op.create_index("ix_memory_deletions_status_lease", "memory_deletions", ["status", "lease_expires_at"])


def downgrade() -> None:
    op.drop_index("ix_memory_deletions_status_lease", table_name="memory_deletions")
    op.drop_table("memory_deletions")
    op.drop_index("ix_memory_ingestions_character", table_name="memory_ingestions")
    op.drop_index("ix_memory_ingestions_status_lease", table_name="memory_ingestions")
    op.drop_table("memory_ingestions")
