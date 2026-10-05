"""Durable training jobs and chunk checkpoints.

Revision ID: 0002_training_jobs
Revises: 0001_initial_schema
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_training_jobs"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_type", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("training_source_type", sa.String(), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("source_char_count", sa.Integer(), nullable=False),
        sa.Column("series_name", sa.String(), nullable=True),
        sa.Column("episode_number", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("completed_chunks", sa.Integer(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("source_tokens", sa.Integer(), nullable=True),
        sa.Column("direct_mode", sa.Boolean(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message_safe", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("target_type IN ('character', 'world')", name="ck_training_job_target_type"),
        sa.CheckConstraint("source_type IN ('text', 'file')", name="ck_training_job_source_type"),
        sa.CheckConstraint("status IN ('queued', 'chunking', 'extracting', 'synthesizing', 'completed', 'failed', 'cancelled')", name="ck_training_job_status"),
    )
    op.create_index("ix_training_jobs_user_created", "training_jobs", ["user_id", "created_at"])
    op.create_index(
        "uq_training_jobs_active_target", "training_jobs", ["user_id", "target_type", "target_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'chunking', 'extracting', 'synthesizing')"),
        sqlite_where=sa.text("status IN ('queued', 'chunking', 'extracting', 'synthesizing')"),
    )
    op.create_table(
        "training_job_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("job_id", sa.String(), sa.ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_start", sa.Integer(), nullable=False),
        sa.Column("core_start", sa.Integer(), nullable=False),
        sa.Column("core_end", sa.Integer(), nullable=False),
        sa.Column("token_start", sa.Integer(), nullable=False),
        sa.Column("token_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("overlap_tokens", sa.Integer(), nullable=False),
        sa.Column("extraction_result", sa.JSON(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("job_id", "chunk_index", name="uq_training_job_chunk_index"),
        sa.CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')", name="ck_training_job_chunk_status"),
    )
    op.create_index("ix_training_job_chunks_job_status", "training_job_chunks", ["job_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_training_job_chunks_job_status", table_name="training_job_chunks")
    op.drop_table("training_job_chunks")
    op.drop_index("uq_training_jobs_active_target", table_name="training_jobs")
    op.drop_index("ix_training_jobs_user_created", table_name="training_jobs")
    op.drop_table("training_jobs")
