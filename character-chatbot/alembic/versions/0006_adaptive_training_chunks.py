"""Represent bounded adaptive child chunks without rewriting successful evidence.

Revision ID: 0006_adaptive_training_chunks
Revises: 0005_llm_response_diagnostics
"""

from alembic import op
import sqlalchemy as sa

revision = "0006_adaptive_training_chunks"
down_revision = "0005_llm_response_diagnostics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_job_chunks") as batch:
        batch.drop_constraint("ck_training_job_chunk_status", type_="check")
        batch.add_column(sa.Column("parent_chunk_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("split_depth", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("child_order", sa.Integer(), nullable=True))
        batch.create_check_constraint(
            "ck_training_job_chunk_status",
            "status IN ('queued', 'processing', 'completed', 'failed', 'split')",
        )
        batch.create_unique_constraint(
            "uq_training_job_chunk_child", ["job_id", "parent_chunk_id", "child_order"],
        )
        batch.create_index("ix_training_job_chunks_parent", ["parent_chunk_id"])


def downgrade() -> None:
    connection = op.get_bind()
    adaptive_rows = connection.execute(sa.text(
        "SELECT 1 FROM training_job_chunks WHERE parent_chunk_id IS NOT NULL OR status = 'split' LIMIT 1"
    )).first()
    if adaptive_rows:
        raise RuntimeError("Cannot downgrade adaptive chunks while split training rows exist")
    with op.batch_alter_table("training_job_chunks") as batch:
        batch.drop_index("ix_training_job_chunks_parent")
        batch.drop_constraint("uq_training_job_chunk_child", type_="unique")
        batch.drop_constraint("ck_training_job_chunk_status", type_="check")
        batch.create_check_constraint(
            "ck_training_job_chunk_status", "status IN ('queued', 'processing', 'completed', 'failed')",
        )
        batch.drop_column("child_order")
        batch.drop_column("split_depth")
        batch.drop_column("parent_chunk_id")
