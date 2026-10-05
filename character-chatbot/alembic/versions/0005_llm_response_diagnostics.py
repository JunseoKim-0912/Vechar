"""Add non-sensitive LLM response diagnostics and logical reservation identity.

Revision ID: 0005_llm_response_diagnostics
Revises: 0004_user_roles
"""

from alembic import op
import sqlalchemy as sa

revision = "0005_llm_response_diagnostics"
down_revision = "0004_user_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("llm_usage") as batch:
        batch.add_column(sa.Column("failure_class", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("operation_key", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("provider_response_id", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("provider_status", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("provider_error_code", sa.String(length=100), nullable=True))
        batch.add_column(sa.Column("incomplete_reason", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("reasoning_tokens", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("reservation_expires_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("reconciled_at", sa.DateTime(), nullable=True))
        batch.create_index("uq_llm_usage_operation_key", ["operation_key"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("llm_usage") as batch:
        batch.drop_index("uq_llm_usage_operation_key")
        for name in ("reconciled_at", "reservation_expires_at", "reasoning_tokens",
                     "incomplete_reason", "provider_error_code", "provider_status", "provider_response_id",
                     "operation_key", "failure_class"):
            batch.drop_column(name)
