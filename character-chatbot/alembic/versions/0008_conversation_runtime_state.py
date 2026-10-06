"""Persist bounded conversation runtime hints alongside the visible turn.

Revision ID: 0008_conversation_runtime_state
Revises: 0007_character_conversations
"""

from alembic import op
import sqlalchemy as sa

revision = "0008_conversation_runtime_state"
down_revision = "0007_character_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("conversations") as batch:
        batch.add_column(sa.Column("runtime_state", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("conversations") as batch:
        batch.drop_column("runtime_state")
