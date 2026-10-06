"""Extend durable conversations for two-character rooms and fenced turns.

Revision ID: 0007_character_conversations
Revises: 0006_adaptive_training_chunks
"""

from alembic import op
import sqlalchemy as sa

revision = "0007_character_conversations"
down_revision = "0006_adaptive_training_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("conversations") as batch:
        batch.add_column(sa.Column("secondary_character_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("kind", sa.String(20), nullable=False,
                                   server_default=sa.text("'user_character'")))
        batch.add_column(sa.Column("name", sa.String(120), nullable=True))
        batch.add_column(sa.Column("language", sa.String(2), nullable=False, server_default=sa.text("'en'")))
        batch.add_column(sa.Column("turn_index", sa.Integer(), nullable=False, server_default=sa.text("0")))
        batch.add_column(sa.Column("generation_token", sa.String(36), nullable=True))
        batch.add_column(sa.Column("generation_lease_expires_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("updated_at", sa.DateTime(), nullable=False,
                                   server_default=sa.text("CURRENT_TIMESTAMP")))
        batch.create_foreign_key("fk_conversations_secondary_character", "characters",
                                 ["secondary_character_id"], ["id"], ondelete="CASCADE")
        batch.create_index("ix_conversations_secondary_character_id", ["secondary_character_id"])
        batch.create_check_constraint("ck_conversations_kind", "kind IN ('user_character', 'character_pair')")
        batch.create_check_constraint("ck_conversations_language", "language IN ('en', 'ko')")
        batch.create_check_constraint("ck_conversations_turn_index", "turn_index >= 0")
        batch.create_check_constraint(
            "ck_conversations_shape",
            "(kind = 'user_character' AND secondary_character_id IS NULL) OR "
            "(kind = 'character_pair' AND secondary_character_id IS NOT NULL "
            "AND character_id <> secondary_character_id AND name IS NOT NULL AND trim(name) <> '')",
        )

    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("speaker_character_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("turn_index", sa.Integer(), nullable=True))
        batch.create_foreign_key("fk_messages_speaker_character", "characters",
                                 ["speaker_character_id"], ["id"], ondelete="SET NULL")
        batch.create_unique_constraint("uq_messages_conversation_turn", ["conversation_id", "turn_index"])


def downgrade() -> None:
    connection = op.get_bind()
    if connection.execute(sa.text("SELECT 1 FROM conversations WHERE kind = 'character_pair' LIMIT 1")).first():
        raise RuntimeError("Cannot downgrade while character conversation rooms exist")
    with op.batch_alter_table("messages") as batch:
        batch.drop_constraint("uq_messages_conversation_turn", type_="unique")
        batch.drop_constraint("fk_messages_speaker_character", type_="foreignkey")
        batch.drop_column("turn_index")
        batch.drop_column("speaker_character_id")
    with op.batch_alter_table("conversations") as batch:
        batch.drop_constraint("ck_conversations_shape", type_="check")
        batch.drop_constraint("ck_conversations_turn_index", type_="check")
        batch.drop_constraint("ck_conversations_language", type_="check")
        batch.drop_constraint("ck_conversations_kind", type_="check")
        batch.drop_index("ix_conversations_secondary_character_id")
        batch.drop_constraint("fk_conversations_secondary_character", type_="foreignkey")
        batch.drop_column("updated_at")
        batch.drop_column("generation_lease_expires_at")
        batch.drop_column("generation_token")
        batch.drop_column("turn_index")
        batch.drop_column("language")
        batch.drop_column("name")
        batch.drop_column("kind")
        batch.drop_column("secondary_character_id")
