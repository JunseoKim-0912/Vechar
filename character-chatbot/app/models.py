import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON, Numeric, Index, CheckConstraint, UniqueConstraint, text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import relationship
from .database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChangeReason(str, enum.Enum):
    TRAINING_INGEST = "TRAINING_INGEST"
    USER_CORRECTION = "USER_CORRECTION"
    MANUAL_EDIT = "MANUAL_EDIT"
    WORLD_EDIT = "WORLD_EDIT"


class SourceType(str, enum.Enum):
    STORY = "STORY"
    DIALOGUE = "DIALOGUE"
    MANUAL_DESCRIPTION = "MANUAL_DESCRIPTION"
    WORLD_DERIVED = "WORLD_DERIVED"  # 세계관에서 자동 추출되어 만들어진 캐릭터


class WorldSourceType(str, enum.Enum):
    NOVEL_EPISODE = "NOVEL_EPISODE"   # 시리즈의 한 화
    DESCRIPTION = "DESCRIPTION"        # 세계관 설명 (등장인물 정보 없음)


class IngestStatus(str, enum.Enum):
    PENDING = "PENDING"
    EXTRACTED = "EXTRACTED"
    MERGED = "MERGED"
    FAILED = "FAILED"


class MessageRole(str, enum.Enum):
    USER = "USER"
    CHARACTER = "CHARACTER"


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),)
    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    is_premium = Column(Boolean, default=False, nullable=False)
    # Permission role is independent of the existing premium product tier.
    role = Column(String(16), default=UserRole.USER.value, server_default=text("'user'"), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    characters = relationship("Character", back_populates="user", cascade="all, delete-orphan")
    worlds = relationship("World", cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    world_id = Column(String, ForeignKey("worlds.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String, nullable=False)
    profile_image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="characters")
    world = relationship("World", back_populates="characters")  # cascade 없음 — 세계관이 지워져도 캐릭터는 남음
    profile = relationship(
        "CharacterProfile", back_populates="character", uselist=False, cascade="all, delete-orphan"
    )
    training_sources = relationship("TrainingSource", back_populates="character", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="character", cascade="all, delete-orphan",
                                 foreign_keys="Conversation.character_id")
    correction_logs = relationship("CorrectionLog", back_populates="character", cascade="all, delete-orphan")


# The "memory". `data` shape (validated with Pydantic in schemas.py, not by Postgres):
# {
#   "personality_summary": str,
#   "speech_style": str,
#   "background_facts": [str],
#   "relationships": [str],
#   "sample_dialogues": [str],
#   "do_not_do": [str]
# }
class CharacterProfile(Base):
    __tablename__ = "character_profiles"
    id = Column(String, primary_key=True, default=gen_uuid)
    character_id = Column(String, ForeignKey("characters.id", ondelete="CASCADE"), unique=True, nullable=False)
    data = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    character = relationship("Character", back_populates="profile")
    history = relationship("CharacterProfileHistory", back_populates="profile", cascade="all, delete-orphan")


# Snapshot taken before every write, so a bad merge/correction can be rolled back.
class CharacterProfileHistory(Base):
    __tablename__ = "character_profile_history"
    id = Column(String, primary_key=True, default=gen_uuid)
    character_profile_id = Column(
        String, ForeignKey("character_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    data = Column(JSON, nullable=False)
    version = Column(Integer, nullable=False)
    change_reason = Column(SAEnum(ChangeReason), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    profile = relationship("CharacterProfile", back_populates="history")


class TrainingSource(Base):
    __tablename__ = "training_sources"
    id = Column(String, primary_key=True, default=gen_uuid)
    character_id = Column(String, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True)
    source_type = Column(SAEnum(SourceType), nullable=False)
    raw_text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    extracted_data = Column(JSON, nullable=True)
    status = Column(SAEnum(IngestStatus), default=IngestStatus.PENDING)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    character = relationship("Character", back_populates="training_sources")


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint("kind IN ('user_character', 'character_pair')", name="ck_conversations_kind"),
        CheckConstraint("language IN ('en', 'ko')", name="ck_conversations_language"),
        CheckConstraint("turn_index >= 0", name="ck_conversations_turn_index"),
        CheckConstraint(
            "(kind = 'user_character' AND secondary_character_id IS NULL) OR "
            "(kind = 'character_pair' AND secondary_character_id IS NOT NULL "
            "AND character_id <> secondary_character_id AND name IS NOT NULL AND trim(name) <> '')",
            name="ck_conversations_shape",
        ),
    )
    id = Column(String, primary_key=True, default=gen_uuid)
    character_id = Column(String, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True)
    secondary_character_id = Column(String, ForeignKey("characters.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(20), nullable=False, default="user_character", server_default=text("'user_character'"))
    name = Column(String(120), nullable=True)
    language = Column(String(2), nullable=False, default="en", server_default=text("'en'"))
    turn_index = Column(Integer, nullable=False, default=0, server_default=text("0"))
    generation_token = Column(String(36), nullable=True)
    generation_lease_expires_at = Column(DateTime, nullable=True)
    # Versioned, bounded conversation hints; canonical profile data is never stored here.
    runtime_state = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow, onupdate=utcnow,
                        server_default=text("CURRENT_TIMESTAMP"))

    character = relationship("Character", back_populates="conversations", foreign_keys=[character_id])
    secondary_character = relationship("Character", foreign_keys=[secondary_character_id])
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("conversation_id", "turn_index", name="uq_messages_conversation_turn"),)
    id = Column(String, primary_key=True, default=gen_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(SAEnum(MessageRole), nullable=False)
    speaker_character_id = Column(String, ForeignKey("characters.id", ondelete="SET NULL"), nullable=True)
    turn_index = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    is_correction_cmd = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class CorrectionLog(Base):
    __tablename__ = "correction_logs"
    id = Column(String, primary_key=True, default=gen_uuid)
    character_id = Column(String, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True)
    user_instruction = Column(Text, nullable=False)
    resulting_version = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    character = relationship("Character", back_populates="correction_logs")


# ── World (세계관) ──────────────────────────────────────────────
# Character와 나란한 엔티티. 여러 캐릭터가 같은 World를 공유해서,
# 소설 전체를 캐릭터마다 반복 학습시키지 않아도 되게 하는 게 목적.

class World(Base):
    __tablename__ = "worlds"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    characters = relationship("Character", back_populates="world")  # cascade 없음
    profile = relationship("WorldProfile", back_populates="world", uselist=False, cascade="all, delete-orphan")
    sources = relationship("WorldSource", back_populates="world", cascade="all, delete-orphan")


# CharacterProfile과 동일한 패턴의 "기억". data 필드:
# {
#   "world_summary": str,
#   "key_facts": [str],
#   "timeline_notes": [str],       # 각 항목에 어느 시리즈/시기인지 명시하도록 추출 프롬프트에서 유도
#   "mentioned_characters": [str]  # 캐릭터 생성 시 추천용
# }
class WorldProfile(Base):
    __tablename__ = "world_profiles"
    id = Column(String, primary_key=True, default=gen_uuid)
    world_id = Column(String, ForeignKey("worlds.id", ondelete="CASCADE"), unique=True, nullable=False)
    data = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    world = relationship("World", back_populates="profile")
    history = relationship("WorldProfileHistory", back_populates="profile", cascade="all, delete-orphan")


class WorldProfileHistory(Base):
    __tablename__ = "world_profile_history"
    id = Column(String, primary_key=True, default=gen_uuid)
    world_profile_id = Column(String, ForeignKey("world_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    data = Column(JSON, nullable=False)
    version = Column(Integer, nullable=False)
    change_reason = Column(SAEnum(ChangeReason), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    profile = relationship("WorldProfile", back_populates="history")


class WorldSource(Base):
    __tablename__ = "world_sources"
    id = Column(String, primary_key=True, default=gen_uuid)
    world_id = Column(String, ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False, index=True)
    source_type = Column(SAEnum(WorldSourceType), nullable=False)
    series_name = Column(String, nullable=True)     # DESCRIPTION 타입이면 비워도 됨
    episode_number = Column(Integer, nullable=True)  # 시리즈 내 순서 (시간대 혼동 방지용 힌트)
    raw_text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    extracted_data = Column(JSON, nullable=True)
    status = Column(SAEnum(IngestStatus), default=IngestStatus.PENDING)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    world = relationship("World", back_populates="sources")


# 내보내기(export) 횟수 제한(월 단위)을 추적하기 위한 로그. 캐릭터/세계관 내보내기를 합쳐서 셉니다.
class ExportLog(Base):
    __tablename__ = "export_logs"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    export_type = Column(String, nullable=False)  # "character" | "world"
    entity_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)


# 가져오기(import) 횟수 제한(월 단위)을 추적하기 위한 로그. 캐릭터/세계관 가져오기를 합쳐서 셉니다.
class ImportLog(Base):
    __tablename__ = "import_logs"
    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    import_type = Column(String, nullable=False)  # "character" | "world"
    entity_id = Column(String, nullable=False)  # 새로 만들어진 character_id/world_id
    created_at = Column(DateTime, default=utcnow)


class LLMUsage(Base):
    __tablename__ = "llm_usage"
    __table_args__ = (
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
        Index("uq_llm_usage_operation_key", "operation_key", unique=True),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    request_type = Column(String, nullable=False)
    model = Column(String, nullable=False)
    status = Column(String, nullable=False)  # reserved | completed | failed | failed_response | usage_unavailable | preflight_failed
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    cached_input_tokens = Column(Integer, nullable=False, default=0)
    reserved_total_tokens = Column(Integer, nullable=False, default=0)
    budget_tokens = Column(Integer, nullable=False, default=0)
    # Reserved upper-bound cost until provider usage arrives; actual estimated cost afterward.
    # Unknown billing outcomes retain the reservation to avoid understating the limit.
    estimated_cost_usd = Column(Numeric(12, 8), nullable=True)
    error_type = Column(String, nullable=True)
    failure_class = Column(String(32), nullable=True)
    operation_key = Column(String(200), nullable=True)
    provider_response_id = Column(String(200), nullable=True)
    provider_status = Column(String(32), nullable=True)
    provider_error_code = Column(String(100), nullable=True)
    incomplete_reason = Column(String(64), nullable=True)
    reasoning_tokens = Column(Integer, nullable=True)
    reservation_expires_at = Column(DateTime, nullable=True)
    reconciled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    __table_args__ = (
        CheckConstraint("target_type IN ('character', 'world')", name="ck_training_job_target_type"),
        CheckConstraint("source_type IN ('text', 'file')", name="ck_training_job_source_type"),
        CheckConstraint(
            "status IN ('queued', 'chunking', 'extracting', 'synthesizing', 'completed', 'failed', 'cancelled')",
            name="ck_training_job_status",
        ),
        Index(
            "uq_training_jobs_active_target", "user_id", "target_type", "target_id", unique=True,
            postgresql_where=text("status IN ('queued', 'chunking', 'extracting', 'synthesizing')"),
            sqlite_where=text("status IN ('queued', 'chunking', 'extracting', 'synthesizing')"),
        ),
        Index("ix_training_jobs_user_created", "user_id", "created_at"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    target_type = Column(String, nullable=False)
    target_id = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    training_source_type = Column(String, nullable=False)
    source_text = Column(Text, nullable=True)
    source_hash = Column(String(64), nullable=False)
    source_char_count = Column(Integer, nullable=False)
    series_name = Column(String, nullable=True)
    episode_number = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="queued")
    stage = Column(String, nullable=False, default="queued")
    total_chunks = Column(Integer, nullable=False, default=0)
    completed_chunks = Column(Integer, nullable=False, default=0)
    progress = Column(Integer, nullable=False, default=0)
    source_tokens = Column(Integer, nullable=True)
    direct_mode = Column(Boolean, nullable=False, default=False)
    attempt_count = Column(Integer, nullable=False, default=0)
    lease_token = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    error_code = Column(String, nullable=True)
    error_message_safe = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    chunks = relationship("TrainingJobChunk", cascade="all, delete-orphan", back_populates="job")


class TrainingJobChunk(Base):
    __tablename__ = "training_job_chunks"
    __table_args__ = (
        UniqueConstraint("job_id", "chunk_index", name="uq_training_job_chunk_index"),
        UniqueConstraint("job_id", "parent_chunk_id", "child_order", name="uq_training_job_chunk_child"),
        CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed', 'split')", name="ck_training_job_chunk_status"),
        Index("ix_training_job_chunks_job_status", "job_id", "status"),
        Index("ix_training_job_chunks_parent", "parent_chunk_id"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    job_id = Column(String, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    parent_chunk_id = Column(String, nullable=True)
    split_depth = Column(Integer, nullable=False, default=0, server_default="0")
    child_order = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="queued")
    source_start = Column(Integer, nullable=False)
    core_start = Column(Integer, nullable=False)
    core_end = Column(Integer, nullable=False)
    token_start = Column(Integer, nullable=False)
    token_end = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    overlap_tokens = Column(Integer, nullable=False, default=0)
    extraction_result = Column(JSON, nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    lease_token = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    job = relationship("TrainingJob", back_populates="chunks")


class MemoryIngestion(Base):
    __tablename__ = "memory_ingestions"
    __table_args__ = (
        UniqueConstraint("provider", "assistant_message_id", name="uq_memory_ingestion_turn"),
        CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')", name="ck_memory_ingestion_status"),
        Index("ix_memory_ingestions_status_lease", "status", "lease_expires_at"),
        Index("ix_memory_ingestions_character", "user_id", "character_id"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    # Scope and source identifiers deliberately survive character/message cascade deletion.
    user_id = Column(String, nullable=False)
    character_id = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    user_message_id = Column(String, nullable=False)
    assistant_message_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    attempt_count = Column(Integer, nullable=False, default=0)
    lease_token = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    last_error_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


class MemoryDeletion(Base):
    __tablename__ = "memory_deletions"
    __table_args__ = (
        UniqueConstraint("provider", "character_id", name="uq_memory_deletion_character"),
        CheckConstraint("status IN ('queued', 'processing', 'completed', 'failed')", name="ck_memory_deletion_status"),
        Index("ix_memory_deletions_status_lease", "status", "lease_expires_at"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=False)
    character_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    attempt_count = Column(Integer, nullable=False, default=0)
    lease_token = Column(String, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    last_error_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
