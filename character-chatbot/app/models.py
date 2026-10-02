import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, JSON, Numeric, Index
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


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    is_premium = Column(Boolean, default=False, nullable=False)
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
    conversations = relationship("Conversation", back_populates="character", cascade="all, delete-orphan")
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
    id = Column(String, primary_key=True, default=gen_uuid)
    character_id = Column(String, ForeignKey("characters.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    character = relationship("Character", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=gen_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(SAEnum(MessageRole), nullable=False)
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
    __table_args__ = (Index("ix_llm_usage_user_created", "user_id", "created_at"),)

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
    estimated_cost_usd = Column(Numeric(12, 8), nullable=True)
    error_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
