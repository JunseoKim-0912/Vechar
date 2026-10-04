from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# This is the contract between the extraction/correction LLM calls and the DB.
# Anything an LLM call returns is validated against this before it's allowed
# to touch CharacterProfile.data.
class CharacterProfileData(BaseModel):
    personality_summary: str = ""
    speech_style: str = ""
    background_facts: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    sample_dialogues: list[str] = Field(default_factory=list)
    do_not_do: list[str] = Field(default_factory=list)


class CharacterSynthesisResult(BaseModel):
    personality_summary: str
    speech_style: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    token: str


class CharacterCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    profile_image_url: Optional[str] = None
    world_id: Optional[str] = None  # 생략 시 이 사용자의 기본 "현실" 세계관에 자동 배정


class CharacterProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    character_id: str
    data: CharacterProfileData
    version: int
    updated_at: datetime


class CharacterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    profile_image_url: Optional[str] = None
    world_id: Optional[str] = None
    created_at: datetime


class CharacterDetailRead(CharacterRead):
    profile: Optional[CharacterProfileRead] = None


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1)


class ConversationCreateRequest(BaseModel):
    character_id: str


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    is_correction_cmd: bool
    created_at: datetime


class RollbackRequest(BaseModel):
    to_version: Optional[int] = None


# ── World (세계관) ──────────────────────────────────────────────

class MentionedCharacter(BaseModel):
    name: str = Field(min_length=1)               # 대표 이름 (가장 널리 알려진 것)
    aliases: list[str] = Field(default_factory=list)  # 소설 속에서 불리는 다른 이름/별명/직함


class WorldProfileData(BaseModel):
    world_summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    timeline_notes: list[str] = Field(default_factory=list)
    mentioned_characters: list[MentionedCharacter] = Field(default_factory=list)


class WorldCharacterRankingResult(BaseModel):
    mentioned_characters: list[MentionedCharacter]


class WorldSynthesisResult(BaseModel):
    world_summary: str



class WorldCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50)


class WorldRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    created_at: datetime


class WorldProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    world_id: str
    data: WorldProfileData
    version: int
    updated_at: datetime


class WorldDetailRead(WorldRead):
    profile: Optional[WorldProfileRead] = None


class WorldEditRequest(BaseModel):
    operation: str  # "add" | "delete" | "modify"
    instruction: str = Field(min_length=1)


class ExtractCharacterFromWorldRequest(BaseModel):
    name: str = Field(min_length=1)


# ── Export / Import (오프라인 공유) ──────────────────────────────

class CharacterExportData(BaseModel):
    export_type: str = "character"
    schema_version: int = 1
    name: str
    profile_data: CharacterProfileData
    exported_at: datetime


class CharacterImportRequest(BaseModel):
    export_type: str
    schema_version: int
    name: str = Field(min_length=1, max_length=50)
    profile_data: CharacterProfileData


class WorldExportData(BaseModel):
    export_type: str = "world"
    schema_version: int = 1
    name: str
    profile_data: WorldProfileData
    exported_at: datetime


class WorldImportRequest(BaseModel):
    export_type: str
    schema_version: int
    name: str = Field(min_length=1, max_length=50)
    profile_data: WorldProfileData
