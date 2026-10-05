from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth import get_current_user_id
from ..models import Character, CharacterProfileHistory, TrainingSource, SourceType, IngestStatus, ChangeReason
from ..schemas import (
    CharacterCreateRequest,
    CharacterRead,
    CharacterDetailRead,
    CharacterProfileRead,
    CharacterProfileData,
    CharacterExportData,
    CharacterImportRequest,
    RollbackRequest,
)
from ..services.extraction_service import extract_profile_from_text
from ..services.character_profile_service import merge_training_source, get_profile, set_initial_profile
from ..training_source import parse_training_request
from ..services.world_profile_service import get_or_create_default_world
from ..services import memory_service
from ..ownership import get_owned_world
from ..tier_limits import get_limits, check_and_log_export, check_import_quota, log_import

router = APIRouter()



@router.post("/", response_model=CharacterRead, status_code=201)
def create_character(
    payload: CharacterCreateRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    limits = get_limits(db, user_id)
    count = db.query(Character).filter(Character.user_id == user_id).count()
    if count >= limits["max_characters"]:
        raise HTTPException(
            status_code=403, detail=f"계정당 최대 {limits['max_characters']}개까지만 캐릭터를 만들 수 있습니다."
        )

    world_id = payload.world_id
    if world_id is None:
        world_id = get_or_create_default_world(db, user_id).id
    else:
        get_owned_world(db, world_id, user_id)

    character = Character(
        user_id=user_id, name=payload.name, profile_image_url=payload.profile_image_url, world_id=world_id
    )
    db.add(character)
    db.commit()
    db.refresh(character)
    return character


@router.post("/import", response_model=CharacterRead, status_code=201)
def import_character(
    payload: CharacterImportRequest, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)
):
    """다른 사람이 내보낸(export) 캐릭터 프로필을 그대로 가져옵니다 — LLM 호출 없이 즉시 사용 가능."""
    if payload.export_type != "character":
        raise HTTPException(status_code=400, detail="세계관 파일이 아니라 캐릭터 파일을 올려주세요.")
    if payload.schema_version != 1:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 버전입니다.")

    check_import_quota(db, user_id)

    limits = get_limits(db, user_id)
    count = db.query(Character).filter(Character.user_id == user_id).count()
    if count >= limits["max_characters"]:
        raise HTTPException(
            status_code=403, detail=f"계정당 최대 {limits['max_characters']}개까지만 캐릭터를 만들 수 있습니다."
        )

    character = Character(user_id=user_id, name=payload.name)
    db.add(character)
    db.commit()
    db.refresh(character)

    set_initial_profile(db, character.id, payload.profile_data)
    db.commit()
    db.refresh(character)

    log_import(db, user_id, "character", character.id)
    return character


@router.get("/", response_model=list[CharacterRead])
def list_characters(user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return db.query(Character).filter(Character.user_id == user_id).all()


@router.get("/{character_id}", response_model=CharacterDetailRead)
def get_character(character_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    character = db.query(Character).filter(Character.id == character_id, Character.user_id == user_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")

    profile_row = get_profile(db, character_id)
    detail = CharacterDetailRead.model_validate(character)
    if profile_row:
        detail.profile = CharacterProfileRead.model_validate(profile_row)
    return detail


@router.get("/{character_id}/export", response_model=CharacterExportData)
def export_character(character_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    """다른 사람에게 공유할 수 있는 파일 형태로 캐릭터 프로필을 내보냅니다 (프로필 사진은 포함되지 않음)."""
    character = db.query(Character).filter(Character.id == character_id, Character.user_id == user_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")

    check_and_log_export(db, user_id, "character", character_id)

    profile_row = get_profile(db, character_id)
    profile_data = CharacterProfileData.model_validate(profile_row.data) if profile_row else CharacterProfileData()

    return CharacterExportData(
        name=character.name,
        profile_data=profile_data,
        exported_at=datetime.now(timezone.utc),
    )


@router.delete("/{character_id}", status_code=204)
def delete_character(character_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    character = db.query(Character).filter(Character.id == character_id, Character.user_id == user_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")

    # Until a durable deletion intent/outbox exists, never delete the DB scope
    # first: a provider failure would otherwise strand external memory.
    memory_deletion = memory_service.delete_character_memories(user_id=user_id, character_id=character_id)
    if not memory_deletion.success:
        raise HTTPException(
            status_code=503 if memory_deletion.retryable else 502,
            detail={"code": "memory_deletion_failed", "retryable": memory_deletion.retryable},
        )

    db.delete(character)  # cascades to profile, training_sources, conversations, correction_logs (see models.py)
    db.commit()
    return None


@router.post("/{character_id}/training-sources", status_code=201)
async def upload_training_source(
    character_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    character = db.query(Character).filter(Character.id == character_id, Character.user_id == user_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")

    fields = await parse_training_request(request)
    try:
        source_type = SourceType(fields["source_type"])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail={"code": "malformed_training_request"}) from exc
    raw_text = fields["raw_text"]

    source = TrainingSource(
        character_id=character.id,
        source_type=source_type,
        raw_text=raw_text,
        char_count=len(raw_text),
        status=IngestStatus.PENDING,
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    try:
        existing_profile = get_profile(db, character.id)
        canonical_profile = (
            CharacterProfileData.model_validate(existing_profile.data) if existing_profile else None
        )
        extracted = extract_profile_from_text(
            db, user_id, raw_text, source_type, character.name, canonical_profile
        )
        source.extracted_data = extracted.model_dump()
        source.status = IngestStatus.EXTRACTED
        db.commit()

        updated_profile = merge_training_source(db, user_id, character.id, extracted, source_id=source.id)

        source.status = IngestStatus.MERGED
        db.commit()

        return {
            "source_id": source.id,
            "status": source.status,
            "profile": CharacterProfileRead.model_validate(updated_profile),
        }
    except HTTPException as e:
        source.status = IngestStatus.FAILED
        source.error_message = str(e.detail)
        db.commit()
        raise
    except Exception as e:
        source.status = IngestStatus.FAILED
        source.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"학습 데이터 처리 중 오류가 발생했습니다: {e}")


# --- Week 5: roll back a runaway merge or correction to a previous snapshot ---
@router.post("/{character_id}/profile/rollback", response_model=CharacterProfileRead)
def rollback_profile(
    character_id: str,
    payload: RollbackRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    character = db.query(Character).filter(Character.id == character_id, Character.user_id == user_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="캐릭터를 찾을 수 없습니다.")

    profile = get_profile(db, character_id)
    if not profile:
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")

    query = db.query(CharacterProfileHistory).filter(CharacterProfileHistory.character_profile_id == profile.id)
    if payload.to_version:
        query = query.filter(CharacterProfileHistory.version == payload.to_version)
    history_entry = query.order_by(CharacterProfileHistory.version.desc()).first()

    if not history_entry:
        raise HTTPException(status_code=404, detail="롤백할 이전 버전을 찾을 수 없습니다.")

    # Snapshot the current (about-to-be-discarded) state too, so a rollback is itself reversible.
    db.add(
        CharacterProfileHistory(
            character_profile_id=profile.id,
            data=profile.data,
            version=profile.version,
            change_reason=ChangeReason.MANUAL_EDIT,
        )
    )
    profile.data = history_entry.data
    profile.version += 1
    db.commit()
    db.refresh(profile)
    return profile
