from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_
from ..database import get_db
from ..auth import get_current_user_id
from ..models import Character, CharacterProfileHistory, Conversation, SourceType, ChangeReason
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
from ..services.character_profile_service import get_profile, set_initial_profile
from ..services.training_jobs import cancel_target_jobs, submit_job
from ..services.training_queue import get_training_queue
from ..training_source import parse_training_request
from ..services.world_profile_service import get_or_create_default_world
from ..services import memory_jobs
from ..services.training_queue import MEMORY_DELETE_TOPIC
from ..ownership import get_owned_world
from ..tier_limits import check_entity_capacity, check_and_log_export, check_import_quota, log_import

router = APIRouter()



@router.post("/", response_model=CharacterRead, status_code=201)
def create_character(
    payload: CharacterCreateRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    check_entity_capacity(db, user_id, "character")

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

    check_entity_capacity(db, user_id, "character")

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

    cancel_target_jobs(db, user_id=user_id, target_type="character", target_id=character_id)
    deletion = memory_jobs.schedule_character_deletion(
        db, user_id=user_id, character_id=character_id,
    )
    # A two-character room is deleted with either participant. PostgreSQL FK
    # cascades are a backstop for non-HTTP deletions; this covers the route and
    # keeps SQLite/test behavior explicit too.
    rooms = db.query(Conversation).filter(
        Conversation.user_id == user_id, Conversation.kind == "character_pair",
        or_(Conversation.character_id == character_id,
            Conversation.secondary_character_id == character_id),
    ).all()
    for room in rooms:
        db.delete(room)
    db.delete(character)  # cascades to profile, training_sources, conversations, correction_logs (see models.py)
    db.commit()
    memory_jobs.publish_safe(MEMORY_DELETE_TOPIC, deletion.id if deletion else None)
    return None


@router.post("/{character_id}/training-sources", status_code=202)
async def upload_training_source(
    character_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    queue = Depends(get_training_queue),
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

    job = await submit_job(
        db, queue, user_id=user_id, target_type="character", target_id=character.id,
        source_type=fields["_source_type"], training_source_type=source_type.value,
        raw_text=raw_text,
    )
    return {"job_id": job.id, "status": job.status}


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
