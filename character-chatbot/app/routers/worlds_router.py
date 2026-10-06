from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth import get_current_user_id
from ..models import World, WorldSource, WorldSourceType, IngestStatus, Character, TrainingSource, SourceType
from ..schemas import (
    WorldCreateRequest,
    WorldRead,
    WorldDetailRead,
    WorldProfileRead,
    WorldProfileData,
    WorldEditRequest,
    ExtractCharacterFromWorldRequest,
    CharacterRead,
    WorldExportData,
    WorldImportRequest,
)
from ..services.training_jobs import cancel_target_jobs, submit_job
from ..services.training_queue import get_training_queue
from ..services.world_profile_service import (
    get_world_profile,
    apply_world_edit,
    summarize_world,
    compact_world_profile,
    set_initial_world_profile,
)
from ..services.character_extraction_service import extract_character_from_world_text
from ..services.character_profile_service import set_initial_profile
from ..training_source import parse_training_request
from ..tier_limits import check_entity_capacity, check_and_log_export, check_import_quota, log_import
from ..schemas import WorldProfileData, MentionedCharacter

router = APIRouter()

VALID_OPERATIONS = {"add", "delete", "modify"}


@router.post("/", response_model=WorldRead, status_code=201)
def create_world(payload: WorldCreateRequest, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    check_entity_capacity(db, user_id, "world")

    world = World(user_id=user_id, name=payload.name)
    db.add(world)
    db.commit()
    db.refresh(world)
    return world


@router.post("/import", response_model=WorldRead, status_code=201)
def import_world(
    payload: WorldImportRequest, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)
):
    """다른 사람이 내보낸(export) 세계관 프로필을 그대로 가져옵니다 — LLM 호출 없이 즉시 사용 가능."""
    if payload.export_type != "world":
        raise HTTPException(status_code=400, detail="캐릭터 파일이 아니라 세계관 파일을 올려주세요.")
    if payload.schema_version != 1:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 버전입니다.")

    check_import_quota(db, user_id)

    check_entity_capacity(db, user_id, "world")

    world = World(user_id=user_id, name=payload.name)
    db.add(world)
    db.commit()
    db.refresh(world)

    set_initial_world_profile(db, world.id, payload.profile_data)
    db.commit()
    db.refresh(world)

    log_import(db, user_id, "world", world.id)
    return world


@router.get("/", response_model=list[WorldRead])
def list_worlds(user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return db.query(World).filter(World.user_id == user_id).all()


@router.get("/{world_id}", response_model=WorldDetailRead)
def get_world(world_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")

    profile_row = get_world_profile(db, world_id)
    detail = WorldDetailRead.model_validate(world)
    if profile_row:
        detail.profile = WorldProfileRead.model_validate(profile_row)
    return detail


@router.get("/{world_id}/export", response_model=WorldExportData)
def export_world(world_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    """다른 사람에게 공유할 수 있는 파일 형태로 세계관 프로필을 내보냅니다."""
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")

    check_and_log_export(db, user_id, "world", world_id)

    profile_row = get_world_profile(db, world_id)
    profile_data = WorldProfileData.model_validate(profile_row.data) if profile_row else WorldProfileData()

    return WorldExportData(
        name=world.name,
        profile_data=profile_data,
        exported_at=datetime.now(timezone.utc),
    )


@router.delete("/{world_id}", status_code=204)
def delete_world(world_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")

    # cascades to profile/history/sources (models.py). Characters linked to this
    # world are NOT deleted — their world_id just becomes NULL (ondelete="SET NULL").
    cancel_target_jobs(db, user_id=user_id, target_type="world", target_id=world_id)
    db.delete(world)
    db.commit()
    return None


@router.post("/{world_id}/sources", status_code=202)
async def upload_world_source(
    world_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    queue = Depends(get_training_queue),
):
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")

    fields = await parse_training_request(request)
    try:
        source_type = WorldSourceType(fields["source_type"])
        series_name = fields.get("series_name")
        if series_name is not None and not isinstance(series_name, str):
            raise TypeError("series_name must be text")
        episode_number = int(fields["episode_number"]) if fields.get("episode_number") else None
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail={"code": "malformed_training_request"}) from exc
    raw_text = fields["raw_text"]

    job = await submit_job(
        db, queue, user_id=user_id, target_type="world", target_id=world.id,
        source_type=fields["_source_type"], training_source_type=source_type.value,
        raw_text=raw_text, series_name=series_name, episode_number=episode_number,
    )
    return {"job_id": job.id, "status": job.status}


@router.get("/{world_id}/summary")
def get_world_summary(world_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")
    return {"summary": summarize_world(db, user_id, world_id)}


@router.post("/{world_id}/edit", response_model=WorldProfileRead)
def edit_world(
    world_id: str,
    payload: WorldEditRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")
    if payload.operation not in VALID_OPERATIONS:
        raise HTTPException(status_code=400, detail=f"operation은 {sorted(VALID_OPERATIONS)} 중 하나여야 합니다.")

    try:
        return apply_world_edit(db, user_id, world_id, payload.operation, payload.instruction)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세계관 편집 중 오류가 발생했습니다: {e}")


@router.post("/{world_id}/compact", response_model=WorldProfileRead)
def compact_world(world_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    """세계관이 여러 화를 거치며 key_facts/timeline_notes가 너무 길어졌을 때 압축."""
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")

    try:
        return compact_world_profile(db, user_id, world_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세계관 압축 중 오류가 발생했습니다: {e}")


@router.post("/{world_id}/extract-character", response_model=CharacterRead, status_code=201)
def extract_character_from_world(
    world_id: str,
    payload: ExtractCharacterFromWorldRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """세계관 텍스트에서 특정 인물(mentioned_characters 중 하나)에 대한 내용만 뽑아 캐릭터를 만듭니다."""
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if not world:
        raise HTTPException(status_code=404, detail="World not found")

    check_entity_capacity(db, user_id, "character")

    # mentioned_characters에 저장된 별명이 있으면, 대표 이름뿐 아니라 별명으로 등장한 화도 같이 찾습니다.
    profile_row = get_world_profile(db, world_id)
    search_terms = [payload.name]
    if profile_row:
        profile_data = WorldProfileData.model_validate(profile_row.data)
        for mc in profile_data.mentioned_characters:
            if mc.name == payload.name:
                search_terms.extend(mc.aliases)
                break

    sources = db.query(WorldSource).filter(WorldSource.world_id == world_id).all()
    matching_texts = [s.raw_text for s in sources if any(term in s.raw_text for term in search_terms)]
    if not matching_texts:
        raise HTTPException(status_code=404, detail=f"'{payload.name}'이(가) 언급된 세계관 텍스트를 찾지 못했습니다.")

    try:
        extracted = extract_character_from_world_text(db, user_id, payload.name, matching_texts)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐릭터 추출 중 오류가 발생했습니다: {e}")

    character = Character(user_id=user_id, name=payload.name, world_id=world_id)
    db.add(character)
    db.commit()
    db.refresh(character)

    training_source = TrainingSource(
        character_id=character.id,
        source_type=SourceType.WORLD_DERIVED,
        raw_text=f"[세계관 '{world.name}'에서 자동 추출됨]",
        char_count=0,
        extracted_data=extracted.model_dump(),
        status=IngestStatus.MERGED,
    )
    db.add(training_source)
    db.flush()
    for event in extracted.timeline:
        event.source_ids = list(dict.fromkeys(event.source_ids + [training_source.id]))
    training_source.extracted_data = extracted.model_dump()
    set_initial_profile(db, character.id, extracted)
    db.commit()

    return character
