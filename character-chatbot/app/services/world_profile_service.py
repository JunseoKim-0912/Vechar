import json
from sqlalchemy.orm import Session
from ..llm import generate_text, generate_structured
from ..models import World, WorldProfile, WorldProfileHistory, ChangeReason
from ..schemas import WorldProfileData, MentionedCharacter, WorldCharacterRankingResult, WorldSynthesisResult

# 가드레일 모듈 (World 버전)
# -----------------
# CharacterProfile과 동일한 원칙: WorldProfile.data를 쓸 수 있는 곳은 이 파일뿐입니다.
# 캐릭터 대화 로직(chat_service.py)은 get_world_profile()만 호출해서 읽기만 합니다.


def get_world_profile(db: Session, world_id: str) -> WorldProfile | None:
    return db.query(WorldProfile).filter(WorldProfile.world_id == world_id).first()


def set_initial_world_profile(db: Session, world_id: str, data: WorldProfileData) -> WorldProfile:
    """외부(예: import) 호출자를 위한 공개 함수 — 내부적으로 _snapshot_and_save를 씀."""
    return _snapshot_and_save(db, world_id, data, ChangeReason.TRAINING_INGEST)


def get_or_create_default_world(db: Session, user_id: str) -> World:
    """캐릭터 생성 시 world_id를 안 정해주면, 이 사용자의 "현실" 세계관에 자동 배정."""
    from ..models import User
    # Lock before testing existence so concurrent first-character requests
    # cannot each create a different persisted Reality world.
    db.query(User.id).filter(User.id == user_id).with_for_update().first()
    world = db.query(World).filter(World.user_id == user_id, World.name == "현실").first()
    if world:
        return world

    from ..tier_limits import check_entity_capacity
    check_entity_capacity(db, user_id, "world")

    world = World(user_id=user_id, name="현실")
    db.add(world)
    db.flush()

    db.add(
        WorldProfile(
            world_id=world.id,
            data=WorldProfileData(world_summary="우리가 살고 있는 현실 세계.").model_dump(),
            version=1,
        )
    )
    db.commit()
    db.refresh(world)
    return world


def _snapshot_and_save(
    db: Session, world_id: str, next_data: WorldProfileData, reason: ChangeReason,
    *, commit: bool = True,
) -> WorldProfile:
    existing = get_world_profile(db, world_id)

    if existing is None:
        profile = WorldProfile(world_id=world_id, data=next_data.model_dump(), version=1)
        db.add(profile)
        if commit:
            db.commit()
            db.refresh(profile)
        else:
            db.flush()
        return profile

    # 내용이 실질적으로 안 바뀌었으면 버전을 올리지 않고 그대로 반환 (히스토리 오염 방지)
    if WorldProfileData.model_validate(existing.data) == next_data:
        return existing

    db.add(
        WorldProfileHistory(
            world_profile_id=existing.id,
            data=existing.data,
            version=existing.version,
            change_reason=reason,
        )
    )
    existing.data = next_data.model_dump()
    existing.version += 1
    if commit:
        db.commit()
        db.refresh(existing)
    else:
        db.flush()
    return existing


def _dedupe_merge(old: list[str], new: list[str]) -> list[str]:
    seen = {s.strip() for s in old if s.strip()}
    result = list(old)
    for item in new:
        trimmed = item.strip()
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            result.append(trimmed)
    return result


CHARACTER_RANKING_SYSTEM_PROMPT = """당신은 세계관 편집자입니다. 세계관에 등장하는 인물 목록이 주어집니다.
각 항목은 {"name": "...", "aliases": ["..."]} 형태입니다. 같은 인물이 이름이나 별명이 겹치거나 사실상
같은 사람을 가리킨다면 하나로 합치고, 지금까지 확인된 모든 이름/별명을 aliases에 전부 모으세요 (중복 제거).
name은 그중 가장 널리 알려진 대표 이름 하나로 정하세요.

정리한 인물들을 이야기에서 비중이 큰 주연/중요 인물 순서로 정렬하고, 이름 없이 직함만 있는 스쳐가는
단역은 제외하세요. 최대 20명까지만 반환하세요.

반드시 아래 JSON으로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력합니다.
{"mentioned_characters": [{"name": "...", "aliases": ["..."]}, ...]}

Understand entries in any model-supported language. Prefer spellings already present in the earlier canonical entries, and never translate proper names merely to match an output language."""


def _rank_and_dedupe_characters(
    db: Session, user_id: str, existing: list[MentionedCharacter], new: list[MentionedCharacter]
) -> list[MentionedCharacter]:
    """이름/별명이 겹치는 인물을 하나로 합치고, 중요도 순으로 정렬해서 최대 20명까지만 남깁니다."""
    combined = [c.model_dump() for c in existing] + [c.model_dump() for c in new]
    if not combined:
        return []

    ranked = generate_structured(
        db=db,
        user_id=user_id,
        request_type="world_character_ranking",
        task="analysis",
        instructions=CHARACTER_RANKING_SYSTEM_PROMPT,
        input_messages=[{"role": "user", "content": json.dumps({"characters": combined}, ensure_ascii=False)}],
        max_output_tokens=1500,
        response_model=WorldCharacterRankingResult,
    )
    return ranked.mentioned_characters[:20]

WORLD_SYNTHESIS_SYSTEM_PROMPT = """당신은 세계관 편집자입니다. 기존 세계관 요약과 새로 추출된 정보가 주어지면,
둘을 자연스럽게 통합한 하나의 요약을 만듭니다. 서로 다른 시리즈/시기의 내용이 섞여 있어도 요약 안에서
시기 구분이 흐려지지 않도록 주의하세요.

반드시 아래 JSON으로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력합니다.
{"world_summary": "..."}

The existing summary is canonical. Preserve its dominant language, register, and style while understanding new information in any model-supported language. Preserve established proper-name and fictional-term spellings, and avoid an accidental bilingual patchwork."""
def merge_world_source(
    db: Session, user_id: str, world_id: str, newly_extracted: WorldProfileData,
    synthesized_summary: WorldSynthesisResult | None = None,
    *, commit: bool = True,
) -> WorldProfile:
    """새 WorldSource(화/설명)가 추출된 뒤 호출됩니다."""
    existing_row = get_world_profile(db, world_id)
    existing_data = (
        WorldProfileData.model_validate(existing_row.data) if existing_row else WorldProfileData()
    )

    cleaned_characters = _rank_and_dedupe_characters(
        db, user_id, existing_data.mentioned_characters, newly_extracted.mentioned_characters
    )

    merged_arrays_only = WorldProfileData(
        world_summary=existing_data.world_summary,
        key_facts=_dedupe_merge(existing_data.key_facts, newly_extracted.key_facts),
        timeline_notes=_dedupe_merge(existing_data.timeline_notes, newly_extracted.timeline_notes),
        mentioned_characters=cleaned_characters,
    )

    if not existing_data.world_summary:
        initial = merged_arrays_only.model_copy(update={
            "world_summary": synthesized_summary.world_summary if synthesized_summary else newly_extracted.world_summary,
        })
        return _snapshot_and_save(db, world_id, initial, ChangeReason.TRAINING_INGEST, commit=commit)

    synthesized = synthesized_summary or generate_structured(
        db=db,
        user_id=user_id,
        request_type="world_synthesis",
        task="analysis",
        instructions=WORLD_SYNTHESIS_SYSTEM_PROMPT,
        input_messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "existing_summary": existing_data.world_summary,
                        "newly_extracted_summary": newly_extracted.world_summary,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        max_output_tokens=1500,
        response_model=WorldSynthesisResult,
    )

    final_data = merged_arrays_only.model_copy(
        update={"world_summary": synthesized.world_summary or existing_data.world_summary}
    )
    return _snapshot_and_save(db, world_id, final_data, ChangeReason.TRAINING_INGEST, commit=commit)

WORLD_EDIT_SYSTEM_PROMPTS = {
    "add": """당신은 세계관 편집자입니다. 사용자가 세계관에 새 정보를 "추가"하라고 요청했습니다.
기존 세계관 정보는 그대로 유지하면서, 사용자가 말한 내용을 알맞은 필드에 추가하세요. 기존 내용을 지우지 마세요.""",
    "delete": """당신은 세계관 편집자입니다. 사용자가 세계관에서 특정 정보를 "삭제"하라고 요청했습니다.
사용자가 지목한 정보만 제거하고, 나머지는 그대로 유지하세요.""",
    "modify": """당신은 세계관 편집자입니다. 사용자가 세계관의 특정 정보를 "수정"하라고 요청했습니다.
사용자가 지목한 부분만 새 내용으로 바꾸고, 나머지는 그대로 유지하세요.""",
}

WORLD_EDIT_JSON_SPEC = """
반드시 아래와 동일한 JSON 스키마로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력합니다.
{
  "world_summary": "...",
  "key_facts": ["..."],
  "timeline_notes": ["..."],
  "mentioned_characters": [{"name": "...", "aliases": ["..."]}]
}

Language policy: understand instructions in any model-supported language; keep the existing canonical world's dominant language and style unless the user explicitly requests a language/style change; preserve established proper-name spellings."""


def apply_world_edit(db: Session, user_id: str, world_id: str, operation: str, instruction: str) -> WorldProfile:
    """세계관 관리 화면에서만 호출됩니다 (add/delete/modify). operation은 반드시 이 셋 중 하나."""
    if operation not in WORLD_EDIT_SYSTEM_PROMPTS:
        raise ValueError(f"Unknown world edit operation: {operation}")

    existing_row = get_world_profile(db, world_id)
    existing_data = (
        WorldProfileData.model_validate(existing_row.data) if existing_row else WorldProfileData()
    )

    parsed = generate_structured(
        db=db,
        user_id=user_id,
        request_type="world_edit",
        task="analysis",
        instructions=WORLD_EDIT_SYSTEM_PROMPTS[operation] + WORLD_EDIT_JSON_SPEC,
        input_messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {"existing_world_profile": existing_data.model_dump(), "user_instruction": instruction},
                    ensure_ascii=False,
                ),
            }
        ],
        max_output_tokens=8000,  # 세계관 프로필 전체를 다시 써야 하므로 여유 있게 (character 쪽과 같은 이유)
        response_model=WorldProfileData,
    )

    return _snapshot_and_save(db, world_id, parsed, ChangeReason.WORLD_EDIT)


WORLD_SUMMARY_SYSTEM_PROMPT = """당신은 세계관 안내자입니다. 주어진 구조화된 세계관 정보를 사람이 읽기 좋은
자연스러운 문단 몇 개로 요약해서 설명하세요. 목록을 나열하듯 말하지 말고, 이야기하듯 풀어서 설명하세요.
입력 canonical profile의 주 언어와 표현 스타일을 유지하고, 고유명사 표기를 바꾸지 마세요."""


def summarize_world(db: Session, user_id: str, world_id: str) -> str:
    """Return 작업 — 지금까지 알고 있는 세계관 정보를 사람이 읽기 좋게 요약."""
    profile_row = get_world_profile(db, world_id)
    if not profile_row:
        return "아직 이 세계관에 대해 알고 있는 정보가 없습니다."

    data = WorldProfileData.model_validate(profile_row.data)
    response_text = generate_text(
        db=db,
        user_id=user_id,
        request_type="world_summary",
        task="analysis",
        instructions=WORLD_SUMMARY_SYSTEM_PROMPT,
        input_messages=[{"role": "user", "content": json.dumps(data.model_dump(), ensure_ascii=False)}],
        max_output_tokens=1000,
    )
    return response_text


COMPACT_SYSTEM_PROMPT = """당신은 세계관 편집자입니다. key_facts와 timeline_notes 배열이 화를 거듭 학습하면서
너무 길고 중복이 많아졌습니다. 같은 내용을 가리키는 항목들을 하나로 합치고, 사소하거나 중복된 항목은
제거해서 더 짧고 밀도 높은 목록으로 압축하세요. 단, 시리즈/시기 구분(timeline_notes의 대괄호 표시)은
반드시 유지하세요 — 서로 다른 시기의 정보를 하나로 뭉개면 안 됩니다. world_summary와 mentioned_characters는
그대로 유지하세요 (mentioned_characters는 별도로 정리됩니다).

반드시 아래와 동일한 JSON 스키마로만 응답하세요.
{
  "world_summary": "...",
  "key_facts": ["..."],
  "timeline_notes": ["..."],
  "mentioned_characters": [{"name": "...", "aliases": ["..."]}]
}

Understand content in any model-supported language. Keep the existing canonical profile's dominant language and style, preserve established proper names and fictional terms, and do not create a bilingual patchwork while compacting."""


def compact_world_profile(db: Session, user_id: str, world_id: str) -> WorldProfile:
    """세계관이 여러 시리즈/화를 거치며 커졌을 때, 쌓인 정보를 중복 없이 압축. 사용자가 명시적으로 요청할 때만 호출.
    mentioned_characters는 이 김에 이름/별명 기준으로도 다시 한번 확실하게 정리합니다."""
    existing_row = get_world_profile(db, world_id)
    if not existing_row:
        raise ValueError("압축할 세계관 프로필이 없습니다.")
    existing_data = WorldProfileData.model_validate(existing_row.data)

    parsed = generate_structured(
        db=db,
        user_id=user_id,
        request_type="world_compaction",
        task="analysis",
        instructions=COMPACT_SYSTEM_PROMPT,
        input_messages=[{"role": "user", "content": json.dumps(existing_data.model_dump(), ensure_ascii=False)}],
        max_output_tokens=8000,
        response_model=WorldProfileData,
    )

    # mentioned_characters는 COMPACT_SYSTEM_PROMPT의 결과를 믿지 않고, 전담 함수로 다시 한번 확실하게 정리
    re_ranked = _rank_and_dedupe_characters(db, user_id, existing_data.mentioned_characters, [])
    parsed = parsed.model_copy(update={"mentioned_characters": re_ranked})
    return _snapshot_and_save(db, world_id, parsed, ChangeReason.WORLD_EDIT)


