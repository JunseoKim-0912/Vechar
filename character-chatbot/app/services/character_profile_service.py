import json
from sqlalchemy.orm import Session
from ..llm import client, MODELS, extract_text
from ..models import CharacterProfile, CharacterProfileHistory, CorrectionLog, ChangeReason
from ..schemas import CharacterProfileData


# 가드레일 모듈
# -----------------
# 이 파일이 CharacterProfile.data를 쓸 수 있는 코드베이스 내 유일한 곳입니다.
# chat_service.py는 get_profile()만 호출합니다 — merge_training_source나
# apply_user_correction은 아예 import하지 않습니다.


def get_profile(db: Session, character_id: str) -> CharacterProfile | None:
    return db.query(CharacterProfile).filter(CharacterProfile.character_id == character_id).first()


def _snapshot_and_save(
    db: Session, character_id: str, next_data: CharacterProfileData, reason: ChangeReason
) -> CharacterProfile:
    existing = get_profile(db, character_id)

    if existing is None:
        profile = CharacterProfile(character_id=character_id, data=next_data.model_dump(), version=1)
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return profile
    # 내용이 실질적으로 안 바뀌었으면 버전을 올리지 않고 그대로 반환 (히스토리 오염 방지)
    if CharacterProfileData.model_validate(existing.data) == next_data:
        return existing

    db.add(
        CharacterProfileHistory(
            character_profile_id=existing.id,
            data=existing.data,
            version=existing.version,
            change_reason=reason,
        )
    )
    existing.data = next_data.model_dump()
    existing.version += 1
    db.commit()
    db.refresh(existing)
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


SYNTHESIS_SYSTEM_PROMPT = """당신은 캐릭터 프로필 편집자입니다. 기존 캐릭터 프로필과 새로 추출된 정보가 주어지면,
둘을 자연스럽게 통합한 최종 프로필을 만듭니다. 기존 정보와 새 정보가 충돌하면(성격이나 말투가 다르게 묘사되는 경우),
더 최근에 제공된 정보(새 정보)를 우선하되 기존 정보를 함부로 버리지 마세요.

반드시 아래 JSON으로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력합니다.
{"personality_summary": "...", "speech_style": "..."}"""


def merge_training_source(db: Session, character_id: str, newly_extracted: CharacterProfileData) -> CharacterProfile:
    """새 TrainingSource가 추출된 뒤 호출됩니다 (extraction_service.py 참고)."""
    existing_row = get_profile(db, character_id)
    existing_data = (
        CharacterProfileData.model_validate(existing_row.data) if existing_row else CharacterProfileData()
    )

    # 배열 필드는 LLM 호출 없이 결정적으로 병합됩니다.
    merged_arrays_only = CharacterProfileData(
        personality_summary=existing_data.personality_summary,
        speech_style=existing_data.speech_style,
        background_facts=_dedupe_merge(existing_data.background_facts, newly_extracted.background_facts),
        relationships=_dedupe_merge(existing_data.relationships, newly_extracted.relationships),
        sample_dialogues=_dedupe_merge(existing_data.sample_dialogues, newly_extracted.sample_dialogues),
        do_not_do=existing_data.do_not_do,  # 학습으로는 절대 안 바뀜, 정정으로만 바뀜
    )

    # 첫 학습 소스라면 합성할 필요 없이 새 값을 그대로 사용합니다.
    if not existing_data.personality_summary and not existing_data.speech_style:
        initial = merged_arrays_only.model_copy(
            update={
                "personality_summary": newly_extracted.personality_summary,
                "speech_style": newly_extracted.speech_style,
            }
        )
        return _snapshot_and_save(db, character_id, initial, ChangeReason.TRAINING_INGEST)

    # 두 번째 소스부터는 기존 요약과 새 요약을 자연스럽게 통합하도록 LLM에 위임합니다.
    response = client.messages.create(
        model=MODELS["extraction"],
        max_tokens=1500,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "existing_profile": {
                            "personality_summary": existing_data.personality_summary,
                            "speech_style": existing_data.speech_style,
                        },
                        "newly_extracted": {
                            "personality_summary": newly_extracted.personality_summary,
                            "speech_style": newly_extracted.speech_style,
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        ],
    )
    synthesized = json.loads(extract_text(response).strip().removeprefix("```json").removesuffix("```").strip())

    final_data = merged_arrays_only.model_copy(
        update={
            "personality_summary": synthesized.get("personality_summary") or existing_data.personality_summary,
            "speech_style": synthesized.get("speech_style") or existing_data.speech_style,
        }
    )
    return _snapshot_and_save(db, character_id, final_data, ChangeReason.TRAINING_INGEST)


CORRECTION_SYSTEM_PROMPT = """당신은 캐릭터 프로필 편집자입니다. 사용자가 대화 중 캐릭터가 자신이 생각하는 모습과 다르다고 느껴서
직접 정정 지시를 내렸습니다. 기존 프로필을 사용자의 지시에 맞게 수정하세요.

규칙:
- 사용자가 명시적으로 말하지 않은 부분은 최대한 그대로 유지하세요 (임의로 다른 부분을 바꾸지 마세요).
- 사용자가 "이런 말투/행동은 하지 않는다"고 하면 do_not_do 배열에 추가하세요.
- 반드시 아래와 동일한 JSON 스키마로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력합니다.
{
  "personality_summary": "...",
  "speech_style": "...",
  "background_facts": ["..."],
  "relationships": ["..."],
  "sample_dialogues": ["..."],
  "do_not_do": ["..."]
}"""


def apply_user_correction(db: Session, character_id: str, user_instruction: str) -> CharacterProfile:
    """오직 /수정 명령어 경로에서만 호출됩니다 — 일반 대화 턴에서는 절대 호출되지 않습니다."""
    existing_row = get_profile(db, character_id)
    existing_data = (
        CharacterProfileData.model_validate(existing_row.data) if existing_row else CharacterProfileData()
    )

    response = client.messages.create(
        model=MODELS["extraction"],
        max_tokens=4000,
        system=CORRECTION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {"existing_profile": existing_data.model_dump(), "user_instruction": user_instruction},
                    ensure_ascii=False,
                ),
            }
        ],
    )
    parsed = CharacterProfileData.model_validate(
        json.loads(extract_text(response).strip().removeprefix("```json").removesuffix("```").strip())
    )

    updated = _snapshot_and_save(db, character_id, parsed, ChangeReason.USER_CORRECTION)

    db.add(
        CorrectionLog(character_id=character_id, user_instruction=user_instruction, resulting_version=updated.version)
    )
    db.commit()

    return updated

def set_initial_profile(db: Session, character_id: str, data: CharacterProfileData) -> CharacterProfile:
    """외부(예: 세계관에서 캐릭터 추출) 호출자를 위한 공개 함수 — 내부적으로 _snapshot_and_save를 씀."""
    return _snapshot_and_save(db, character_id, data, ChangeReason.TRAINING_INGEST)