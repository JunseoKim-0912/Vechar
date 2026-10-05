import json
from sqlalchemy.orm import Session
from ..llm import generate_structured
from ..models import CharacterProfile, CharacterProfileHistory, CorrectionLog, ChangeReason
from ..schemas import CharacterProfileData, CharacterSynthesisResult, TimelineEvent
from .character_timeline import derive_chat_reference, reconcile_timeline


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
    if next_data.timeline:
        ordered = reconcile_timeline([], next_data.timeline)
        reference = derive_chat_reference(ordered, next_data.chat_reference_point)
        if reference is not None:
            # The canonical persona is already synthesized for the reference point.
            # A newly ingested historical personality must not replace that persona.
            if next_data.personality_summary:
                reference.state.personality = next_data.personality_summary
            if next_data.speech_style:
                reference.state.speech_style = next_data.speech_style
        next_data = next_data.model_copy(update={
            "timeline": ordered,
            "chat_reference_point": reference,
        })
    elif next_data.chat_reference_point is not None:
        next_data = next_data.model_copy(update={"chat_reference_point": None})
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
입력 순서가 아닌 canonical chronology와 제공된 chat reference point를 우선하고, 근거가 모순되면
임의로 확정하지 말고 기존 정보를 함부로 버리지 마세요.
Do not treat narrative order as chronological order. A flashback age/state must not replace the latest
living canonical state. Synthesize personality and speech for the supplied chat reference point, not
for an earlier flashback or a post-death event.

반드시 아래 JSON으로만 응답하세요. 다른 텍스트 없이 순수 JSON만 출력합니다.
{"personality_summary": "...", "speech_style": "..."}

Language policy:
- The existing profile is canonical. Keep its dominant language, register, and writing style even if the new information is in another language.
- Understand English, Korean, and mixed-language input without omitting facts.
- Preserve established spellings for names and fictional terms. Do not create an English/Korean patchwork unless the canonical profile itself intentionally uses both."""


def merge_training_source(
    db: Session, user_id: str, character_id: str, newly_extracted: CharacterProfileData,
    source_id: str | None = None,
) -> CharacterProfile:
    """새 TrainingSource가 추출된 뒤 호출됩니다 (extraction_service.py 참고)."""
    existing_row = get_profile(db, character_id)
    existing_data = (
        CharacterProfileData.model_validate(existing_row.data) if existing_row else CharacterProfileData()
    )

    ordered_timeline = reconcile_timeline(
        existing_data.timeline, newly_extracted.timeline, source_id=source_id,
    )
    reference = derive_chat_reference(ordered_timeline, existing_data.chat_reference_point)
    current_relationships = (
        [f"{item.name}: {item.status}" for item in reference.state.relationships]
        if reference and reference.state.relationships else None
    )

    # Timeline reconciliation precedes persona synthesis. Historical arrays remain as evidence;
    # the chat prompt uses reference-point state instead of treating them as current facts.
    merged_arrays_only = CharacterProfileData(
        personality_summary=existing_data.personality_summary,
        speech_style=existing_data.speech_style,
        background_facts=_dedupe_merge(existing_data.background_facts, newly_extracted.background_facts),
        relationships=current_relationships or _dedupe_merge(existing_data.relationships, newly_extracted.relationships),
        sample_dialogues=_dedupe_merge(existing_data.sample_dialogues, newly_extracted.sample_dialogues),
        do_not_do=existing_data.do_not_do,  # 학습으로는 절대 안 바뀜, 정정으로만 바뀜
        timeline=ordered_timeline,
        chat_reference_point=reference,
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

    # An out-of-order prequel or post-death source adds historical evidence, not a new persona.
    if (existing_data.chat_reference_point and reference
            and existing_data.chat_reference_point.event_key == reference.event_key
            and newly_extracted.timeline
            and all(event.event_key != reference.event_key for event in newly_extracted.timeline)):
        return _snapshot_and_save(db, character_id, merged_arrays_only, ChangeReason.TRAINING_INGEST)

    # 두 번째 소스부터는 기존 요약과 새 요약을 자연스럽게 통합하도록 LLM에 위임합니다.
    synthesized = generate_structured(
        db=db,
        user_id=user_id,
        request_type="character_synthesis",
        task="analysis",
        instructions=SYNTHESIS_SYSTEM_PROMPT,
        input_messages=[
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
                        "chat_reference_point": {
                            "event_key": reference.event_key, "age": reference.age,
                            "status": reference.status, "summary": reference.summary,
                        } if reference else None,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        max_output_tokens=1500,
        response_model=CharacterSynthesisResult,
    )

    final_data = merged_arrays_only.model_copy(
        update={
            "personality_summary": synthesized.personality_summary or existing_data.personality_summary,
            "speech_style": synthesized.speech_style or existing_data.speech_style,
        }
    )
    return _snapshot_and_save(db, character_id, final_data, ChangeReason.TRAINING_INGEST)


CORRECTION_SYSTEM_PROMPT = """당신은 캐릭터 프로필 편집자입니다. 사용자가 대화 중 캐릭터가 자신이 생각하는 모습과 다르다고 느껴서
직접 정정 지시를 내렸습니다. 기존 프로필을 사용자의 지시에 맞게 수정하세요.

규칙:
- 사용자가 명시적으로 말하지 않은 부분은 최대한 그대로 유지하세요 (임의로 다른 부분을 바꾸지 마세요).
- 사용자가 "이런 말투/행동은 하지 않는다"고 하면 do_not_do 배열에 추가하세요.
- 제공된 Structured Outputs schema의 모든 필드를 반환하세요. timeline의 기존 event_key를 유지하고,
  시간 관련 정정은 해당 event의 age/date/state_changes를 함께 수정하세요. chat_reference_point는 서버가 재계산합니다.
- Do not treat narrative order as chronological order. Flashback ages and states are historical, not current.
- A correction of current age/role/relationship must be reflected in timeline and the latest living state together.

Language policy:
- Understand correction instructions in English, Korean, or both.
- Keep the existing canonical profile's dominant language and style unless the user explicitly asks to change that canonical language or style.
- Preserve established spellings of proper names and fictional terms."""


def apply_user_correction(db: Session, user_id: str, character_id: str, user_instruction: str) -> CharacterProfile:
    """오직 /수정 명령어 경로에서만 호출됩니다 — 일반 대화 턴에서는 절대 호출되지 않습니다."""
    existing_row = get_profile(db, character_id)
    existing_data = (
        CharacterProfileData.model_validate(existing_row.data) if existing_row else CharacterProfileData()
    )

    parsed = generate_structured(
        db=db,
        user_id=user_id,
        request_type="character_correction",
        task="analysis",
        instructions=CORRECTION_SYSTEM_PROMPT,
        input_messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {"existing_profile": existing_data.model_dump(), "user_instruction": user_instruction},
                    ensure_ascii=False,
                ),
            }
        ],
        max_output_tokens=8000,
        response_model=CharacterProfileData,
        input_policy="training",
    )
    timeline = reconcile_timeline(existing_data.timeline, parsed.timeline, prefer_new=True)
    reference = derive_chat_reference(timeline, existing_data.chat_reference_point)
    # A structured correction may explicitly supply a corrected current age even if it
    # failed to amend the event. Keep the timeline authoritative by amending that event.
    proposed = parsed.chat_reference_point
    if proposed and proposed.age is not None and (not reference or proposed.age != reference.age):
        if reference and reference.status != "deceased_in_canon":
            revised = [event.model_copy(deep=True) for event in timeline]
            for event in revised:
                if event.event_key == reference.event_key:
                    event.age = proposed.age
                    event.precision = "exact"
                    event.temporal_uncertainty = "User-corrected current age"
                    break
            timeline = reconcile_timeline([], revised)
        elif not timeline:
            timeline = reconcile_timeline([], [TimelineEvent(
                event_key="user-corrected-current-age", age=proposed.age, precision="exact",
                summary="User-corrected current age",
            )])
        reference = derive_chat_reference(timeline, reference)
    corrected = parsed.model_copy(update={"timeline": timeline, "chat_reference_point": reference})
    updated = _snapshot_and_save(db, character_id, corrected, ChangeReason.USER_CORRECTION)

    db.add(
        CorrectionLog(character_id=character_id, user_instruction=user_instruction, resulting_version=updated.version)
    )
    db.commit()

    return updated

def set_initial_profile(db: Session, character_id: str, data: CharacterProfileData) -> CharacterProfile:
    """외부(예: 세계관에서 캐릭터 추출) 호출자를 위한 공개 함수 — 내부적으로 _snapshot_and_save를 씀."""
    return _snapshot_and_save(db, character_id, data, ChangeReason.TRAINING_INGEST)
