from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy.orm import Session
from ..models import Character, Message, MessageRole, CorrectionLog
from ..llm import generate_text, make_chat_input_counter
from ..chat_context_config import ROLEPLAY_MAX_TOKENS
from ..schemas import CharacterProfileData, WorldProfileData
from .character_profile_service import get_profile, apply_user_correction
from .chat_prompt_builder import build_chat_instructions
from .chat_context_budget import select_chat_context
from . import memory_service, memory_jobs
from .training_queue import MEMORY_INGEST_TOPIC
from .world_profile_service import get_world_profile
from ..tier_limits import correction_quota_message
from ..ownership import get_owned_world
from .chat_latency import stage

CORRECTION_PREFIX = "/수정"



def send_message(
    db: Session,
    character_id: str,
    conversation_id: str,
    user_message: str,
    user_id: str,
    locale: str = "en",
) -> dict:
    with stage("character_profile_retrieval"):
        character = db.query(Character).filter(Character.id == character_id, Character.user_id == user_id).first()
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.world_id:
        get_owned_world(db, character.world_id, user_id)

    # Route explicit corrections to the guarded write path — never to the roleplay model.
    if user_message.strip().startswith(CORRECTION_PREFIX):
        instruction = user_message.strip()[len(CORRECTION_PREFIX):].strip()
        if not instruction:
            return {"role": "SYSTEM_NOTE", "content": f"사용할 형식: {CORRECTION_PREFIX} <바꾸고 싶은 내용 설명>"}

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = (
            db.query(CorrectionLog)
            .join(Character, CorrectionLog.character_id == Character.id)
            .filter(Character.user_id == user_id, CorrectionLog.created_at >= today_start)
            .count()
        )
        quota_message = correction_quota_message(db, user_id, today_count)
        if quota_message:
            return {
                "role": "SYSTEM_NOTE",
                "content": quota_message,
            }

        db.add(
            Message(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_message,
                is_correction_cmd=True,
            )
        )
        db.commit()

        updated = apply_user_correction(db, user_id, character_id, instruction)
        return {"role": "SYSTEM_NOTE", "content": f"캐릭터 프로필이 업데이트되었습니다 (v{updated.version})."}

    with stage("character_profile_retrieval"):
        profile_row = get_profile(db, character_id)  # READ ONLY — this module never writes.
        profile_data = (
            CharacterProfileData.model_validate(profile_row.data) if profile_row else CharacterProfileData()
        )

    world_data = None
    with stage("world_retrieval"):
        if character.world_id:
            world_row = get_world_profile(db, character.world_id)  # READ ONLY
            if world_row:
                world_data = WorldProfileData.model_validate(world_row.data)

    # BUGFIX: this used to be .order_by(asc()).limit(30), which returns the
    # OLDEST 30 messages once a conversation exceeds 30 — freezing context at
    # the start of the conversation instead of tracking recent turns. Fetch the
    # most recent 30 (desc + limit), then reverse back to chronological order.
    with stage("session_history"):
        history = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(30)
            .all()
        )
        history.reverse()

    with stage("memory_retrieval"):
        memory_result = memory_service.retrieve_for_turn(
            user_id=user_id, character_id=character_id,
            conversation_id=conversation_id, current_message=user_message,
        )

    user_turn = Message(conversation_id=conversation_id, role=MessageRole.USER, content=user_message)
    db.add(user_turn)
    db.commit()
    user_message_id = user_turn.id

    with stage("canonical_state_reconstruction"):
        # Prompt builder derives the lived temporal slice from the stored timeline.
        instructions = build_chat_instructions(
            character.name, profile_data, world_data, correction_prefix=CORRECTION_PREFIX, locale=locale,
            current_message=user_message, recent_messages=[(m.role, m.content) for m in history],
        )
    with stage("prompt_builder"):
        budget = select_chat_context(
            instructions,
            [(m.role, m.content) for m in history],
            user_message,
            memories=[candidate.content for candidate in memory_result.candidates],
            count_input_tokens=make_chat_input_counter(db, user_id, ROLEPLAY_MAX_TOKENS),
        )

    reply_text = generate_text(
        db=db,
        user_id=user_id,
        request_type="chat",
        task="chat",
        instructions=instructions,
        input_messages=budget.input_messages,
        max_output_tokens=ROLEPLAY_MAX_TOKENS,
    )

    assistant_turn = Message(conversation_id=conversation_id, role=MessageRole.CHARACTER, content=reply_text)
    db.add(assistant_turn)
    db.flush()
    ingestion = memory_jobs.schedule_ingestion(
        db, user_id=user_id, character_id=character_id, conversation_id=conversation_id,
        user_message_id=user_message_id, assistant_message_id=assistant_turn.id,
    )
    db.commit()
    memory_jobs.publish_safe(MEMORY_INGEST_TOPIC, ingestion.id if ingestion else None)

    return {"role": "CHARACTER", "content": reply_text}
