from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from fastapi import HTTPException
from sqlalchemy.orm import Session
from ..models import Character, Conversation, Message, MessageRole, CorrectionLog, LLMUsage
from ..llm import generate_chat_turn as generate_text, make_chat_input_counter
from ..llm_failures import LLMStructuredOutputError, classify_llm_failure
from ..llm_config import model_for_task
from ..llm_operation import LLMOperation, llm_operation
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
from .chat_latency import set_safe_metadata, stage
from .chat_language import response_language
from .conversation_loop_guard import is_obvious_loop
from .response_quality import explicit_repetition_request, select_quality_response
from .conversation_runtime import ConversationRuntimeState, advance_runtime, turn_output_instructions

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
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id,
                                                      Conversation.user_id == user_id).first()
        runtime = ConversationRuntimeState.from_storage(
            conversation.runtime_state if conversation else None, history,
        )
        set_safe_metadata(conversation_runtime_version=runtime.version,
                          open_thread_count=len(runtime.open_threads),
                          resolved_thread_count=len(runtime.resolved_threads))

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
        instructions += runtime.prompt_block(character_id) + turn_output_instructions()
    language = response_language(user_message, [(m.role, m.content) for m in history], locale)
    set_safe_metadata(conversation_type="user_character", response_language=language,
                      conversation_id=conversation_id)
    counter = make_chat_input_counter(db, user_id, ROLEPLAY_MAX_TOKENS)
    prior_history = [(m.role, m.content) for m in history]
    with stage("prompt_builder"):
        budget = select_chat_context(
            instructions,
            prior_history,
            user_message,
            memories=[candidate.content for candidate in memory_result.candidates],
            count_input_tokens=counter,
        )

    token = str(uuid4())

    def generate(attempt: int, instruction_text: str, input_messages: list[dict]):
        operation = LLMOperation(job_id=conversation_id, stage="reply", attempt=attempt,
                                 chunk_key=token, operation_type="user_chat")
        with llm_operation(operation):
            try:
                return generate_text(
                    db=db, user_id=user_id, request_type="chat", task="chat",
                    instructions=instruction_text, input_messages=input_messages,
                    max_output_tokens=ROLEPLAY_MAX_TOKENS,
                )
            except Exception as exc:
                set_safe_metadata(provider_error_category=classify_llm_failure(exc).kind.value)
                raise

    schema_retried = False
    try:
        first_response = generate(1, instructions, budget.input_messages)
    except LLMStructuredOutputError:
        schema_retried = True
        first_response = generate(2, instructions + "\nReturn the required structured turn fields exactly.",
                                  budget.input_messages)
    same_speaker_recent = [m.content for m in history if m.role == MessageRole.CHARACTER][-5:]
    previous_user = [m.content for m in history if m.role == MessageRole.USER][-5:]
    allow_repetition = explicit_repetition_request(user_message) or any(
        user_message.strip().casefold() == previous.strip().casefold()
        or is_obvious_loop(user_message, [previous]) for previous in previous_user
    )

    def retry(correction: str):
        retry_instructions = instructions + correction
        with stage("prompt_builder"):
            retry_budget = select_chat_context(
                retry_instructions, prior_history, user_message,
                memories=[candidate.content for candidate in memory_result.candidates],
                count_input_tokens=counter,
            )
        return generate(2, retry_instructions, retry_budget.input_messages)

    outcome = select_quality_response(
        first_response, same_speaker_recent=same_speaker_recent,
        language=language, allow_repetition=allow_repetition, retry=retry,
        runtime_state=runtime, speaker=character_id, room=False,
        retry_allowed=not schema_retried,
    )
    set_safe_metadata(**outcome.safe_metadata("user_character", language))
    set_safe_metadata(retry_count=max(int(schema_retried), outcome.retry_count),
                      corrective_retry=bool(schema_retried or outcome.retry_count))
    reply_text = outcome.text

    assistant_turn = Message(conversation_id=conversation_id, role=MessageRole.CHARACTER, content=reply_text)
    db.add(assistant_turn)
    db.flush()
    advance_runtime(runtime, outcome.turn, reply_text, character_id, assistant_turn.id)
    if conversation:
        conversation.runtime_state = runtime.to_storage()
    ingestion = memory_jobs.schedule_ingestion(
        db, user_id=user_id, character_id=character_id, conversation_id=conversation_id,
        user_message_id=user_message_id, assistant_message_id=assistant_turn.id,
    )
    db.commit()
    memory_jobs.publish_safe(MEMORY_INGEST_TOPIC, ingestion.id if ingestion else None)

    keys = [LLMOperation(job_id=conversation_id, stage="reply", attempt=attempt,
                         chunk_key=token, operation_type="user_chat").key("chat")
            for attempt in range(1, max(int(schema_retried), outcome.retry_count) + 2)]
    usage = db.query(LLMUsage).filter(LLMUsage.user_id == user_id,
                                     LLMUsage.operation_key.in_(keys)).all()
    set_safe_metadata(model=model_for_task("chat"),
                      input_tokens=sum(row.input_tokens for row in usage),
                      output_tokens=sum(row.output_tokens for row in usage),
                      estimated_cost_usd=str(sum(
                          (row.estimated_cost_usd or Decimal(0) for row in usage), Decimal(0),
                      )))

    return {"role": "CHARACTER", "content": reply_text}
