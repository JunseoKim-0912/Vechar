"""Durable two-character rooms sharing the existing Conversation/Message ledger."""

from datetime import timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from ..chat_context_config import ROLEPLAY_MAX_TOKENS
from ..llm import generate_chat_turn as generate_text, make_chat_input_counter
from ..llm_config import model_for_task
from ..llm_failures import LLMStructuredOutputError, classify_llm_failure
from ..llm_operation import LLMOperation, llm_operation
from ..models import Character, Conversation, LLMUsage, Message, MessageRole, utcnow
from ..schemas import CharacterProfileData, WorldProfileData
from .character_profile_service import get_profile
from .chat_actions import normalize_assistant_actions
from .chat_context_budget import select_chat_context
from .chat_latency import set_safe_metadata, stage
from .chat_prompt_builder import build_chat_instructions
from .response_quality import select_quality_response
from .conversation_runtime import ConversationRuntimeState, advance_runtime, turn_output_instructions
from .knowledge_provenance import KnowledgeProvenance
from .character_fidelity import CharacterFidelityContract, build_fidelity_contract
from .world_profile_service import get_world_profile

ROOM_KIND = "character_pair"
ROOM_LEASE = timedelta(minutes=20)
ROOM_HISTORY_LIMIT = 30
REQUEST_TYPE = "character_conversation"


def _owned_room(db: Session, room_id: str, user_id: str) -> Conversation:
    room = db.query(Conversation).filter(
        Conversation.id == room_id, Conversation.user_id == user_id,
        Conversation.kind == ROOM_KIND,
    ).first()
    if room is None:
        raise HTTPException(status_code=404, detail="Character conversation not found")
    return room


def _participants(db: Session, room: Conversation, user_id: str) -> tuple[Character, Character]:
    characters = db.query(Character).filter(
        Character.id.in_([room.character_id, room.secondary_character_id]),
        Character.user_id == user_id,
    ).all()
    by_id = {character.id: character for character in characters}
    if room.character_id not in by_id or room.secondary_character_id not in by_id:
        raise HTTPException(status_code=404, detail="Room participant not found")
    return by_id[room.character_id], by_id[room.secondary_character_id]


def _room_data(db: Session, room: Conversation, user_id: str) -> dict:
    first, second = _participants(db, room, user_id)
    return {
        "id": room.id, "name": room.name, "language": room.language,
        "participants": [
            {"id": first.id, "name": first.name, "profile_image_url": first.profile_image_url, "position": 0},
            {"id": second.id, "name": second.name, "profile_image_url": second.profile_image_url, "position": 1},
        ],
        "turn_index": room.turn_index,
        "next_speaker_character_id": (first if room.turn_index % 2 == 0 else second).id,
        "created_at": room.created_at, "updated_at": room.updated_at,
    }


def create_room(db: Session, user_id: str, first_id: str, second_id: str,
                name: str, language: str) -> dict:
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail={"code": "room_name_required"})
    if first_id == second_id:
        raise HTTPException(status_code=422, detail={"code": "duplicate_room_participant"})
    owned = db.query(Character.id).filter(
        Character.id.in_([first_id, second_id]), Character.user_id == user_id,
    ).all()
    if len(owned) != 2:
        raise HTTPException(status_code=404, detail="Room participant not found")
    room = Conversation(user_id=user_id, character_id=first_id,
                        secondary_character_id=second_id, kind=ROOM_KIND,
                        name=name, language=language, turn_index=0)
    db.add(room)
    db.commit()
    db.refresh(room)
    return _room_data(db, room, user_id)


def list_rooms(db: Session, user_id: str) -> list[dict]:
    rooms = db.query(Conversation).filter(
        Conversation.user_id == user_id, Conversation.kind == ROOM_KIND,
    ).order_by(Conversation.updated_at.desc(), Conversation.id.desc()).all()
    return [_room_data(db, room, user_id) for room in rooms]


def get_room(db: Session, room_id: str, user_id: str) -> dict:
    return _room_data(db, _owned_room(db, room_id, user_id), user_id)


def list_messages(db: Session, room_id: str, user_id: str) -> list[Message]:
    _owned_room(db, room_id, user_id)
    return db.query(Message).filter(Message.conversation_id == room_id).order_by(
        Message.turn_index.asc(), Message.id.asc(),
    ).all()


def rename_room(db: Session, room_id: str, user_id: str, name: str) -> dict:
    room = _owned_room(db, room_id, user_id)
    room.name = name.strip()
    if not room.name:
        raise HTTPException(status_code=422, detail={"code": "room_name_required"})
    room.updated_at = utcnow()
    db.commit()
    db.refresh(room)
    return _room_data(db, room, user_id)


def delete_room(db: Session, room_id: str, user_id: str) -> None:
    room = _owned_room(db, room_id, user_id)
    db.delete(room)  # Existing Message cascade covers every room message.
    db.commit()


def _claim_turn(db: Session, room_id: str, user_id: str, expected: int, token: str) -> None:
    now = utcnow()
    prior = db.query(Conversation.generation_token, Conversation.generation_lease_expires_at).filter(
        Conversation.id == room_id, Conversation.user_id == user_id,
    ).first()
    # The existing DateTime column is timezone-naive on SQLite/PostgreSQL even
    # though utcnow() is aware. Interpret stored lease timestamps as UTC.
    prior_expiry = prior.generation_lease_expires_at if prior else None
    if prior_expiry is not None and prior_expiry.tzinfo is None:
        prior_expiry = prior_expiry.replace(tzinfo=timezone.utc)
    recovered = bool(prior and prior.generation_token and prior_expiry and prior_expiry < now)
    claimed = db.execute(update(Conversation).where(
        Conversation.id == room_id, Conversation.user_id == user_id,
        Conversation.kind == ROOM_KIND, Conversation.turn_index == expected,
        or_(Conversation.generation_token.is_(None),
            Conversation.generation_lease_expires_at < now),
    ).values(generation_token=token, generation_lease_expires_at=now + ROOM_LEASE))
    db.commit()  # Release the row lock before any provider work.
    if claimed.rowcount != 1:
        _owned_room(db, room_id, user_id)
        raise HTTPException(status_code=409, detail={"code": "room_turn_stale_or_busy"})
    set_safe_metadata(generation_claim_recovered=recovered)


def _release_claim(db: Session, room_id: str, user_id: str, token: str) -> None:
    db.rollback()
    db.execute(update(Conversation).where(
        Conversation.id == room_id, Conversation.user_id == user_id,
        Conversation.generation_token == token,
    ).values(generation_token=None, generation_lease_expires_at=None))
    db.commit()


def _usage_totals(db: Session, room_id: str, token: str, attempts: int, user_id: str) -> dict:
    keys = [LLMOperation(job_id=room_id, stage="next", attempt=attempt,
                         chunk_key=token, operation_type="character_room").key(REQUEST_TYPE)
            for attempt in range(1, attempts + 1)]
    rows = db.query(LLMUsage).filter(LLMUsage.user_id == user_id,
                                     LLMUsage.operation_key.in_(keys)).all()
    return {
        "model": model_for_task("chat"),
        "input_tokens": sum(row.input_tokens for row in rows),
        "output_tokens": sum(row.output_tokens for row in rows),
        "estimated_cost_usd": str(sum(
            (row.estimated_cost_usd or Decimal(0) for row in rows), Decimal(0),
        )),
    }


def _room_instructions(speaker_name: str, profile: CharacterProfileData,
                       world: WorldProfileData | None, other_name: str,
                       language: str, last_message: str,
                       history: list[tuple[MessageRole, str]], runtime_block: str = "",
                       provenance_block: str = "",
                       fidelity_contract: CharacterFidelityContract | None = None) -> str:
    base = build_chat_instructions(
        speaker_name, profile, world, correction_prefix="/수정",
        current_message=last_message,
        recent_messages=history,
        response_language_override="Korean" if language == "ko" else "English",
        fidelity_contract=fidelity_contract,
    )
    return base + f"""

[CHARACTER CONVERSATION ROOM]
You are {speaker_name}, speaking directly to {other_name}. Only write your own single turn; never write
the other character's dialogue or hidden thoughts. The other character's private profile and world
are not your canon. This is a crossover conversation, not a merge of either story's timeline or world.
Use only your own canonical state, what the other character actually said in this room, and shared
room-visible context. The room response language is {'Korean' if language == 'ko' else 'English'}.

Each response should change the conversational state in some meaningful way when possible: follow through
on an action, resolve or deepen an open thread, introduce a consequence, change the relationship, or take
a related new perspective. A wish already expressed, advice already given, and a limitation already named
are established; do not paraphrase them as if new. The same canonical motif may recur only to make a new
point. Preserve continuity and voice; do not jump to an unrelated topic just to be novel.
{runtime_block}{provenance_block}{turn_output_instructions()}"""


def next_turn(db: Session, room_id: str, user_id: str, expected_turn_index: int) -> dict:
    token = str(uuid4())
    set_safe_metadata(operation="character_conversation_turn", conversation_id=room_id,
                      turn_index=expected_turn_index + 1, retry_count=0,
                      loop_guard_triggered=False)
    with stage("session_lookup"):
        _claim_turn(db, room_id, user_id, expected_turn_index, token)
    attempts = 0
    try:
        room = _owned_room(db, room_id, user_id)
        first, second = _participants(db, room, user_id)
        speaker, other = (first, second) if expected_turn_index % 2 == 0 else (second, first)
        speaker_id, speaker_name = speaker.id, speaker.name
        other_name = other.name
        room_language = room.language
        set_safe_metadata(speaker_character_id=speaker_id)

        with stage("character_profile_retrieval"):
            profile_row = get_profile(db, speaker_id)
            profile = CharacterProfileData.model_validate(profile_row.data) if profile_row else CharacterProfileData()
        with stage("world_retrieval"):
            world_row = get_world_profile(db, speaker.world_id) if speaker.world_id else None
            world = WorldProfileData.model_validate(world_row.data) if world_row else None
        with stage("session_history"):
            messages = db.query(Message).filter(Message.conversation_id == room_id).order_by(
                Message.turn_index.desc(), Message.id.desc(),
            ).limit(ROOM_HISTORY_LIMIT).all()
            messages.reverse()
            history = [
                (MessageRole.CHARACTER, message.content) if message.speaker_character_id == speaker_id
                else (MessageRole.USER, f"[PEER_CLAIM] {other_name} said: "
                      f"{normalize_assistant_actions(message.content)}")
                for message in messages
            ]
            same_speaker_recent = [message.content for message in messages
                                   if message.speaker_character_id == speaker_id][-5:]
            last_content = messages[-1].content if messages else ""
            runtime = ConversationRuntimeState.from_storage(room.runtime_state, messages)
            provenance = KnowledgeProvenance.for_room(messages, speaker_id)
            set_safe_metadata(conversation_runtime_version=runtime.version,
                              open_thread_count=len(runtime.open_threads),
                              resolved_thread_count=len(runtime.resolved_threads),
                              **provenance.safe_counts())

        with stage("canonical_state_reconstruction"):
            fidelity = build_fidelity_contract(speaker_name, profile, world)
            set_safe_metadata(**fidelity.safe_metadata())
            instructions = _room_instructions(speaker_name, profile, world, other_name,
                                              room_language, last_content, history,
                                              runtime.prompt_block(speaker_id, room=True),
                                              provenance.prompt_block(), fidelity)
        prompt_turn = f"It is {speaker_name}'s turn. Respond to {other_name} with one natural turn."
        with stage("prompt_builder"):
            counter = make_chat_input_counter(db, user_id, ROLEPLAY_MAX_TOKENS)
            budget = select_chat_context(instructions, history, prompt_turn,
                                         count_input_tokens=counter)
        # No room/profile read transaction remains open across network latency.
        db.rollback()

        def generate(attempt: int, instruction_text: str, input_messages: list[dict]):
            operation = LLMOperation(job_id=room_id, stage="next", attempt=attempt,
                                     chunk_key=token, operation_type="character_room")
            with llm_operation(operation):
                try:
                    return generate_text(db=db, user_id=user_id, request_type=REQUEST_TYPE,
                                         task="chat", instructions=instruction_text,
                                         input_messages=input_messages,
                                         max_output_tokens=ROLEPLAY_MAX_TOKENS)
                except Exception as exc:
                    set_safe_metadata(provider_error_category=classify_llm_failure(exc).kind.value)
                    raise

        attempts = 1
        schema_retried = False
        try:
            first_response = generate(1, instructions, budget.input_messages)
        except LLMStructuredOutputError:
            schema_retried = True
            attempts = 2
            first_response = generate(2, instructions + "\nReturn the required structured turn fields exactly.",
                                      budget.input_messages)

        def retry(correction: str):
            nonlocal attempts
            attempts = 2
            retry_instructions = instructions + correction
            with stage("prompt_builder"):
                retry_budget = select_chat_context(retry_instructions, history, prompt_turn,
                                                   count_input_tokens=counter)
            return generate(2, retry_instructions, retry_budget.input_messages)

        outcome = select_quality_response(
            first_response, same_speaker_recent=same_speaker_recent,
            recent_visible=[message.content for message in messages[-4:]],
            language="Korean" if room_language == "ko" else "English",
            allow_repetition=False, retry=retry, runtime_state=runtime,
            speaker=speaker_id, room=True, retry_allowed=not schema_retried,
            fidelity_contract=fidelity, request_text=last_content,
            visible_history=(), provenance=provenance,
        )
        attempts = max(attempts, 1 + outcome.retry_count)
        set_safe_metadata(**outcome.safe_metadata("character_pair",
                                                   "Korean" if room_language == "ko" else "English"),
                          loop_guard_triggered=outcome.initial.semantic_loop_detected)
        set_safe_metadata(retry_count=attempts - 1, corrective_retry=attempts > 1)
        set_safe_metadata(provider_call_count=attempts)
        response = outcome.text

        persisted_turn = Message(conversation_id=room_id, role=MessageRole.CHARACTER,
                                 speaker_character_id=speaker_id, turn_index=expected_turn_index + 1,
                                 content=response)
        db.add(persisted_turn)
        db.flush()
        advance_runtime(runtime, outcome.turn, response, speaker_id, persisted_turn.id,
                        strict_room=True)
        runtime.knowledge_provenance_summary = provenance.safe_counts()
        advanced = db.execute(update(Conversation).where(
            Conversation.id == room_id, Conversation.user_id == user_id,
            Conversation.generation_token == token, Conversation.turn_index == expected_turn_index,
        ).values(turn_index=expected_turn_index + 1, runtime_state=runtime.to_storage(), generation_token=None,
                 generation_lease_expires_at=None, updated_at=utcnow()))
        if advanced.rowcount != 1:
            raise HTTPException(status_code=409, detail={"code": "room_turn_stale_or_busy"})
        db.commit()
        persisted = db.query(Message).filter(Message.conversation_id == room_id,
                                             Message.turn_index == expected_turn_index + 1).one()
        room = _owned_room(db, room_id, user_id)
        result = {"room": _room_data(db, room, user_id), "message": persisted}
        set_safe_metadata(**_usage_totals(db, room_id, token, attempts, user_id))
        return result
    except Exception as exc:
        set_safe_metadata(provider_call_count=attempts)
        set_safe_metadata(provider_error_category=classify_llm_failure(exc).kind.value)
        _release_claim(db, room_id, user_id, token)
        if attempts:
            set_safe_metadata(**_usage_totals(db, room_id, token, attempts, user_id))
        raise
