from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth import get_current_user_id
from ..models import Character, Conversation, Message, User
from ..ownership import get_owned_world
from ..schemas import MessageCreateRequest, MessageRead, ConversationCreateRequest
from ..services.chat_service import send_message
from ..services.chat_latency import stage

router = APIRouter()


@router.get("/characters/{character_id}/conversations")
def list_character_conversations(
    character_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db),
):
    character = db.query(Character.id).filter(Character.id == character_id,
                                              Character.user_id == user_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    rows = (db.query(Conversation)
            .filter(Conversation.user_id == user_id, Conversation.character_id == character_id,
                    Conversation.kind == "user_character")
            .order_by(Conversation.created_at.desc(), Conversation.id.desc()).all())
    return [{"id": row.id, "character_id": row.character_id, "created_at": row.created_at} for row in rows]


@router.post("/conversations", status_code=201)
def create_conversation(
    payload: ConversationCreateRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    character = (
        db.query(Character)
        .filter(Character.id == payload.character_id, Character.user_id == user_id)
        .first()
    )
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.world_id:
        get_owned_world(db, character.world_id, user_id)

    if payload.reuse_existing:
        # Backend-owned get-or-create survives refresh, remount and new processes.
        # The user row lock prevents duplicate initial sessions under concurrent mounts.
        db.query(User.id).filter(User.id == user_id).with_for_update().first()
        existing = (db.query(Conversation)
                    .filter(Conversation.user_id == user_id,
                            Conversation.character_id == character.id,
                            Conversation.kind == "user_character")
                    .order_by(Conversation.created_at.desc(), Conversation.id.desc()).first())
        if existing:
            result = {"id": existing.id, "character_id": existing.character_id}
            db.commit()
            return result

    conversation = Conversation(character_id=character.id, user_id=user_id)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return {"id": conversation.id, "character_id": conversation.character_id}


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageRead])
def get_messages(conversation_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    conversation = (
        db.query(Conversation)
        .join(Character, Conversation.character_id == Character.id)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id,
                Conversation.kind == "user_character", Character.user_id == user_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )


@router.post("/conversations/{conversation_id}/messages")
def post_message(
    conversation_id: str,
    payload: MessageCreateRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    with stage("session_lookup"):
        conversation = (
            db.query(Conversation)
            .join(Character, Conversation.character_id == Character.id)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id,
                    Conversation.kind == "user_character", Character.user_id == user_id)
            .first()
        )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        return send_message(db, conversation.character_id, conversation.id, payload.content, user_id, payload.locale)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="메시지 처리 중 오류가 발생했습니다.") from exc
