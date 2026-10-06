from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth import get_current_user_id
from ..models import Character, Conversation, Message
from ..ownership import get_owned_world
from ..schemas import MessageCreateRequest, MessageRead, ConversationCreateRequest
from ..services.chat_service import send_message
from ..services.chat_latency import stage

router = APIRouter()


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
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id, Character.user_id == user_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
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
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id, Character.user_id == user_id)
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
