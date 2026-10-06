"""Authenticated, owner-scoped two-character room API."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user_id
from ..database import get_db
from ..schemas import (
    CharacterConversationCreateRequest, CharacterConversationNextRequest,
    CharacterConversationRead, CharacterConversationRenameRequest,
    CharacterConversationTurnRead, MessageRead,
)
from ..services import character_conversations as rooms

router = APIRouter()


@router.post("", response_model=CharacterConversationRead, status_code=201)
def create_room(payload: CharacterConversationCreateRequest,
                user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return rooms.create_room(db, user_id, payload.character_a_id, payload.character_b_id,
                             payload.name, payload.language)


@router.get("", response_model=list[CharacterConversationRead])
def list_rooms(user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return rooms.list_rooms(db, user_id)


@router.get("/{room_id}", response_model=CharacterConversationRead)
def get_room(room_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return rooms.get_room(db, room_id, user_id)


@router.get("/{room_id}/messages", response_model=list[MessageRead])
def list_messages(room_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return rooms.list_messages(db, room_id, user_id)


@router.patch("/{room_id}", response_model=CharacterConversationRead)
def rename_room(room_id: str, payload: CharacterConversationRenameRequest,
                user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return rooms.rename_room(db, room_id, user_id, payload.name)


@router.delete("/{room_id}", status_code=204)
def delete_room(room_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    rooms.delete_room(db, room_id, user_id)
    return None


@router.post("/{room_id}/next", response_model=CharacterConversationTurnRead)
def next_turn(room_id: str, payload: CharacterConversationNextRequest,
              user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    try:
        return rooms.next_turn(db, room_id, user_id, payload.expected_turn_index)
    except HTTPException:
        raise
    except Exception as exc:
        # No provider details or raw dialogue cross the API boundary.
        raise HTTPException(status_code=503, detail={"code": "room_generation_failed"}) from exc
