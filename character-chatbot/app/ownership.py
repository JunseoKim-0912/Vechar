from fastapi import HTTPException
from sqlalchemy.orm import Session

from .models import World


def get_owned_world(db: Session, world_id: str, user_id: str) -> World:
    world = db.query(World).filter(World.id == world_id, World.user_id == user_id).first()
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")
    return world
