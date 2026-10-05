"""Private, source-free training progress API."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user_id
from ..database import get_db
from ..models import TrainingJob


router = APIRouter()


@router.get("/{job_id}")
def get_training_job(job_id: str, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id, TrainingJob.user_id == user_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "training_job_not_found"})
    return {
        "id": job.id, "target_type": job.target_type, "target_id": job.target_id,
        "status": job.status, "stage": job.stage, "progress": job.progress,
        "total_chunks": job.total_chunks, "completed_chunks": job.completed_chunks,
        "error_code": job.error_code, "error_message_safe": job.error_message_safe,
        "created_at": job.created_at, "completed_at": job.completed_at,
    }
