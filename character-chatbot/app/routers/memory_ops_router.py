"""Authenticated scheduled outbox repair; no memory content leaves this route."""

import hmac
import os

from fastapi import APIRouter, Header, HTTPException

from ..services.memory_jobs import reconcile_pending

router = APIRouter()


@router.get("/reconcile", include_in_schema=False)
def reconcile_memory(authorization: str | None = Header(default=None)):
    secret = os.getenv("CRON_SECRET")
    if not secret or not authorization or not hmac.compare_digest(authorization, f"Bearer {secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return reconcile_pending()
