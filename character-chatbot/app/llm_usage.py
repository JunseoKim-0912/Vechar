from datetime import datetime, timezone
from decimal import Decimal

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .llm_config import MODEL_PRICES_USD_PER_MILLION
from .models import LLMUsage, User
from .tier_limits import TIER_LIMITS


def _period_usage(db: Session, user_id: str, start: datetime) -> int:
    return int(
        db.query(func.coalesce(func.sum(LLMUsage.budget_tokens), 0))
        .filter(LLMUsage.user_id == user_id, LLMUsage.created_at >= start)
        .scalar()
    )


def check_capacity(bind, user_id: str, output_cap: int) -> None:
    """Reject exhausted accounts before contacting the token-count endpoint."""
    with Session(bind=bind) as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail={"code": "user_not_found"})
        limits = TIER_LIMITS["premium" if user.is_premium else "free"]
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if _period_usage(db, user_id, day_start) + output_cap > limits["max_llm_tokens_per_day"]:
            raise HTTPException(
                status_code=429,
                detail={"code": "daily_limit_reached", "message": "Daily LLM token limit reached."},
            )
        if _period_usage(db, user_id, day_start.replace(day=1)) + output_cap > limits["max_llm_tokens_per_month"]:
            raise HTTPException(
                status_code=429,
                detail={"code": "monthly_limit_reached", "message": "Monthly LLM token limit reached."},
            )


def reserve_usage(bind, user_id: str, request_type: str, model: str, input_tokens: int, output_cap: int) -> str:
    """Atomically reserve the counted input plus the maximum possible output."""
    if input_tokens < 0 or output_cap <= 0:
        raise ValueError("Invalid LLM token reservation")

    with Session(bind=bind) as db:
        # SQLite ignores SELECT FOR UPDATE; BEGIN IMMEDIATE serializes writers instead.
        if db.bind.dialect.name == "sqlite":
            db.execute(text("BEGIN IMMEDIATE"))
            user = db.query(User).filter(User.id == user_id).first()
        else:
            user = db.query(User).filter(User.id == user_id).with_for_update().first()
        if user is None:
            raise HTTPException(status_code=401, detail={"code": "user_not_found"})

        limits = TIER_LIMITS["premium" if user.is_premium else "free"]
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = day_start.replace(day=1)
        reservation = input_tokens + output_cap

        if _period_usage(db, user_id, day_start) + reservation > limits["max_llm_tokens_per_day"]:
            raise HTTPException(
                status_code=429,
                detail={"code": "daily_limit_reached", "message": "Daily LLM token limit reached."},
            )
        if _period_usage(db, user_id, month_start) + reservation > limits["max_llm_tokens_per_month"]:
            raise HTTPException(
                status_code=429,
                detail={"code": "monthly_limit_reached", "message": "Monthly LLM token limit reached."},
            )

        row = LLMUsage(
            user_id=user_id,
            request_type=request_type,
            model=model,
            status="reserved",
            reserved_total_tokens=reservation,
            budget_tokens=reservation,
            created_at=now,
        )
        db.add(row)
        db.commit()
        return row.id


def record_preflight_failure(bind, user_id: str, request_type: str, model: str, error_type: str) -> None:
    """Token counting returned no usage, and no generation was started."""
    with Session(bind=bind) as db:
        db.add(
            LLMUsage(
                user_id=user_id,
                request_type=request_type,
                model=model,
                status="preflight_failed",
                error_type=error_type,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()


def _estimated_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int):
    prices = MODEL_PRICES_USD_PER_MILLION.get(model)
    if prices is None:
        return None
    uncached = max(0, input_tokens - cached_tokens)
    return (
        Decimal(uncached) * prices["input"]
        + Decimal(cached_tokens) * prices["cached_input"]
        + Decimal(output_tokens) * prices["output"]
    ) / Decimal(1_000_000)


def record_response(bind, usage_id: str, response) -> None:
    """Record provider-reported tokens even when the response has no usable text."""
    with Session(bind=bind) as db:
        row = db.query(LLMUsage).filter(LLMUsage.id == usage_id).with_for_update().one()
        usage = response.usage
        if usage is None:
            row.status = "usage_unavailable"
            row.error_type = "MissingUsage"
        else:
            row.model = response.model or row.model
            row.input_tokens = int(usage.input_tokens)
            row.output_tokens = int(usage.output_tokens)
            row.total_tokens = int(usage.total_tokens)
            details = getattr(usage, "input_tokens_details", None)
            row.cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)
            row.budget_tokens = max(0, row.total_tokens)
            row.estimated_cost_usd = _estimated_cost(
                row.model, row.input_tokens, row.cached_input_tokens, row.output_tokens
            )
            row.status = "completed" if response.status == "completed" and response.output_text else "failed_response"
            if row.status != "completed":
                row.error_type = response.status or "EmptyOutput"
        db.commit()


def record_failure(bind, usage_id: str, error_type: str, definitely_unbilled: bool = False) -> None:
    """Unknown failures keep their reservation; known pre-generation 4xx errors release it."""
    with Session(bind=bind) as db:
        row = db.query(LLMUsage).filter(LLMUsage.id == usage_id).with_for_update().one()
        row.status = "failed"
        row.error_type = error_type
        if definitely_unbilled:
            row.budget_tokens = 0
        db.commit()
