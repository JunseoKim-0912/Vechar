import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import HTTPException
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from .llm_config import MODEL_PRICES_USD_PER_MILLION
from .llm_failures import DuplicateLLMOperation
from .llm_operation import current_operation
from .models import LLMUsage, User, UserRole
from .tier_limits import TIER_LIMITS


logger = logging.getLogger(__name__)
RESERVATION_RECOVERY_AFTER = timedelta(hours=2)


def _period_usage(db: Session, user_id: str, start: datetime) -> Decimal:
    return Decimal(
        db.query(func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0))
        .filter(LLMUsage.user_id == user_id, LLMUsage.created_at >= start)
        .scalar()
    )


def _reservation_cost(model: str, input_tokens: int, output_cap: int) -> Decimal:
    """Reserve uncached input plus the full output cap; cached input is reconciled later."""
    cost = _estimated_cost(model, input_tokens, 0, output_cap)
    if cost is None:
        raise ValueError(f"No pricing configured for model: {model}")
    return cost


def _check_limits(db: Session, user_id: str, limits: dict, now: datetime, additional_cost: Decimal) -> None:
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    daily_used = _period_usage(db, user_id, day_start)
    daily_limit = limits["max_llm_cost_usd_per_day"]
    if daily_used + additional_cost > daily_limit:
        exhausted = daily_used >= daily_limit
        raise HTTPException(
            status_code=429,
            detail={"code": "daily_limit_reached" if exhausted else "request_exceeds_remaining_daily_budget",
                    "message": "Daily LLM cost limit reached." if exhausted
                    else "Request exceeds remaining daily LLM cost allowance."},
        )
    month_start = day_start.replace(day=1)
    monthly_used = _period_usage(db, user_id, month_start)
    monthly_limit = limits["max_llm_cost_usd_per_month"]
    if monthly_used + additional_cost > monthly_limit:
        exhausted = monthly_used >= monthly_limit
        raise HTTPException(
            status_code=429,
            detail={"code": "monthly_limit_reached" if exhausted else "request_exceeds_remaining_monthly_budget",
                    "message": "Monthly LLM cost limit reached." if exhausted
                    else "Request exceeds remaining monthly LLM cost allowance."},
        )


def check_capacity(bind, user_id: str, model: str, output_cap: int) -> None:
    """Reject exhausted accounts before contacting the token-count endpoint."""
    minimum_reservation = _reservation_cost(model, 0, output_cap)
    with Session(bind=bind) as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail={"code": "user_not_found"})
        if user.role != UserRole.ADMIN.value:
            limits = TIER_LIMITS["premium" if user.is_premium else "free"]
            _check_limits(db, user_id, limits, datetime.now(timezone.utc), minimum_reservation)


def reserve_usage(bind, user_id: str, request_type: str, model: str, input_tokens: int,
                  output_cap: int, *, operation_key: str | None = None) -> str:
    """Atomically reserve maximum estimated USD cost and retain token counts for accounting."""
    if input_tokens < 0 or output_cap <= 0:
        raise ValueError("Invalid LLM token reservation")
    reserved_cost = _reservation_cost(model, input_tokens, output_cap)

    with Session(bind=bind) as db:
        # SQLite ignores SELECT FOR UPDATE; BEGIN IMMEDIATE serializes writers instead.
        if db.bind.dialect.name == "sqlite":
            db.execute(text("BEGIN IMMEDIATE"))
            user = db.query(User).filter(User.id == user_id).first()
        else:
            user = db.query(User).filter(User.id == user_id).with_for_update().first()
        if user is None:
            raise HTTPException(status_code=401, detail={"code": "user_not_found"})

        now = datetime.now(timezone.utc)
        # This is detection, not forgiveness: unknown provider billing retains its
        # reservation in the period SUM until an operator can reconcile it.
        stale = db.query(LLMUsage).filter(
            LLMUsage.user_id == user_id, LLMUsage.status == "reserved",
            or_(LLMUsage.reservation_expires_at <= now,
                (LLMUsage.reservation_expires_at.is_(None)
                 & (LLMUsage.created_at <= now - RESERVATION_RECOVERY_AFTER))),
        ).all()
        for old in stale:
            old.status = "recovery_required"
            logger.warning("llm_stale_reservation operation_id=%s model=%s reserved_usd=%s",
                           old.operation_key, old.model, old.estimated_cost_usd)
        if operation_key and db.query(LLMUsage.id).filter(LLMUsage.operation_key == operation_key).first():
            if stale:
                db.commit()
            raise DuplicateLLMOperation("Logical LLM attempt already reserved")
        reservation = input_tokens + output_cap
        if user.role != UserRole.ADMIN.value:
            limits = TIER_LIMITS["premium" if user.is_premium else "free"]
            try:
                _check_limits(db, user_id, limits, now, reserved_cost)
            except HTTPException:
                if stale:
                    db.commit()
                raise

        row = LLMUsage(
            user_id=user_id,
            request_type=request_type,
            model=model,
            status="reserved",
            reserved_total_tokens=reservation,
            budget_tokens=reservation,
            estimated_cost_usd=reserved_cost,
            operation_key=operation_key,
            reservation_expires_at=now + RESERVATION_RECOVERY_AFTER,
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
        logging.warning("No configured price for reported OpenAI model %s; estimated cost unavailable", model)
        return None
    uncached = max(0, input_tokens - cached_tokens)
    threshold = prices.get("long_context_threshold")
    long_context = threshold is not None and input_tokens > threshold
    input_multiplier = prices["long_context_input_multiplier"] if long_context else Decimal(1)
    output_multiplier = prices["long_context_output_multiplier"] if long_context else Decimal(1)
    return (
        (Decimal(uncached) * prices["input"] + Decimal(cached_tokens) * prices["cached_input"])
        * input_multiplier
        + Decimal(output_tokens) * prices["output"] * output_multiplier
    ) / Decimal(1_000_000)


def record_response(bind, usage_id: str, response, *, output_error: str | None = None,
                    failure_class: str | None = None) -> None:
    """Record provider-reported tokens even when the response has no usable text."""
    with Session(bind=bind) as db:
        row = db.query(LLMUsage).filter(LLMUsage.id == usage_id).with_for_update().one()
        row.provider_response_id = getattr(response, "id", None)
        row.provider_status = getattr(response, "status", None)
        row.provider_error_code = getattr(getattr(response, "error", None), "code", None)
        row.incomplete_reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        row.failure_class = failure_class
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
            output_details = getattr(usage, "output_tokens_details", None)
            reasoning = getattr(output_details, "reasoning_tokens", None)
            row.reasoning_tokens = int(reasoning) if reasoning is not None else None
            row.budget_tokens = max(0, row.total_tokens)
            actual_cost = _estimated_cost(
                row.model, row.input_tokens, row.cached_input_tokens, row.output_tokens
            )
            # An unpriced provider-reported model must not erase the reservation.
            if actual_cost is not None:
                row.estimated_cost_usd = actual_cost
                row.reconciled_at = datetime.now(timezone.utc)
            row.status = (
                "completed" if output_error is None and response.status == "completed" and response.output_text
                else "failed_response"
            )
            if row.status != "completed":
                row.error_type = output_error or (
                    response.status if response.status != "completed" else "EmptyOutput"
                )
        db.commit()
        operation = current_operation()
        logger.info("llm_response operation_id=%s job_id=%s chunk_index=%s attempt=%s model=%s "
                    "response_status=%s provider_error_code=%s incomplete_reason=%s "
                    "input_tokens=%s cached_input_tokens=%s "
                    "output_tokens=%s reasoning_tokens=%s estimated_usd=%s retry_class=%s",
                    row.operation_key, operation.job_id if operation else None,
                    operation.chunk_index if operation else None, operation.attempt if operation else None,
                    row.model, row.provider_status, row.provider_error_code, row.incomplete_reason, row.input_tokens,
                    row.cached_input_tokens, row.output_tokens, row.reasoning_tokens,
                    row.estimated_cost_usd, row.failure_class)


def record_failure(bind, usage_id: str, error_type: str, definitely_unbilled: bool = False,
                   failure_class: str | None = None) -> None:
    """Unknown failures keep their reservation; known pre-generation 4xx errors release it."""
    with Session(bind=bind) as db:
        row = db.query(LLMUsage).filter(LLMUsage.id == usage_id).with_for_update().one()
        row.status = "failed"
        row.error_type = error_type
        row.failure_class = failure_class
        if definitely_unbilled:
            row.budget_tokens = 0
            row.estimated_cost_usd = Decimal(0)
            row.reconciled_at = datetime.now(timezone.utc)
        db.commit()
        operation = current_operation()
        logger.warning("llm_call_failed operation_id=%s job_id=%s chunk_index=%s attempt=%s model=%s "
                       "estimated_usd=%s retry_class=%s released=%s",
                       row.operation_key, operation.job_id if operation else None,
                       operation.chunk_index if operation else None, operation.attempt if operation else None,
                       row.model, row.estimated_cost_usd, failure_class, definitely_unbilled)
