from datetime import datetime, timezone
from decimal import Decimal
from fastapi import HTTPException
from sqlalchemy.orm import Session
from .models import User, ExportLog, ImportLog

# 프리미엄은 지금은 상업적 판매용이 아니라, 본인 계정 전용 플래그입니다.
# set_premium.py로 직접 DB에서 켜고 끕니다 (공개 API로 노출하지 않음).
TIER_LIMITS = {
    "free": {
        "max_worlds": 2,
        "max_characters": 5,
        "max_corrections_per_day": 5,
        "max_exports_per_month": 5,
        "max_imports_per_month": 5,
        "max_llm_cost_usd_per_day": Decimal("1.00"),
        "max_llm_cost_usd_per_month": Decimal("20.00"),
    },
    "premium": {
        "max_worlds": 5,
        "max_characters": 25,
        "max_corrections_per_day": 25,
        "max_exports_per_month": 10,
        "max_imports_per_month": 10,
        "max_llm_cost_usd_per_day": Decimal("5.00"),
        "max_llm_cost_usd_per_month": Decimal("100.00"),
    },
}


def get_limits(db: Session, user_id: str) -> dict:
    user = db.query(User).filter(User.id == user_id).first()
    tier = "premium" if user and user.is_premium else "free"
    return TIER_LIMITS[tier]


def _month_start() -> datetime:
    return datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def check_and_log_export(db: Session, user_id: str, export_type: str, entity_id: str) -> None:
    """내보내기 이번 달 한도를 확인하고, 통과하면 로그를 남깁니다. 한도 초과 시 403을 던집니다.
    캐릭터/세계관 내보내기를 합쳐서 하나의 월간 한도로 셉니다. 매월 1일 자정(UTC)에 초기화됩니다."""
    limits = get_limits(db, user_id)
    month_count = (
        db.query(ExportLog)
        .filter(ExportLog.user_id == user_id, ExportLog.created_at >= _month_start())
        .count()
    )
    if month_count >= limits["max_exports_per_month"]:
        raise HTTPException(
            status_code=403,
            detail=f"이번 달 내보내기 횟수({limits['max_exports_per_month']}회)를 다 쓰셨습니다. 다음 달에 다시 시도해주세요.",
        )

    db.add(ExportLog(user_id=user_id, export_type=export_type, entity_id=entity_id))
    db.commit()


def check_import_quota(db: Session, user_id: str) -> None:
    """가져오기 대상(캐릭터/세계관)을 실제로 만들기 *전에* 호출하세요 — 한도 초과 시 아무것도
    만들지 않고 403을 던집니다. 캐릭터/세계관 가져오기를 합쳐서 하나의 월간 한도로 셉니다."""
    limits = get_limits(db, user_id)
    month_count = (
        db.query(ImportLog)
        .filter(ImportLog.user_id == user_id, ImportLog.created_at >= _month_start())
        .count()
    )
    if month_count >= limits["max_imports_per_month"]:
        raise HTTPException(
            status_code=403,
            detail=f"이번 달 가져오기 횟수({limits['max_imports_per_month']}회)를 다 쓰셨습니다. 다음 달에 다시 시도해주세요.",
        )


def log_import(db: Session, user_id: str, import_type: str, entity_id: str) -> None:
    """check_import_quota 통과 후, 실제로 캐릭터/세계관을 만든 *다음에* 호출해서 이력을 남깁니다."""
    db.add(ImportLog(user_id=user_id, import_type=import_type, entity_id=entity_id))
    db.commit()
