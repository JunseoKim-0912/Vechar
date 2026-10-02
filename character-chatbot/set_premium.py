"""
계정을 프리미엄으로 설정/해제하는 스크립트. 공개 API가 아니라 직접 실행하는 관리 스크립트입니다
(아직 상업적으로 판매하는 기능이 아니라 본인 계정 전용으로만 쓰는 걸 전제로 합니다).

사용법:
  python set_premium.py <email>            # 프리미엄으로 설정
  python set_premium.py <email> --off       # 프리미엄 해제
"""

import sys
from app.database import SessionLocal
from app.models import User


def main():
    if len(sys.argv) < 2:
        print("사용법: python set_premium.py <email> [--off]")
        sys.exit(1)

    email = sys.argv[1]
    turn_on = "--off" not in sys.argv

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"'{email}' 계정을 찾을 수 없습니다.")
            sys.exit(1)

        user.is_premium = turn_on
        db.commit()
        status = "프리미엄으로 설정" if turn_on else "프리미엄 해제"
        print(f"'{email}' 계정을 {status}했습니다.")
    finally:
        db.close()


if __name__ == "__main__":
    main()