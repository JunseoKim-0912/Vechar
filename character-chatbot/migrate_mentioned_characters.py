"""
mentioned_characters 필드를 예전 형식(문자열 목록)에서 새 형식({name, aliases} 목록)으로 변환하는
일회성 마이그레이션 스크립트. alias(별명) 기능을 추가하면서 필드 구조가 바뀌어서 필요합니다.

이 스크립트는 이름을 구조만 바꿔줄 뿐 중복을 알아서 합쳐주지는 않습니다 — 돌린 다음, 각 세계관에서
"압축하기" 버튼을 한 번씩 눌러주시면 그때 이름/별명이 겹치는 인물이 실제로 하나로 정리됩니다.

사용법: python migrate_mentioned_characters.py
"""

from app.database import SessionLocal
from app.models import WorldProfile


def main():
    db = SessionLocal()
    try:
        profiles = db.query(WorldProfile).all()
        migrated = 0
        for profile in profiles:
            chars = profile.data.get("mentioned_characters", [])
            if chars and isinstance(chars[0], str):
                new_data = dict(profile.data)
                new_data["mentioned_characters"] = [{"name": c, "aliases": []} for c in chars]
                profile.data = new_data
                migrated += 1
        db.commit()
        print(f"{migrated}개의 세계관 프로필을 새 형식으로 변환했습니다.")
        if migrated:
            print("각 세계관 상세 페이지에서 '압축하기'를 한 번씩 눌러주시면 중복된 이름들이 실제로 합쳐집니다.")
    finally:
        db.close()


if __name__ == "__main__":
    main()