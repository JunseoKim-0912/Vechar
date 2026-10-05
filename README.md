# Vechar

Vechar는 사용자가 캐릭터와 세계관을 만들고 학습 자료로 프로필을 구성한 뒤 캐릭터와 대화하는 프로토타입입니다. 캐릭터·세계관 프로필, 대화 기록, 사용자별 LLM 사용량 제한을 FastAPI가 관리합니다. 장기 메모리(MemMachine 등)는 아직 구현되지 않았습니다.

## 구성

| 디렉터리 | 역할 | 기술 |
| --- | --- | --- |
| `character-chatbot/` | API, 인증, 프로필·대화·사용량 관리 | Python 3.12, FastAPI, SQLAlchemy, OpenAI Responses API |
| `character-chatbot-frontend/` | 브라우저 UI | React, Vite, React Router |

로컬에서는 SQLite를 사용합니다. 프런트엔드는 기본적으로 `http://localhost:4000`의 API를 호출합니다.

## 로컬 실행

백엔드 (`character-chatbot/`에서 실행):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# .env에 로컬 DATABASE_URL, JWT_SECRET, 필요 시 OPENAI_API_KEY를 직접 설정
.\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 4000
```

이미 `.env`가 있다면 덮어쓰지 마세요. LLM 기능을 사용할 때만 본인의 OpenAI API key가 필요합니다. `/health`와 `/docs`에서 기본 서버 상태를 확인할 수 있습니다.

프런트엔드 (`character-chatbot-frontend/`에서 별도 터미널로 실행):

```powershell
npm ci
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm run dev
```

프런트엔드의 `.env`가 이미 있다면 덮어쓰지 마세요. 개발 서버 주소는 Vite 출력에서 확인합니다.

## 환경변수

| 프로젝트 | 이름 | 용도 |
| --- | --- | --- |
| Backend | `DATABASE_URL` | 로컬 SQLite 또는 배포용 PostgreSQL 연결 URL. 배포에는 영속적인 PostgreSQL이 필요합니다. |
| Backend | `JWT_SECRET` | 토큰 서명 비밀 값. Vercel에서는 반드시 별도로 설정해야 합니다. |
| Backend | `OPENAI_API_KEY` | OpenAI 호출용 비밀 값. 브라우저에 전달하지 않습니다. |
| Backend | `OPENAI_ANALYSIS_MODEL` | 캐릭터·세계관 분석 및 프로필 처리용 모델. 예시는 backend `.env.example`에 있습니다. |
| Backend | `OPENAI_CHAT_MODEL` | 실시간 캐릭터 채팅용 모델. 예시는 backend `.env.example`에 있습니다. |
| Backend | `CORS_ORIGINS` | 허용할 프런트엔드 origin의 쉼표 구분 목록. Vercel에서는 실제 프런트엔드 URL을 명시합니다. |
| Frontend | `VITE_API_BASE_URL` | 백엔드의 공개 base URL. 로컬 기본값은 `http://localhost:4000`입니다. |

실제 비밀 값은 `.env.example`이나 Git에 넣지 마세요. `VITE_` 환경변수는 브라우저 빌드에 포함되므로 비밀 값으로 사용하면 안 됩니다. 기존 로컬 `VITE_API_BASE`도 호환 목적으로 읽지만 새 설정은 `VITE_API_BASE_URL`을 사용합니다.

## 테스트

```powershell
# character-chatbot/
py -m unittest discover -s tests -v

# character-chatbot-frontend/
npm ci
npm run lint
npm run build
```

백엔드 테스트는 OpenAI를 mock 처리하며 실제 API 호출이 필요하지 않습니다.

## 배포 구조와 아직 필요한 작업

같은 Git 저장소에서 Vercel 프로젝트를 두 개 만들고 Root Directory를 각각 `character-chatbot`과 `character-chatbot-frontend`로 지정하는 구성을 전제로 합니다. 백엔드는 `server.py`가 기존 `app.main:app`을 노출하고, `.python-version`과 `requirements.txt`를 사용합니다. 프런트엔드는 Vite `dist` 빌드와 `vercel.json`의 SPA rewrite를 사용합니다. 프런트엔드 프로젝트에는 `VITE_API_BASE_URL`, 백엔드 프로젝트에는 실제 frontend origin을 넣은 `CORS_ORIGINS`를 설정해야 합니다.

이 설정만으로 공개 서비스가 완성되는 것은 아닙니다. 배포 전 영속적인 PostgreSQL과 스키마 관리 방식을 준비해야 합니다. 앱 시작 시 스키마를 자동 생성하거나 migration을 자동 실행하지 않습니다. `/ready`는 DB 연결만 확인합니다. Vercel Functions의 파일시스템은 영속적 업로드 저장소가 아니므로 Vercel에서는 로컬 이미지 업로드 및 `/uploads` 정적 제공을 비활성화합니다. 로컬 업로드는 계속 동작하지만 파일별 접근 제어가 없으므로 민감한 이미지를 올리면 안 됩니다. 향후 `app/routers/upload_router.py`의 파일 저장·URL 생성 부분을 작은 storage service로 분리하고, 객체 저장소의 URL과 접근 정책을 연결해야 합니다. 프런트엔드는 기존 `/uploads/...` 상대 경로를 backend origin으로 해석하며, 향후 객체 저장소 URL도 별도로 검토해야 합니다.

## 데이터베이스 migration

백엔드 `character-chatbot/`에서 실행합니다. Alembic은 기존 `app.database`의 `DATABASE_URL` 로딩과 `postgres://` 정규화를 사용합니다. `alembic.ini`에 URL/암호를 적지 않습니다. 로컬 SQLite와 **비어 있는** 새 PostgreSQL DB는 `python -m alembic -c alembic.ini upgrade head`로 초기화합니다. 현재 버전은 `python -m alembic -c alembic.ini current`, 저장소의 head는 `python -m alembic -c alembic.ini heads`로 확인합니다. 새 revision은 `python -m alembic -c alembic.ini revision --autogenerate -m "설명"`으로 생성한 뒤 PostgreSQL enum·제약조건·DDL을 반드시 검토합니다. 테스트의 isolated SQLite DB는 계속 자체 `create_all`을 사용할 수 있습니다.

이미 사용자 데이터와 테이블이 있는 **production Neon DB에서는 초기 revision을 `upgrade`하지 마세요.** 최초 도입은 담당자가 명시적으로 아래 순서로 수행합니다.

1. Neon backup/branch를 확보하고, 먼저 해당 branch에서 절차를 검증합니다.
2. 실제 DB의 테이블 15개, PostgreSQL enum 5종 및 값, 컬럼·nullability·인덱스·unique·foreign key/삭제 규칙을 `0001_initial_schema`와 대조합니다. 차이가 있으면 stamp하지 말고 별도 수정 계획을 세웁니다.
3. 일치함을 확인한 뒤 production `DATABASE_URL`을 관리자 환경에만 설정하고 `python -m alembic -c alembic.ini stamp 0001_initial_schema`를 **한 번만** 실행합니다. 이는 기존 테이블을 재생성하지 않고 `alembic_version`만 기록합니다.
4. `python -m alembic -c alembic.ini current`가 `0001_initial_schema`인지 확인하고 앱의 `/ready` 및 주요 기능을 smoke test합니다.

현재 revision은 기존 테이블을 그대로 사용하므로 stamp 전에도 현재 앱 동작에 migration 버전 검사는 필요하지 않습니다. 앞으로 실제 스키마 변경 revision을 배포할 때는 백업/branch 검증 후 별도 관리 단계에서 `upgrade head`를 실행하고 앱을 배포하세요. Vercel build/start에서는 migration이나 stamp를 실행하지 않습니다. `downgrade base`는 **모든 앱 테이블을 삭제하므로 production에서 실행하지 마세요.**

첫 GitHub push 전에 과거 커밋의 `.env`, 로컬 DB, 업로드 파일을 Git history에서 제거하고 노출된 적이 있는 자격 증명을 회전해야 합니다. 현재 작업 트리의 ignore 설정만으로 과거 기록은 지워지지 않습니다.
