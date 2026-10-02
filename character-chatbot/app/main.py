import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .routers import auth_router, characters_router, upload_router
from .routers import chat_router
from .routers import worlds_router

app = FastAPI(title="Character Chatbot API")

raw_cors_origins = os.getenv("CORS_ORIGINS")
if raw_cors_origins is None:
    cors_origins = [] if os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV") else [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
else:
    cors_origins = [origin.strip().rstrip("/") for origin in raw_cors_origins.split(",") if origin.strip()]
if (os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV")) and "*" in cors_origins:
    raise RuntimeError("CORS_ORIGINS must list explicit origins on Vercel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

if upload_router.LOCAL_UPLOADS_ENABLED:
    os.makedirs("uploads", exist_ok=True)
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.on_event("startup")
def on_startup():
    # 프로토타입 단계의 테이블 생성 방식입니다. 스키마가 안정되면
    # Alembic 마이그레이션으로 바꿔서 데이터를 지우지 않고 스키마를 바꿀 수 있게 하세요.
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"ok": True}


app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(characters_router.router, prefix="/characters", tags=["characters"])
app.include_router(upload_router.router, prefix="/upload", tags=["upload"])
app.include_router(chat_router.router, prefix="/chat", tags=["chat"])
app.include_router(worlds_router.router, prefix="/worlds", tags=["worlds"])
