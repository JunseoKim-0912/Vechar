import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .routers import auth_router, characters_router, upload_router
from .routers import chat_router
from .routers import worlds_router

app = FastAPI(title="Character Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 전에 실제 프론트엔드 주소로 좁히세요
    allow_methods=["*"],
    allow_headers=["*"],
)

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