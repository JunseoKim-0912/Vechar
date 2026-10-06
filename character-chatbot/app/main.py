import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from .database import engine
from .routers import auth_router, characters_router, upload_router
from .routers import chat_router
from .routers import worlds_router
from .routers import training_jobs_router
from .routers import memory_ops_router
from .services.chat_latency import start_chat_latency, end_chat_latency

app = FastAPI(title="Character Chatbot API")


@app.middleware("http")
async def chat_latency_middleware(request, call_next):
    if request.method != "POST" or not request.url.path.startswith("/chat/conversations/") or not request.url.path.endswith("/messages"):
        return await call_next(request)
    tracker, token = start_chat_latency()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        tracker.finish(status_code)
        end_chat_latency(token)

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


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    """Database-dependent readiness without exposing connection details."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"ready": True}


app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(characters_router.router, prefix="/characters", tags=["characters"])
app.include_router(upload_router.router, prefix="/upload", tags=["upload"])
app.include_router(chat_router.router, prefix="/chat", tags=["chat"])
app.include_router(worlds_router.router, prefix="/worlds", tags=["worlds"])
app.include_router(training_jobs_router.router, prefix="/training-jobs", tags=["training-jobs"])
app.include_router(memory_ops_router.router, prefix="/internal/memory", tags=["internal"])
