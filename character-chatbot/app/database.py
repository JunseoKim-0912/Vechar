import os
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()


def normalize_database_url(raw_url: str | None) -> str:
    """Normalize supported provider URLs without altering credentials or query options."""
    if not raw_url or not raw_url.strip():
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")
    url = raw_url.strip()
    # Some providers still expose SQLAlchemy's retired legacy alias.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def engine_kwargs_for_url(database_url: str) -> dict:
    """Keep SQLite's thread option isolated from PostgreSQL/managed DB drivers."""
    if make_url(database_url).get_backend_name() == "sqlite":
        return {"connect_args": {"check_same_thread": False}}
    return {}


DATABASE_URL = normalize_database_url(os.getenv("DATABASE_URL"))
database_backend = make_url(DATABASE_URL).get_backend_name()
if (os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV")) and database_backend == "sqlite":
    raise RuntimeError("Vercel deployments require a persistent PostgreSQL DATABASE_URL")

engine = create_engine(DATABASE_URL, **engine_kwargs_for_url(DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
