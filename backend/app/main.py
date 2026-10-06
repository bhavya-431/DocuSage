from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.database import engine
from app.models import Base  # noqa: F401 — registers all tables on Base.metadata

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Idempotent schema bootstrap: pgvector extension + all tables.
    # Degrades to a warning when the DB is unreachable so the API still boots
    # (GET /health reports database status).
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Schema bootstrap complete")
    except Exception as e:
        logger.warning(f"Schema bootstrap skipped (database unreachable): {e}")
    yield
    # Shutdown actions


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Enterprise Document Intelligence Agent (RAG) with dual-gate abstention, page-level citations, and strict user isolation.",
        lifespan=lifespan,
    )

    # CORS configuration for React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    return app


app = create_app()
