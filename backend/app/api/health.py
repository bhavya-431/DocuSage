from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import Settings, get_settings
from app.core.database import async_session_factory
from app.providers.embeddings import get_embedding_provider
from app.providers.llm import get_llm_provider

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)) -> dict:
    """Comprehensive health check for API, DB, pgvector, and provider configurations.

    Per PRD: Provider credentials/configuration are validated without making paid
    or rate-limited inference calls.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    overall_status = "ok"

    # 1. Database & pgvector validation
    db_status = "connected"
    pgvector_status = "unknown"
    db_error = None

    try:
        async with async_session_factory() as session:
            # Check basic query connectivity
            await session.execute(text("SELECT 1;"))

            # Check pgvector extension
            vector_res = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            )
            row = vector_res.fetchone()
            if row:
                pgvector_status = f"installed (v{row[0]})"
            else:
                pgvector_status = "missing"
                overall_status = "degraded"
    except Exception as e:
        db_status = "unreachable"
        pgvector_status = "unreachable"
        db_error = str(e)
        overall_status = "degraded"

    # 2. Embedding provider credential & configuration validation (No inference calls)
    emb_provider = get_embedding_provider(settings)
    emb_valid, emb_msg = await emb_provider.validate_credentials()
    if not emb_valid:
        overall_status = "degraded"

    # 3. LLM provider credential & configuration validation (No inference calls)
    llm_provider = get_llm_provider(settings)
    llm_valid, llm_msg = await llm_provider.validate_credentials()
    if not llm_valid:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": timestamp,
        "app": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "database": {
            "status": db_status,
            "pgvector": pgvector_status,
            "error": db_error,
        },
        "providers": {
            "embedding": {
                "provider": settings.EMBEDDING_PROVIDER,
                "model": settings.EMBEDDING_MODEL,
                "dimension": emb_provider.dimension,
                "valid": emb_valid,
                "detail": emb_msg,
            },
            "llm": {
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL,
                "valid": llm_valid,
                "detail": llm_msg,
            },
        },
    }
