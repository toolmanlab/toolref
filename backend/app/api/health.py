"""Health check endpoints."""

from fastapi import APIRouter
from sqlalchemy import text

from app.cache.redis import check_redis
from app.db.engine import async_session
from app.vectorstore.milvus import check_milvus

router = APIRouter(tags=["health"])


async def _check_postgres() -> bool:
    """Return True if PostgreSQL is reachable."""
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@router.get("/health")
async def health() -> dict[str, object]:
    """Liveness / readiness probe returning component status.

    Returns a dict with overall status and per-component details.
    """
    pg_ok = await _check_postgres()
    redis_ok = await check_redis()
    milvus_ok = check_milvus()

    all_ok = pg_ok and redis_ok and milvus_ok

    return {
        "status": "healthy" if all_ok else "degraded",
        "components": {
            "postgres": "up" if pg_ok else "down",
            "redis": "up" if redis_ok else "down",
            "milvus": "up" if milvus_ok else "down",
        },
    }
