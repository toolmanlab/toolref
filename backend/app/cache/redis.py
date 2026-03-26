"""Redis async connection manager."""

import logging

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return the global Redis client (creates one if needed)."""
    global _redis
    if _redis is None:
        _redis = await connect_redis()
    return _redis


async def connect_redis() -> aioredis.Redis:
    """Create and verify a Redis connection."""
    global _redis
    logger.info("Connecting to Redis at %s:%s", settings.redis_host, settings.redis_port)
    _redis = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password or None,
        db=settings.redis_db,
        decode_responses=True,
    )
    await _redis.ping()
    logger.info("Redis connection established")
    return _redis


async def close_redis() -> None:
    """Gracefully close the Redis connection."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("Redis connection closed")


async def check_redis() -> bool:
    """Return True if Redis is reachable."""
    try:
        r = await get_redis()
        return await r.ping()
    except Exception:
        return False
