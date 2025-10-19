"""Redis caching utilities."""

import json
from typing import Any

import redis.asyncio as redis

from src.core.config import get_config

# Global Redis client
_redis_client = None


async def get_redis_client() -> redis.Redis | None:
    """Get or create Redis client."""
    global _redis_client

    if _redis_client is None:
        config = get_config()
        try:
            _redis_client = redis.from_url(config.redis_url)
            await _redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None

    return _redis_client


async def cache_get(key: str) -> Any | None:
    """Get value from cache."""
    client = await get_redis_client()
    if not client:
        return None

    try:
        value = await client.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        print(f"Cache get failed: {e}")

    return None


async def cache_set(key: str, value: Any, ttl: int | None = None) -> bool:
    """Set value in cache with optional TTL."""
    client = await get_redis_client()
    if not client:
        return False

    try:
        serialized = json.dumps(value)
        if ttl:
            await client.setex(key, ttl, serialized)
        else:
            await client.set(key, serialized)
        return True
    except Exception as e:
        print(f"Cache set failed: {e}")
        return False

