"""Redis 客户端封装 — 缓存 / 会话 / 任务队列。

用法::

    from backend.storage.redis.manager import get_redis
    r = await get_redis()
    await r.set("key", "value", ex=3600)
"""

from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """返回 Redis 客户端单例（自动连接）。"""
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=cfg.REDIS_HOST,
            port=cfg.REDIS_PORT,
            db=cfg.REDIS_DB,
            password=cfg.REDIS_PASSWORD or None,
            decode_responses=True,
        )
        await _redis.ping()
        logger.info("[redis] connected to %s:%d", cfg.REDIS_HOST, cfg.REDIS_PORT)
    return _redis


async def close_redis() -> None:
    """关闭 Redis 连接（应用关闭时调用）。"""
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
        logger.info("[redis] connection closed")
