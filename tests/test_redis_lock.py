"""RedisLock 分布式锁测试（集成，需本机 Redis；连不上自动 skip）。

覆盖：获取/互斥/释放、自动续期、Lua 原子释放（token 校验）、上下文管理器。
每个测试用独立 Redis 客户端（不用进程级单例，避免跨测试事件循环冲突）。
"""
from __future__ import annotations

import asyncio

import pytest

from backend.storage.redis.lock import RedisLock

pytestmark = pytest.mark.anyio


def _redis_available():
    try:
        import redis

        r = redis.Redis(host="127.0.0.1", port=6379, socket_connect_timeout=1)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def r():
    from redis import asyncio as aioredis

    client = aioredis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    for key in ("lock:test:1", "lock:test:2", "lock:test:ctx", "lock:test:stolen"):
        await client.delete(key)
    yield client
    await client.close()


@pytest.mark.skipif(not _redis_available(), reason="Redis 不可用")
async def test_acquire_creates_key_and_release_deletes(r):
    lock = await RedisLock.acquire(r, "lock:test:1", ttl=30)
    assert lock is not None
    assert await r.exists("lock:test:1") == 1
    await lock.release()
    assert await r.exists("lock:test:1") == 0


@pytest.mark.skipif(not _redis_available(), reason="Redis 不可用")
async def test_mutual_exclusion_second_acquire_returns_none(r):
    lock_a = await RedisLock.acquire(r, "lock:test:2", ttl=30)
    assert lock_a is not None
    lock_b = await RedisLock.acquire(r, "lock:test:2", ttl=30)
    assert lock_b is None  # 非阻塞互斥
    await lock_a.release()
    lock_c = await RedisLock.acquire(r, "lock:test:2", ttl=30)
    assert lock_c is not None  # 释放后可再获取
    await lock_c.release()


@pytest.mark.skipif(not _redis_available(), reason="Redis 不可用")
async def test_renewal_keeps_lock_alive_past_ttl(r):
    lock = await RedisLock.acquire(r, "lock:test:1", ttl=2)
    assert lock is not None
    try:
        # 不续期的话 2 秒后锁过期；续期任务每 ~0.7s 刷新，4 秒后应仍在
        await asyncio.sleep(4.0)
        assert await r.exists("lock:test:1") == 1
    finally:
        await lock.release()


@pytest.mark.skipif(not _redis_available(), reason="Redis 不可用")
async def test_release_is_owner_scoped_when_lock_stolen(r):
    """锁过期被抢走后，原持有者 release 不删新持有者的锁（Lua token 校验）。"""
    lock_a = await RedisLock.acquire(r, "lock:test:stolen", ttl=1)
    assert lock_a is not None
    await asyncio.sleep(1.2)  # 等 A 的锁自然过期
    lock_b = await RedisLock.acquire(r, "lock:test:stolen", ttl=30)
    assert lock_b is not None  # B 抢到锁
    await lock_a.release()  # A 的残留释放（token 不匹配 → 不删）
    assert await r.exists("lock:test:stolen") == 1  # B 的锁完好
    await lock_b.release()
    assert await r.exists("lock:test:stolen") == 0


@pytest.mark.skipif(not _redis_available(), reason="Redis 不可用")
async def test_context_manager_usage(r):
    lock = await RedisLock.acquire(r, "lock:test:ctx", ttl=30)
    assert lock is not None
    async with lock:
        assert await r.exists("lock:test:ctx") == 1
    # 退出 with 后锁已释放
    assert await r.exists("lock:test:ctx") == 0
