"""Redis 分布式锁（短 TTL + 自动续期 + Lua 原子释放/续期）。

设计（2026-08-27，替代内嵌在 ingestion_worker 里的裸锁逻辑）：

- **获取**：``SET key token NX EX ttl`` —— 单命令原子；value 为每次获取生成的
  唯一 token（uuid4.hex），是持有者校验的基础。
- **续期**：持有者后台任务每 ``ttl/3`` 秒经 Lua ``_RENEW_SCRIPT`` 比对 token 后
  ``EXPIRE``——处理多久锁都有效；token 不一致（锁已被新持有者获取）则不续期，
  不误伤新持有者。
- **释放**：Lua ``_UNLOCK_SCRIPT`` 比对 token 一致才 ``DEL``——锁过期被抢走后，
  原持有者的释放不会删掉新持有者的锁（GET+DEL 两步原子化，消除竞态）。
- **崩溃恢复**：worker 崩溃后锁在 ttl 内自动过期，配合消息认领超时可快速重跑。

用法::

    lock = await RedisLock.acquire(redis, key, ttl=30)
    if lock is None:
        ...  # 未获取到锁（他人持有），由调用方决定跳过/等待
    try:
        ...
    finally:
        await lock.release()

也支持上下文管理器::

    async with lock:   # 仅限已成功 acquire 的锁对象
        ...
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Optional

# 释放锁：比对 value（持有者 token）一致才删除——防锁过期被抢走后，
# 原持有者误删新持有者的锁（GET+DEL 两条命令之间有竞态，脚本原子化）。
_UNLOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""
# 续期锁：比对 value 一致才 EXPIRE——防原持有者的续期任务误续新持有者的锁。
_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""


class RedisLock:
    """Redis 分布式锁：工厂 ``acquire`` 获取，``release`` / ``async with`` 释放。

    未获取到锁时 ``acquire`` 返回 ``None``（非阻塞）。
    """

    __slots__ = (
        "_redis", "_key", "_ttl", "_token",
        "_renewal_stop", "_renewal_task", "_acquired",
    )

    def __init__(self, redis, key: str, ttl: int) -> None:
        self._redis = redis
        self._key = key
        self._ttl = max(1, ttl)
        self._token = uuid.uuid4().hex
        self._renewal_stop: Optional[asyncio.Event] = None
        self._renewal_task: Optional[asyncio.Task] = None
        self._acquired = False

    # ── 工厂 ────────────────────────────────────────────────────────────────

    @classmethod
    async def acquire(cls, redis, key: str, ttl: int) -> "Optional[RedisLock]":
        """尝试获取锁；成功返回锁对象（已启动自动续期），失败返回 None。"""
        lock = cls(redis, key, ttl)
        ok = await redis.set(key, lock._token, nx=True, ex=lock._ttl)
        if not ok:
            return None
        lock._acquired = True
        lock._start_renewal()
        return lock

    # ── 续期 ────────────────────────────────────────────────────────────────

    def _start_renewal(self) -> None:
        self._renewal_stop = asyncio.Event()
        self._renewal_task = asyncio.create_task(self._renew())

    async def _renew(self) -> None:
        interval = max(1, self._ttl / 3)
        while not self._renewal_stop.is_set():
            await asyncio.sleep(interval)
            try:
                # Lua 比对 token 才续期，防止续到新持有者的锁上
                await self._redis.eval(
                    _RENEW_SCRIPT, 1, self._key, self._token, self._ttl
                )
            except Exception:
                pass  # Redis 短暂不可用：锁自然过期，由上层幂等/认领兜底

    # ── 释放 ────────────────────────────────────────────────────────────────

    async def release(self) -> None:
        """停止续期并原子释放锁（幂等，可重复调用）。"""
        if self._renewal_stop is not None:
            self._renewal_stop.set()
        if self._renewal_task is not None:
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except (asyncio.CancelledError, Exception):
                pass
            self._renewal_task = None
        if self._acquired:
            try:
                # Lua 比对 token 才删除：只释放自己持有的锁
                await self._redis.eval(_UNLOCK_SCRIPT, 1, self._key, self._token)
            except Exception:
                pass
            self._acquired = False

    # ── 上下文管理器 ────────────────────────────────────────────────────────

    async def __aenter__(self) -> "RedisLock":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        await self.release()
        return False
