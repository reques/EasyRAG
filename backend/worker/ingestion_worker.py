"""文件索引 worker — 消费 Redis Stream 任务。

两种运行方式（2026-08-27）：
1. **内嵌后端进程**（推荐）：FastAPI lifespan 自动启动/停止（main.py），
   前后端两个终端即可跑通全链路，无需额外终端。
2. **独立进程**：``python -m backend.worker.ingestion_worker [--concurrency N]``，
   适合多机/多 worker 扩展（消费组内多消费者自动负载均衡）。

为什么用 Redis Stream（替代 BackgroundTasks 进程内任务）：
- 原 BackgroundTasks 在 API 进程内，uvicorn 重启/崩溃 → 任务丢失，文件状态卡死
- 现在消息在 Redis Stream 中持久化，worker 崩溃未确认的消息留在 PEL，
  由 XAUTOCLAIM 超时认领重跑 → 任务不丢
"""
from __future__ import annotations

import argparse
import asyncio
import os
import socket
import uuid
from typing import Set

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.repositories.knowledge_repository import KnowledgeFileRepository
from backend.services.ingestion_queue import (
    GROUP,
    STREAM,
    ack_message,
    ensure_group,
)
from backend.services.ingestion_service import (
    fetch_raw_from_minio,
    run_ingestion,
)
from backend.services.knowledge_service import update_file_progress
from backend.storage.postgres.manager import get_session
from backend.storage.redis.lock import RedisLock
from backend.storage.redis.manager import get_redis

logger = get_logger(__name__)
cfg = get_settings()

_stopping = False


async def handle_message(redis_client, message_id: str, fields: dict, sem: asyncio.Semaphore) -> None:
    """处理单条消息：幂等检查 → 并发闸门 → MinIO 拉取 → run_ingestion → ACK。

    任何失败路径都以「标记 failed + ACK」收尾，保证消息不会在 PEL 中死循环；
    用户从文件列表看到 failed 后手动重试（与旧 BackgroundTasks 体验一致）。
    """
    try:
        file_id = uuid.UUID(fields.get("file_id", ""))
        kb_id = uuid.UUID(fields.get("kb_id", ""))
        filename = fields.get("filename", "unknown")

        # ── 幂等检查 + 取 MinIO 定位（一次查询） ──
        async with get_session() as s:
            repo = KnowledgeFileRepository(s)
            f = await repo.get_by_id(file_id)
            if f is None:
                logger.warning("[ingestion_worker] file %s not found, ack skip", file_id)
                await ack_message(message_id)
                return
            if f.status in ("completed", "failed"):
                logger.info("[ingestion_worker] skip %s (status=%s), ack", filename, f.status)
                await ack_message(message_id)
                return
            minio_bucket = f.minio_bucket or cfg.MINIO_BUCKET
            minio_object = f.minio_object

        if not minio_object:
            logger.error("[ingestion_worker] %s has no minio_object, mark failed", filename)
            async with get_session() as s:
                await update_file_progress(
                    s, file_id, 100, status="failed",
                    error_message="文件未写入对象存储，无法处理",
                    stage="failed", message="处理失败",
                )
            await ack_message(message_id)
            return

        # ── 文件处理锁（2026-08-27：RedisLock 工厂，短 TTL + 自动续期 + Lua 原子性） ──
        # 防 XAUTOCLAIM 认领重入 / 多 worker 并发处理同一文件：
        # 锁存在 = 另一实例正在处理 → 跳过。
        lock = await RedisLock.acquire(
            redis_client, f"ingestion:lock:{file_id}", cfg.INGESTION_LOCK_TTL
        )
        if lock is None:
            logger.info("[ingestion_worker] %s already being processed, skip (lock held)", filename)
            await ack_message(message_id)
            return

        try:
            # ── 全局并发闸门：同时最多 INGESTION_CONCURRENCY 个文件 ──
            async with sem:
                raw = await fetch_raw_from_minio(minio_bucket, minio_object)
                if raw is None:
                    async with get_session() as s:
                        await update_file_progress(
                            s, file_id, 100, status="failed",
                            error_message="从对象存储读取文件失败",
                            stage="failed", message="处理失败",
                        )
                else:
                    await run_ingestion(
                        file_id,
                        kb_id,
                        raw,
                        filename,
                        fields.get("strategy") or None,
                        fields.get("content_type") or None,
                        fields.get("parser") or "auto",
                    )
                await ack_message(message_id)
        finally:
            # 停止续期并原子释放锁（仅释放自己持有的；异常路径由外层 except 兜底）
            await lock.release()

    except Exception as exc:
        logger.error("[ingestion_worker] message %s failed: %s", message_id, exc)
        try:
            fid = fields.get("file_id")
            if fid:
                async with get_session() as s:
                    await update_file_progress(
                        s, uuid.UUID(fid), 100, status="failed",
                        error_message=f"worker 处理异常: {str(exc)[:400]}",
                        stage="failed", message="处理失败",
                    )
        except Exception:
            logger.exception("[ingestion_worker] failed to mark error status")
        await ack_message(message_id)


async def main_loop(concurrency: int) -> None:
    global _stopping
    r = await get_redis()
    await ensure_group()
    consumer = f"worker-{socket.gethostname()}-{os.getpid()}"
    sem = asyncio.Semaphore(concurrency)
    pending: Set[asyncio.Task] = set()
    in_flight: Set[str] = set()  # 正在处理的消息 ID（防 XAUTOCLAIM 重入同一消息）
    logger.info("[ingestion_worker] consumer=%s concurrency=%d stream=%s group=%s",
                consumer, concurrency, STREAM, GROUP)

    def spawn(coro, message_id: str) -> None:
        if message_id in in_flight:
            # 已在处理中（比如刚被 XREADGROUP 投递，又被 XAUTOCLAIM 认领到）→ 不重复处理
            logger.info("[ingestion_worker] message %s already in flight, skip respawn", message_id)
            return
        if len(pending) >= concurrency * 4:  # 简单背压：排队任务过多时等一等
            return
        in_flight.add(message_id)
        task = asyncio.create_task(coro)
        pending.add(task)
        task.add_done_callback(lambda t, mid=message_id: (pending.discard(t), in_flight.discard(mid)))

    while not _stopping:
        # 1) 认领超时 pending 消息（worker 崩溃遗留），优先处理
        try:
            claimed = await r.xautoclaim(
                STREAM, GROUP, consumer, cfg.INGESTION_PENDING_CLAIM_MS, "0-0", count=8
            )
            for mid, flds in claimed[1]:
                spawn(handle_message(r, mid, flds, sem), mid)
        except Exception as exc:
            logger.warning("[ingestion_worker] xautoclaim failed: %s", exc)

        # 2) 读新消息（阻塞 5s，无消息则继续循环做认领）
        try:
            resp = await r.xreadgroup(
                GROUP, consumer, {STREAM: ">"}, count=concurrency * 2, block=5000
            )
        except Exception as exc:
            logger.warning("[ingestion_worker] xreadgroup failed: %s", exc)
            await asyncio.sleep(1)
            continue

        if resp:
            for _, msgs in resp:
                for mid, flds in msgs:
                    spawn(handle_message(r, mid, flds, sem), mid)
        await asyncio.sleep(0)  # 让出事件循环

    # 优雅停机：等待已领取的任务完成（新消息不再接收）
    if pending:
        logger.info("[ingestion_worker] draining %d in-flight task(s)...", len(pending))
        await asyncio.gather(*pending, return_exceptions=True)
    logger.info("[ingestion_worker] stopped")


def stop_worker() -> None:
    """请求 worker 停止（lifespan shutdown / 独立进程 signal handler 调用）。"""
    global _stopping
    _stopping = True


async def _run_worker_guarded(concurrency: int) -> None:
    """worker 主循环包装：异常只记日志不向上抛，避免拖垮宿主（API 进程）。"""
    try:
        await main_loop(concurrency)
    except Exception as exc:
        logger.error("[ingestion_worker] worker loop crashed: %s", exc)


def start_worker(concurrency: int) -> asyncio.Task:
    """以后台任务方式启动 worker（FastAPI lifespan 集成用）。

    返回 task 句柄，shutdown 时先 ``stop_worker()`` 再 ``await task`` 排空存量。
    """
    return asyncio.create_task(_run_worker_guarded(max(1, concurrency)))


def main() -> None:
    parser = argparse.ArgumentParser(description="EasyRAG ingestion worker")
    parser.add_argument("--concurrency", type=int, default=cfg.INGESTION_CONCURRENCY)
    args = parser.parse_args()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        for sig in (__import__("signal").SIGINT, __import__("signal").SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_worker)
            except NotImplementedError:
                pass  # Windows: signal handler 受限，fallback 到 KeyboardInterrupt
        try:
            await main_loop(max(1, args.concurrency))
        finally:
            from backend.storage.redis.manager import close_redis
            await close_redis()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("[ingestion_worker] interrupted")


if __name__ == "__main__":
    main()
