"""文件索引消息队列 — Redis Stream 封装。

架构（2026-08-27，替代 FastAPI BackgroundTasks 进程内任务）：

- Stream: ``kb:ingestion``（单 stream，消费者组 ``kb:ingestion:grp`` 内多 worker 负载均衡）
- 消息字段: file_id / kb_id / filename / strategy / content_type / parser / attempts
- 语义: **at-least-once** —— XREADGROUP 消费 + 处理成功 XACK 确认；
  worker 崩溃未确认的消息留在 PEL（pending entries list），由其他 worker
  通过 XAUTOCLAIM 超时认领重跑（超时 = INGESTION_PENDING_CLAIM_MS）
- 幂等: 消费端按 file 状态跳过已 completed/failed 的消息（防重复消费跑两遍）
- 文件字节不入消息：上传时已存 MinIO，消费端按 minio_object 拉取，消息体仅几 KB
- Redis 连接复用 backend.storage.redis.manager 的进程级单例（与缓存等共用一条连接池）
"""
from __future__ import annotations

import uuid
from typing import Optional

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.storage.redis.manager import get_redis

logger = get_logger(__name__)

STREAM = "kb:ingestion"
GROUP = "kb:ingestion:grp"


async def ensure_group() -> None:
    """确保消费者组存在（幂等）。

    id="0"：组首次创建时从 stream 头部开始投递——worker 首次启动前积压的
    上传任务不会丢；重复投递由消费端幂等检查（completed/failed 跳过）兜底。
    """
    r = await get_redis()
    try:
        await r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        logger.info("[ingestion_queue] consumer group '%s' ready on '%s'", GROUP, STREAM)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            logger.warning("[ingestion_queue] xgroup_create failed: %s", exc)


async def publish_ingestion(
    file_id: uuid.UUID,
    kb_id: uuid.UUID,
    filename: str,
    strategy: Optional[str] = None,
    content_type: Optional[str] = None,
    parser: str = "auto",
) -> bool:
    """发布一条文件索引任务。返回是否成功（失败时调用方决定回退方案）。

    消息体不含文件字节（MinIO 已存），只携带定位信息，保证发布快且小。
    """
    try:
        await ensure_group()
        r = await get_redis()
        await r.xadd(STREAM, {
            "file_id": str(file_id),
            "kb_id": str(kb_id),
            "filename": filename,
            "strategy": strategy or "",
            "content_type": content_type or "",
            "parser": parser or "auto",
            "attempts": "0",
        })
        logger.info("[ingestion_queue] published: %s (file_id=%s)", filename, file_id)
        return True
    except Exception as exc:
        logger.error("[ingestion_queue] publish failed for '%s': %s", filename, exc)
        return False


async def ack_message(message_id: str) -> None:
    """确认一条消息（处理后调用，消息移出 PEL）。"""
    try:
        r = await get_redis()
        await r.xack(STREAM, GROUP, message_id)
    except Exception as exc:
        logger.warning("[ingestion_queue] xack failed %s: %s", message_id, exc)
