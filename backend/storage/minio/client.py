"""MinIO 客户端封装 — 文件/文档对象存储。

用法::

    from backend.storage.minio.client import get_minio_client, ensure_bucket
    client = get_minio_client()
    await ensure_bucket()
    client.put_object(bucket, object_name, data, length)
"""

from __future__ import annotations

from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_settings()

_client: Minio | None = None


def get_minio_client() -> Minio:
    """返回 MinIO 客户端单例。"""
    global _client
    if _client is None:
        _client = Minio(
            endpoint=cfg.MINIO_ENDPOINT,
            access_key=cfg.MINIO_ACCESS_KEY,
            secret_key=cfg.MINIO_SECRET_KEY,
            secure=cfg.MINIO_SECURE,
        )
        logger.info("[minio] client created, endpoint=%s", cfg.MINIO_ENDPOINT)
    return _client


def ensure_bucket(bucket_name: str | None = None) -> None:
    """确保存储桶存在，不存在则创建。"""
    client = get_minio_client()
    name = bucket_name or cfg.MINIO_BUCKET
    if not client.bucket_exists(name):
        client.make_bucket(name)
        logger.info("[minio] bucket '%s' created", name)
    else:
        logger.debug("[minio] bucket '%s' already exists", name)
