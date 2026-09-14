"""知识库删除接口单测 — DELETE /bases/{kb_id} 与 DELETE /bases/{kb_id}/files。

纯 mock，不连数据库/Minio/Milvus。重点验证：
- 归属校验（404，不触发任何清理）；
- 「删行优先」顺序：PG commit 必须发生在向量/图谱/MinIO 清理之前；
- 单步清理失败不阻塞、返回计数正确。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.server.routers.knowledge_router import (
    delete_all_kb_files,
    delete_kb,
)
from backend.storage.postgres.models_knowledge import KnowledgeBase
from backend.storage.postgres.models_user import User
from app.rag.retriever import MemoryRetriever

KB_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OWNER_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
OTHER_OWNER = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _make_kb(owner_id: uuid.UUID = OWNER_ID) -> KnowledgeBase:
    return KnowledgeBase(
        id=KB_ID,
        name="测试库",
        description=None,
        owner_id=owner_id,
        collection_name="kb_test",
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _make_user() -> User:
    return User(
        id=OWNER_ID,
        username="tester",
        hashed_password="x",
        role="user",
        is_active=True,
        is_superuser=False,
    )


def _make_file_row(index: int, minio_object: str | None = "obj") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.UUID(f"{index:032x}"),
        filename=f"file{index}.md",
        minio_bucket="easyrag-files",
        minio_object=(
            f"kb/{KB_ID}/{index:032x}/file{index}.md" if minio_object else None
        ),
    )


class FakeSession:
    """记录 execute/commit 调用序，供「删行优先」顺序断言。"""

    def __init__(self, calls: list) -> None:
        self.calls = calls

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def execute(self, stmt) -> None:
        self.calls.append("db_delete")

    async def commit(self) -> None:
        self.calls.append("db_commit")


class FakeKbRepo:
    def __init__(self, kb: KnowledgeBase | None, calls: list) -> None:
        self.kb = kb
        self.calls = calls

    async def get_by_id(self, kb_id):
        return self.kb

    async def delete(self, entity) -> None:
        self.calls.append("kb_row_delete")


class FakeFileRepo:
    def __init__(self, files, calls: list) -> None:
        self.files = files
        self.calls = calls

    async def list_all_by_kb(self, kb_id):
        self.calls.append("list_files")
        return self.files


class FakeRetriever:
    def __init__(self, calls: list, count: int = 7, fail: bool = False) -> None:
        self.calls = calls
        self.count = count
        self.fail = fail

    def delete_documents_by_kb(self, kb_id: str) -> int:
        self.calls.append("vectors")
        if self.fail:
            raise RuntimeError("milvus down")
        return self.count


class FakeNeo4jClearer:
    def __init__(self, calls: list, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def clear_kb(self, kb_str: str) -> None:
        self.calls.append("graph")
        if self.fail:
            raise RuntimeError("neo4j down")


class FakeMinio:
    def __init__(self, calls: list, keys: list[str]) -> None:
        self.calls = calls
        self.keys = keys
        self.removed: list[tuple[str, str]] = []
        self.listed_prefix: str | None = None

    def list_objects(self, bucket, prefix=None, recursive=False):
        self.listed_prefix = prefix
        self.calls.append("minio_list")
        return [SimpleNamespace(object_name=k) for k in self.keys]

    def remove_object(self, bucket, name) -> None:
        self.removed.append((bucket, name))
        self.calls.append("minio_remove")


def _patch_stack(kb, files, calls, *, vectors_fail=False, keys=None):
    """构造被测代码所需的全部 mock（context manager 集合）。"""
    keys = keys if keys is not None else [f"kb/{KB_ID}/row/fileN.md"]
    ctx_minio = FakeMinio(calls, keys)

    async def fake_reset(kb_id):
        calls.append("graph")

    stack = [
        patch(
            "backend.server.routers.knowledge_router.get_session",
            return_value=FakeSession(calls),
        ),
        patch(
            "backend.server.routers.knowledge_router.KnowledgeBaseRepository",
            return_value=FakeKbRepo(kb, calls),
        ),
        patch(
            "backend.server.routers.knowledge_router.KnowledgeFileRepository",
            return_value=FakeFileRepo(files, calls),
        ),
        patch(
            "backend.services.graph_build_service.reset_kb_graph",
            fake_reset,
        ),
        patch(
            "app.rag.retriever.get_retriever",
            return_value=FakeRetriever(calls, count=7, fail=vectors_fail),
        ),
        patch(
            "backend.storage.minio.client.get_minio_client",
            return_value=ctx_minio,
        ),
    ]
    return stack, ctx_minio


def _enter(stack):
    from contextlib import ExitStack

    exit_stack = ExitStack()
    for p in stack:
        exit_stack.enter_context(p)
    return exit_stack


# ── 归属与存在性 ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [delete_all_kb_files, delete_kb])
async def test_foreign_or_missing_kb_returns_404_without_cleanup(endpoint):
    calls: list = []
    kb = None if endpoint is delete_kb else _make_kb(owner_id=OTHER_OWNER)
    if endpoint is delete_all_kb_files:
        kb = _make_kb(owner_id=OTHER_OWNER)
    stack, _ = _patch_stack(kb, [], calls)
    with _enter(stack):
        with pytest.raises(HTTPException) as exc:
            await endpoint(str(KB_ID), _make_user())
    assert exc.value.status_code == 404
    assert calls == []  # 未触发任何清理


# ── 删除全部文件 ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_all_files_rows_committed_before_cleanups():
    calls: list = []
    files = [_make_file_row(1), _make_file_row(2)]
    stack, minio = _patch_stack(_make_kb(), files, calls,
                                keys=[f"kb/{KB_ID}/a/x.md", f"kb/{KB_ID}/b/y.md"])
    with _enter(stack):
        resp = await delete_all_kb_files(str(KB_ID), _make_user())

    assert resp.deleted_files == 2
    assert resp.deleted_vectors == 7
    # 删行 → commit 必须早于向量/图谱/MinIO 清理
    assert calls.index("db_commit") < calls.index("vectors")
    assert calls.index("db_commit") < calls.index("graph")
    assert calls.index("db_commit") < calls.index("minio_list")
    # MinIO 按 kb 前缀清理，两对象全删
    assert minio.listed_prefix == f"kb/{KB_ID}/"
    assert len(minio.removed) == 2


@pytest.mark.asyncio
async def test_delete_all_files_survives_vector_failure():
    calls: list = []
    files = [_make_file_row(1)]
    stack, _ = _patch_stack(_make_kb(), files, calls, vectors_fail=True)
    with _enter(stack):
        resp = await delete_all_kb_files(str(KB_ID), _make_user())

    assert resp.deleted_files == 1
    assert resp.deleted_vectors == 0
    assert "db_commit" in calls
    assert "graph" in calls  # 后续步骤照常执行


@pytest.mark.asyncio
async def test_delete_all_files_empty_kb_is_idempotent():
    calls: list = []
    stack, minio = _patch_stack(_make_kb(), [], calls, keys=[])
    with _enter(stack):
        resp = await delete_all_kb_files(str(KB_ID), _make_user())

    assert resp.deleted_files == 0
    assert resp.deleted_vectors == 7  # mock 返回值；真实空库为 0
    assert minio.removed == []


# ── 删除知识库 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_kb_purges_contents_then_deletes_row():
    calls: list = []
    files = [_make_file_row(1), _make_file_row(2, minio_object=None)]
    stack, _ = _patch_stack(_make_kb(), files, calls)
    with _enter(stack):
        resp = await delete_kb(str(KB_ID), _make_user())

    assert resp.deleted_files == 2
    assert "kb_row_delete" in calls
    assert calls.index("db_commit") < calls.index("kb_row_delete")
    assert calls.count("vectors") == 1
    assert calls.count("graph") == 1


# ── MemoryRetriever.delete_documents_by_kb 回归 ───────────────────────────────

def test_memory_retriever_delete_by_kb_only_touches_target_kb():
    kb_a, kb_b = "aaaa", "bbbb"
    r = MemoryRetriever()
    r._texts = ["a1", "a2", "b1"]
    r._metas = [
        {"knowledge_base_id": kb_a},
        {"knowledge_base_id": kb_a},
        {"knowledge_base_id": kb_b},
    ]
    r._vecs = [[0.1], [0.2], [0.3]]

    deleted = r.delete_documents_by_kb(kb_a)

    assert deleted == 2
    assert r._texts == ["b1"]
    assert r._metas == [{"knowledge_base_id": kb_b}]
    assert r._vecs == [[0.3]]
