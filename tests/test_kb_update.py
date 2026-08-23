"""知识库更新接口(PATCH /knowledge/bases/{kb_id})单测 — 纯 mock, 不连数据库。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.server.routers.knowledge_router import KBUpdateRequest, update_kb
from backend.services.knowledge_service import update_knowledge_base
from backend.storage.postgres.models_knowledge import KnowledgeBase
from backend.storage.postgres.models_user import User

KB_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
OWNER_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
OTHER_OWNER = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _make_kb(
    name: str = "旧名字",
    description: str | None = "旧描述",
    kb_id: uuid.UUID = KB_ID,
    owner_id: uuid.UUID = OWNER_ID,
) -> KnowledgeBase:
    return KnowledgeBase(
        id=kb_id,
        name=name,
        description=description,
        owner_id=owner_id,
        collection_name="kb_test_collection",
        created_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
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


class FakeSession:
    def __init__(self) -> None:
        self.committed = False
        self.refreshed: list = []

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def commit(self) -> None:
        self.committed = True

    async def refresh(self, obj) -> None:
        self.refreshed.append(obj)


class FakeRepo:
    def __init__(self, kb: KnowledgeBase | None, duplicate: KnowledgeBase | None = None) -> None:
        self.kb = kb
        self.duplicate = duplicate
        self.get_by_id_calls: list = []
        self.get_by_name_calls: list = []

    async def get_by_id(self, kb_id: uuid.UUID):
        self.get_by_id_calls.append(kb_id)
        return self.kb

    async def get_by_name(self, name: str, owner_id: uuid.UUID):
        self.get_by_name_calls.append((name, owner_id))
        return self.duplicate


# ── Schema 校验 ───────────────────────────────────────────────────────────────

def test_update_request_requires_at_least_one_field():
    with pytest.raises(ValidationError):
        KBUpdateRequest()


def test_update_request_accepts_single_field():
    assert KBUpdateRequest(name="新名字").name == "新名字"
    assert KBUpdateRequest(description="只改描述").description == "只改描述"


def test_update_request_rejects_blank_name():
    with pytest.raises(ValidationError):
        KBUpdateRequest(name="")


# ── Service 层 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_knowledge_base_only_sets_non_none_fields():
    kb = _make_kb()
    await update_knowledge_base(None, kb, name="新名字")
    assert kb.name == "新名字"
    assert kb.description == "旧描述"  # 未提供 → 保持不变


# ── Router 层 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_kb_success_renames_and_sets_description():
    kb = _make_kb()
    session = FakeSession()
    repo = FakeRepo(kb=kb, duplicate=None)
    with (
        patch("backend.server.routers.knowledge_router.get_session", return_value=session),
        patch("backend.server.routers.knowledge_router.KnowledgeBaseRepository", return_value=repo),
    ):
        resp = await update_kb(
            str(KB_ID),
            KBUpdateRequest(name="新名字", description="新描述"),
            _make_user(),
        )

    assert kb.name == "新名字"
    assert kb.description == "新描述"
    assert session.committed
    assert session.refreshed == [kb]
    assert resp.id == str(KB_ID)
    assert resp.name == "新名字"
    assert resp.description == "新描述"
    assert resp.collection_name == "kb_test_collection"


@pytest.mark.asyncio
async def test_update_kb_description_only_keeps_name():
    kb = _make_kb()
    session = FakeSession()
    repo = FakeRepo(kb=kb)
    with (
        patch("backend.server.routers.knowledge_router.get_session", return_value=session),
        patch("backend.server.routers.knowledge_router.KnowledgeBaseRepository", return_value=repo),
    ):
        resp = await update_kb(str(KB_ID), KBUpdateRequest(description="新描述"), _make_user())

    assert kb.name == "旧名字"
    assert kb.description == "新描述"
    assert resp.name == "旧名字"


@pytest.mark.asyncio
async def test_update_kb_404_when_not_found():
    session = FakeSession()
    repo = FakeRepo(kb=None)
    with (
        patch("backend.server.routers.knowledge_router.get_session", return_value=session),
        patch("backend.server.routers.knowledge_router.KnowledgeBaseRepository", return_value=repo),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_kb(str(KB_ID), KBUpdateRequest(name="新名字"), _make_user())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_kb_404_when_not_owner():
    kb = _make_kb(owner_id=OTHER_OWNER)
    session = FakeSession()
    repo = FakeRepo(kb=kb)
    with (
        patch("backend.server.routers.knowledge_router.get_session", return_value=session),
        patch("backend.server.routers.knowledge_router.KnowledgeBaseRepository", return_value=repo),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_kb(str(KB_ID), KBUpdateRequest(name="新名字"), _make_user())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_kb_409_on_duplicate_name():
    kb = _make_kb()
    duplicate = _make_kb(name="新名字", kb_id=OTHER_ID)
    session = FakeSession()
    repo = FakeRepo(kb=kb, duplicate=duplicate)
    with (
        patch("backend.server.routers.knowledge_router.get_session", return_value=session),
        patch("backend.server.routers.knowledge_router.KnowledgeBaseRepository", return_value=repo),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_kb(str(KB_ID), KBUpdateRequest(name="新名字"), _make_user())

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_kb_same_name_is_not_duplicate():
    """名字未变化时不触发重名检查(排除自身)。"""
    kb = _make_kb(name="旧名字")
    session = FakeSession()
    repo = FakeRepo(kb=kb, duplicate=kb)  # get_by_name 会返回自身
    with (
        patch("backend.server.routers.knowledge_router.get_session", return_value=session),
        patch("backend.server.routers.knowledge_router.KnowledgeBaseRepository", return_value=repo),
    ):
        resp = await update_kb(
            str(KB_ID),
            KBUpdateRequest(name="旧名字", description="新描述"),
            _make_user(),
        )

    assert resp.name == "旧名字"
    assert kb.description == "新描述"
    assert repo.get_by_name_calls == []  # 名字没变 → 不查询重名
