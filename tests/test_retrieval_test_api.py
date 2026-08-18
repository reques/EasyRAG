"""Tests for the lightweight, knowledge-base-scoped retrieval test API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
import uuid

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from backend.server.routers import knowledge_router


KB_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OWNER_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


@asynccontextmanager
async def fake_session():
    yield object()


class FakeKnowledgeBaseRepository:
    knowledge_base = SimpleNamespace(id=KB_ID, owner_id=OWNER_ID)

    def __init__(self, _session):
        pass

    async def get_by_id(self, _kb_id):
        return self.knowledge_base


class SpyRetriever:
    def __init__(self):
        self.call = None

    def retrieve(self, query, **kwargs):
        self.call = (query, kwargs)
        return [
            {
                "content": "完整的命中文档片段",
                "metadata": {
                    "score": 0.87,
                    "source": "paper.pdf",
                    "file_id": "file-1",
                    "chunk_index": "12",
                    "page_start": 3,
                    "page_end": 4,
                    "section_path": "Results",
                    "parser_name": "mineru",
                    "parent_text": "should not be repeated in metadata",
                },
            }
        ]


def configure_owned_kb(monkeypatch):
    monkeypatch.setattr(knowledge_router, "get_session", fake_session)
    monkeypatch.setattr(
        knowledge_router,
        "KnowledgeBaseRepository",
        FakeKnowledgeBaseRepository,
    )


@pytest.mark.asyncio
async def test_retrieval_test_returns_ranked_hits_and_propagates_scope(monkeypatch):
    configure_owned_kb(monkeypatch)
    retriever = SpyRetriever()
    monkeypatch.setattr("app.rag.retriever.get_retriever", lambda: retriever)

    response = await knowledge_router.test_retrieval(
        KB_ID,
        knowledge_router.RetrievalTestRequest(
            query="  核心结论是什么？  ",
            top_k=5,
            score_threshold=0,
        ),
        current_user=SimpleNamespace(id=OWNER_ID),
    )

    query, kwargs = retriever.call
    assert query == "核心结论是什么？"
    assert kwargs["top_k"] == 5
    assert kwargs["knowledge_base_ids"] == [str(KB_ID)]
    assert kwargs["score_threshold"] == -1.0

    assert response.knowledge_base_id == str(KB_ID)
    assert response.total == 1
    assert response.elapsed_ms >= 0
    assert response.results[0].rank == 1
    assert len(response.results[0].chunk_id) == 64
    assert response.results[0].score == pytest.approx(0.87)
    assert response.results[0].source == "paper.pdf"
    assert response.results[0].chunk_index == 12
    assert "parent_text" not in response.results[0].metadata


@pytest.mark.asyncio
async def test_retrieval_test_hides_unowned_knowledge_base(monkeypatch):
    configure_owned_kb(monkeypatch)
    monkeypatch.setattr(
        FakeKnowledgeBaseRepository,
        "knowledge_base",
        SimpleNamespace(
            id=KB_ID,
            owner_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await knowledge_router.test_retrieval(
            KB_ID,
            knowledge_router.RetrievalTestRequest(query="query"),
            current_user=SimpleNamespace(id=OWNER_ID),
        )

    assert exc_info.value.status_code == 404


def test_retrieval_test_request_rejects_blank_query_and_invalid_parameters():
    with pytest.raises(ValidationError):
        knowledge_router.RetrievalTestRequest(query="   ")
    with pytest.raises(ValidationError):
        knowledge_router.RetrievalTestRequest(query="query", top_k=0)
    with pytest.raises(ValidationError):
        knowledge_router.RetrievalTestRequest(
            query="query",
            score_threshold=1.1,
        )
