"""Security regression tests for knowledge-base-scoped evaluations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import uuid

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from backend.server.routers import evaluation_router
from backend.services import evaluation_service


KB_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
FILE_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
OWNER_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
OTHER_OWNER_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


class ScopeSpyRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [{
            "content": "matched",
            "metadata": {
                "source": "expected.pdf",
                "file_id": str(FILE_ID),
                "chunk_id": "chunk-expected",
                "score": 0.9,
            },
        }]


def test_run_evaluation_scopes_every_query_to_one_knowledge_base(monkeypatch):
    retriever = ScopeSpyRetriever()
    monkeypatch.setattr("app.rag.retriever.get_retriever", lambda: retriever)

    metrics = evaluation_service.run_evaluation(
        [
            evaluation_service.EvaluationCase(
                question="query one",
                expected_file_id=str(FILE_ID),
                expected_chunk_id="chunk-expected",
                reference_answer="reference one",
                expected_source="expected.pdf",
            ),
            evaluation_service.EvaluationCase(
                question="query two",
                expected_file_id=str(FILE_ID),
                expected_chunk_id="chunk-missing",
                reference_answer="reference two",
                expected_source="expected.pdf",
            ),
        ],
        top_k=5,
        knowledge_base_id=KB_ID,
    )

    assert metrics["hit_rate"] == 0.5
    assert metrics["hit_rate_at_k"] == 0.5
    assert metrics["mrr_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 0.1
    assert metrics["ndcg_at_k"] == 0.5
    assert metrics["file_hit_rate"] == 1.0
    assert metrics["file_recall_at_k"] == 1.0
    assert metrics["details"][0]["reference_answer"] == "reference one"
    assert metrics["details"][0]["expected_file_id"] == str(FILE_ID)
    assert metrics["details"][0]["expected_chunk_id"] == "chunk-expected"
    assert len(retriever.calls) == 2
    assert all(
        call[1]["knowledge_base_ids"] == [str(KB_ID)]
        for call in retriever.calls
    )


@asynccontextmanager
async def fake_session():
    yield SimpleNamespace(commit=_async_noop)


async def _async_noop():
    return None


class OwnedKnowledgeBaseRepository:
    owner_id = OWNER_ID

    def __init__(self, _session):
        pass

    async def get_by_id(self, kb_id):
        return SimpleNamespace(id=kb_id, owner_id=self.owner_id)


class ExpectedFileRepository:
    def __init__(self, _session):
        pass

    async def list_by_ids_for_kb(self, _kb_id, file_ids):
        return [
            SimpleNamespace(id=file_id, filename="expected.pdf")
            for file_id in file_ids
        ]


def evaluation_request():
    return evaluation_router.EvalRunRequest(
        name="安全范围测试",
        kb_id=KB_ID,
        top_k=4,
        cases=[{
            "question": "query",
            "expected_file_id": FILE_ID,
            "expected_chunk_id": "chunk-expected",
            "reference_answer": "reference answer",
        }],
    )


@pytest.mark.asyncio
async def test_create_run_validates_owner_and_passes_kb_scope(monkeypatch):
    monkeypatch.setattr(evaluation_router, "get_session", fake_session)
    monkeypatch.setattr(
        evaluation_router,
        "KnowledgeBaseRepository",
        OwnedKnowledgeBaseRepository,
    )
    monkeypatch.setattr(
        evaluation_router,
        "KnowledgeFileRepository",
        ExpectedFileRepository,
    )
    calls = []

    def fake_run(cases, top_k, *, knowledge_base_id, ragas_metrics=None):
        calls.append((cases, top_k, knowledge_base_id))
        return {
            "hit_rate": 1.0,
            "mrr": 1.0,
            "avg_score": 0.9,
            "details": [{"query": "query", "hit_rank": 1}],
        }

    async def fake_save(_session, name, metrics, top_k, kb_id, dataset_id=None):
        return SimpleNamespace(
            id=uuid.uuid4(),
            name=name,
            knowledge_base_id=kb_id,
            top_k=top_k,
            query_count=1,
            hit_rate=metrics["hit_rate"],
            mrr=metrics["mrr"],
            avg_score=metrics["avg_score"],
            metrics_json=json.dumps(metrics),
            created_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(evaluation_router, "run_evaluation", fake_run)
    monkeypatch.setattr(evaluation_router, "save_run", fake_save)

    result = await evaluation_router.create_run(
        evaluation_request(),
        current_user=SimpleNamespace(id=OWNER_ID),
    )

    assert len(calls) == 1
    cases, top_k, knowledge_base_id = calls[0]
    assert top_k == 4
    assert knowledge_base_id == KB_ID
    assert cases == [
        evaluation_service.EvaluationCase(
            question="query",
            expected_file_id=str(FILE_ID),
            expected_chunk_id="chunk-expected",
            reference_answer="reference answer",
            expected_source="expected.pdf",
        )
    ]
    assert result.knowledge_base_id == str(KB_ID)


@pytest.mark.asyncio
async def test_create_run_hides_unowned_knowledge_base(monkeypatch):
    monkeypatch.setattr(evaluation_router, "get_session", fake_session)
    monkeypatch.setattr(
        evaluation_router,
        "KnowledgeBaseRepository",
        OwnedKnowledgeBaseRepository,
    )
    monkeypatch.setattr(OwnedKnowledgeBaseRepository, "owner_id", OTHER_OWNER_ID)

    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(evaluation_router, "run_evaluation", should_not_run)

    with pytest.raises(HTTPException) as exc_info:
        await evaluation_router.create_run(
            evaluation_request(),
            current_user=SimpleNamespace(id=OWNER_ID),
        )

    assert exc_info.value.status_code == 404
    assert called is False


@pytest.mark.asyncio
async def test_create_run_rejects_expected_file_outside_selected_kb(monkeypatch):
    class MissingFileRepository:
        def __init__(self, _session):
            pass

        async def list_by_ids_for_kb(self, _kb_id, _file_ids):
            return []

    monkeypatch.setattr(evaluation_router, "get_session", fake_session)
    monkeypatch.setattr(
        evaluation_router,
        "KnowledgeBaseRepository",
        OwnedKnowledgeBaseRepository,
    )
    monkeypatch.setattr(
        evaluation_router,
        "KnowledgeFileRepository",
        MissingFileRepository,
    )

    with pytest.raises(HTTPException) as exc_info:
        await evaluation_router.create_run(
            evaluation_request(),
            current_user=SimpleNamespace(id=OWNER_ID),
        )

    assert exc_info.value.status_code == 422
    assert str(FILE_ID) in exc_info.value.detail["file_ids"]


def test_evaluation_case_requires_the_new_dataset_fields():
    with pytest.raises(ValidationError):
        evaluation_router.EvalCaseIn(question="question")

    case = evaluation_router.EvalCaseIn(
        question="  question  ",
        expected_file_id=FILE_ID,
        expected_chunk_id="  chunk-id  ",
        reference_answer="  answer  ",
    )
    assert case.question == "question"
    assert case.expected_chunk_id == "chunk-id"
    assert case.reference_answer == "answer"


class EmptyScalars:
    def all(self):
        return []


class QueryResult:
    def scalars(self):
        return EmptyScalars()

    def scalar_one_or_none(self):
        return None


class CapturingSession:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return QueryResult()


@pytest.mark.asyncio
async def test_evaluation_history_queries_are_filtered_by_owner():
    session = CapturingSession()

    await evaluation_service.list_runs(session, OWNER_ID)
    await evaluation_service.get_run(session, uuid.uuid4(), OWNER_ID)

    for statement in session.statements:
        sql = str(statement)
        assert "JOIN knowledge_bases" in sql
        assert "knowledge_bases.owner_id" in sql
