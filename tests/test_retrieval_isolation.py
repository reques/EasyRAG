"""Regression tests for fail-closed, knowledge-base-scoped retrieval."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.rag.bm25 import BM25Retriever
from app.rag.enhanced_retriever import EnhancedRetriever
from app.rag.graph_cache import GraphCache
from app.rag.retriever import (
    ChromaRetriever,
    MemoryRetriever,
    MilvusRetriever,
    get_document_chunk_id,
)
from app.graph import nodes


KB_A = "11111111-1111-1111-1111-111111111111"
KB_B = "22222222-2222-2222-2222-222222222222"


class FakeEmbedder:
    def embed_query(self, _query):
        return [1.0, 0.0]


def test_public_chunk_id_is_stable_and_honours_an_explicit_id():
    metadata = {"source": "paper.pdf", "chunk_index": 3}
    first = get_document_chunk_id(KB_A, "content", metadata)
    second = get_document_chunk_id(KB_A, "content", dict(metadata))

    assert first == second
    assert len(first) == 64
    assert get_document_chunk_id(
        KB_A,
        "content",
        {**metadata, "chunk_id": "stored-chunk-id"},
    ) == "stored-chunk-id"


def test_memory_retriever_denies_empty_scope_and_filters_before_top_k():
    retriever = MemoryRetriever()
    retriever._texts = ["tenant A", "tenant B"]
    retriever._vecs = [[1.0, 0.0], [1.0, 0.0]]
    retriever._metas = [
        {"knowledge_base_id": KB_A, "source": "a.txt"},
        {"knowledge_base_id": KB_B, "source": "b.txt"},
    ]

    assert retriever.retrieve("query") == []
    with patch("app.rag.retriever.get_embedder", return_value=FakeEmbedder()):
        docs = retriever.retrieve("query", top_k=10, knowledge_base_ids=[KB_A])

    assert [doc["content"] for doc in docs] == ["tenant A"]


def test_memory_retriever_accepts_a_per_request_score_threshold():
    retriever = MemoryRetriever()
    retriever._texts = ["negative but requested for debugging"]
    retriever._vecs = [[-1.0, 0.0]]
    retriever._metas = [{"knowledge_base_id": KB_A, "source": "debug.txt"}]

    with patch("app.rag.retriever.get_embedder", return_value=FakeEmbedder()):
        default_docs = retriever.retrieve("query", knowledge_base_ids=[KB_A])
        unfiltered_docs = retriever.retrieve(
            "query",
            knowledge_base_ids=[KB_A],
            score_threshold=-1.0,
        )

    assert default_docs == []
    assert [doc["content"] for doc in unfiltered_docs] == [
        "negative but requested for debugging"
    ]


def test_invalid_scope_id_is_rejected_before_query_execution():
    retriever = MemoryRetriever()
    retriever._texts = ["tenant A"]
    retriever._vecs = [[1.0, 0.0]]
    retriever._metas = [{"knowledge_base_id": KB_A}]

    with pytest.raises(ValueError):
        retriever.retrieve("query", knowledge_base_ids=["not-a-uuid"])


def test_enhanced_retriever_denies_empty_scope_before_any_backend_work():
    retriever = EnhancedRetriever.__new__(EnhancedRetriever)

    result = retriever.retrieve("query")

    assert result.raw_docs == []
    assert result.knowledge_blocks == []


class FakeHit:
    def __init__(self, content, kb_id):
        self.score = 1.0
        self.entity = {
            "content": content,
            "source": f"{content}.txt",
            "knowledge_base_id": kb_id,
        }


class FakeMilvusCollection:
    def __init__(self):
        self.search_kwargs = None

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        # Return an out-of-scope hit deliberately to verify defence in depth.
        return [[FakeHit("tenant A", KB_A), FakeHit("tenant B", KB_B)]]


def test_milvus_retriever_pushes_scope_into_query_and_rechecks_hits():
    retriever = MilvusRetriever.__new__(MilvusRetriever)
    retriever._col = FakeMilvusCollection()

    with patch("app.rag.retriever.get_embedder", return_value=FakeEmbedder()):
        docs = retriever.retrieve("query", knowledge_base_ids=[KB_A])

    assert retriever._col.search_kwargs["expr"] == (
        f'knowledge_base_id in ["{KB_A}"]'
    )
    assert [doc["content"] for doc in docs] == ["tenant A"]


class FakeChromaCollection:
    def __init__(self):
        self.query_kwargs = None

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [["tenant A", "tenant B"]],
            "metadatas": [[
                {"knowledge_base_id": KB_A},
                {"knowledge_base_id": KB_B},
            ]],
            "distances": [[0.0, 0.0]],
        }


def test_chroma_retriever_pushes_scope_into_query_and_rechecks_hits():
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever._col = FakeChromaCollection()

    with patch("app.rag.retriever.get_embedder", return_value=FakeEmbedder()):
        docs = retriever.retrieve("query", knowledge_base_ids=[KB_A])

    assert retriever._col.query_kwargs["where"] == {"knowledge_base_id": KB_A}
    assert [doc["content"] for doc in docs] == ["tenant A"]


def test_bm25_filters_candidates_before_ranking():
    retriever = BM25Retriever()
    retriever.index([
        {
            "id": "a",
            "content": "alpha tenant A",
            "metadata": {"knowledge_base_id": KB_A},
        },
        {
            "id": "b",
            "content": "alpha tenant B",
            "metadata": {"knowledge_base_id": KB_B},
        },
    ])

    assert retriever.search("alpha") == []
    docs = retriever.search("alpha", knowledge_base_ids=[KB_A])
    assert [doc["id"] for doc in docs] == ["a"]


def test_graph_cache_separates_same_entity_name_by_knowledge_base():
    cache = GraphCache()
    cache.upsert_entity("Acme", description="tenant A", kb_id=KB_A)
    cache.upsert_entity("Acme", description="tenant B", kb_id=KB_B)
    cache.add_relation("Acme", "Product A", "owns", kb_id=KB_A)
    cache.add_relation("Acme", "Product B", "owns", kb_id=KB_B)

    assert cache.match_entities(["Acme"]) == []
    entities = cache.match_entities(["Acme"], knowledge_base_ids=[KB_A])
    relations = cache.get_neighbor_relations(
        "Acme", knowledge_base_ids=[KB_A]
    )

    assert [entity["description"] for entity in entities] == ["tenant A"]
    assert [relation["target"] for relation in relations] == ["Product A"]


class ScopeSpyRetriever:
    def __init__(self, documents=None):
        self.documents = documents or []
        self.knowledge_base_ids = None

    def retrieve(self, _query, top_k=4, knowledge_base_ids=None):
        self.knowledge_base_ids = list(knowledge_base_ids or [])
        return self.documents[:top_k]


def test_graph_node_propagates_authorised_scope(monkeypatch):
    spy = ScopeSpyRetriever([{
        "content": "tenant A",
        "metadata": {"knowledge_base_id": KB_A, "source": ""},
    }])
    monkeypatch.setattr(nodes.cfg, "ENHANCED_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(nodes.cfg, "GRAPH_ENABLED", False)
    monkeypatch.setattr("app.rag.retriever.get_retriever", lambda: spy)

    result = nodes.knowledge_retrieval({
        "query": "query",
        "knowledge_base_ids": [KB_A],
        "steps": [],
    })

    assert spy.knowledge_base_ids == [KB_A]
    assert [doc["content"] for doc in result["retrieved_docs"]] == ["tenant A"]


def test_kb_search_scope_comes_from_request_authorisation(monkeypatch):
    """Deep 路径等价物（取代 RagWorker）：kb_search 的检索范围只来自请求级授权上下文。

    多智能体统一到 DeepAgents 后，子任务的隔离不再依赖 TaskBrief 传参，而是
    工具从 ContextVar 读取授权范围（模型无法通过工具参数越权）。
    """
    from types import SimpleNamespace

    from app.services.knowledge_context import use_authorised_kb_ids
    from app.tools.registry import get_tool_registry

    spy = ScopeSpyRetriever()

    class _FakeEnhanced:
        def retrieve(self, query, history=None, knowledge_base_ids=None):
            spy.retrieve(query, knowledge_base_ids=knowledge_base_ids)
            return SimpleNamespace(knowledge_blocks=[], raw_docs=[], sources=[])

    monkeypatch.setattr(
        "app.rag.enhanced_retriever.get_enhanced_retriever", lambda: _FakeEnhanced()
    )

    with use_authorised_kb_ids([KB_A]):
        get_tool_registry().invoke("kb_search", query="query")

    assert spy.knowledge_base_ids == [KB_A]
