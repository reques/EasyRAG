from __future__ import annotations

import asyncio
import json
import uuid

from app.core.config import Settings
from app.rag.extractors.base import (
    EntityExtraction,
    ExtractionResult,
    GraphExtractor,
    RelationExtraction,
)
from app.rag.extractors.llm_extractor import LLMExtractor


class PackedFakeLLM:
    model = "graph-fast-test"

    def __init__(self, *, missing_ids: set[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.missing_ids = missing_ids or set()

    async def chat_json(self, messages, **kwargs):
        prompt = messages[0]["content"]
        payload_text = prompt.split("<items_json>", 1)[1].split(
            "</items_json>", 1
        )[0]
        payload = json.loads(payload_text)
        self.calls.append({"payload": payload, "kwargs": kwargs})
        items = []
        for item in payload["items"]:
            if item["id"] in self.missing_ids:
                continue
            items.append({
                "id": item["id"],
                "entities": [{
                    "name": f"entity-{item['id']}",
                    "type": "concept",
                    "description": "",
                }],
                "relations": [],
            })
        return {"items": items}


def test_llm_extractor_packs_chunks_and_preserves_result_order():
    llm = PackedFakeLLM()
    extractor = LLMExtractor(
        llm=llm,
        pack_max_chars=650,
        pack_max_chunks=3,
        max_tokens=777,
    )
    chunks = [
        (f"chunk-{i}-" + ("x" * 290), {"chunk_id": f"c{i}"})
        for i in range(5)
    ]

    results = asyncio.run(extractor.extract_batch(chunks, concurrency=2))

    assert len(llm.calls) == 3
    assert [len(call["payload"]["items"]) for call in llm.calls] == [2, 2, 1]
    assert all(call["kwargs"]["max_tokens"] == 777 for call in llm.calls)
    assert [result.entities[0].name for result in results] == [
        "entity-0",
        "entity-1",
        "entity-2",
        "entity-3",
        "entity-4",
    ]


def test_missing_packed_item_is_not_cacheable():
    llm = PackedFakeLLM(missing_ids={"1"})
    extractor = LLMExtractor(
        llm=llm,
        pack_max_chars=2_000,
        pack_max_chunks=4,
    )
    chunks = [
        ("a" * 100, {"chunk_id": "c0"}),
        ("b" * 100, {"chunk_id": "c1"}),
    ]

    results = asyncio.run(extractor.extract_batch(chunks))

    assert results[0].cacheable is True
    assert results[1].empty
    assert results[1].cacheable is False


def test_extraction_result_serialization_round_trip():
    result = ExtractionResult(
        entities=[EntityExtraction("A", "technology", "entity description")],
        relations=[RelationExtraction("A", "B", "uses", "relation description")],
    )

    restored = ExtractionResult.from_dict(result.to_dict())

    assert restored == result


class CacheAwareFakeExtractor(GraphExtractor):
    name = "fake"
    prompt_version = "fake-v1"
    model_name = "fake-model"

    def __init__(self) -> None:
        self.seen_chunks: list[tuple] = []

    def cache_fingerprint(self) -> str:
        return f"{self.name}:{self.model_name}:{self.prompt_version}"

    async def extract(self, text: str, meta=None) -> ExtractionResult:
        return ExtractionResult(entities=[EntityExtraction(name=text[:8])])

    async def extract_batch(self, chunks, **_kwargs):
        self.seen_chunks = list(chunks)
        return [
            ExtractionResult(entities=[EntityExtraction(name=text[:8])])
            for text, _meta in chunks
        ]


def test_chunk_cache_extracts_only_misses_and_restores_order(monkeypatch):
    from backend.services import graph_build_service as service

    kb_id = uuid.uuid4()
    extractor = CacheAwareFakeExtractor()
    chunks = [
        ("cached content " * 6, {"chunk_id": "cached"}),
        ("new content " * 6, {"chunk_id": "new"}),
    ]
    cached_key = service._graph_cache_key(kb_id, extractor, chunks[0][0])
    cached_result = ExtractionResult(
        entities=[EntityExtraction(name="from-cache")]
    )
    stored = []

    async def fake_load(_kb_id, _keys):
        return {cached_key: cached_result}

    async def fake_store(records):
        stored.extend(records)

    monkeypatch.setattr(service, "_load_cached_extractions", fake_load)
    monkeypatch.setattr(service, "_store_cached_extractions", fake_store)

    results = asyncio.run(
        service._extract_chunks_with_cache(
            kb_id,
            extractor,
            chunks,
            concurrency=2,
            cache_enabled=True,
        )
    )

    assert extractor.seen_chunks == [chunks[1]]
    assert [result.entities[0].name for result in results] == [
        "from-cache",
        "new cont",
    ]
    assert len(stored) == 1
    assert stored[0]["chunk_id"] == "new"


def test_graph_cache_key_invalidates_on_extractor_fingerprint_change():
    from backend.services.graph_build_service import _graph_cache_key

    kb_id = uuid.uuid4()
    extractor = CacheAwareFakeExtractor()
    first = _graph_cache_key(kb_id, extractor, "same content")
    extractor.prompt_version = "fake-v2"
    second = _graph_cache_key(kb_id, extractor, "same content")

    assert first != second


def test_acceleration_configuration_defaults():
    fields = Settings.model_fields

    assert fields["GRAPH_EXTRACT_CONCURRENCY"].default == 8
    assert fields["GRAPH_EXTRACT_PACK_MAX_CHARS"].default == 1_800
    assert fields["GRAPH_EXTRACT_PACK_MAX_CHUNKS"].default == 4
    assert fields["GRAPH_EXTRACT_MAX_TOKENS"].default == 1_024
    assert fields["GRAPH_EXTRACT_CACHE_ENABLED"].default is True
    assert fields["GRAPH_BUILD_BATCH_SIZE"].default == 32


def test_graph_embedding_fallback_isolates_bad_items():
    from backend.services.graph_build_service import (
        _embed_graph_items_with_fallback,
    )

    class SelectivelyBrokenEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_texts(self, texts):
            self.calls.append(list(texts))
            if any("raises" in text for text in texts):
                raise RuntimeError("synthetic batch failure")
            return [
                [float("nan"), 0.0] if "nan-vector" in text else [1.0, 2.0]
                for text in texts
            ]

    embedder = SelectivelyBrokenEmbedder()
    items = [
        {"id": "good-1", "kind": "entity"},
        {"id": "bad-exception", "kind": "entity"},
        {"id": "bad-nan", "kind": "triple"},
        {"id": "good-2", "kind": "triple"},
    ]
    texts = ["normal text", "this raises", "nan-vector", "another normal text"]

    kept_items, vectors = _embed_graph_items_with_fallback(
        items,
        texts,
        embedder,
        batch_size=4,
    )

    assert [item["id"] for item in kept_items] == ["good-1", "good-2"]
    assert vectors == [[1.0, 2.0], [1.0, 2.0]]
    assert len(embedder.calls) > 1
