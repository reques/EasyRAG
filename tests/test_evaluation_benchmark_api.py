"""Phase-two benchmark and Golden Set API contract tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import uuid

from pydantic import ValidationError
import pytest

from backend.server.routers.evaluation_router import (
    BenchmarkCreateRequest,
    BenchmarkMetrics,
    BenchmarkUpdateRequest,
    DatasetUpdateRequest,
    router,
)
from backend.services.evaluation_benchmarks import (
    assess_benchmark,
    build_benchmark_snapshot,
    export_benchmark_json,
    update_benchmark,
)
from backend.services.evaluation_import import parse_dataset_import


KB_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
DATASET_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def make_benchmark(**overrides):
    values = {
        "id": uuid.uuid4(),
        "knowledge_base_id": KB_ID,
        "dataset_id": DATASET_ID,
        "name": "法律检索基准",
        "description": "desc",
        "metrics_json": json.dumps({
            "recall_at_k": {"enabled": True, "threshold": 0.8},
            "mrr_at_k": {"enabled": True, "threshold": 0.7},
            "context_relevance": {"enabled": False, "threshold": 0.75},
            "faithfulness": {"enabled": False, "threshold": 0.9},
        }),
        "retrieval_config_json": json.dumps({
            "top_k": 5,
            "score_threshold": 0.3,
            "mode": "basic",
        }),
        "version": 2,
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_benchmark_request_validation_and_defaults():
    request = BenchmarkCreateRequest(
        knowledge_base_id=KB_ID,
        dataset_id=DATASET_ID,
        name="  法律基准  ",
    )
    assert request.name == "法律基准"
    assert request.retrieval_config.top_k == 5
    assert request.metrics.recall_at_k.enabled is True
    assert request.metrics.faithfulness.enabled is False

    with pytest.raises(ValidationError):
        BenchmarkMetrics(
            recall_at_k={"enabled": False},
            mrr_at_k={"enabled": False},
            context_relevance={"enabled": False},
            faithfulness={"enabled": False},
        )
    with pytest.raises(ValidationError):
        BenchmarkUpdateRequest()
    with pytest.raises(ValidationError):
        DatasetUpdateRequest()


def test_benchmark_export_snapshot_and_threshold_assessment():
    benchmark = make_benchmark()
    dataset = SimpleNamespace(
        id=DATASET_ID,
        name="golden-set",
        version=3,
        case_count=1,
        cases_json=json.dumps([{"question": "q"}]),
    )

    exported = export_benchmark_json(benchmark)
    snapshot = build_benchmark_snapshot(benchmark, dataset)
    assessment = assess_benchmark(
        exported["metrics"],
        {"recall_at_k": 0.9, "mrr_at_k": 0.6},
    )

    assert exported["retrieval_config"]["score_threshold"] == 0.3
    assert snapshot["benchmark"]["version"] == 2
    assert snapshot["dataset"]["version"] == 3
    assert snapshot["dataset"]["cases"] == [{"question": "q"}]
    assert assessment["passed"] is False
    assert assessment["metrics"]["recall_at_k"]["passed"] is True
    assert assessment["metrics"]["mrr_at_k"]["passed"] is False


def test_update_benchmark_only_advances_version_for_real_changes():
    class FakeSession:
        async def flush(self):
            return None

    benchmark = make_benchmark()
    asyncio.run(update_benchmark(FakeSession(), benchmark, changes={"name": benchmark.name}))
    assert benchmark.version == 2

    asyncio.run(update_benchmark(FakeSession(), benchmark, changes={"status": "archived"}))
    assert benchmark.version == 3
    assert benchmark.status == "archived"


def test_json_and_csv_imports_are_normalised():
    json_payload = {
        "name": "法律集",
        "cases": [{
            "question": "什么是合同？",
            "expected_filename": "合同法.pdf",
            "expected_chunk_ids": ["chunk-1", "chunk-1", "chunk-2"],
        }],
    }
    parsed_json = parse_dataset_import(
        "legal.json",
        json.dumps(json_payload, ensure_ascii=False).encode(),
    )
    parsed_csv = parse_dataset_import(
        "legal.csv",
        (
            "question,expected_filename,expected_chunk_ids,expect_miss\n"
            '负样本,合同法.pdf,"chunk-1|chunk-2",true\n'
        ).encode(),
    )

    assert parsed_json["name"] == "法律集"
    assert parsed_json["cases"][0]["expected_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert parsed_csv["cases"][0]["expect_miss"] is True
    assert parsed_csv["cases"][0]["expected_chunk_ids"] == ["chunk-1", "chunk-2"]


def test_import_routes_are_registered_before_dynamic_dataset_route():
    paths = [route.path for route in router.routes]
    dynamic_index = paths.index("/evaluation/datasets/{dataset_id}")
    assert paths.index("/evaluation/datasets/import/preview") < dynamic_index
    assert paths.index("/evaluation/datasets/import") < dynamic_index
    assert "/evaluation/metric-catalog" in paths
    assert "/evaluation/benchmarks/{benchmark_id}/runs" in paths
