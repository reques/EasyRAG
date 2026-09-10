"""Evaluation benchmark persistence contract tests."""

from __future__ import annotations

import asyncio
import json
import uuid

from backend.services.evaluation_service import save_run
from backend.storage.postgres.models_knowledge import (
    EvaluationBenchmark,
    EvaluationRun,
)


def test_evaluation_benchmark_schema_contract():
    table = EvaluationBenchmark.__table__

    assert {
        "id",
        "knowledge_base_id",
        "dataset_id",
        "name",
        "description",
        "metrics_json",
        "retrieval_config_json",
        "version",
        "status",
        "created_at",
        "updated_at",
    } <= set(table.columns.keys())

    constraint_names = {constraint.name for constraint in table.constraints}
    assert "uq_evaluation_benchmarks_kb_name" in constraint_names
    assert "ck_evaluation_benchmarks_version" in constraint_names
    assert "ck_evaluation_benchmarks_status" in constraint_names

    dataset_fk = next(iter(table.c.dataset_id.foreign_keys))
    knowledge_base_fk = next(iter(table.c.knowledge_base_id.foreign_keys))
    assert dataset_fk.target_fullname == "evaluation_datasets.id"
    assert dataset_fk.ondelete == "RESTRICT"
    assert knowledge_base_fk.target_fullname == "knowledge_bases.id"
    assert knowledge_base_fk.ondelete == "CASCADE"


def test_evaluation_run_has_reproducible_benchmark_fields():
    table = EvaluationRun.__table__

    assert {
        "benchmark_id",
        "benchmark_version",
        "benchmark_snapshot_json",
        "status",
        "started_at",
        "finished_at",
        "error_message",
    } <= set(table.columns.keys())

    benchmark_fk = next(iter(table.c.benchmark_id.foreign_keys))
    assert benchmark_fk.target_fullname == "evaluation_benchmarks.id"
    assert benchmark_fk.ondelete == "SET NULL"


def test_save_run_freezes_benchmark_snapshot_and_completes_lifecycle():
    class FakeSession:
        def __init__(self):
            self.saved = None

        def add(self, value):
            self.saved = value

        async def flush(self):
            return None

    session = FakeSession()
    benchmark_id = uuid.uuid4()
    snapshot = {
        "benchmark_version": 3,
        "dataset": {"id": str(uuid.uuid4()), "version": 2},
        "metrics": {"recall_at_k": {"enabled": True, "threshold": 0.8}},
        "retrieval_config": {"top_k": 5, "score_threshold": 0.3},
    }
    metrics = {
        "hit_rate": 1.0,
        "mrr": 0.75,
        "avg_score": 0.82,
        "details": [{"question": "q"}],
    }

    run = asyncio.run(
        save_run(
            session,
            name="benchmark-run",
            metrics=metrics,
            top_k=5,
            benchmark_id=benchmark_id,
            benchmark_version=3,
            benchmark_snapshot=snapshot,
        )
    )

    assert session.saved is run
    assert run.benchmark_id == benchmark_id
    assert run.benchmark_version == 3
    assert json.loads(run.benchmark_snapshot_json) == snapshot
    assert run.status == "completed"
    assert run.started_at is not None
    assert run.finished_at == run.started_at
    assert run.error_message is None
