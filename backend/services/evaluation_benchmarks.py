"""Evaluation benchmark persistence and reproducible snapshot helpers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.postgres.models_knowledge import (
    EvaluationBenchmark,
    EvaluationDataset,
    KnowledgeBase,
)


DEFAULT_METRICS: dict[str, dict[str, Any]] = {
    "recall_at_k": {"enabled": True, "threshold": 0.8},
    "mrr_at_k": {"enabled": True, "threshold": 0.7},
    "context_relevance": {"enabled": False, "threshold": 0.75},
    "faithfulness": {"enabled": False, "threshold": 0.9},
}

DEFAULT_RETRIEVAL_CONFIG: dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.0,
    "mode": "basic",
}


def default_metrics() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_METRICS)


def default_retrieval_config() -> dict[str, Any]:
    return deepcopy(DEFAULT_RETRIEVAL_CONFIG)


def _json_object(raw: Optional[str], fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return deepcopy(fallback)
    return value if isinstance(value, dict) else deepcopy(fallback)


def export_benchmark_json(benchmark: EvaluationBenchmark) -> dict[str, Any]:
    return {
        "id": str(benchmark.id),
        "knowledge_base_id": str(benchmark.knowledge_base_id),
        "dataset_id": str(benchmark.dataset_id),
        "name": benchmark.name,
        "description": benchmark.description,
        "metrics": _json_object(benchmark.metrics_json, DEFAULT_METRICS),
        "retrieval_config": _json_object(
            benchmark.retrieval_config_json,
            DEFAULT_RETRIEVAL_CONFIG,
        ),
        "version": benchmark.version,
        "status": benchmark.status,
        "created_at": (
            benchmark.created_at.isoformat() if benchmark.created_at else ""
        ),
        "updated_at": (
            benchmark.updated_at.isoformat() if benchmark.updated_at else ""
        ),
    }


async def create_benchmark(
    session: AsyncSession,
    *,
    knowledge_base_id: uuid.UUID,
    dataset_id: uuid.UUID,
    name: str,
    description: str,
    metrics: dict[str, Any],
    retrieval_config: dict[str, Any],
    status: str = "active",
) -> EvaluationBenchmark:
    benchmark = EvaluationBenchmark(
        knowledge_base_id=knowledge_base_id,
        dataset_id=dataset_id,
        name=name,
        description=description,
        metrics_json=json.dumps(metrics, ensure_ascii=False),
        retrieval_config_json=json.dumps(retrieval_config, ensure_ascii=False),
        version=1,
        status=status,
    )
    session.add(benchmark)
    await session.flush()
    return benchmark


async def list_benchmarks(
    session: AsyncSession,
    owner_id: uuid.UUID,
    *,
    knowledge_base_id: Optional[uuid.UUID] = None,
) -> list[EvaluationBenchmark]:
    stmt = (
        select(EvaluationBenchmark)
        .join(
            KnowledgeBase,
            EvaluationBenchmark.knowledge_base_id == KnowledgeBase.id,
        )
        .where(KnowledgeBase.owner_id == owner_id)
    )
    if knowledge_base_id is not None:
        stmt = stmt.where(
            EvaluationBenchmark.knowledge_base_id == knowledge_base_id
        )
    stmt = stmt.order_by(EvaluationBenchmark.updated_at.desc())
    return list((await session.execute(stmt)).scalars().all())


async def get_benchmark(
    session: AsyncSession,
    benchmark_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> Optional[EvaluationBenchmark]:
    stmt = (
        select(EvaluationBenchmark)
        .join(
            KnowledgeBase,
            EvaluationBenchmark.knowledge_base_id == KnowledgeBase.id,
        )
        .where(
            EvaluationBenchmark.id == benchmark_id,
            KnowledgeBase.owner_id == owner_id,
        )
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def update_benchmark(
    session: AsyncSession,
    benchmark: EvaluationBenchmark,
    *,
    changes: dict[str, Any],
) -> bool:
    changed = False
    json_fields = {
        "metrics": "metrics_json",
        "retrieval_config": "retrieval_config_json",
    }
    for field, value in changes.items():
        target = json_fields.get(field, field)
        stored = (
            json.dumps(value, ensure_ascii=False)
            if field in json_fields
            else value
        )
        if getattr(benchmark, target) != stored:
            setattr(benchmark, target, stored)
            changed = True
    if changed:
        benchmark.version += 1
    await session.flush()
    return changed


async def delete_benchmark(
    session: AsyncSession,
    benchmark_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> bool:
    benchmark = await get_benchmark(session, benchmark_id, owner_id)
    if benchmark is None:
        return False
    await session.delete(benchmark)
    return True


async def duplicate_benchmark(
    session: AsyncSession,
    source: EvaluationBenchmark,
    *,
    name: str,
) -> EvaluationBenchmark:
    return await create_benchmark(
        session,
        knowledge_base_id=source.knowledge_base_id,
        dataset_id=source.dataset_id,
        name=name,
        description=source.description,
        metrics=_json_object(source.metrics_json, DEFAULT_METRICS),
        retrieval_config=_json_object(
            source.retrieval_config_json,
            DEFAULT_RETRIEVAL_CONFIG,
        ),
        status="draft",
    )


def build_benchmark_snapshot(
    benchmark: EvaluationBenchmark,
    dataset: EvaluationDataset,
) -> dict[str, Any]:
    try:
        cases = json.loads(dataset.cases_json or "[]")
    except (TypeError, ValueError):
        cases = []
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": export_benchmark_json(benchmark),
        "dataset": {
            "id": str(dataset.id),
            "name": dataset.name,
            "version": dataset.version,
            "case_count": dataset.case_count,
            "cases": cases if isinstance(cases, list) else [],
        },
    }


def assess_benchmark(
    metrics_config: dict[str, Any],
    run_metrics: dict[str, Any],
) -> dict[str, Any]:
    values = {
        "recall_at_k": run_metrics.get("recall_at_k"),
        "mrr_at_k": run_metrics.get("mrr_at_k"),
        "context_relevance": (
            (run_metrics.get("ragas") or {}).get("metrics") or {}
        ).get("context_precision"),
        "faithfulness": (
            (run_metrics.get("ragas") or {}).get("metrics") or {}
        ).get("faithfulness"),
    }
    results: dict[str, Any] = {}
    all_passed = True
    evaluated = 0
    for metric, config in metrics_config.items():
        if not isinstance(config, dict) or not config.get("enabled"):
            continue
        threshold = float(config.get("threshold", 0.0))
        value = values.get(metric)
        passed = value is not None and float(value) >= threshold
        results[metric] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }
        evaluated += 1
        all_passed = all_passed and passed
    return {
        "passed": bool(evaluated) and all_passed,
        "evaluated_metric_count": evaluated,
        "metrics": results,
    }
