from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from backend.services import evaluation_service, ragas_evaluator, ragas_worker
from backend.services.ragas_evaluator import RagasEvaluationSample, RagasEvaluator


def sample() -> RagasEvaluationSample:
    return RagasEvaluationSample(
        question="question",
        retrieved_context_ids=["chunk-a", "chunk-b"],
        reference_context_ids=["chunk-a"],
        retrieved_contexts=["context a", "context b"],
        reference_answer="answer",
    )


def test_process_mode_sends_json_to_dedicated_python(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["payload"] = json.loads(kwargs["input"])
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "worker log\n"
                '{"status":"completed","ragas_version":"0.4.3",'
                '"metrics":{"id_context_recall":1.0},"details":[]}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(ragas_evaluator.subprocess, "run", fake_run)
    evaluator = RagasEvaluator(
        python_executable="C:/ragas/python.exe",
        metrics=["id_context_recall"],
    )

    result = evaluator.evaluate([sample()])

    assert captured["command"] == [
        "C:/ragas/python.exe",
        "-m",
        "backend.services.ragas_worker",
    ]
    assert captured["payload"]["samples"][0]["question"] == "question"
    assert result["status"] == "completed"
    assert result["execution_mode"] == "process"
    assert result["metrics"]["id_context_recall"] == 1.0


def test_missing_process_environment_does_not_raise(monkeypatch):
    def missing_python(*_args, **_kwargs):
        raise FileNotFoundError(2, "not found", "C:/missing/python.exe")

    monkeypatch.setattr(ragas_evaluator.subprocess, "run", missing_python)
    result = RagasEvaluator(
        python_executable="C:/missing/python.exe",
    ).evaluate([sample()])

    assert result["status"] == "unavailable"
    assert "C:/missing/python.exe" in result["error"]


def test_in_process_mode_uses_the_same_worker_contract(monkeypatch):
    async def fake_evaluate(payload):
        return {
            "status": "completed",
            "metrics": {"id_context_precision": 0.5},
            "received": payload["samples"][0]["question"],
        }

    monkeypatch.setattr(ragas_worker, "evaluate_payload", fake_evaluate)
    result = RagasEvaluator(
        execution_mode="in_process",
        metrics=["id_context_precision"],
    ).evaluate([sample()])

    assert result["received"] == "question"
    assert result["metrics"]["id_context_precision"] == 0.5


def test_unknown_ragas_metric_is_rejected():
    with pytest.raises(ValueError):
        RagasEvaluator(metrics=["made_up_metric"])


@pytest.mark.asyncio
async def test_worker_calculates_id_metrics_through_ragas_contract(monkeypatch):
    class SingleTurnSample:
        def __init__(self, **kwargs):
            self.retrieved_context_ids = kwargs["retrieved_context_ids"]
            self.reference_context_ids = kwargs["reference_context_ids"]

    class Precision:
        async def single_turn_ascore(self, sample):
            retrieved = set(sample.retrieved_context_ids)
            relevant = set(sample.reference_context_ids)
            return len(retrieved & relevant) / len(retrieved)

    class Recall:
        async def single_turn_ascore(self, sample):
            retrieved = set(sample.retrieved_context_ids)
            relevant = set(sample.reference_context_ids)
            return len(retrieved & relevant) / len(relevant)

    monkeypatch.setitem(
        sys.modules,
        "ragas",
        SimpleNamespace(SingleTurnSample=SingleTurnSample),
    )
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        SimpleNamespace(
            IDBasedContextPrecision=Precision,
            IDBasedContextRecall=Recall,
        ),
    )

    result = await ragas_worker.evaluate_payload({
        "metrics": ["id_context_precision", "id_context_recall"],
        "samples": [sample().to_dict()],
    })

    assert result["status"] == "completed"
    assert result["metrics"]["id_context_precision"] == 0.5
    assert result["metrics"]["id_context_recall"] == 1.0


def test_ragas_failure_never_discards_local_metrics(monkeypatch):
    class Retriever:
        def retrieve(self, _question, **_kwargs):
            return [{
                "content": "context",
                "metadata": {
                    "source": "paper.pdf",
                    "file_id": "33333333-3333-3333-3333-333333333333",
                    "chunk_id": "chunk-a",
                    "score": 0.8,
                },
            }]

    class FailedRagasEvaluator:
        def evaluate(self, samples):
            assert samples[0].retrieved_context_ids == ["chunk-a"]
            return {"status": "failed", "error": "isolated failure", "metrics": {}}

    monkeypatch.setattr("app.rag.retriever.get_retriever", lambda: Retriever())
    monkeypatch.setattr(evaluation_service.cfg, "RAGAS_ENABLED", True)
    monkeypatch.setattr(
        evaluation_service,
        "get_ragas_evaluator",
        lambda _settings, _metrics=None: FailedRagasEvaluator(),
    )

    metrics = evaluation_service.run_evaluation(
        [evaluation_service.EvaluationCase(
            question="question",
            expected_file_id="33333333-3333-3333-3333-333333333333",
            expected_chunk_id="chunk-a",
            reference_answer="answer",
            expected_source="paper.pdf",
        )],
        top_k=3,
        knowledge_base_id="11111111-1111-1111-1111-111111111111",
    )

    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["ragas"]["status"] == "failed"
