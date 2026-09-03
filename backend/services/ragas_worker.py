"""JSON stdin/stdout worker for the optional Ragas environment."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import sys
from typing import Any
import warnings


def _score_value(result: Any) -> float | None:
    value = getattr(result, "value", result)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(score) or math.isinf(score) else score


async def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ragas import SingleTurnSample
        # Ragas 0.4.3 emits a deprecation warning pointing to collections,
        # although those ID classes are not exported there yet.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    except ImportError as exc:
        return {
            "status": "unavailable",
            "error": (
                "Ragas is not installed in the evaluation environment. "
                "Install requirements-ragas.txt. "
                f"Original error: {exc}"
            ),
        }

    metric_names = list(payload.get("metrics") or [])
    samples = list(payload.get("samples") or [])
    llm_config = payload.get("llm") or {}
    scorers: dict[str, Any] = {}

    if "id_context_precision" in metric_names:
        scorers["id_context_precision"] = IDBasedContextPrecision()
    if "id_context_recall" in metric_names:
        scorers["id_context_recall"] = IDBasedContextRecall()

    llm_metric_names = {"context_precision", "context_recall"} & set(metric_names)
    if llm_metric_names:
        api_key = str(llm_config.get("api_key") or "")
        model = str(llm_config.get("model") or "")
        if not api_key or not model:
            return {
                "status": "failed",
                "error": "Ragas LLM metrics require an API key and model",
            }
        try:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.metrics.collections import ContextPrecision, ContextRecall

            client_kwargs = {"api_key": api_key}
            if llm_config.get("base_url"):
                client_kwargs["base_url"] = llm_config["base_url"]
            llm = llm_factory(model, client=AsyncOpenAI(**client_kwargs))
            if "context_precision" in llm_metric_names:
                scorers["context_precision"] = ContextPrecision(llm=llm)
            if "context_recall" in llm_metric_names:
                scorers["context_recall"] = ContextRecall(llm=llm)
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"Failed to initialise Ragas LLM metrics: {exc}",
            }

    details = []
    values_by_metric: dict[str, list[float]] = {
        name: [] for name in metric_names
    }
    had_errors = False
    for index, raw_sample in enumerate(samples):
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for name in metric_names:
            scorer = scorers.get(name)
            if scorer is None:
                scores[name] = None
                errors[name] = "Metric was not initialised"
                had_errors = True
                continue
            try:
                if name.startswith("id_"):
                    sample = SingleTurnSample(
                        retrieved_context_ids=raw_sample.get(
                            "retrieved_context_ids", []
                        ),
                        reference_context_ids=raw_sample.get(
                            "reference_context_ids", []
                        ),
                    )
                    result = await scorer.single_turn_ascore(sample)
                else:
                    result = await scorer.ascore(
                        user_input=raw_sample.get("question", ""),
                        reference=raw_sample.get("reference_answer", ""),
                        retrieved_contexts=raw_sample.get(
                            "retrieved_contexts", []
                        ),
                    )
                value = _score_value(result)
                scores[name] = value
                if value is not None:
                    values_by_metric[name].append(value)
            except Exception as exc:
                scores[name] = None
                errors[name] = str(exc)[:500]
                had_errors = True

        detail = {"index": index, "scores": scores}
        if errors:
            detail["errors"] = errors
        details.append(detail)

    aggregate = {
        name: (
            round(sum(values) / len(values), 6)
            if values
            else None
        )
        for name, values in values_by_metric.items()
    }
    try:
        version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "status": "partial" if had_errors else "completed",
        "ragas_version": version,
        "metrics": aggregate,
        "details": details,
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        result = asyncio.run(evaluate_payload(payload))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)[:1000]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
