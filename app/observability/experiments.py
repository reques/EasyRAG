"""Native OTel experiments over local evaluation data, one actual trace per case.

Attribute contract: https://langfuse.com/integrations/native/opentelemetry/experiments
No remote dataset mutation, SDK private attributes, or duplicate task execution.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import uuid

try:
    from opentelemetry.sdk.trace import SpanProcessor
except ImportError:
    class SpanProcessor:
        pass

from app.observability.context import ExecutionContext, execution_context, execution_scope
from app.observability.redaction import content, redact

_experiment = ContextVar("evaluation_experiment", default=None)


def current_experiment():
    return _experiment.get()


@contextmanager
def use_evaluation_experiment(cases, kb_id, *, name, user_id, configuration=None):
    experiment = new_experiment(cases, kb_id, name=name, user_id=user_id)
    details = json.loads(experiment["metadata"])
    # Explicit non-secret, reproducibility fields only.
    details.update({key: value for key, value in (configuration or {}).items()
                    if key in {"top_k", "score_threshold", "generate_answers", "ragas_metrics", "dataset_version", "benchmark_version"}})
    experiment["metadata"] = json.dumps(details)
    token = _experiment.set(experiment)
    try:
        yield
    finally:
        _experiment.reset(token)


def snapshot_identity(cases, kb_id):
    from dataclasses import asdict
    # Immutable content identity; edits never silently replace historical cases.
    encoded = json.dumps({"kb_id": str(kb_id), "cases": [asdict(case) for case in cases]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class ExperimentSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context=None):
        context = execution_context()
        if context is None or not context.experiment:
            return
        experiment = context.experiment
        with context.lock:
            root_id = experiment.setdefault("root_id", format(span.context.span_id, "016x"))
        attributes = {
            "langfuse.experiment.id": experiment["id"],
            "langfuse.experiment.name": experiment["name"],
            "langfuse.experiment.dataset.id": experiment["dataset_id"],
            "langfuse.experiment.item.id": experiment["item_id"],
            "langfuse.experiment.item.root_observation_id": root_id,
            "langfuse.environment": "experiment",
            "langfuse.experiment.metadata": experiment["metadata"],
        }
        if format(span.context.span_id, "016x") == root_id:
            expected = content(experiment.get("expected_output"))
            if expected is not None:
                attributes["langfuse.experiment.item.expected_output"] = json.dumps(expected, ensure_ascii=False)
        span.set_attributes(attributes)

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


def new_experiment(cases, kb_id, *, name="evaluation", user_id=""):
    from app.core.config import get_settings
    cfg = get_settings()
    identifier = str(uuid.uuid4())
    return {"id": identifier, "name": f"{redact(name, 100)}-{identifier[:8]}",
            "dataset_id": snapshot_identity(cases, kb_id), "user_id": str(user_id),
            "metadata": json.dumps({"source": "easyrag_evaluation", "release": cfg.APP_VERSION,
                                    "model": redact(cfg.LLM_MODEL, 120), "metrics_version": "local-v2"})}


@contextmanager
def evaluation_case_scope(experiment, index, case):
    from app.observability.langfuse_client import get_client
    if not experiment or get_client() is None:
        yield {}
        return
    binding = {**experiment, "item_id": hashlib.sha256(
        f"{experiment['dataset_id']}:{index}".encode()).hexdigest(), "expected_output": case.reference_answer}
    context = ExecutionContext(
        execution_id=experiment["id"], trace_id=uuid.uuid5(uuid.UUID(experiment["id"]), str(index)).hex,
        user_id=experiment["user_id"], sampled=True, question=case.question, experiment=binding,
        metadata={"evaluationRunId": experiment["id"], "evaluationItemId": binding["item_id"],
                  "datasetId": experiment["dataset_id"]},
    )
    with execution_scope(context, name="evaluation.case") as root:
        yield {"trace_id": context.trace_id, "observation_id": root.id,
               "experiment_id": experiment["id"], "item_id": binding["item_id"]} if root.raw else {}
