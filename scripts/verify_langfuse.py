"""Synthetic SDK verification; no keys, business data or remote service needed.

Run: python scripts/verify_langfuse.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import json
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify():
    from langfuse import Langfuse
    from langfuse.span_filter import is_langfuse_span
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.agents.cancellation import AgentRunCancelled
    from app.core.config import get_settings
    from app.observability import langfuse_client
    from app.observability.context import ExecutionContext, execution_scope, observation_scope
    from app.observability.langchain import with_model_callback
    from app.observability.redaction import mask_otel_spans

    cfg = get_settings()
    old_enabled = cfg.LANGFUSE_ENABLED
    old_client, old_init = langfuse_client._client, langfuse_client._initialized
    exporter = InMemorySpanExporter()
    # A custom exporter ensures no traces or API calls leave this process.
    client = Langfuse(public_key="pk-lf-verify-" + uuid.uuid4().hex,
                      secret_key="synthetic", base_url="http://127.0.0.1:1",
                      tracer_provider=TracerProvider(), span_exporter=exporter,
                      should_export_span=is_langfuse_span, mask_otel_spans=mask_otel_spans)
    cfg.LANGFUSE_ENABLED = True
    langfuse_client._client, langfuse_client._initialized = client, True
    trace_id = uuid.uuid4().hex
    context = ExecutionContext("synthetic-execution", trace_id, "local-trace",
                               "synthetic-user", "synthetic-session", True)
    try:
        with execution_scope(context) as root:
            with observation_scope("agent.synthetic", kind="agent"):
                model = FakeListChatModel(responses=["synthetic answer"])
                model.invoke("PRIVATE-PROMPT sk-test-secret-1234", config=with_model_callback({}))
                def worker():
                    with observation_scope("tool.synthetic", kind="tool"):
                        with observation_scope("llm.synthetic", kind="generation", model="synthetic") as generation:
                            generation.update(output="PRIVATE-OUTPUT", usage_details={"input": 2, "output": 3})
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(copy_context().run, worker) for _ in range(2)]
                    for future in futures:
                        future.result()
                try:
                    with observation_scope("llm.cancelled", kind="generation", model="synthetic"):
                        raise AgentRunCancelled()
                except AgentRunCancelled:
                    pass
            root.end("completed")
        client.flush()
        spans = list(exporter.get_finished_spans())
        assert len(spans) == 8, [(s.name, s.parent) for s in spans]
        by_id = {s.context.span_id: s for s in spans}
        roots = [s for s in spans if s.attributes.get("langfuse.internal.as_root")]
        assert len(roots) == 1 and roots[0].name == "execution.chat"
        assert {s.context.trace_id for s in spans} == {int(trace_id, 16)}
        assert all(s.parent and s.parent.span_id in by_id for s in spans if s not in roots)
        assert all(s.attributes.get("user.id") == "synthetic-user" for s in spans)
        assert all(s.attributes.get("session.id") == "synthetic-session" for s in spans)
        cancelled = next(s for s in spans if s.name == "llm.cancelled")
        assert cancelled.attributes.get("langfuse.observation.level") != "ERROR"
        serialized = json.dumps([dict(s.attributes) for s in spans])
        assert "PRIVATE-" not in serialized and "sk-test-secret-1234" not in serialized
        assert not context.handles
        return {"passed": True, "observations": len(spans), "roots": len(roots),
                "parent_links": "valid", "identity_propagation": "valid", "content": "filtered",
                "cancelled": "not_error", "remote_requests": 0}
    finally:
        client.shutdown()
        cfg.LANGFUSE_ENABLED = old_enabled
        langfuse_client._client, langfuse_client._initialized = old_client, old_init


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
