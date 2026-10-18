"""Model-only callback: one generation per model run, no tool/graph duplicates."""
from __future__ import annotations

import threading
from langchain_core.callbacks import BaseCallbackHandler
from app.observability.context import Observation, execution_context


class ModelCallback(BaseCallbackHandler):
    run_inline = True

    def __init__(self):
        self.pending = {}
        self.lock = threading.RLock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or (serialized or {}).get("name", "unknown")
        with self.lock:
            if run_id not in self.pending:
                self.pending[run_id] = Observation("llm.generate", kind="generation", model=str(model),
                    metadata={"modelCallId": str(run_id), "stream": bool(params.get("stream")),
                              "messageCount": sum(len(batch) for batch in messages)})

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self.on_chat_model_start(serialized, [prompts], run_id=run_id, **kwargs)

    def on_llm_new_token(self, token, *, run_id, chunk=None, **kwargs):
        with self.lock:
            handle = self.pending.get(run_id)
        if handle:
            message = getattr(chunk, "message", None)
            handle.token(content=bool(token and not (getattr(message, "additional_kwargs", {}) or {}).get("reasoning_content")))

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self.lock:
            handle = self.pending.pop(run_id, None)
        if handle is None:
            return
        from app.observability.models import response_usage
        usage, finish, text = response_usage(response)
        handle.update(usage_details=usage or None, output=text,
                      metadata={"usageStatus": "known" if usage else "unknown", "finishReason": finish})
        handle.end()

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            handle = self.pending.pop(run_id, None)
        if handle:
            from app.agents.cancellation import AgentRunCancelled
            import asyncio
            status = "cancelled" if isinstance(error, AgentRunCancelled) else (
                "interrupted" if isinstance(error, asyncio.CancelledError) else "error")
            handle.error(error, status=status)
            handle.end()


def with_model_callback(config):
    result = dict(config or {})
    context = execution_context()
    if context is None or not context.sampled:
        return result
    # LangChain inherits a parent's callbacks into nested graph calls. Reuse
    # one handler per execution so callback-manager merging can deduplicate it;
    # creating a new handler here counts every subagent generation twice.
    with context.lock:
        if context.model_callback is None:
            context.model_callback = ModelCallback()
        handler = context.model_callback
    callbacks = result.get("callbacks")
    if callbacks is None:
        result["callbacks"] = [handler]
    elif isinstance(callbacks, (list, tuple)):
        result["callbacks"] = [*callbacks] if handler in callbacks else [*callbacks, handler]
    else:
        result["callbacks"] = callbacks.copy()
        result["callbacks"].add_handler(handler, inherit=True)
    return result
