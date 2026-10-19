"""Estimate request components at the last Agent middleware boundary.

The provider's response usage remains the only authoritative token total.
These categories use a generic tokenizer and must always be labelled estimates.
Only counts and model metadata are retained; prompt contents are not persisted.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.llm.client import get_active_chat_model_profile


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _count(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=())) if text else 0


def _message_text(content: Any) -> tuple[str, bool]:
    if isinstance(content, str):
        return content, True
    if isinstance(content, list):
        texts: list[str] = []
        complete = True
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            else:
                complete = False  # Images/files require provider-specific counting.
        return "\n".join(texts), complete
    return "", False


def estimate_request_breakdown(request: Any) -> dict[str, Any]:
    """Count final system/messages/tools separately without claiming exactness."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    result: dict[str, Any] = {
        "system": 0, "tools": 0, "conversation": 0, "knowledge": 0,
        "method": "cl100k_base_estimate", "complete_text": True,
    }
    messages = ([request.system_message] if request.system_message is not None else []) + list(request.messages)
    for message in messages:
        content, complete = _message_text(getattr(message, "content", ""))
        result["complete_text"] &= complete
        extra = ""
        if getattr(message, "tool_calls", None):
            extra = json.dumps(message.tool_calls, ensure_ascii=False, sort_keys=True, default=str)
        if getattr(message, "tool_call_id", None):
            extra += str(message.tool_call_id)
        if str(getattr(message, "id", "") or "").startswith("context:"):
            category = "knowledge"
        elif getattr(message, "type", "") == "system":
            category = "system"
        else:
            category = "conversation"
        result[category] += _count(content + extra)
    for tool in request.tools:
        try:
            schema = convert_to_openai_tool(tool)
            result["tools"] += _count(json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str))
        except Exception:
            result["complete_text"] = False
    return result


def build_context_usage_middleware():
    """Install after prompt/tool modifying middleware so it sees their output."""
    from langchain.agents.middleware import AgentMiddleware

    class ContextUsageMiddleware(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            try:
                breakdown = estimate_request_breakdown(request)
            except Exception:
                breakdown = None
            model = str(getattr(request.model, "model_name", "") or "")
            base = str(getattr(request.model, "openai_api_base", "") or "").rstrip("/")
            profile = get_active_chat_model_profile()
            matched = bool(profile and profile.model == model and profile.base_url.rstrip("/") == base)
            metadata = {
                "api_model": model,
                "profile_id": profile.id if matched else "",
                "context_window": profile.context_window if matched else None,
                "context_source": profile.context_source if matched else "unknown",
                "breakdown": breakdown,
            }
            response = handler(request)
            for message in response.result:
                if getattr(message, "type", "") == "ai":
                    message.response_metadata["context_meter"] = metadata
            return response

    return ContextUsageMiddleware()
