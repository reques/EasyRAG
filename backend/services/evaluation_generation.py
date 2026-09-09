"""Answer generation used by semantic evaluation metrics."""

from __future__ import annotations

from typing import Any, Iterable

from app.llm.client import get_llm_client


def generate_evaluation_answer(question: str, documents: Iterable[dict[str, Any]]) -> str:
    """Generate a grounded answer for Faithfulness evaluation.

    This deliberately uses the existing main LLM client so evaluation follows
    the same provider configuration as the application without introducing a
    second agent implementation.
    """
    contexts = [str(item.get("content") or "").strip() for item in documents]
    contexts = [item for item in contexts if item]
    if not contexts:
        return ""
    joined = "\n\n".join(f"[{idx}] {text}" for idx, text in enumerate(contexts, 1))
    messages = [
        {
            "role": "system",
            "content": (
                "你是检索评测中的回答生成器。只能依据给定上下文回答问题；"
                "上下文没有答案时明确说不知道，不要补充外部事实。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：{question}\n\n上下文：\n{joined}\n\n请给出简洁回答。",
        },
    ]
    return get_llm_client("main").chat_sync(messages).strip()
