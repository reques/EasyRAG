from __future__ import annotations
from typing import Any, Dict, List
from app.core.config import get_settings
from app.core.exceptions import EmptyRetrievalError, LLMClientError, ToolError
from app.core.logger import get_logger
from app.graph.state import AgentState
from app.llm.client import get_llm_client
from app.prompts.templates import (
    ANSWER_NO_CONTEXT, ANSWER_VALIDATION, ANSWER_WITH_CONTEXT,
    FALLBACK_ANSWER, INTENT_RECOGNITION, TASK_PLANNING,
)
from app.tools.registry import get_tool_registry
logger = get_logger(__name__)
cfg = get_settings()

def _append_step(state, msg):
    steps = list(state.get("steps") or [])
    steps.append(msg)
    return steps


def intent_recognition(state):
    """Node 1: Classify user intent and set routing flags."""
    query = state["query"]
    logger.info("[intent_recognition] query=%r", query[:80])
    client = get_llm_client()
    prompt = INTENT_RECOGNITION.format(query=query)
    try:
        data = client.chat_json_sync([{"role": "user", "content": prompt}])
        intent = str(data.get("intent", "knowledge_qa"))
        confidence = float(data.get("confidence", 0.8))
        requires_retrieval = bool(data.get("requires_retrieval", True))
        requires_tool = bool(data.get("requires_tool", False))
        tool_name = data.get("tool_name") or None
        tool_args = data.get("tool_args") or {}
        logger.info("[intent_recognition] intent=%s conf=%.2f", intent, confidence)
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "requires_retrieval": requires_retrieval,
            "requires_tool": requires_tool,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "steps": _append_step(state, "intent_recognition -> " + intent),
        }
    except Exception as exc:
        logger.warning("[intent_recognition] failed: %s", exc)
        return {
            "intent": "knowledge_qa",
            "intent_confidence": 0.5,
            "requires_retrieval": True,
            "requires_tool": False,
            "tool_name": None,
            "tool_args": {},
            "steps": _append_step(state, "intent_recognition -> fallback"),
        }

def task_planning(state):
    """Node 2: Decompose complex request into ordered sub-tasks."""
    query = state["query"]
    intent = state.get("intent", "complex_task")
    logger.info("[task_planning] query=%r intent=%s", query[:60], intent)
    client = get_llm_client()
    prompt = TASK_PLANNING.format(query=query, intent=intent)
    try:
        data = client.chat_json_sync([{"role": "user", "content": prompt}])
        sub_tasks = data.get("sub_tasks") or []
        needs_retrieval = bool(data.get("needs_retrieval", True))
        needs_tool = bool(data.get("needs_tool", False))
        logger.info("[task_planning] %d sub-tasks", len(sub_tasks))
        return {
            "sub_tasks": sub_tasks,
            "requires_retrieval": needs_retrieval,
            "requires_tool": needs_tool,
            "steps": _append_step(state, "task_planning -> " + str(len(sub_tasks)) + " sub-tasks"),
        }
    except Exception as exc:
        logger.warning("[task_planning] failed: %s", exc)
        return {
            "sub_tasks": [query],
            "requires_retrieval": True,
            "requires_tool": False,
            "steps": _append_step(state, "task_planning -> error, single task"),
        }


def knowledge_retrieval(state):
    """Node 3: Run RAG retrieval and populate retrieved_docs."""
    query = state["query"]
    logger.info("[knowledge_retrieval] query=%r", query[:80])
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.retrieve(query, top_k=cfg.RETRIEVER_TOP_K)
        logger.info("[knowledge_retrieval] retrieved %d docs", len(docs))
        if not docs:
            raise EmptyRetrievalError("No documents matched.")
        return {
            "retrieved_docs": docs,
            "retrieval_triggered": True,
            "error_message": None,
            "steps": _append_step(state, "knowledge_retrieval -> " + str(len(docs)) + " docs"),
        }
    except EmptyRetrievalError:
        logger.warning("[knowledge_retrieval] empty result")
        return {
            "retrieved_docs": [],
            "retrieval_triggered": True,
            "error_message": None,
            "steps": _append_step(state, "knowledge_retrieval -> empty"),
        }
    except Exception as exc:
        logger.error("[knowledge_retrieval] error: %s", exc)
        return {
            "retrieved_docs": [],
            "retrieval_triggered": True,
            "error_message": str(exc),
            "steps": _append_step(state, "knowledge_retrieval -> ERROR"),
        }

def tool_selection(state):
    """Node 4: Validate chosen tool; infer from query if needed."""
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args") or {}
    logger.info("[tool_selection] tool=%s", tool_name)
    registry = get_tool_registry()
    available = registry.list_names()
    if tool_name and tool_name in available:
        return {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "steps": _append_step(state, "tool_selection -> " + tool_name),
        }
    q = state["query"].lower()
    if any(w in q for w in ["calculat", "compute", "sqrt", "pow"]):
        return {
            "tool_name": "calculator",
            "tool_args": {"expression": state["query"]},
            "steps": _append_step(state, "tool_selection -> inferred calculator"),
        }
    if any(w in q for w in ["time", "date", "today", "now", "weekday"]):
        return {
            "tool_name": "datetime_tool",
            "tool_args": {},
            "steps": _append_step(state, "tool_selection -> inferred datetime_tool"),
        }
    if any(w in q for w in ["word count", "char count", "text stat"]):
        return {
            "tool_name": "text_tool",
            "tool_args": {"operation": "stats", "text": state["query"]},
            "steps": _append_step(state, "tool_selection -> inferred text_tool"),
        }
    logger.warning("[tool_selection] no valid tool found")
    return {
        "requires_tool": False,
        "tool_name": None,
        "steps": _append_step(state, "tool_selection -> no valid tool"),
    }


def tool_execution(state):
    """Node 5: Execute selected tool and capture result."""
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args") or {}
    logger.info("[tool_execution] tool=%s", tool_name)
    if not tool_name:
        return {
            "tool_result": None,
            "tool_triggered": False,
            "tool_error": "No tool selected",
            "steps": _append_step(state, "tool_execution -> skipped"),
        }
    registry = get_tool_registry()
    try:
        result = registry.invoke(tool_name, **tool_args)
        logger.info("[tool_execution] result=%r", str(result)[:120])
        return {
            "tool_result": result,
            "tool_triggered": True,
            "tool_error": None,
            "steps": _append_step(state, "tool_execution -> " + tool_name + " OK"),
        }
    except ToolError as exc:
        logger.error("[tool_execution] ToolError: %s", exc)
        return {
            "tool_result": None,
            "tool_triggered": True,
            "tool_error": str(exc),
            "steps": _append_step(state, "tool_execution -> " + tool_name + " FAILED"),
        }
    except Exception as exc:
        logger.error("[tool_execution] unexpected: %s", exc)
        return {
            "tool_result": None,
            "tool_triggered": True,
            "tool_error": "Unexpected: " + str(exc),
            "steps": _append_step(state, "tool_execution -> ERROR"),
        }

def answer_generation(state):
    """Node 6: Generate a draft answer from retrieved docs and/or tool results."""
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    tool_result = state.get("tool_result") or ""
    tool_error = state.get("tool_error") or ""
    regen_count = state.get("regeneration_count") or 0
    history = state.get("history") or []
    logger.info(
        "[answer_generation] docs=%d tool_result=%s regen=%d history_turns=%d",
        len(docs), bool(tool_result), regen_count, len(history) // 2,
    )
    client = get_llm_client()
    effective_tool = tool_result or ("Tool failed: " + tool_error if tool_error else "N/A")
    try:
        messages = [{"role": t["role"], "content": t["content"]} for t in history]
        if docs:
            context = "\n\n".join(
                "[" + str(i + 1) + "] " + d["content"] for i, d in enumerate(docs)
            )
            messages.append({
                "role": "user",
                "content": ANSWER_WITH_CONTEXT.format(
                    query=query, context=context, tool_result=effective_tool
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": ANSWER_NO_CONTEXT.format(
                    query=query, tool_result=effective_tool
                ),
            })
        draft = client.chat_sync(messages)
        logger.info("[answer_generation] draft length=%d", len(draft))
        return {
            "draft_answer": draft,
            "regeneration_count": regen_count + 1,
            "error_message": None,
            "steps": _append_step(state, "answer_generation (attempt " + str(regen_count + 1) + ")"),
        }
    except LLMClientError as exc:
        logger.error("[answer_generation] LLM error: %s", exc)
        return {
            "draft_answer": "",
            "error_message": str(exc),
            "steps": _append_step(state, "answer_generation -> LLM ERROR"),
        }
    except Exception as exc:
        logger.error("[answer_generation] unexpected: %s", exc)
        return {
            "draft_answer": "",
            "error_message": str(exc),
            "steps": _append_step(state, "answer_generation -> ERROR"),
        }

def answer_validation(state):
    """Node 7: Check whether draft answer is sufficient."""
    if not cfg.ANSWER_VALIDATION_ENABLED:
        logger.info("[answer_validation] disabled by config")
        return {
            "validation_passed": True,
            "validation_feedback": "Validation disabled",
            "final_answer": state.get("draft_answer", ""),
            "steps": _append_step(state, "answer_validation -> skipped"),
        }
    query = state["query"]
    draft = state.get("draft_answer", "")
    logger.info("[answer_validation] draft length=%d", len(draft))
    if len(draft.strip()) < cfg.ANSWER_MIN_LENGTH:
        return {
            "validation_passed": False,
            "validation_feedback": "Answer too short.",
            "final_answer": draft,
            "steps": _append_step(state, "answer_validation -> FAILED (too short)"),
        }
    client = get_llm_client()
    prompt = ANSWER_VALIDATION.format(query=query, draft_answer=draft)
    try:
        data = client.chat_json_sync([{"role": "user", "content": prompt}])
        passed = bool(data.get("passed", True))
        feedback = str(data.get("feedback", ""))
        logger.info("[answer_validation] passed=%s feedback=%r", passed, feedback)
        return {
            "validation_passed": passed,
            "validation_feedback": feedback,
            "final_answer": draft if passed else "",
            "steps": _append_step(state, "answer_validation -> " + ("PASSED" if passed else "FAILED")),
        }
    except Exception as exc:
        logger.warning("[answer_validation] error (accepting draft): %s", exc)
        return {
            "validation_passed": True,
            "validation_feedback": "Validation error; accepting draft.",
            "final_answer": draft,
            "steps": _append_step(state, "answer_validation -> ERROR (accepted)"),
        }


def fallback_handler(state):
    """Node 8: Produce a safe fallback response on any failure."""
    query = state.get("query", "")
    error = state.get("error_message", "An unknown error occurred.")
    logger.warning("[fallback_handler] error=%r", error)
    fallback_text = FALLBACK_ANSWER.format(query=query, error=error)
    return {
        "final_answer": fallback_text,
        "is_fallback": True,
        "validation_passed": False,
        "steps": _append_step(state, "fallback_handler -> fallback answer generated"),
    }
