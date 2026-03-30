"""LangGraph routing functions.

Each router receives the current AgentState and returns the name
of the next node to visit.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logger import get_logger
from app.graph.state import AgentState

logger = get_logger(__name__)
cfg = get_settings()

# Node name constants
INTENT_RECOGNITION  = "intent_recognition"
TASK_PLANNING       = "task_planning"
KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
TOOL_SELECTION      = "tool_selection"
TOOL_EXECUTION      = "tool_execution"
ANSWER_GENERATION   = "answer_generation"
ANSWER_VALIDATION   = "answer_validation"
FALLBACK_HANDLER    = "fallback_handler"
END                 = "__end__"


def route_after_intent(state: AgentState) -> str:
    """Dispatch after intent classification."""
    if state.get("error_message") and not state.get("intent"):
        return FALLBACK_HANDLER
    intent = state.get("intent", "knowledge_qa")
    logger.info("[router] after_intent -> %s", intent)
    if intent == "complex_task":
        return TASK_PLANNING
    if intent == "tool_use":
        return TOOL_SELECTION
    if intent == "knowledge_qa":
        return KNOWLEDGE_RETRIEVAL
    return ANSWER_GENERATION  # chitchat / unknown


def route_after_planning(state: AgentState) -> str:
    """After task planning choose retrieval, tool, or generation."""
    requires_retrieval = state.get("requires_retrieval", True)
    requires_tool = state.get("requires_tool", False)
    logger.info("[router] after_planning retrieval=%s tool=%s",
                requires_retrieval, requires_tool)
    if requires_tool:
        return TOOL_SELECTION
    if requires_retrieval:
        return KNOWLEDGE_RETRIEVAL
    return ANSWER_GENERATION


def route_after_retrieval(state: AgentState) -> str:
    """After retrieval optionally run a tool, else generate."""
    if state.get("error_message"):
        return FALLBACK_HANDLER
    if state.get("requires_tool"):
        return TOOL_SELECTION
    return ANSWER_GENERATION


def route_after_tool_execution(state: AgentState) -> str:
    """After tool execution always proceed to generation."""
    return ANSWER_GENERATION


def route_after_generation(state: AgentState) -> str:
    """After generation: error -> fallback, else validate."""
    if state.get("error_message") or not state.get("draft_answer"):
        return FALLBACK_HANDLER
    return ANSWER_VALIDATION


def route_after_validation(state: AgentState) -> str:
    """If passed -> END; if failed and retries left -> re-generate; else -> END."""
    if state.get("validation_passed"):
        return END
    regen = state.get("regeneration_count") or 0
    max_regen = 1  # allow one regeneration
    if regen < max_regen:
        logger.info("[router] after_validation -> re-generate (attempt %d)", regen + 1)
        return ANSWER_GENERATION
    logger.info("[router] after_validation -> END (max retries reached)")
    return END
