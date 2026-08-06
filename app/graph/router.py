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
AGENT_REASONING     = "agent_reasoning"
END                 = "__end__"


def route_after_intent(state: AgentState) -> str:
    """Dispatch after intent classification."""
    if state.get("error_message") and not state.get("intent"):
        return FALLBACK_HANDLER
    # ReAct 循环子图: complex_task / 低置信度优先走 agent_reasoning
    if state.get("use_react"):
        logger.info("[router] after_intent -> agent_reasoning (use_react)")
        return AGENT_REASONING
    intent = state.get("intent", "knowledge_qa")
    logger.info("[router] after_intent -> %s", intent)
    if intent == "complex_task":
        return TASK_PLANNING
    if intent == "tool_use":
        return TOOL_SELECTION
    if intent == "knowledge_qa":
        return KNOWLEDGE_RETRIEVAL
    return ANSWER_GENERATION  # chitchat / unknown


def route_after_reasoning(state: AgentState) -> str:
    """ReAct 推理后: 有待执行工具 -> tool_execution, 有 draft_answer -> validation,
    推理连续失败(is_fallback) -> fallback_handler。"""
    if state.get("is_fallback") or state.get("error_message"):
        return FALLBACK_HANDLER
    if state.get("draft_answer"):
        logger.info("[router] after_reasoning -> answer_validation (final_answer)")
        return ANSWER_VALIDATION
    if state.get("pending_tool"):
        logger.info("[router] after_reasoning -> tool_execution")
        return TOOL_EXECUTION
    # 无 pending 也无 answer（不应出现）→ fallback
    logger.warning("[router] after_reasoning: no pending_tool & no draft_answer -> fallback")
    return FALLBACK_HANDLER


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
    """After tool execution: ReAct 模式循环回 agent_reasoning, 快速路径进 generation。"""
    if state.get("use_react"):
        logger.info("[router] after_tool_execution -> agent_reasoning (react loop)")
        return AGENT_REASONING
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
    max_regen = 2  # allow one regeneration (answer_generation increments to 1, retry→2)
    if regen < max_regen:
        logger.info("[router] after_validation -> re-generate (attempt %d)", regen + 1)
        return ANSWER_GENERATION
    logger.info("[router] after_validation -> END (max retries reached)")
    return END
