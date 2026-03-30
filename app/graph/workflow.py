"""LangGraph workflow assembly.

Builds a StateGraph from nodes and routing functions,
then compiles it into a runnable app.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.logger import get_logger
from app.graph.state import AgentState
from app.graph import nodes
from app.graph.router import (
    route_after_intent,
    route_after_planning,
    route_after_retrieval,
    route_after_tool_execution,
    route_after_generation,
    route_after_validation,
    INTENT_RECOGNITION,
    TASK_PLANNING,
    KNOWLEDGE_RETRIEVAL,
    TOOL_SELECTION,
    TOOL_EXECUTION,
    ANSWER_GENERATION,
    ANSWER_VALIDATION,
    FALLBACK_HANDLER,
)

logger = get_logger(__name__)


def build_graph():
    """Construct and compile the LangGraph agent workflow.

    Graph topology:
        intent_recognition
            |--(complex_task)--> task_planning --> [retrieval | tool_selection | generation]
            |--(tool_use)-------> tool_selection --> tool_execution --> answer_generation
            |--(knowledge_qa)---> knowledge_retrieval --> answer_generation
            |--(chitchat)-------> answer_generation
        answer_generation --> answer_validation --> END
                                                --> answer_generation (1 retry)
        any error --> fallback_handler --> END
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node(INTENT_RECOGNITION,  nodes.intent_recognition)
    graph.add_node(TASK_PLANNING,       nodes.task_planning)
    graph.add_node(KNOWLEDGE_RETRIEVAL, nodes.knowledge_retrieval)
    graph.add_node(TOOL_SELECTION,      nodes.tool_selection)
    graph.add_node(TOOL_EXECUTION,      nodes.tool_execution)
    graph.add_node(ANSWER_GENERATION,   nodes.answer_generation)
    graph.add_node(ANSWER_VALIDATION,   nodes.answer_validation)
    graph.add_node(FALLBACK_HANDLER,    nodes.fallback_handler)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point(INTENT_RECOGNITION)

    # ── Conditional edges ────────────────────────────────────────────────
    graph.add_conditional_edges(
        INTENT_RECOGNITION,
        route_after_intent,
        {
            TASK_PLANNING:       TASK_PLANNING,
            TOOL_SELECTION:      TOOL_SELECTION,
            KNOWLEDGE_RETRIEVAL: KNOWLEDGE_RETRIEVAL,
            ANSWER_GENERATION:   ANSWER_GENERATION,
            FALLBACK_HANDLER:    FALLBACK_HANDLER,
        },
    )

    graph.add_conditional_edges(
        TASK_PLANNING,
        route_after_planning,
        {
            TOOL_SELECTION:      TOOL_SELECTION,
            KNOWLEDGE_RETRIEVAL: KNOWLEDGE_RETRIEVAL,
            ANSWER_GENERATION:   ANSWER_GENERATION,
        },
    )

    graph.add_conditional_edges(
        KNOWLEDGE_RETRIEVAL,
        route_after_retrieval,
        {
            TOOL_SELECTION:    TOOL_SELECTION,
            ANSWER_GENERATION: ANSWER_GENERATION,
            FALLBACK_HANDLER:  FALLBACK_HANDLER,
        },
    )

    # tool_selection always proceeds to tool_execution
    graph.add_edge(TOOL_SELECTION, TOOL_EXECUTION)

    graph.add_conditional_edges(
        TOOL_EXECUTION,
        route_after_tool_execution,
        {ANSWER_GENERATION: ANSWER_GENERATION},
    )

    graph.add_conditional_edges(
        ANSWER_GENERATION,
        route_after_generation,
        {
            ANSWER_VALIDATION: ANSWER_VALIDATION,
            FALLBACK_HANDLER:  FALLBACK_HANDLER,
        },
    )

    graph.add_conditional_edges(
        ANSWER_VALIDATION,
        route_after_validation,
        {
            ANSWER_GENERATION: ANSWER_GENERATION,
            END:               END,
        },
    )

    # fallback always terminates
    graph.add_edge(FALLBACK_HANDLER, END)

    compiled = graph.compile()
    logger.info("[workflow] LangGraph compiled successfully")
    return compiled


# Module-level singleton
_graph = None


def get_graph():
    """Return the compiled graph (built once per process)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
