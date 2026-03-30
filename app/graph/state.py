"""LangGraph shared state definition for the Agent workflow."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every LangGraph node.

    Fields are optional (total=False) so nodes only need to write
    the keys they own; readers must use `.get()` with a default.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    query: str                          # original user query
    session_id: str                     # conversation session ID
    history: List[Dict[str, str]]       # prior turns [{"role":…,"content":…}]

    # ── Intent / Planning ─────────────────────────────────────────────────
    intent: str                         # e.g. "knowledge_qa", "tool_use", "complex_task"
    intent_confidence: float            # 0.0-1.0
    requires_retrieval: bool            # should RAG be invoked?
    requires_tool: bool                 # should a tool be invoked?
    tool_name: Optional[str]            # chosen tool name
    tool_args: Dict[str, Any]           # arguments for the tool
    sub_tasks: List[str]                # decomposed sub-tasks (complex flow)

    # ── RAG ───────────────────────────────────────────────────────────────
    retrieved_docs: List[Dict[str, Any]]  # [{"content":…, "metadata":…}]
    retrieval_triggered: bool

    # ── Tool ──────────────────────────────────────────────────────────────
    tool_result: Optional[str]          # raw output of tool execution
    tool_triggered: bool
    tool_error: Optional[str]           # error message if tool failed

    # ── Generation ────────────────────────────────────────────────────────
    draft_answer: str                   # first-pass generated answer
    final_answer: str                   # validated / refined answer

    # ── Validation ────────────────────────────────────────────────────────
    validation_passed: bool
    validation_feedback: str
    regeneration_count: int             # how many times we re-generated

    # ── Audit trail ───────────────────────────────────────────────────────
    steps: List[str]                    # human-readable log of each node visited
    error_message: Optional[str]        # set by fallback_handler
    is_fallback: bool                   # True when fallback path was taken
