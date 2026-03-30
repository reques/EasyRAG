"""Domain-specific exception hierarchy.

Raise these inside business logic; catch them at the API boundary.
"""
from __future__ import annotations


# ── Base ─────────────────────────────────────────────────────────────────────

class AgentError(Exception):
    """Base for all agent errors."""


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLMError(AgentError):
    """LLM call failed or returned unusable output."""


# Aliases used in llm/client.py
LLMClientError = LLMError


class LLMFormatError(LLMError):
    """LLM returned output that could not be parsed into expected format."""


# Alias used in llm/client.py
LLMOutputParseError = LLMFormatError


class LLMTimeoutError(LLMError):
    """LLM call timed out."""


# ── Retrieval ────────────────────────────────────────────────────────────────

class RetrievalError(AgentError):
    """Vector store query failed."""


class EmptyRetrievalError(RetrievalError):
    """Query returned zero documents above threshold."""


# ── Embedding ────────────────────────────────────────────────────────────────

class EmbeddingError(AgentError):
    """Embedding model call failed."""


# ── Vector Store ─────────────────────────────────────────────────────────────

class VectorStoreError(AgentError):
    """Vector store operation failed."""


# ── Tools ────────────────────────────────────────────────────────────────────

class ToolError(AgentError):
    """A tool invocation failed."""

    def __init__(self, tool_name: str, detail: str):
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"Tool '{tool_name}' failed: {detail}")


class ToolNotFoundError(ToolError):
    """Requested tool does not exist in the registry."""

    def __init__(self, tool_name: str = "unknown", detail: str = "tool not registered"):
        self.tool_name = tool_name
        self.detail = detail
        AgentError.__init__(self, f"Tool '{tool_name}' not found: {detail}")


class ToolExecutionError(AgentError):
    """Tool raised an error during execution (non-ToolNotFoundError)."""

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(detail)


# ── Validation ───────────────────────────────────────────────────────────────

class ValidationError(AgentError):
    """Answer failed quality / validation check."""


# ── Planning ─────────────────────────────────────────────────────────────────

class PlanningError(AgentError):
    """Task planner could not produce a valid plan."""


# ── Session ──────────────────────────────────────────────────────────────────

class SessionNotFoundError(AgentError):
    """Requested session ID does not exist."""
