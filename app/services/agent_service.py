"""Agent service layer - wraps the LangGraph workflow."""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from app.core.logger import get_logger
from app.graph.workflow import get_graph

logger = get_logger(__name__)
cfg = get_settings()


class SessionStore:
    """Lightweight in-memory session history with TTL expiry."""

    def __init__(self, ttl: int = 3600):
        self._ttl = ttl
        # {session_id: {"history": [...], "last_access": float}}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s["last_access"] > self._ttl]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("[session] evicted expired session: %s", sid)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the conversation history for *session_id* (may be empty)."""
        self._evict_expired()
        session = self._sessions.get(session_id)
        if session:
            session["last_access"] = time.time()
            return list(session["history"])
        return []

    def append(self, session_id: str, query: str, answer: str) -> None:
        """Append a (query, answer) turn to the session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {"history": [], "last_access": time.time()}
        session = self._sessions[session_id]
        session["history"].append({"role": "user",      "content": query})
        session["history"].append({"role": "assistant", "content": answer})
        session["last_access"] = time.time()
        # Keep last 20 turns (40 messages) to bound memory
        if len(session["history"]) > 40:
            session["history"] = session["history"][-40:]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class AgentService:
    def __init__(self):
        self._graph = get_graph()
        self._sessions = SessionStore(ttl=cfg.SESSION_TTL)

    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        logger.info("[agent_service] session=%s query=%r", session_id, query[:80])
        start = time.perf_counter()
        history = self._sessions.get_history(session_id)
        initial: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "steps": [],
            "retrieved_docs": [],
            "tool_args": {},
            "sub_tasks": [],
            "regeneration_count": 0,
            "retrieval_triggered": False,
            "tool_triggered": False,
            "is_fallback": False,
        }
        try:
            final: Dict[str, Any] = self._graph.invoke(
                initial,
                config={"recursion_limit": cfg.AGENT_MAX_ITERATIONS},
            )
        except Exception as exc:
            logger.error("[agent_service] graph error: %s", exc)
            final = {
                **initial,
                "final_answer": "An unexpected error occurred: " + str(exc),
                "is_fallback": True,
                "error_message": str(exc),
                "steps": ["graph_invoke -> FATAL ERROR"],
            }
        elapsed = time.perf_counter() - start
        logger.info("[agent_service] done in %.2fs", elapsed)
        return self._build_response(final, elapsed)

    @staticmethod
    def _build_response(state: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        docs = state.get("retrieved_docs") or []
        return {
            "query": state.get("query", ""),
            "session_id": state.get("session_id", ""),
            "intent": state.get("intent", "unknown"),
            "intent_confidence": state.get("intent_confidence", 0.0),
            "retrieval_triggered": state.get("retrieval_triggered", False),
            "retrieved_docs_count": len(docs),
            "tool_triggered": state.get("tool_triggered", False),
            "tool_name": state.get("tool_name"),
            "tool_result": state.get("tool_result"),
            "tool_error": state.get("tool_error"),
            "sub_tasks": state.get("sub_tasks") or [],
            "steps": state.get("steps") or [],
            "validation_passed": state.get("validation_passed", False),
            "validation_feedback": state.get("validation_feedback", ""),
            "is_fallback": state.get("is_fallback", False),
            "final_answer": state.get("final_answer") or state.get("draft_answer", ""),
            "elapsed_seconds": round(elapsed, 3),
        }


_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service
