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

    def run(
        self,
        query: str,
        session_id: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        logger.info("[agent_service] session=%s query=%r", session_id, query[:80])
        start = time.perf_counter()
        # 优先使用传入的 DB 历史，否则回退到内存 SessionStore
        if history is None:
            history = self._sessions.get_history(session_id)
        initial: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "history": history,
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

    # ── 流式路径 (SSE) ────────────────────────────────────────────────────
    def prepare_context(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """同步检索 + 构建生成消息, 为流式生成准备上下文。

        先做意图识别(intent_recognition), 按 intent 分流:
          chitchat      — 跳过检索, 直接对话
          tool_use      — 走 tool_selection + tool_execution(web_search/calculator/datetime),
                          工具结果注入上下文; requires_retrieval 时叠加知识库检索
          knowledge_qa  — 向量检索(现状)
          complex_task  — 检索 + 可选工具组合
        返回 dict 含: messages / sources(含 file_id) / intent / tool_result。
        这是同步阻塞调用, 在 async 端点里需用 run_in_executor 包裹。
        """
        from app.graph.nodes import (
            intent_recognition, knowledge_retrieval, tool_selection, tool_execution,
        )
        from app.prompts.templates import ANSWER_NO_CONTEXT, ANSWER_WITH_CONTEXT

        history = history or []
        state: Dict[str, Any] = {"query": query, "steps": []}

        # 1. 意图识别(失败时 fallback knowledge_qa, 与完整路径一致)
        state.update(intent_recognition(state))
        intent = state.get("intent", "knowledge_qa")
        logger.info("[prepare_context] intent=%s conf=%.2f", intent, state.get("intent_confidence", 0))

        sources: List[Dict[str, Any]] = []
        tool_result_text = "N/A"

        # 2. 工具路径: tool_use 或 requires_tool 时执行工具
        if state.get("requires_tool") or intent == "tool_use":
            state.update(tool_selection(state))
            state.update(tool_execution(state))
            if state.get("tool_triggered") and state.get("tool_result") is not None:
                tool_result_text = str(state["tool_result"])
            sources.extend(state.get("sources") or [])  # web_search 的引用

        # 3. 检索路径: 需要检索时做向量检索(chitchat 通常 requires_retrieval=False)
        docs = []
        if state.get("requires_retrieval", True) or intent in ("knowledge_qa", "complex_task"):
            state.update(knowledge_retrieval(state))
            docs = state.get("retrieved_docs") or []
            sources.extend(state.get("kb_sources") or [])
            # knowledge_retrieval 内 web fallback 的 sources 也合并
            for s in (state.get("sources") or []):
                if s not in sources:
                    sources.append(s)

        # 4. 拼装生成消息
        messages = [{"role": t["role"], "content": t["content"]} for t in history]
        if docs:
            context = "\n\n".join(
                "[" + str(i + 1) + "] " + d["content"] for i, d in enumerate(docs)
            )
            messages.append({
                "role": "user",
                "content": ANSWER_WITH_CONTEXT.format(
                    query=query, context=context, tool_result=tool_result_text
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": ANSWER_NO_CONTEXT.format(query=query, tool_result=tool_result_text),
            })

        return {
            "messages": messages,
            "sources": sources,
            "intent": intent,
            "tool_result": state.get("tool_result"),
        }

    @staticmethod
    def _build_response(state: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        docs = state.get("retrieved_docs") or []
        # 合并知识库引用与 web 搜索引用, 前端统一渲染引用块
        sources = list(state.get("kb_sources") or [])
        sources.extend(state.get("sources") or [])
        response = {
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
            "sources": sources,
            "final_answer": state.get("final_answer") or state.get("draft_answer", ""),
            "elapsed_seconds": round(elapsed, 3),
        }
        # 增强检索附加字段
        if state.get("knowledge_blocks"):
            response["knowledge_blocks"] = state["knowledge_blocks"]
        if state.get("query_decomposition"):
            response["query_decomposition"] = state["query_decomposition"]
        if state.get("gap_rounds"):
            response["gap_rounds"] = state["gap_rounds"]
            response["gap_details"] = state.get("gap_details") or []
        return response


_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service
