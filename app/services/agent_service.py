"""Agent service layer - wraps the LangGraph workflow."""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Sequence
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

    # ── 智能路由：auto 模式下判断是否走多智能体 ──────────────────────────────
    @staticmethod
    def _should_use_multi(query: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
        """auto 模式下按查询特征判断是否走 Orchestrator-Worker 多智能体。

        规则（轻量，不调用 LLM）：
        1. 查询含多领域关键词组合（法律+代码、检索+计算、查询+生成等）→ multi
        2. 查询长度 > 100 字符且含「然后」「再」「并且」「同时」等连词 → multi
        3. 其余 → single（快速路径）
        """
        q = query.lower()

        # 多领域关键词组合（覆盖更全面的跨域场景）
        domain_pairs = [
            # 法律 + 代码/计算
            (("法律", "法条", "合同", "劳动", "赔偿", "安全生产", "民法典", "刑法", "行政处罚", "诉讼"),
             ("代码", "脚本", "python", "计算", "程序", "算法", "开发")),
            # 检索/查询 + 生成/写作
            (("查询", "检索", "搜索", "查一下", "查找", "了解"),
             ("写", "生成", "创作", "编写", "撰写", "起草", "整理")),
            # 分析/解读 + 代码/实现
            (("分析", "解读", "解释", "说明", "比较", "对比"),
             ("代码", "脚本", "程序", "算法", "实现", "开发")),
            # 法律 + 法律（跨法域比较，如安全生产法 vs 民法典）
            (("安全生产", "刑法", "行政处罚", "民法", "合同", "劳动"),
             ("民法典", "赔偿", "责任", "适用", "关系", "区别")),
        ]
        for domain_a, domain_b in domain_pairs:
            if any(k in q for k in domain_a) and any(k in q for k in domain_b):
                return True

        # 长查询 + 连词
        connectors = ("然后", "接着", "再", "并且", "同时", "以及", "之后")
        if len(query) > 80 and any(c in query for c in connectors):
            return True

        return False

    def run(
        self,
        query: str,
        session_id: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
        user_id=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        logger.info("[agent_service] session=%s query=%r", session_id, query[:80])
        start = time.perf_counter()

    # ── 多智能体分支（AGENT_MODE=multi 或 auto 智能判断）────────────────────
        if cfg.AGENT_MODE == "multi" or (
            cfg.AGENT_MODE == "auto" and self._should_use_multi(query, history)
        ):
            try:
                from app.agents.orchestrator import get_orchestrator

                orchestrator = get_orchestrator()
                result = orchestrator.run(
                    query,
                    history=history,
                    knowledge_base_ids=knowledge_base_ids,
                    knowledge_catalog=knowledge_catalog,
                )
                # 拆解器判定单一意图 → 回退单 Agent 快速路径
                if not result.get("degenerate_to_single"):
                    result["session_id"] = session_id
                    return result
                logger.info("[agent_service] orchestrator degenerated to single, fast path")
            except Exception as exc:
                logger.error("[agent_service] multi-agent failed, fallback single: %s", exc)
                # 崩溃回退 single，可用性永不回退

        # ── 单 Agent 路径（默认，行为不变）─────────────────────────────────
        # 优先使用传入的 DB 历史，否则回退到内存 SessionStore
        if history is None:
            history = self._sessions.get_history(session_id)
        initial: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "history": history,
            "user_id": str(user_id) if user_id else "",
            "knowledge_base_ids": list(knowledge_base_ids or []),
            "knowledge_catalog": list(knowledge_catalog or []),
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
        user_id=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
        on_step=None,
    ) -> Dict[str, Any]:
        """同步检索 + 构建生成消息, 为流式生成准备上下文。

        编排流程（每一步都可通过 on_step 回调透传到前端实时展示）:
          0. 查询改写  — 追问/指代结合历史还原成自包含问题（"今天呢"→"无锡今天天气"）
          1. 意图识别  — 带历史分类, 决定走哪条编排分支
          2. 分支执行:
             chitchat      — 跳过检索/工具, 直接对话
             tool_use      — tool_selection + tool_execution, 工具结果注入上下文
             knowledge_qa  — 向量检索知识库
             complex_task  — 工具 + 检索组合
        返回 dict 含: messages / sources / intent / tool_result / resolved_query。
        on_step: 可选回调 fn(step, detail)，在关键步骤调用。
        """
        from app.graph.nodes import (
            intent_recognition, knowledge_retrieval, tool_selection, tool_execution,
            rewrite_query_with_history,
        )
        from app.prompts.templates import (
            ANSWER_NO_CONTEXT, ANSWER_WITH_CONTEXT,
            ANSWER_WITH_ENHANCED_CONTEXT,
        )

        def _step(step: str, detail: str = ""):
            if on_step:
                try:
                    on_step(step, detail)
                except Exception:
                    pass

        history = history or []

        # 0. 查询改写：把"今天呢"这类追问结合历史还原成自包含问题
        _step("understand", "理解问题中...")
        resolved_query = rewrite_query_with_history(query, history)
        if resolved_query != query:
            _step("understand_done", f"理解为：{resolved_query}")
        else:
            _step("understand_done", query[:60])

        state: Dict[str, Any] = {
            "query": resolved_query, "steps": [], "user_id": user_id,
            "knowledge_base_ids": list(knowledge_base_ids or []),
            "knowledge_catalog": list(knowledge_catalog or []),
            "history": history,
        }

        # 1. 意图识别（带历史）
        _step("intent", "判断问题类型...")
        state.update(intent_recognition(state))
        intent = state.get("intent", "knowledge_qa")
        intent_label = {
            "knowledge_qa": "知识库问答", "tool_use": "联网/工具查询",
            "complex_task": "复杂任务", "chitchat": "闲聊",
        }.get(intent, intent)
        _step("intent_done", f"{intent_label}（置信度 {state.get('intent_confidence', 0):.0%}）")
        logger.info("[prepare_context] intent=%s conf=%.2f", intent, state.get("intent_confidence", 0))

        sources: List[Dict[str, Any]] = []
        tool_result_text = "N/A"

        # 2. 工具路径: tool_use 或 requires_tool 时执行工具
        if state.get("requires_tool") or intent == "tool_use":
            state.update(tool_selection(state))
            tool_name = state.get("tool_name")
            if tool_name:
                _step("tool", f"调用工具 {tool_name}...")
            state.update(tool_execution(state))
            if state.get("tool_triggered") and state.get("tool_result") is not None:
                tool_result_text = str(state["tool_result"])
                _step("tool_done", f"{tool_name} 返回结果")
            elif state.get("tool_error"):
                _step("tool_done", f"{tool_name} 失败：{state['tool_error'][:50]}")
            sources.extend(state.get("sources") or [])  # web_search 的引用

        # 3. 检索路径: 需要检索时做向量检索(chitchat 通常 requires_retrieval=False)
        docs = []
        if state.get("requires_retrieval", True) or intent in ("knowledge_qa", "complex_task"):
            _step("retrieve", "检索知识库...")
            state.update(knowledge_retrieval(state))
            docs = state.get("retrieved_docs") or []
            _step("retrieve_done", f"命中 {len(docs)} 条知识" if docs else "知识库无相关内容")
            sources.extend(state.get("kb_sources") or [])
            # knowledge_retrieval 内 web fallback 的 sources 也合并
            for s in (state.get("sources") or []):
                if s not in sources:
                    sources.append(s)

        # 4. 拼装生成消息（语义记忆: 注入用户 facts 到 system prompt）
        _step("generate", "生成回答中...")
        messages = [{"role": t["role"], "content": t["content"]} for t in history]

        from app.services.knowledge_catalog import format_knowledge_catalog

        messages.insert(0, {
            "role": "system",
            "content": format_knowledge_catalog(state.get("knowledge_catalog")),
        })

        # 语义记忆注入: 跨会话用户事实（偏好/身份/历史结论）
        # prepare_context 在 executor 线程跑, DB 查询走隔离 engine 避免连接池污染
        user_id = state.get("user_id")
        if user_id:
            try:
                from app.graph.nodes import _run_in_thread_isolated

                async def _fetch_facts(s):
                    from app.memory.manager import get_user_facts
                    return await get_user_facts(s, user_id)

                facts = _run_in_thread_isolated(_fetch_facts)
                if facts:
                    messages.insert(0, {
                        "role": "system",
                        "content": "关于这位用户的已知信息：\n" + "\n".join(f"- {f}" for f in facts),
                    })
            except Exception as exc:
                logger.warning("[prepare_context] user facts inject failed: %s", exc)

        knowledge_blocks = state.get("knowledge_blocks")
        if knowledge_blocks and docs:
            from app.rag.enhanced_retriever import format_blocks_for_prompt
            from app.graph.nodes import _rebuild_blocks
            context = format_blocks_for_prompt(_rebuild_blocks(knowledge_blocks, docs))
            # 截断保护：防止超大 context 导致 LLM 流式输出为空
            MAX_CONTEXT_CHARS = 8000
            if len(context) > MAX_CONTEXT_CHARS:
                logger.warning(
                    "[prepare_context] context too long (%d chars), truncating to %d",
                    len(context), MAX_CONTEXT_CHARS,
                )
                context = context[:MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"
            messages.append({
                "role": "user",
                "content": ANSWER_WITH_ENHANCED_CONTEXT.format(
                    query=resolved_query, context=context, tool_result=tool_result_text
                ),
            })
        elif docs:
            context = "\n\n".join(
                "[" + str(i + 1) + "] " + d["content"] for i, d in enumerate(docs)
            )
            messages.append({
                "role": "user",
                "content": ANSWER_WITH_CONTEXT.format(
                    query=resolved_query, context=context, tool_result=tool_result_text
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": ANSWER_NO_CONTEXT.format(query=resolved_query, tool_result=tool_result_text),
            })

        return {
            "messages": messages,
            "sources": sources,
            "intent": intent,
            "tool_result": state.get("tool_result"),
            "resolved_query": resolved_query,
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
