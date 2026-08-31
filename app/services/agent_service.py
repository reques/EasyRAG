"""Agent service layer - wraps the LangGraph workflow."""
from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    # 领域 → 关键词（2026-08-15 修订：从"围绕法律手工列举的组合"改为领域字典，
    # 保证知识库类型多样时跨领域查询仍能触发多智能体。新增领域只需加一行）。
    _DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "法律": ("法律", "法条", "合同", "劳动", "赔偿", "安全生产", "民法典",
                 "刑法", "行政处罚", "诉讼", "合规", "仲裁", "条款"),
        "代码/计算": ("代码", "脚本", "python", "计算", "程序", "算法", "开发",
                     "调试", "接口", "部署", "实现"),
        "写作/整理": ("写", "生成", "创作", "编写", "撰写", "起草", "整理",
                     "总结", "报告", "摘要"),
        "检索/查询": ("查询", "检索", "搜索", "查一下", "查找", "了解", "调研"),
        "分析/解读": ("分析", "解读", "解释", "说明", "比较", "对比", "评估", "研究"),
        "金融/财经": ("股票", "基金", "投资", "理财", "汇率", "利率", "税务",
                     "保险", "贷款", "财务"),
        "医疗/健康": ("医院", "医生", "药品", "症状", "诊断", "治疗", "健康",
                     "饮食", "营养", "运动"),
        "科技/互联网": ("人工智能", "机器学习", "大模型", "芯片", "互联网",
                       "软件", "硬件", "数据"),
        "职场/人力": ("简历", "面试", "职场", "加班", "裁员", "晋升", "绩效", "沟通"),
        "生活/消费": ("菜谱", "做饭", "装修", "旅游", "机票", "酒店", "攻略", "购物"),
        "教育/学术": ("考试", "学习", "课程", "论文", "考研", "留学", "培训", "教材"),
        "历史/人文": ("历史", "文化", "朝代", "文物", "哲学", "名著", "人物"),
    }

    @staticmethod
    def _should_use_multi(query: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
        """auto 模式下按查询特征判断是否走多智能体路径。

        2026-08-26（阶段 3）：命中后路由到 DeepAgents（主 Agent + SubAgent +
        spawn_tasks DAG），Orchestrator-Worker 已退役。规则本身不变：

        规则（轻量，不调用 LLM）：
        1. 查询命中 ≥2 个**不同领域**的关键词 → multi
           （领域来自 _DOMAIN_KEYWORDS 字典，法律/代码/金融/医疗/教育/生活…
           任意跨领域组合都命中，不再围绕单一领域列举 pair）
        2. 查询长度 > 80 字符且含「然后」「再」「并且」「同时」等连词 → multi
        3. 其余 → single（快速路径）
        """
        q = query.lower()
        hit_domains = {
            name
            for name, keywords in AgentService._DOMAIN_KEYWORDS.items()
            if any(k in q for k in keywords)
        }
        if len(hit_domains) >= 2:
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
        image_data: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("[agent_service] session=%s query=%r", session_id, query[:80])
        start = time.perf_counter()

        # ── DeepAgents 模式（AGENT_MODE=deepagents）：主 Agent + SubAgent ──
        if cfg.AGENT_MODE == "deepagents":
            return self._run_deep(
                query,
                session_id=session_id,
                history=history,
                user_id=user_id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
            )

        # ── 多智能体分支：multi / auto 命中均路由到 DeepAgents（统一实现）──
        # AGENT_MODE=multi 作为 deepagents 的兼容别名保留（2026-08-26 阶段 5，
        # Orchestrator-Worker 已退役删除）。
        if cfg.AGENT_MODE == "multi":
            logger.warning(
                "[agent_service] AGENT_MODE=multi 已废弃（deepagents 别名），"
                "建议改用 AGENT_MODE=deepagents"
            )
            return self._run_deep(
                query,
                session_id=session_id,
                history=history,
                user_id=user_id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
            )
        if cfg.AGENT_MODE == "auto" and self._should_use_multi(query, history):
            logger.info(
                "[agent_service] auto 判定复杂任务，路由至 deepagents（主 Agent 自行拆解/委派）"
            )
            return self._run_deep(
                query,
                session_id=session_id,
                history=history,
                user_id=user_id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
            )

        # ── 单 Agent 旧固定管线（AGENT_MODE=single）─────────────────────────────────
        # 优先使用传入的 DB 历史，否则回退到内存 SessionStore
        # ── 轻量动态 Agent（auto 普通问题 / AGENT_MODE=dynamic）──
        # 模型每轮自行决策：直接回答 / 调工具（web_search、calculator…）/ 检索
        # 知识库（kb_search）；简单问题只消耗一次 LLM 调用，不走固定管线。
        if cfg.AGENT_MODE != "single":
            logger.info(
                "[agent_service] 路由至 dynamic agent（动态工具/检索决策）",
            )
            return self._run_dynamic(
                query,
                session_id=session_id,
                history=history,
                user_id=user_id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
                image_data=image_data,
            )

        if history is None:
            history = self._sessions.get_history(session_id)
        initial: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "history": history,
            "user_id": str(user_id) if user_id else "",
            "knowledge_base_ids": list(knowledge_base_ids or []),
            "knowledge_catalog": list(knowledge_catalog or []),
            "image_data": image_data,
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

    # ── DeepAgents 模式（AGENT_MODE=deepagents）──────────────────────────
    def _run_dynamic(
        self,
        query: str,
        session_id: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
        user_id=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        on_step=None,
        on_artifact=None,
    ) -> Dict[str, Any]:
        """轻量动态 Agent 入口：请求级 trace + 统一事件流。

        复用 ``app/agents/dynamic.run_dynamic_agent``：模型通过函数调用自行决定
        是否调工具、是否检索、是否直接回答；返回与单 Agent 兼容的响应结构。
        """
        from app.agents.dynamic import run_dynamic_agent
        from app.agents.events import use_request_trace

        with use_request_trace(session_id=session_id) as request_trace:
            if history is None:
                history = self._sessions.get_history(session_id)
            result = run_dynamic_agent(
                query,
                session_id=session_id,
                history=history,
                user_id=user_id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
                image_data=image_data,
                on_step=on_step,
                on_artifact=on_artifact,
            )
            result["trace_id"] = request_trace.trace.trace_id
            result["events"] = list(request_trace.events)
            return result


    def _run_deep(
        self,
        query: str,
        session_id: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
        user_id=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
        on_step=None,
        on_artifact=None,
    ) -> Dict[str, Any]:
        """DeepAgents 入口：建立请求级 trace（统一事件流，2026-08-26 阶段 1）。

        作用域内所有结构化事件（步骤/工具调用/委派）进入同一事件流，随响应
        返回 ``trace_id`` + ``events``（内存级执行轨迹，供诊断/前端消费；
        持久化在后续阶段接入）。SSE 步骤透传行为不变（见 _run_deep_inner）。
        """
        from app.agents.events import use_request_trace

        with use_request_trace(session_id=session_id) as request_trace:
            result = self._run_deep_inner(
                query,
                session_id=session_id,
                history=history,
                user_id=user_id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
                on_step=on_step,
                on_artifact=on_artifact,
            )
            result["trace_id"] = request_trace.trace.trace_id
            result["events"] = list(request_trace.events)
            return result

    def _run_deep_inner(
        self,
        query: str,
        session_id: str = "default",
        history: Optional[List[Dict[str, str]]] = None,
        user_id=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
        on_step=None,
        on_artifact=None,
    ) -> Dict[str, Any]:
        """DeepAgents 风格：主 Agent（create_react_agent + task 工具）→ SubAgent。

        同步执行（子 Agent 内联在 task 工具中，无异步任务系统）。
        on_step: 可选回调 fn(step, detail)，供 SSE 流式端点实时透传阶段状态。
        on_artifact: 可选回调 fn(dict)，推送 ReAct 每轮推理思考 / 工具输入输出
          等中间产出（{kind, stage, title, content}），实时透传给前端。
        返回与单 Agent 兼容的响应结构。（由 _run_deep 包裹统一事件流。）
        """
        from app.agents.deep.agent import get_main_agent
        from app.agents.events import emit
        from app.services.knowledge_catalog import format_knowledge_catalog
        from app.skills.context import get_active_skill_prompt
        from langgraph.errors import GraphRecursionError

        start = time.perf_counter()
        steps: List[str] = []
        artifacts: List[Dict[str, Any]] = []

        def _step(step: str, detail: str = ""):
            steps.append(f"{step}: {detail}")
            emit("step", step, step, detail)
            if on_step:
                try:
                    on_step(step, detail)
                except Exception:
                    pass

        def _artifact(kind: str, stage: str, title: str, content: str):
            if not content:
                return
            ev = {"kind": kind, "stage": stage, "title": title[:80], "content": content}
            artifacts.append(ev)
            emit("artifact", stage, title, content, artifact_kind=kind)
            if on_artifact:
                try:
                    on_artifact(dict(ev))
                except Exception:
                    pass

        if history is None:
            history = self._sessions.get_history(session_id)

        # ── 组装消息（复用 prepare_context 的注入链，保证行为一致）──────
        messages: List[Dict[str, str]] = []
        skill_prompt = get_active_skill_prompt()
        if skill_prompt:
            messages.append({"role": "system", "content": skill_prompt})
        if knowledge_catalog:
            messages.append({
                "role": "system",
                "content": format_knowledge_catalog(list(knowledge_catalog)),
            })
        if user_id:
            try:
                from app.graph.nodes import _run_in_thread_isolated

                async def _fetch_facts(s):
                    from app.memory.manager import get_user_facts
                    return await get_user_facts(s, user_id)

                facts = _run_in_thread_isolated(_fetch_facts)
                if facts:
                    messages.append({
                        "role": "system",
                        "content": "关于这位用户的已知信息：\n" + "\n".join(f"- {f}" for f in facts),
                    })
            except Exception as exc:
                logger.warning("[run_deep] user facts inject failed: %s", exc)

        # 先向流式客户端报告研究规划，再进入可能较慢的知识库检索，避免
        # 深度研究启动后长时间没有任何高层进度反馈。
        _step("understand", "DeepAgents 主 Agent 开始处理...")

        # ── 知识库前置检索（2026-08-21, S1：此前 DeepAgents 从不检索知识库，
        #    系统提示却声称"检索结果会作为上下文提供"——知识库问答退化成纯 LLM
        #    生成。此处与 prepare_context 对齐：生成前检索并注入上下文）────
        sources: List[Dict[str, str]] = []
        if knowledge_base_ids:
            try:
                from app.rag.enhanced_retriever import (
                    format_blocks_for_prompt,
                    format_flat_for_prompt,
                    get_enhanced_retriever,
                )

                _kb_result = get_enhanced_retriever().retrieve(
                    query,
                    history=history,
                    knowledge_base_ids=list(knowledge_base_ids),
                )
                _kb_context = ""
                if _kb_result.knowledge_blocks:
                    _kb_context = format_blocks_for_prompt(_kb_result.knowledge_blocks)
                elif _kb_result.raw_docs:
                    _kb_context = format_flat_for_prompt(_kb_result.raw_docs)
                if _kb_context:
                    messages.append({
                        "role": "system",
                        "content": (
                            "以下是知识库检索到的相关内容（回答时优先采用，并标注来源）：\n"
                            + _kb_context
                        ),
                    })
                    _step("retrieve", f"知识库命中 {len(_kb_result.raw_docs)} 条")
                    for _doc in _kb_result.raw_docs[:4]:
                        _src = (_doc.metadata or {}).get("source") or "知识片段"
                        _doc_text = str(_doc.content or "").strip()
                        if _doc_text:
                            _snippet = " ".join(_doc_text.split())[:90]
                            _artifact(
                                "retrieve", "retrieve", _src,
                                _snippet + ("…" if len(_doc_text) > 90 else ""),
                            )
                else:
                    _step("retrieve_done", "知识库无相关内容")
                for _s in _kb_result.sources:
                    if _s not in sources:
                        sources.append(_s)
            except Exception as exc:
                logger.warning("[run_deep] kb retrieval failed: %s", exc)
                _step("retrieve_done", f"检索失败: {str(exc)[:50]}")

        messages.extend(history)
        messages.append({"role": "user", "content": query})

        # 请求级知识库授权：作用域内 kb_search 工具（含 task 委派的 SubAgent）
        # 都能读取当前用户授权范围，避免越权；contextvars 对同线程同步调用链可见
        from app.services.knowledge_context import use_authorised_kb_ids
        # S3 步骤透传：task 工具读取该观察者，把子 Agent 中间步骤透传 SSE
        from app.agents.deep.observe import use_task_observers

        with (
            use_authorised_kb_ids(knowledge_base_ids),
            use_task_observers(_step, _artifact),
        ):
            agent = get_main_agent()
            tool_called: Optional[str] = None
            final_state: Optional[Dict[str, Any]] = None
            recursion_hit = False
            try:
                # stream_mode="values"：每个 chunk 是全量 state，最后一个即最终状态
                for chunk in agent.stream(
                    {"messages": messages},
                    config={"recursion_limit": cfg.DEEP_MAIN_RECURSION_LIMIT},
                    stream_mode="values",
                ):
                    final_state = chunk
                    msgs = chunk.get("messages") or []
                    if not msgs:
                        continue
                    last = msgs[-1]
                    mtype = getattr(last, "type", "")
                    tc = getattr(last, "tool_calls", None)
                    if tc:
                        # ReAct 一步：AI 消息正文 = 这一步的推理思考，tool_calls = 动作
                        # （act and reasoning：思考内容直接透出到步骤流，而非固定短语）
                        tool_called = tc[0].get("name", "")
                        _ai_thought = str(getattr(last, "content", "") or "").strip()
                        if _ai_thought:
                            # 思考过长时截断，避免刷屏（主对话框只展示思考要点）
                            _thought_snippet = " ".join(_ai_thought.split())[:400]
                            if len(_ai_thought) > 400:
                                _thought_snippet += "…"
                            _step("agent_reasoning", _thought_snippet)
                            _artifact("thought", "reason", "推理", _thought_snippet)
                        _args = tc[0].get("args") or {}
                        try:
                            _args_text = json.dumps(_args, ensure_ascii=False)[:800]
                        except Exception:
                            _args_text = str(_args)[:800]
                        _args_short = " ".join(_args_text.split())[:160]
                        if len(_args_text) > 160:
                            _args_short += "…"
                        _step("tool", f"调用 {tool_called} {_args_short}".rstrip())
                        _artifact(
                            "delegate" if tool_called == "task" else "tool",
                            "tool",
                            f"调用 {tool_called}",
                            _args_text,
                        )
                    elif mtype == "tool":
                        _t_content = str(getattr(last, "content", "") or "")
                        _step("tool_done", f"工具返回: {_t_content[:120]}")
                        # 工具返回 → 总结性截断（全文只在需要时可用，不进入主对话框）
                        _tool_flat = " ".join(_t_content.split())
                        _artifact(
                            "tool_result", "tool", "工具返回",
                            _tool_flat[:300] + ("…" if len(_tool_flat) > 300 else ""),
                        )
                        _step("agent_reasoning_done", "推理完成")
                    elif mtype == "ai" and getattr(last, "content", ""):
                        # 无 tool_calls 的 AI 消息 = 最终回答（循环末尾），不是中间思考
                        _step("generate", "主 Agent 生成回答中...")
            except GraphRecursionError:
                # S4 超限降级（2026-08-26，阶段 1）：基于已积累的 messages 强制收尾，
                # 对齐单 Agent 图的 "max iterations, forced answer"——不再直接返回错误。
                # final_state 保留了最后一个成功 chunk（部分执行状态）。
                recursion_hit = True
                steps.append("deep agent hit recursion limit, forced answer from partial state")
                _step("fallback", "已达推理步数上限，基于已有信息收尾")
            except Exception as exc:
                logger.error("[run_deep] deep agent error: %s", exc)
                steps.append(f"deep agent error: {exc}")
                _step("fallback", "深度研究执行失败，准备返回可用结果")
                return {
                    "query": query,
                    "session_id": session_id,
                    "final_answer": f"处理请求时发生错误: {exc}",
                    "intent": "deepagents",
                    "intent_confidence": 0.0,
                    "retrieval_triggered": False,
                    "retrieved_docs_count": 0,
                    "tool_triggered": False,
                    "tool_name": None,
                    "tool_result": None,
                    "tool_error": str(exc),
                    "sub_tasks": [],
                    "steps": steps,
                    "artifacts": artifacts,
                    "sources": sources,
                    "is_fallback": True,
                    "degraded": False,
                    "error_message": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                }

        msgs = (final_state or {}).get("messages") or []
        answer = ""
        answer_is_final_ai = False
        for m in reversed(msgs):
            content = getattr(m, "content", "") or ""
            if getattr(m, "type", "") == "ai" and content:
                answer = content if isinstance(content, str) else str(content)
                answer_is_final_ai = True
                break
        if not answer and msgs:
            answer = str(getattr(msgs[-1], "content", "") or "")

        if recursion_hit:
            # S4：部分状态里通常没有最终 AI 消息——用已收集的工具结果拼接
            # forced answer（与单 Agent 图 agent_reasoning 的收尾一致）。
            tool_texts = [
                str(getattr(m, "content", "") or "")
                for m in msgs
                if getattr(m, "type", "") == "tool"
            ]
            observations = "\n".join(t for t in tool_texts if t.strip())
            if not answer_is_final_ai:
                if observations:
                    answer = f"基于已有信息：\n{observations[:600]}"
                else:
                    # 无 AI 回答也无工具产出：answer 只是残留的原始消息文本，不可用
                    answer = "（已达推理步数上限，未能收集到足够信息；请尝试简化问题后重新提问。）"
            _step("generate_done", "超限收尾完成（降级回答）")
        else:
            _step("generate_done", f"回答完成（{len(answer)} 字符）")
        return {
            "query": query,
            "session_id": session_id,
            "final_answer": answer,
            "intent": "deepagents",
            "intent_confidence": 0.0,
            "retrieval_triggered": False,
            "retrieved_docs_count": 0,
            "tool_triggered": bool(tool_called),
            "tool_name": tool_called,
            "tool_result": None,
            "tool_error": None,
            "sub_tasks": [],
            "steps": steps,
            "artifacts": artifacts,
            "sources": sources,
            "is_fallback": False,
            "degraded": recursion_hit,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }

    # ── 流式路径 (SSE) ────────────────────────────────────────────────────
    def prepare_context(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        user_id=None,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        on_step=None,
        on_artifact=None,
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
        on_artifact: 可选回调 fn(dict)，推送检索片段/工具结果等中间产出
          （{kind, stage, title, content}），供 SSE 实时透传给前端。
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

        def _artifact(kind: str, stage: str, title: str, content: str):
            if not on_artifact or not content:
                return
            try:
                on_artifact({
                    "kind": kind, "stage": stage,
                    "title": title[:80], "content": content,
                })
            except Exception:
                pass

        def _summarize_tool_result(tool: str, text: str) -> str:
            """工具返回 → 总结性描述（不贴全文，主对话框只展示要点）。"""
            # web_search 返回 JSON：提取结果条数与标题做总结
            if tool == "web_search":
                try:
                    data = json.loads(text)
                    results = data.get("results") or []
                    if isinstance(results, list) and results:
                        titles = [
                            str(r.get("title") or r.get("url") or "")
                            for r in results if isinstance(r, dict)
                        ]
                        titles = [t for t in titles if t][:3]
                        tail = "…" if len(results) > 3 else ""
                        return f"搜索到 {len(results)} 条结果：" + "、".join(titles) + tail
                except Exception:
                    pass
            flat = " ".join(text.split())
            return flat[:200] + ("…" if len(flat) > 200 else "")

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
            "complex_task": "复杂任务", "chitchat": "闲聊", "direct": "直接回答",
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
                # 结果摘要进 detail，前端可直接展示工具返回了什么
                _tool_summary = tool_result_text.strip().replace("\n", " ")
                if len(_tool_summary) > 60:
                    _tool_summary = _tool_summary[:60] + "…"
                _step("tool_done", f"{tool_name} 返回：{_tool_summary}" if _tool_summary else f"{tool_name} 返回结果")
                # 工具返回 → 总结性要点（主对话框不贴全文）
                _artifact("tool_result", "tool", f"{tool_name} 返回", _summarize_tool_result(tool_name, tool_result_text))
            elif state.get("tool_error"):
                _step("tool_done", f"{tool_name} 失败：{state['tool_error'][:50]}")
                _artifact("tool_result", "tool", f"{tool_name} 失败", " ".join(str(state["tool_error"]).split())[:200])
            sources.extend(state.get("sources") or [])  # web_search 的引用

        # 3. 检索路径: 需要检索时做向量检索(chitchat 通常 requires_retrieval=False)
        docs = []
        if state.get("requires_retrieval", True) or intent in ("knowledge_qa", "complex_task"):
            _kb_count = len(list(knowledge_base_ids or []))
            _step(
                "retrieve",
                f"在 {_kb_count} 个知识库中检索..." if _kb_count else "检索知识库...",
            )
            state.update(knowledge_retrieval(state))
            docs = state.get("retrieved_docs") or []
            if docs:
                # 来源文件名摘要（最多 3 个），让"命中 N 条"落到具体文件上
                _src_titles: List[str] = []
                for _s in (state.get("kb_sources") or []):
                    _t = _s.get("title") or ""
                    if _t and _t not in _src_titles:
                        _src_titles.append(_t)
                    if len(_src_titles) >= 3:
                        break
                _retrieve_detail = f"命中 {len(docs)} 条知识"
                if _src_titles:
                    _retrieve_detail += "，来源：" + "、".join(_src_titles)
                _step("retrieve_done", _retrieve_detail)
                # 检索命中 → 推送总结性摘要（来源名 + 首句），不贴全文
                for _doc in docs[:4]:
                    _doc_meta = _doc.get("metadata") or {}
                    _doc_title = (
                        _doc.get("source")
                        or _doc.get("title")
                        or _doc.get("filename")
                        or _doc_meta.get("source")
                        or _doc_meta.get("title")
                        or _doc_meta.get("filename")
                        or "知识片段"
                    )
                    _doc_content = str(_doc.get("content") or "").strip()
                    # 取首句（前 90 字）作为一句话摘要
                    _doc_snippet = " ".join(_doc_content.split())[:90]
                    if _doc_snippet:
                        _tail = "…" if len(_doc_content) > 90 else ""
                        _artifact("retrieve", "retrieve", f"{_doc_title}", _doc_snippet + _tail)
            else:
                _step("retrieve_done", "知识库无相关内容")
            sources.extend(state.get("kb_sources") or [])
            # knowledge_retrieval 内 web fallback 的 sources 也合并
            for s in (state.get("sources") or []):
                if s not in sources:
                    sources.append(s)

        # 4. 拼装生成消息（语义记忆: 注入用户 facts 到 system prompt）
        #    生成步骤 detail 带上当前模型名，前端"生成回答"节点更具体
        try:
            from app.llm.client import get_active_chat_model_profile
            _active_profile = get_active_chat_model_profile()
            _gen_model = _active_profile.name if _active_profile else ""
        except Exception:
            _gen_model = ""
        _step("generate", f"生成回答中..." + (f"（{_gen_model}）" if _gen_model else ""))
        messages = [{"role": t["role"], "content": t["content"]} for t in history]

        from app.services.knowledge_catalog import format_knowledge_catalog
        from app.skills.context import get_active_skill_prompt

        skill_prompt = get_active_skill_prompt()
        if skill_prompt:
            messages.insert(0, {"role": "system", "content": skill_prompt})

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
                "content": (
                    [
                        {"type": "text", "text": ANSWER_WITH_ENHANCED_CONTEXT.format(
                            query=resolved_query, context=context, tool_result=tool_result_text
                        )},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]
                    if image_data else
                    ANSWER_WITH_ENHANCED_CONTEXT.format(
                        query=resolved_query, context=context, tool_result=tool_result_text
                    )
                ),
            })
        elif docs:
            context = "\n\n".join(
                "[" + str(i + 1) + "] " + d["content"] for i, d in enumerate(docs)
            )
            messages.append({
                "role": "user",
                "content": (
                    [
                        {"type": "text", "text": ANSWER_WITH_CONTEXT.format(
                            query=resolved_query, context=context, tool_result=tool_result_text
                        )},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]
                    if image_data else
                    ANSWER_WITH_CONTEXT.format(
                        query=resolved_query, context=context, tool_result=tool_result_text
                    )
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    [
                        {"type": "text", "text": ANSWER_NO_CONTEXT.format(
                            query=resolved_query, tool_result=tool_result_text
                        )},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]
                    if image_data else
                    ANSWER_NO_CONTEXT.format(query=resolved_query, tool_result=tool_result_text)
                ),
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
