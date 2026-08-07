from __future__ import annotations
from typing import Any, Dict, List
from app.core.config import get_settings
from app.core.exceptions import EmptyRetrievalError, LLMClientError, ToolError
from app.core.logger import get_logger
from app.graph.state import AgentState
from app.llm.client import get_llm_client
from app.prompts.templates import (
    ANSWER_NO_CONTEXT, ANSWER_VALIDATION, ANSWER_WITH_CONTEXT,
    ANSWER_WITH_ENHANCED_CONTEXT, ANSWER_WITH_ENHANCED_NO_CONTEXT,
    FALLBACK_ANSWER, INTENT_RECOGNITION, REACT_REASONING, TASK_PLANNING,
)
from app.tools.registry import get_tool_registry
logger = get_logger(__name__)
cfg = get_settings()

def _append_step(state, msg):
    steps = list(state.get("steps") or [])
    steps.append(msg)
    return steps


def _format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 4) -> str:
    """把最近几轮对话压缩成分类器可读的一行一行文本。"""
    if not history:
        return "（无历史对话）"
    recent = history[-(max_turns * 2):]
    lines = []
    for t in recent:
        role = "用户" if t.get("role") == "user" else "助手"
        content = (t.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{role}: {content[:120]}")
    return "\n".join(lines) if lines else "（无历史对话）"


# 短指代词触发 query 重写。消息含这些特征且很短时，大概率依赖上文。
_FOLLOWUP_HINTS = ("呢", "那", "它", "他", "她", "这", "那", "还有", "再", "呢？", "吗", "怎么样", "如何")


def _needs_rewrite(query: str, history: List[Dict[str, str]]) -> bool:
    """判断是否需要结合历史做指代消解。只有存在历史时才可能改写。"""
    if not history:
        return False
    q = query.strip()
    if len(q) <= 12:  # 很短 → 大概率是追问
        return True
    return any(h in q for h in _FOLLOWUP_HINTS) and len(q) <= 30


def rewrite_query_with_history(query: str, history: List[Dict[str, str]]) -> str:
    """结合历史把追问改写成自包含问题。失败或无历史时返回原 query。"""
    if not _needs_rewrite(query, history):
        return query
    try:
        client = get_llm_client(tier="fast")
        from app.prompts.templates import QUERY_REWRITE
        prompt = QUERY_REWRITE.format(
            history=_format_history_for_prompt(history),
            query=query,
        )
        rewritten = client.chat_sync(
            [{"role": "user", "content": prompt}], temperature=0.0
        ).strip().strip('"').strip("'").strip()
        # 防御：改写结果为空或反而更长/更怪 → 用原句
        if rewritten and 0 < len(rewritten) <= 200:
            logger.info("[query_rewrite] %r -> %r", query[:40], rewritten[:60])
            return rewritten
    except Exception as exc:
        logger.warning("[query_rewrite] failed, use original: %s", exc)
    return query


def intent_recognition(state):
    """Node 1: Classify user intent and set routing flags. 带历史上下文。"""
    query = state["query"]
    history = state.get("history") or []
    logger.info("[intent_recognition] query=%r history_turns=%d", query[:80], len(history) // 2)
    client = get_llm_client(tier="fast")
    # 动态注入当前可用工具（含 MCP 工具）——prompt 里硬编码枚举会漏掉
    # 后注册的工具，导致 LLM 把未知工具名当参数传给别的工具（如 "echo" → text_tool）
    from app.tools.registry import get_tool_registry
    available_tools = get_tool_registry().to_react_prompt()
    prompt = INTENT_RECOGNITION.format(
        history=_format_history_for_prompt(history),
        query=query,
        available_tools=available_tools,
    )
    try:
        data = client.chat_json_sync([{"role": "user", "content": prompt}], temperature=0.0)
        intent = str(data.get("intent", "knowledge_qa"))
        confidence = float(data.get("confidence", 0.8))
        requires_retrieval = bool(data.get("requires_retrieval", True))
        requires_tool = bool(data.get("requires_tool", False))
        tool_name = data.get("tool_name") or None
        tool_args = data.get("tool_args") or {}
        logger.info("[intent_recognition] intent=%s conf=%.2f", intent, confidence)
        # ReAct 分流: complex_task 或低置信度 → 走 ReAct 循环子图, 其余走快速路径
        use_react = intent == "complex_task" or confidence < 0.6
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "requires_retrieval": requires_retrieval,
            "requires_tool": requires_tool,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "use_react": use_react,
            "steps": _append_step(state, "intent_recognition -> " + intent + (" [react]" if use_react else "")),
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


def agent_reasoning(state):
    """ReAct 推理节点: LLM 决定下一步是调工具还是给最终答案。

    每轮读取 query + history + observations（过往行动-观察序列）+ 可用工具描述,
    输出 JSON 决定 action:
      - action.type="tool"         → 写 pending_tool, 路由到 tool_execution
      - action.type="final_answer" → 写 draft_answer, 路由到 answer_validation
    非法 JSON / 未知工具 → 记为失败 observation 让 LLM 自我修正, 连续 3 次 → fallback。
    达 AGENT_MAX_ITERATIONS → 强制基于现有观察生成答案。
    """
    query = state["query"]
    observations = state.get("observations") or []
    iterations = state.get("react_iterations", 0)
    max_iter = cfg.AGENT_MAX_ITERATIONS or 5
    logger.info("[agent_reasoning] iter=%d/%d obs=%d", iterations, max_iter, len(observations))

    # 步数耗尽 → 强制基于现有观察给答案
    if iterations >= max_iter:
        obs_text = "\n".join(str(o.get("result", "")) for o in observations if o.get("tool") != "_error") or "（无有效观察）"
        return {
            "draft_answer": f"基于已有信息：\n{obs_text[:600]}",
            "react_iterations": iterations + 1,
            "steps": _append_step(state, "agent_reasoning -> max iterations, forced answer"),
        }

    client = get_llm_client()
    registry = get_tool_registry()
    obs_text = "\n".join(
        f"{i+1}. 思考: {o.get('thought','')} | 工具: {o.get('tool','')} | 结果: {str(o.get('result',''))[:200]}"
        for i, o in enumerate(observations)
    ) or "（暂无观察）"
    prompt = REACT_REASONING.format(
        tools=registry.to_react_prompt(),
        observations=obs_text,
        query=query,
    )
    try:
        data = client.chat_json_sync([{"role": "user", "content": prompt}])
        action = data.get("action") or {}
        thought = str(data.get("thought", ""))
        if action.get("type") == "final_answer":
            return {
                "draft_answer": str(action.get("answer", "")),
                "react_iterations": iterations + 1,
                "steps": _append_step(state, f"agent_reasoning iter{iterations} -> final_answer"),
            }
        # tool 调用
        tool_name = action.get("tool_name")
        if tool_name not in registry.list_names():
            raise ValueError(f"unknown or unavailable tool: {tool_name}")
        return {
            "pending_tool": {"tool_name": tool_name, "args": action.get("args") or {}, "thought": thought},
            "observations": list(observations),
            "react_iterations": iterations + 1,
            "steps": _append_step(state, f"agent_reasoning iter{iterations} -> tool:{tool_name}"),
        }
    except Exception as exc:
        logger.warning("[agent_reasoning] failed: %s", exc)
        new_obs = list(observations)
        new_obs.append({"thought": "", "tool": "_error", "args": {},
                        "result": f"推理失败: {exc}。请输出合法 JSON。"})
        errors = sum(1 for o in new_obs if o.get("tool") == "_error")
        if errors >= 3:
            return {
                "is_fallback": True,
                "error_message": "ReAct 推理连续失败",
                "react_iterations": iterations + 1,
                "steps": _append_step(state, "agent_reasoning -> 3 failures, fallback"),
            }
        return {
            "observations": new_obs,
            "react_iterations": iterations + 1,
            "pending_tool": {"tool_name": "_retry", "args": {}},
            "steps": _append_step(state, f"agent_reasoning iter{iterations} -> retry after error"),
        }


async def lookup_file_ids_async(
    pairs: List[tuple],
) -> Dict[tuple, str]:
    """async 版 file_id 反查 — 在 FastAPI 协程里直接 await(不走 executor)。

    pairs 为 [(knowledge_base_id, source), ...]。返回 {(kb_id, source): file_id}。
    供 SSE 流式端点在主协程调用, 避免 executor 线程里 asyncio.run 与
    主线程 async engine 的事件循环冲突。
    """
    pairs = [(kb, src) for kb, src in pairs if kb and src]
    if not pairs:
        return {}
    try:
        import uuid as _uuid
        from sqlalchemy import select
        from backend.storage.postgres.manager import get_session
        from backend.storage.postgres.models_knowledge import KnowledgeFile

        kb_uuids = {_uuid.UUID(kb) for kb, _ in pairs}
        out: Dict[tuple, str] = {}
        async with get_session() as session:
            rows = (
                await session.execute(
                    select(
                        KnowledgeFile.id,
                        KnowledgeFile.knowledge_base_id,
                        KnowledgeFile.filename,
                    ).where(KnowledgeFile.knowledge_base_id.in_(kb_uuids))
                )
            ).all()
            for fid, kb_id, filename in rows:
                key = (str(kb_id), filename)
                if key in pairs:
                    out[key] = str(fid)
        return out
    except Exception as exc:
        logger.warning("[lookup_file_ids_async] failed: %s", exc)
        return {}


# ── executor 线程安全的一次性 DB 访问 ────────────────────────────────────────
# 背景：knowledge_retrieval / answer_generation 等节点被 FastAPI 端点经
# run_in_executor 丢到 worker 线程跑同步代码。线程内若 asyncio.run(...) 复用
# 全局 async engine 的连接池，连接会带着另一个事件循环的 Future 归还池中，
# 污染后续请求（症状：第一个事务静默失效，随后 FK violation / PendingRollback）。
# 因此线程内的 DB 查询一律用随用随建的独立 engine，用完即 dispose，
# 与主 loop 的连接池完全隔离。
async def _run_with_isolated_engine(coro_fn):
    """在独立 engine 上执行 coro_fn(session)，返回其结果。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from backend.storage.postgres.manager import DATABASE_URL

    engine = create_async_engine(DATABASE_URL, pool_size=1, max_overflow=0)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            return await coro_fn(session)
    finally:
        await engine.dispose()


def _run_in_thread_isolated(coro_fn):
    """executor 线程里安全执行 async DB 查询：新事件循环 + 独立 engine。"""
    import asyncio
    return asyncio.run(_run_with_isolated_engine(coro_fn))


def _lookup_file_ids(docs: List[Dict[str, Any]]) -> Dict[tuple, str]:
    """按 (knowledge_base_id, source) 批量反查 knowledge_files.id。

    检索结果只带 kb_id + 文件名, 前端要跳转到文档详情需要 file_id。
    一次性查出所有候选 (kb_id, source) 组合, 返回 {(kb_id, source): file_id}。
    失败时返回空 dict — 引用块退化为纯文本, 不影响主链路。
    """
    pairs = {
        ((d.get("metadata") or {}).get("knowledge_base_id") or "",
         (d.get("metadata") or {}).get("source") or "")
        for d in docs
    }
    pairs = {(kb, src) for kb, src in pairs if kb and src}
    if not pairs:
        return {}

    async def _query(session) -> Dict[tuple, str]:
        import uuid as _uuid
        from sqlalchemy import select
        from backend.storage.postgres.models_knowledge import KnowledgeFile

        kb_uuids = {_uuid.UUID(kb) for kb, _ in pairs}
        out: Dict[tuple, str] = {}
        rows = (
            await session.execute(
                select(
                    KnowledgeFile.id,
                    KnowledgeFile.knowledge_base_id,
                    KnowledgeFile.filename,
                ).where(KnowledgeFile.knowledge_base_id.in_(kb_uuids))
            )
        ).all()
        for fid, kb_id, filename in rows:
            key = (str(kb_id), filename)
            if key in pairs:
                out[key] = str(fid)
        return out

    try:
        return _run_in_thread_isolated(_query)
    except Exception as exc:
        logger.warning("[_lookup_file_ids] failed (refs without file_id): %s", exc)
        return {}


def _lookup_file_ids_by_filename(filenames: List[str]) -> Dict[str, str]:
    """按文件名反查 file_id — 增强检索路径专用。

    enhanced_retriever 返回的 sources 只有 filename 没有 kb_id，
    遍历所有 knowledge_files 按 filename 匹配，返回 {filename: file_id}。
    """
    if not filenames:
        return {}

    async def _query(session) -> Dict[str, str]:
        from sqlalchemy import select
        from backend.storage.postgres.models_knowledge import KnowledgeFile
        rows = (await session.execute(
            select(KnowledgeFile.id, KnowledgeFile.filename)
            .where(KnowledgeFile.filename.in_(filenames))
        )).all()
        return {filename: str(fid) for fid, filename in rows}

    try:
        return _run_in_thread_isolated(_query)
    except Exception as exc:
        logger.warning("[_lookup_file_ids_by_filename] failed: %s", exc)
        return {}


def knowledge_retrieval(state):
    """Node 3: Run RAG retrieval and populate retrieved_docs.

    当 ENHANCED_RETRIEVAL_ENABLED=True 时使用增强检索引擎（查询分解×四路并行×图谱融合重排×知识块聚类）。
    否则走原有路径（向量检索 + 可选图谱旁路）。
    """
    query = state["query"]
    logger.info("[knowledge_retrieval] query=%r", query[:80])

    # ── 增强检索路径 ─────────────────────────────────────────────────────
    if cfg.ENHANCED_RETRIEVAL_ENABLED:
        return _enhanced_knowledge_retrieval(state)

    # ── 原有检索路径 ─────────────────────────────────────────────────────
    return _legacy_knowledge_retrieval(state)


def _enhanced_knowledge_retrieval(state):
    """增强检索：查询分解 × 四路并行检索 × 图谱融合重排 × 知识块聚类 × 迭代补充。"""
    query = state["query"]
    logger.info("[knowledge_retrieval:enhanced] query=%r", query[:80])

    try:
        from app.rag.enhanced_retriever import (
            get_enhanced_retriever,
            format_blocks_for_prompt,
            format_flat_for_prompt,
        )

        retriever = get_enhanced_retriever()
        result = retriever.retrieve(query)

        # 知识块格式化为上下文
        if result.knowledge_blocks:
            context = format_blocks_for_prompt(result.knowledge_blocks)
        elif result.raw_docs:
            context = format_flat_for_prompt(result.raw_docs)
        else:
            context = ""

        # 构建 retrieved_docs（兼容原有格式）
        docs = [{
            "content": d.content,
            "metadata": {**d.metadata, "score": d.score, "path": d.retrieval_path},
        } for d in result.raw_docs]

        # 知识块序列化
        blocks = [{
            "block_id": b.block_id,
            "entities": b.entities,
            "summary": b.summary,
            "score": b.block_score,
        } for b in result.knowledge_blocks]

        # 来源提取
        kb_sources: List[Dict[str, str]] = []
        seen: set = set()
        for s in result.sources:
            title = s.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            kb_sources.append({
                "title": title,
                "url": s.get("url", ""),
                "type": s.get("type", "kb"),
                "score": round(s.get("score", 0.0), 4),
                "knowledge_base_id": s.get("knowledge_base_id", ""),
            })

        # 反查 file_id，使前端引用可点击
        if kb_sources:
            try:
                file_id_map = _lookup_file_ids_by_filename([s["title"] for s in kb_sources])
                for s in kb_sources:
                    s["file_id"] = file_id_map.get(s["title"], "")
            except Exception as exc:
                logger.debug("[knowledge_retrieval:enhanced] file_id lookup failed: %s", exc)

        logger.info(
            "[knowledge_retrieval:enhanced] %d docs, %d blocks, %d sources",
            len(docs), len(blocks), len(kb_sources),
        )

        return {
            "retrieved_docs": docs,
            "retrieval_triggered": True,
            "knowledge_blocks": blocks,
            "query_decomposition": result.query_decomposition.to_dict(),
            "gap_rounds": result.gap_rounds,
            "gap_details": result.gap_details,
            "kb_sources": kb_sources,
            "error_message": None,
            "steps": _append_step(
                state,
                "knowledge_retrieval:enhanced -> "
                + str(len(docs)) + " docs, "
                + str(len(blocks)) + " blocks"
            ),
        }

    except Exception as exc:
        logger.warning("[knowledge_retrieval:enhanced] failed: %s, falling back to legacy", exc)
        return _legacy_knowledge_retrieval(state)


def _legacy_knowledge_retrieval(state):
    """原有检索路径：向量检索 + 可选图谱旁路。"""
    query = state["query"]
    logger.info("[knowledge_retrieval:legacy] query=%r", query[:80])
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        docs = retriever.retrieve(query, top_k=cfg.RETRIEVER_TOP_K)
        logger.info("[knowledge_retrieval] retrieved %d docs", len(docs))

        if cfg.GRAPH_ENABLED:
            try:
                from backend.services.graph_service import query_related, format_subgraph_for_prompt

                async def _graph_query():
                    async with get_session() as session:
                        # 跨知识库全局查询（暂未按会话锁定 kb，待多 kb 路由完善）
                        import uuid as _uuid
                        from sqlalchemy import select
                        from backend.storage.postgres.models_knowledge import KnowledgeBase
                        kb_ids = (await session.execute(select(KnowledgeBase.id))).scalars().all()
                        subgraphs = []
                        for kb_id in kb_ids:
                            subgraphs.extend(await query_related(session, kb_id, query))
                        return subgraphs

                from app.rag.enhanced_retriever import _run_async_in_thread
                subgraphs = _run_async_in_thread(_graph_query())
                if subgraphs:
                    docs.append({
                        "content": format_subgraph_for_prompt(subgraphs),
                        "metadata": {"source": "knowledge_graph", "graph": True, "score": 1.0},
                    })
                    logger.info("[knowledge_retrieval] graph: %d entities injected", len(subgraphs))
            except Exception as exc:
                logger.warning("[knowledge_retrieval] graph query failed (ignored): %s", exc)

        if not docs:
            raise EmptyRetrievalError("No documents matched.")
        # 知识库引用透出: 从 retrieved_docs 提取去重后的来源,
        # 与 web_search 的 sources 并列进入最终响应, 供前端渲染引用块。
        # 方案 B: 检索结果已带 knowledge_base_id, 据此反查 knowledge_files
        # 拿到 file_id, 使前端引用可点击跳转到具体文档详情。
        file_id_map = _lookup_file_ids(docs)
        kb_sources: List[Dict[str, Any]] = []
        seen: set = set()
        for d in docs:
            meta = d.get("metadata") or {}
            src = (meta.get("source") or "").strip()
            if not src or src in seen:
                continue
            seen.add(src)
            kb_id = (meta.get("knowledge_base_id") or "").strip()
            entry: Dict[str, Any] = {
                "title": src,
                "url": "",
                "type": "knowledge_graph" if meta.get("graph") else "kb",
                "score": round(float(meta.get("score", 0.0)), 4),
                "knowledge_base_id": kb_id,
                "file_id": file_id_map.get((kb_id, src), ""),
            }
            kb_sources.append(entry)
        return {
            "retrieved_docs": docs,
            "retrieval_triggered": True,
            "kb_sources": kb_sources,
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
    # web_search first: queries like "今天的新闻" contain both news & date
    # keywords — searching is the more specific intent.
    if any(w in q for w in [
        "search", "news", "latest", "today's", "current events", "look up",
        "搜索", "检索", "新闻", "最新", "最近", "今天的新闻",
    ]):
        return {
            "tool_name": "web_search",
            "tool_args": {"query": state["query"]},
            "steps": _append_step(state, "tool_selection -> inferred web_search"),
        }
    if any(w in q for w in ["calculat", "compute", "sqrt", "pow", "计算", "等于多少", "多少"]):
        return {
            "tool_name": "calculator",
            "tool_args": {"expression": state["query"]},
            "steps": _append_step(state, "tool_selection -> inferred calculator"),
        }
    if any(w in q for w in [
        "time", "date", "today", "now", "weekday",
        "几点", "时间", "日期", "今天", "现在", "星期", "周几",
    ]):
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
    """Node 5: Execute selected tool and capture result.

    ReAct 模式（state["use_react"]）: 从 pending_tool 取工具, 执行结果追加到
    observations（含 thought/tool/args/result）, 循环回 agent_reasoning。
    快速路径: 从 tool_name/tool_args 取（现状）, 只写 tool_result。
    pending_tool.tool_name == "_retry" 时跳过执行直接循环（推理失败重试）。
    """
    use_react = state.get("use_react", False)
    registry = get_tool_registry()

    # ── ReAct 分支 ────────────────────────────────────────────────────────
    if use_react:
        pending = state.get("pending_tool") or {}
        tool_name = pending.get("tool_name")
        tool_args = pending.get("args") or {}
        thought = pending.get("thought", "")
        observations = list(state.get("observations") or [])
        logger.info("[tool_execution/react] tool=%s", tool_name)

        # 推理失败重试标记: 不执行工具, 直接循环回 reasoning
        if tool_name == "_retry":
            return {
                "observations": observations,
                "pending_tool": None,
                "steps": _append_step(state, "tool_execution/react -> retry loop"),
            }

        try:
            result = registry.invoke(tool_name, **tool_args)
            observations.append({"thought": thought, "tool": tool_name, "args": tool_args, "result": result})
            update: Dict[str, Any] = {
                "observations": observations,
                "pending_tool": None,
                "tool_result": result,
                "tool_triggered": True,
                "tool_error": None,
                "steps": _append_step(state, f"tool_execution/react -> {tool_name} OK"),
            }
            if tool_name == "web_search":
                from app.tools.web_search_tool import extract_sources
                update["sources"] = extract_sources(result)
            return update
        except Exception as exc:
            logger.error("[tool_execution/react] failed: %s", exc)
            observations.append({"thought": thought, "tool": tool_name, "args": tool_args,
                                 "result": f"工具执行失败: {exc}"})
            return {
                "observations": observations,
                "pending_tool": None,
                "tool_triggered": True,
                "tool_error": str(exc),
                "steps": _append_step(state, f"tool_execution/react -> {tool_name} FAILED"),
            }

    # ── 快速路径分支（现状）─────────────────────────────────────────────
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
        update: Dict[str, Any] = {
            "tool_result": result,
            "tool_triggered": True,
            "tool_error": None,
            "steps": _append_step(state, "tool_execution -> " + tool_name + " OK"),
        }
        # web_search embeds a machine-readable sources block; extract it so the
        # final answer can list references at the bottom.
        if tool_name == "web_search":
            from app.tools.web_search_tool import extract_sources
            update["sources"] = extract_sources(result)
        return update
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

        # 增强检索：使用知识块格式（截断保护：防止超大 context 导致 LLM 返回空）
        knowledge_blocks = state.get("knowledge_blocks")
        if knowledge_blocks and docs:
            from app.rag.enhanced_retriever import format_blocks_for_prompt
            context = format_blocks_for_prompt(
                _rebuild_blocks(knowledge_blocks, docs)
            )
            # 截断保护：context 超过 8000 字符时截断，避免 LLM 因 prompt 过长返回空
            MAX_CONTEXT_CHARS = 8000
            if len(context) > MAX_CONTEXT_CHARS:
                logger.warning(
                    "[answer_generation] context too long (%d chars), truncating to %d",
                    len(context), MAX_CONTEXT_CHARS,
                )
                context = context[:MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"
            messages.append({
                "role": "user",
                "content": ANSWER_WITH_ENHANCED_CONTEXT.format(
                    query=query, context=context, tool_result=effective_tool
                ),
            })
        elif docs:
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

        # LLM occasionally returns an empty body with HTTP 200 — retry once
        # in-place before giving up, so the user doesn't hit a fallback.
        if not draft.strip():
            logger.warning("[answer_generation] empty draft, retrying once")
            # 如果用的是增强检索格式且答案为空，回退到传统平铺格式重试
            if knowledge_blocks and docs:
                flat_context = "\n\n".join(
                    "[" + str(i + 1) + "] " + d["content"] for i, d in enumerate(docs)
                )
                fallback_msg = {
                    "role": "user",
                    "content": ANSWER_WITH_CONTEXT.format(
                        query=query, context=flat_context, tool_result=effective_tool
                    ),
                }
                retry_messages = messages[:-1] + [fallback_msg]
                logger.info("[answer_generation] retrying with flat context (%d chars)", len(flat_context))
                draft = client.chat_sync(retry_messages)
            else:
                draft = client.chat_sync(messages)
            logger.info("[answer_generation] retry draft length=%d", len(draft))

        if not draft.strip():
            return {
                "draft_answer": "",
                "error_message": "LLM returned an empty response",
                "steps": _append_step(state, "answer_generation -> EMPTY after retry"),
            }

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
    error = state.get("error_message") or "the model could not produce an answer"
    logger.warning("[fallback_handler] error=%r", error)
    fallback_text = FALLBACK_ANSWER.format(query=query, error=error)
    return {
        "final_answer": fallback_text,
        "is_fallback": True,
        "validation_passed": False,
        "steps": _append_step(state, "fallback_handler -> fallback answer generated"),
    }


def _rebuild_blocks(block_data: List[Dict[str, Any]], docs: List[Dict[str, Any]]):
    """从序列化数据重建知识块对象（用于 format_blocks_for_prompt）。"""
    from app.rag.enhanced_retriever import KnowledgeBlock, CandidateDoc

    blocks = []
    for b in block_data:
        block = KnowledgeBlock(
            block_id=b.get("block_id", ""),
            entities=b.get("entities", []),
            summary=b.get("summary", ""),
            block_score=b.get("score", 0.0),
        )
        blocks.append(block)

    # 将 docs 分配到各 block（简化：根据 metadata.path 分配）
    if blocks:
        for d in docs:
            path = d.get("metadata", {}).get("path", "semantic")
            doc = CandidateDoc(
                content=d["content"],
                metadata=d.get("metadata", {}),
                score=d.get("metadata", {}).get("score", 0.5),
                retrieval_path=path,
            )
            # 分配到第一个 block
            first_block = blocks[0]
            first_block.docs.append(doc)

    return blocks
