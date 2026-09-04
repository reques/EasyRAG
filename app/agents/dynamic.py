"""轻量动态 Agent：根据情景由 LLM 自行决定「直接回答 / 调用工具 / 检索知识库」。

与 DeepAgents（``app/agents/deep``）的区别：
- 不引入 task/spawn_tasks/revise_plan 等委派机制，只保留项目注册表里的普通工具；
- 用 create_agent 的函数调用能力替代固定管线（query_rewrite -> intent
  -> retrieval/tool -> validate），模型每轮自行决策下一步动作；
- 简单问题（寒暄、常识、写作等）只需一次 LLM 调用即可完成，不需要走检索/验证等
  复杂流程；需要实时信息时模型会自动调 web_search，知识库内容会自动调 kb_search。

调用方：
- ``AgentService.run``（/chat/send）：``AGENT_MODE=auto`` 的非复杂问题与
  ``AGENT_MODE=dynamic`` 均路由到这里；
- ``chat_router`` 的流式分支（/chat/stream）：与 DeepAgents 分支同构，逐轮
  透出工具调用 / 检索状态。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

from app.agents.context import ChatContext
from app.core.config import get_settings
from app.core.logger import get_logger
from app.tools.registry import get_tool_registry

logger = get_logger(__name__)
cfg = get_settings()


DYNAMIC_SYSTEM_PROMPT = """你是一名智能助手，运行在 EasyRAG 企业知识库问答平台上。

回答原则：
1. 简单、常识、寒暄、写作类问题：直接回答，不要调用任何工具，也不要检索。
2. 问题涉及已上传知识库/内部资料（规章制度、产品文档、合同、法律条文等）：
   先用 kb_search 检索，再基于检索内容回答并标注来源。
3. 问题需要实时或外部信息（新闻、天气、行情、价格、最新事件、知识库没有的内容）：
   用 web_search 搜索。
4. 明确的计算、日期时间问题：用 calculator / datetime_tool。
5. 可以连续调用多个工具：例如先检索知识库，再联网补充；也可以根据上一次工具结果
   决定是否继续调用。
6. 工具调用后若信息仍不充分，可继续调用其他工具；信息已充分或工具无果时，基于已有
   信息给出诚实回答，不要编造来源。
7. 引用来源：回答知识库/联网内容时标注来源。
8. 使用中文回答。

可用工具：
{tools_prompt}

工作方式：每一步你选择「调用工具」或「直接给出最终回答」。简单问题一次调用即可回答，
绝不为简单问题启动检索或工具链等复杂流程。
"""


# 生产路径缓存（model=None 时缓存；测试注入 mock 绕过缓存）
_dynamic_agent_cache: Optional[Any] = None


def _tools_prompt_text() -> str:
    """生成工具清单文本（与意图识别 prompt 一致的动态注册表视图）。"""
    try:
        return get_tool_registry().to_react_prompt()
    except Exception as exc:
        logger.warning("[dynamic] tools prompt failed: %s", exc)
        return "（暂无可用工具）"


def build_dynamic_agent(
    model=None,
    recursion_limit: Optional[int] = None,
):
    """构建轻量动态 Agent（create_agent + 注册表工具，无委派工具）。

    model: 测试可注入 mock；None 时使用项目 LangChain 模型并缓存。
    recursion_limit: 仅作为构建期默认值保留（invoke 时仍可覆盖）。
    """
    from langchain.agents import create_agent

    from app.agents.deep.llm import get_langchain_model
    from app.agents.deep.tools import registry_to_langchain_tools
    from app.skills.middleware import build_skills_middleware

    global _dynamic_agent_cache
    cacheable = model is None
    if cacheable and _dynamic_agent_cache is not None:
        return _dynamic_agent_cache
    if model is None:
        model = get_langchain_model()
    tools = registry_to_langchain_tools()
    prompt = DYNAMIC_SYSTEM_PROMPT.format(tools_prompt=_tools_prompt_text())
    logger.info(
        "[dynamic] agent built: %d tools (registry only, no delegation tools)",
        len(tools),
    )
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        # 2026-09-04 Skill 重构：Skill 注入 + 渐进式门控（含 read_skill）
        middleware=[build_skills_middleware()],
        name="easyrag_dynamic_agent",
    )
    if cacheable:
        _dynamic_agent_cache = agent
    return agent


def get_dynamic_agent():
    """进程级单例动态 Agent。"""
    return build_dynamic_agent()


def run_dynamic_agent(
    query: str,
    context: Optional[ChatContext] = None,
    *,
    session_id: str = "default",
    history: Optional[List[Dict[str, str]]] = None,
    user_id=None,
    knowledge_base_ids: Optional[Sequence[str]] = None,
    knowledge_catalog: Optional[Sequence[Dict[str, Any]]] = None,
    image_data: Optional[str] = None,
    on_step=None,
    on_artifact=None,
    recursion_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """运行轻量动态 Agent 并返回与单 Agent 兼容的结果字典。

    阶段 2：优先传声明式 ``context``（ChatContext）；散装参数保留为过渡期
    兼容（内部合入 context，调用方迁移完成后删除）。
    on_step: 可选回调 fn(step, detail)，供流式端点实时透出阶段状态。
    on_artifact: 可选回调 fn(dict)，透出工具调用/工具返回等中间产出。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langgraph.errors import GraphRecursionError

    from app.agents.events import emit
    from app.services.knowledge_catalog import format_knowledge_catalog
    from app.services.knowledge_context import use_authorised_kb_ids

    context = context or ChatContext(thread_id=session_id)
    if history is not None:
        context.history = tuple(history)
    if user_id is not None:
        context.user_id = str(user_id)
    if knowledge_base_ids is not None:
        context.knowledge_base_ids = tuple(knowledge_base_ids)
    if knowledge_catalog is not None:
        context.knowledge_catalog = tuple(knowledge_catalog)
    if image_data is not None:
        context.image_data = image_data
    if on_step is not None:
        context.on_step = on_step
    if on_artifact is not None:
        context.on_artifact = on_artifact
    session_id = context.thread_id
    history = list(context.history)
    knowledge_base_ids = list(context.knowledge_base_ids)
    knowledge_catalog = list(context.knowledge_catalog)
    image_data = context.image_data
    on_step = context.on_step
    on_artifact = context.on_artifact

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

    # ── 组装消息（对齐 prepare_context / _run_deep 的注入链）──────────────
    # 2026-09-04 Skill 重构：Skill 区块不再在这里手工拼 —— SkillsMiddleware
    # 每轮按 activated_skills 动态渲染（未激活给摘要行、已激活给正文），
    # 手工注入会与之重复且拿不到激活状态。
    messages: List[Any] = []
    if knowledge_catalog:
        messages.append(
            SystemMessage(content=format_knowledge_catalog(list(knowledge_catalog)))
        )
    for t in history:
        content = str(t.get("content") or "")
        if t.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    if image_data:
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ]
            )
        )
    else:
        messages.append(HumanMessage(content=query))

    _step("understand", "动态 Agent 开始处理…")

    sources: List[Dict[str, str]] = []
    tool_names: List[str] = []
    tool_results: List[str] = []
    retrieval_triggered = False
    final_answer = ""
    degraded = False
    final_state: Optional[Dict[str, Any]] = None
    try:
        with use_authorised_kb_ids(knowledge_base_ids):
            agent = get_dynamic_agent()
            for chunk in agent.stream(
                {"messages": messages},
                config={"recursion_limit": recursion_limit or cfg.AGENT_MAX_ITERATIONS},
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
                    name = tc[0].get("name", "")
                    tool_names.append(name)
                    args = tc[0].get("args") or {}
                    try:
                        args_text = json.dumps(args, ensure_ascii=False)[:800]
                    except Exception:
                        args_text = str(args)[:800]
                    args_short = " ".join(args_text.split())[:160]
                    if len(args_text) > 160:
                        args_short += "…"
                    _step("tool", f"调用 {name} {args_short}".rstrip())
                    _artifact("tool", "tool", f"调用 {name}", args_text)
                    if name == "kb_search":
                        retrieval_triggered = True
                elif mtype == "tool":
                    content = str(getattr(last, "content", "") or "")
                    tool_results.append(content)
                    flat = " ".join(content.split())
                    _step("tool_done", f"工具返回: {flat[:120]}")
                    _artifact(
                        "tool_result", "tool", "工具返回",
                        flat[:300] + ("…" if len(flat) > 300 else ""),
                    )
                    # 联网搜索的来源标注 → 前端引用区
                    try:
                        from app.tools.web_search_tool import extract_sources

                        for s in extract_sources(content) or []:
                            if s not in sources:
                                sources.append(s)
                    except Exception:
                        pass
                elif mtype == "ai" and getattr(last, "content", ""):
                    _step("generate", "动态 Agent 生成回答中…")
    except GraphRecursionError:
        degraded = True
        steps.append("dynamic agent hit recursion limit, forced answer from partial state")
        _step("fallback", "已达推理步数上限，基于已有信息收尾")
    except Exception as exc:
        logger.error("[dynamic] agent error: %s", exc)
        steps.append(f"dynamic agent error: {exc}")
        _step("fallback", f"执行失败: {str(exc)[:80]}")
        return {
            "query": query,
            "session_id": session_id,
            "final_answer": f"处理请求时发生错误: {exc}",
            "intent": "dynamic",
            "intent_confidence": 0.0,
            "retrieval_triggered": retrieval_triggered,
            "retrieved_docs_count": 0,
            "tool_triggered": bool(tool_names),
            "tool_name": tool_names[-1] if tool_names else None,
            "tool_result": tool_results[-1] if tool_results else None,
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

    # ── 从最终状态提取回答 ────────────────────────────────────────────────
    msgs = (final_state or {}).get("messages") or []
    for m in reversed(msgs):
        content = getattr(m, "content", "") or ""
        if getattr(m, "type", "") == "ai" and content:
            final_answer = content if isinstance(content, str) else str(content)
            break
    if not final_answer and msgs:
        final_answer = str(getattr(msgs[-1], "content", "") or "")

    if degraded and not final_answer.strip():
        observations = "\n".join(t for t in tool_results if t.strip())
        if observations:
            final_answer = f"基于已有信息：\n{observations[:600]}"
        else:
            final_answer = (
                "（已达推理步数上限，未能收集到足够信息；"
                "请尝试简化问题后重新提问。）"
            )

    _step("generate_done", f"回答完成（{len(final_answer)} 字符）")
    return {
        "query": query,
        "session_id": session_id,
        "final_answer": final_answer,
        "intent": "dynamic",
        "intent_confidence": 0.0,
        "retrieval_triggered": retrieval_triggered,
        "retrieved_docs_count": 0,
        "tool_triggered": bool(tool_names),
        "tool_name": tool_names[-1] if tool_names else None,
        "tool_result": tool_results[-1] if tool_results else None,
        "tool_error": None,
        "sub_tasks": [],
        "steps": steps,
        "artifacts": artifacts,
        "sources": sources,
        "is_fallback": degraded,
        "degraded": degraded,
        "error_message": None,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
