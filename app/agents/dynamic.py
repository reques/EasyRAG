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
from app.agents.action_progress import SUMMARY_ARG, build_action_progress_middleware
from app.agents.response_stream import ProgressEventCollector, ResponseStream, message_text
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

工作方式（根据问题和工具观察结果逐步决策）：
- 简单问题直接输出 <answer>最终回答</answer>，不必写进度，也不必调用工具。
- 需要工具时，先输出 <progress>一句面向用户的行动说明</progress>，随后正常调用工具。
  说明要紧扣当前问题，指出要查什么、核对什么；只写简短行动摘要，不输出内部推理过程。
- 每次工具调用还必须填写 _action_summary 参数，用一句话说明该次行动；
  若模型只支持工具调用参数而无法同时输出正文，就通过此参数提供行动说明。
- 收到工具结果后，判断是否足够回答。若还需补充，用新的 <progress> 简述已有发现或
  缺口及接下来的具体动作，再调用工具。不得预先宣称尚未执行的检索或核验已经完成。
- 例如用户问合同违约责任，可以说“先查合同中的违约条款，确认适用条件”；检索无结果后
  可以说“当前资料没有覆盖逾期付款，我会换用该条款的关键词补查”。例子不应机械照搬。
- 信息足够时直接输出 <answer>最终回答</answer>，保留正常 Markdown 格式。
  <progress> 和 <answer> 是输出通道标记，不要放在代码块里；不要在标记外添加开场白。
绝不为简单问题启动检索或工具链，不为了展示进度增加额外调用。
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
        middleware=[build_skills_middleware(), build_action_progress_middleware()],
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
    collected = ProgressEventCollector()
    artifacts = collected.artifacts

    def _step(step: str, detail: str = ""):
        steps.append(f"{step}: {detail}")
        emit("step", step, step, detail)
        if on_step:
            try:
                on_step(step, detail)
            except Exception:
                pass

    def _artifact(kind: str, stage: str, title: str, content: str, **extra):
        # kind="answer" 且 content 为空 = 正文流结束标记，仍需透出（不落 artifacts）
        if not content and kind != "answer" and "streaming" not in extra:
            return
        ev = {"kind": kind, "stage": stage, "title": title[:80], "content": content, **extra}
        collected.artifact(ev)
        emit("artifact", stage, title, content, artifact_kind=kind, **extra)
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

    sources: List[Dict[str, str]] = []
    tool_names: List[str] = []
    tool_results: List[str] = []
    retrieval_triggered = False
    final_answer = ""
    degraded = False

    # 每轮分别解析行动摘要和回答；未声明通道的文本待 tool_calls 确定后再分类。
    streamed_any = False
    response_stream = None
    round_number = 0
    processed_count = len(messages)
    calls_by_id: dict[str, str] = {}

    def _emit_stream_end() -> None:
        """正文流结束标记（前端把"回答"流标记为完成；未流过则 no-op）。"""
        if not streamed_any:
            return
        _artifact("answer", "generate", "回答", "", streaming=False, stream_id="final-answer")

    def _emit_token(text: str) -> None:
        nonlocal streamed_any
        if not text:
            return
        if not streamed_any:
            _step("generate", "正在组织回答")
        streamed_any = True
        _artifact("answer", "generate", "回答", text, streaming=True, stream_id="final-answer")

    def _new_response_stream() -> ResponseStream:
        nonlocal round_number
        round_number += 1
        stream_id = f"dynamic-progress-{round_number}"

        def _progress(text: str, done: bool):
            _artifact("thought", "reason", "行动说明", text, id=stream_id, streaming=not done)

        return ResponseStream(_progress, _emit_token)

    try:
        with use_authorised_kb_ids(knowledge_base_ids):
            agent = get_dynamic_agent()
            # values 提供完整工具调用和观察；messages 按声明的输出通道流式分发。
            for stream_item in agent.stream(
                {"messages": messages},
                config={"recursion_limit": recursion_limit or cfg.AGENT_MAX_ITERATIONS},
                stream_mode=["values", "messages"],
            ):
                # 真实 langgraph：stream_mode 为列表时产出 (mode, payload)；
                # 测试 fake（旧契约）：直接产出 values chunk。统一适配。
                if isinstance(stream_item, tuple) and len(stream_item) == 2:
                    mode, payload = stream_item
                else:
                    mode, payload = "values", stream_item
                if mode == "messages":
                    token_msg, _meta = payload
                    if getattr(token_msg, "type", "") not in {"ai", "AIMessageChunk"}:
                        continue
                    if _meta.get("langgraph_node") not in {None, "model"}:
                        continue
                    # 只读公开 content 文本；reasoning_content/工具参数不会进入解析器。
                    text = message_text(getattr(token_msg, "content", ""))
                    if text:
                        if response_stream is None:
                            response_stream = _new_response_stream()
                        response_stream.feed(text)
                    continue
                chunk = payload
                msgs = chunk.get("messages") or []
                if not msgs:
                    continue
                new_messages = msgs[processed_count:]
                processed_count = len(msgs)
                for last in new_messages:
                    mtype = getattr(last, "type", "")
                    tc = getattr(last, "tool_calls", None) or []
                    if mtype == "ai":
                        if response_stream is None:
                            response_stream = _new_response_stream()
                        content = message_text(getattr(last, "content", ""))
                        # values-only 模型和未通过 messages 发出的尾部文本都需要补齐。
                        if content.startswith(response_stream.raw):
                            response_stream.feed(content[len(response_stream.raw):])
                        answer = response_stream.finish(tool_calls=bool(tc))
                        had_progress = bool(response_stream.progress.strip())
                        response_stream = None
                        if not tc:
                            final_answer = answer
                        for call in tc:
                            name = call.get("name", "")
                            call_id = call.get("id", "")
                            calls_by_id[call_id] = name
                            tool_names.append(name)
                            args = dict(call.get("args") or {})
                            summary = args.pop(SUMMARY_ARG, "")
                            if not had_progress and isinstance(summary, str) and summary.strip():
                                _artifact("thought", "reason", "行动说明", summary.strip()[:600])
                            args_text = json.dumps(args, ensure_ascii=False)[:800]
                            _step("tool", f"调用 {name}")
                            _artifact("tool", "tool", f"调用 {name}", args_text, tool_call_id=call_id)
                            if name == "kb_search":
                                retrieval_triggered = True
                    elif mtype == "tool":
                        content = message_text(getattr(last, "content", ""))
                        tool_results.append(content)
                        call_id = getattr(last, "tool_call_id", "")
                        name = calls_by_id.get(call_id) or getattr(last, "name", "") or "工具"
                        flat = " ".join(content.split())
                        _step("tool_done", f"{name} 返回: {flat[:120]}")
                        _artifact(
                            "tool_result", "tool", f"{name} 返回",
                            flat[:300] + ("…" if len(flat) > 300 else ""),
                            tool_call_id=call_id,
                            is_error=getattr(last, "status", "") == "error",
                        )
                        try:
                            from app.tools.web_search_tool import extract_sources

                            for source in extract_sources(content) or []:
                                if source not in sources:
                                    sources.append(source)
                        except Exception:
                            pass
    except GraphRecursionError:
        degraded = True
        steps.append("dynamic agent hit recursion limit, forced answer from partial state")
        _step("fallback", "已达推理步数上限，基于已有信息收尾")
    except Exception as exc:
        logger.error("[dynamic] agent error: %s", exc)
        steps.append(f"dynamic agent error: {exc}")
        _step("fallback", f"执行失败: {str(exc)[:80]}")
        _emit_stream_end()
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
    _emit_stream_end()
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
