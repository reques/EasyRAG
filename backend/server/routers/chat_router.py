"""对话路由 — 带 DB 持久化的对话 API。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from backend.services.chat_service import (
    create_conversation,
    add_message,
    get_conversation_history,
    list_user_conversations,
    get_conversation,
    generate_conversation_title,
)
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.models_user import User

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ── Request / Response ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    conversation_id: Optional[str] = None  # None = 创建新会话


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    intent: str = ""
    steps: list[str] = []
    sources: list[dict] = []
    elapsed_seconds: float = 0.0


class ConversationSummary(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/send", response_model=ChatResponse)
async def send_message(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """发送消息并获取 Agent 回复（持久化对话历史）。"""
    start = time.perf_counter()

    async with get_session() as session:
        # 获取或创建会话
        conv_id = None
        is_new = False
        if req.conversation_id:
            conv = await get_conversation(session, uuid.UUID(req.conversation_id))
            if not conv or conv.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            conv_id = conv.id
        else:
            conv = await create_conversation(session, current_user.id)
            conv_id = conv.id
            is_new = True

        # 保存用户消息
        await add_message(session, conv_id, "user", req.query)
        await session.commit()

        # 加载完整对话历史（含刚保存的用户消息）
        db_history = await get_conversation_history(session, conv_id)

    # =====================================================================
    # 调用 LangGraph Agent，传入 DB 中的对话历史
    # =====================================================================
    try:
        from app.services.agent_service import get_agent_service
        agent = get_agent_service()
        result = agent.run(
            query=req.query,
            session_id=str(conv_id),
            history=db_history,          # ← 关键：传入 DB 历史
        )
        answer = result.get("final_answer", "")
    except Exception as exc:
        logger.error("[chat/send] agent error: %s", exc)
        answer = f"处理请求时发生错误: {exc}"

    # 兜底：Agent 返回空答案时，用 LLM 直接生成（跳过检索）
    if not answer.strip():
        logger.warning("[chat/send] agent returned empty answer, fallback to direct LLM")
        try:
            from app.llm.client import get_llm_client
            llm = get_llm_client()
            fallback_answer = llm.chat_sync([{
                "role": "user",
                "content": (
                    f"请简要回答以下问题（200字以内）：\n\n{req.query}\n\n"
                    "如果问题涉及法律条款，请引用具体法条编号。"
                ),
            }])
            if fallback_answer and fallback_answer.strip():
                answer = fallback_answer
                logger.info("[chat/send] direct LLM fallback succeeded (%d chars)", len(fallback_answer))
            else:
                answer = "抱歉，模型暂时无法生成回答，请稍后重试或简化问题。"
        except Exception as fb_exc:
            logger.error("[chat/send] direct LLM fallback failed: %s", fb_exc)
            answer = "抱歉，处理请求时遇到问题，请稍后重试。"

    # 自动设置会话标题（首次对话时用 LLM 生成语义摘要）
    if is_new and req.query.strip() and answer.strip():
        title = await generate_conversation_title(req.query, answer)
        async with get_session() as session:
            conv = await get_conversation(session, conv_id)
            if conv:
                conv.title = title
                await session.commit()

    elapsed = round(time.perf_counter() - start, 3)

    # 保存助手回复
    async with get_session() as session:
        meta = json.dumps({
            "intent": result.get("intent", ""),
            "steps": result.get("steps", []),
            "sources": result.get("sources", []),
        }, ensure_ascii=False)
        await add_message(session, conv_id, "assistant", answer, metadata_json=meta)
        await session.commit()

    return ChatResponse(
        conversation_id=str(conv_id),
        answer=answer,
        intent=result.get("intent", ""),
        steps=result.get("steps", []),
        sources=result.get("sources", []),
        elapsed_seconds=elapsed,
    )


@router.post("/stream")
async def send_message_stream(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """流式对话 — SSE 逐 token 推送 Agent 回复, 结束时推送引用块。

    事件序列:
      data: {"type": "conversation_id", "conversation_id": "..."}
      data: {"type": "delta", "content": "<增量文本>"}   (多次)
      data: {"type": "done", "sources": [...], "intent": "...", "elapsed_seconds": 1.23}
      data: {"type": "error", "detail": "..."}           (仅出错时)

    设计: 检索(同步)用 run_in_executor 跑, 生成用 LLM chat_stream 流式,
    最终答案 + 引用落库与 /chat/send 保持一致。
    """
    import asyncio
    from fastapi.responses import StreamingResponse

    start = time.perf_counter()

    async with get_session() as session:
        # 获取或创建会话
        if req.conversation_id:
            conv = await get_conversation(session, uuid.UUID(req.conversation_id))
            if not conv or conv.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            conv_id = conv.id
            is_new = False
        else:
            conv = await create_conversation(session, current_user.id)
            conv_id = conv.id
            is_new = True

        await add_message(session, conv_id, "user", req.query)
        await session.commit()
        db_history = await get_conversation_history(session, conv_id)

    async def event_gen():
        from app.services.agent_service import get_agent_service
        from app.llm.client import get_llm_client

        loop = asyncio.get_event_loop()
        agent = get_agent_service()

        def _sse(payload: dict) -> str:
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        yield _sse({"type": "conversation_id", "conversation_id": str(conv_id)})

        # 1. 同步检索准备上下文(阻塞, 放 executor)
        try:
            ctx = await loop.run_in_executor(
                None, agent.prepare_context, req.query, db_history
            )
        except Exception as exc:
            logger.error("[chat/stream] prepare_context error: %s", exc)
            yield _sse({"type": "error", "detail": f"检索失败: {exc}"})
            return

        # 1b. 主协程里 async 反查 file_id 并回填到引用(executor 线程里
        #     asyncio.run 会与主线程 async engine 冲突, 故在此统一补齐)。
        try:
            from app.graph.nodes import lookup_file_ids_async
            pairs = [
                (s.get("knowledge_base_id", ""), s.get("title", ""))
                for s in ctx["sources"]
                if s.get("type") in ("kb", "knowledge_graph")
            ]
            fid_map = await lookup_file_ids_async(pairs)
            for s in ctx["sources"]:
                key = (s.get("knowledge_base_id", ""), s.get("title", ""))
                if key in fid_map:
                    s["file_id"] = fid_map[key]
        except Exception as exc:
            logger.warning("[chat/stream] file_id backfill failed: %s", exc)

        # 2. 流式生成（含空响应兜底）
        answer_parts: list[str] = []
        try:
            llm = get_llm_client()
            async for delta in llm.chat_stream(ctx["messages"]):
                answer_parts.append(delta)
                yield _sse({"type": "delta", "content": delta})
        except Exception as exc:
            logger.error("[chat/stream] generation error: %s", exc)
            yield _sse({"type": "error", "detail": f"生成失败: {exc}"})
            return

        # 兜底：流式返回空时用同步调用重试（API 偶发空响应，尤其法律类内容）
        if not answer_parts:
            logger.warning("[chat/stream] stream returned 0 tokens, falling back to sync")
            try:
                fallback_answer = await loop.run_in_executor(
                    None, llm.chat_sync, ctx["messages"]
                )
            except Exception as fb_exc:
                logger.error("[chat/stream] sync fallback also failed: %s", fb_exc)
                yield _sse({"type": "error", "detail": "模型未返回有效回答，请重试"})
                return

            if fallback_answer and fallback_answer.strip():
                answer_parts = [fallback_answer]
                yield _sse({"type": "delta", "content": fallback_answer})
                logger.info("[chat/stream] sync fallback succeeded (%d chars)", len(fallback_answer))
            else:
                logger.warning("[chat/stream] sync fallback also returned empty")
                yield _sse({"type": "error", "detail": "模型未返回有效回答，请尝试简化问题后重试"})
                return

        answer = "".join(answer_parts).strip()
        elapsed = round(time.perf_counter() - start, 3)

        # 3. 落库助手回复(含引用)
        try:
            async with get_session() as session:
                meta = json.dumps({
                    "intent": ctx["intent"],
                    "sources": ctx["sources"],
                }, ensure_ascii=False)
                await add_message(session, conv_id, "assistant", answer, metadata_json=meta)
                await session.commit()
        except Exception as exc:
            logger.warning("[chat/stream] persist answer failed: %s", exc)

        yield _sse({
            "type": "done",
            "sources": ctx["sources"],
            "intent": ctx["intent"],
            "elapsed_seconds": elapsed,
        })

        # 4. 新会话标题生成 — 在 done 之后的后台协程里做，不阻塞 SSE 流。
        #    LLM 生成语义化标题(非原文截取)，前端下次轮询会话列表时即可见。
        if is_new and answer:
            async def _gen_title():
                try:
                    title = await generate_conversation_title(req.query, answer)
                    async with get_session() as session:
                        c = await get_conversation(session, conv_id)
                        if c:
                            c.title = title
                            await session.commit()
                    logger.info("[chat/stream] title generated: %s", title)
                except Exception as exc:
                    logger.warning("[chat/stream] title gen failed: %s", exc)

            asyncio.get_event_loop().create_task(_gen_title())

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    current_user: User = Depends(get_current_user),
):
    """列出当前用户的所有会话。"""
    async with get_session() as session:
        convs = await list_user_conversations(session, current_user.id)
        return [
            ConversationSummary(
                id=str(c.id),
                title=c.title,
                created_at=c.created_at.isoformat() if c.created_at else "",
                updated_at=c.updated_at.isoformat() if c.updated_at else "",
            )
            for c in convs
        ]


@router.get("/conversations/{conversation_id}/history")
async def get_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """获取指定会话的对话历史。"""
    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = await get_conversation_history(session, uuid.UUID(conversation_id))
        return {"conversation_id": conversation_id, "messages": messages}


@router.post("/conversations/{conversation_id}/summarize")
async def summarize_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """用 LLM 为已有对话生成摘要标题。"""
    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msgs = await get_conversation_history(session, uuid.UUID(conversation_id))
        if len(msgs) < 2:
            raise HTTPException(status_code=400, detail="Conversation too short to summarize")

        # 取前 2 轮对话生成标题（复用统一的语义化标题函数）
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), msgs[0]["content"])
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), msgs[-1]["content"])

        title = await generate_conversation_title(user_msg, asst_msg)
        conv.title = title
        await session.commit()
        return {"conversation_id": conversation_id, "title": title}
