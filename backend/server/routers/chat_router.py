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

    # 自动设置会话标题（首次对话时用 LLM 生成摘要）
    if is_new and req.query.strip() and answer.strip():
        try:
            from app.llm.client import get_llm_client
            llm = get_llm_client()
            summary_prompt = (
                "用不超过20个字概括这段对话的主题，只返回概括结果，不要加引号或任何额外说明。\n\n"
                f"用户: {req.query.strip()[:200]}\n"
                f"助手: {answer.strip()[:300]}"
            )
            title = llm.chat_sync(
                [{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=50,
            ).strip().strip('"').strip("'").strip("。").strip("，")
            # 兜底：如果 LLM 返回太短或失败，用原始截断
            if len(title) < 2:
                title = req.query.strip()[:30]
        except Exception as exc:
            logger.warning("[chat/send] title generation failed, using fallback: %s", exc)
            title = req.query.strip()[:30]

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

        # 取前 2 轮对话生成标题
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), msgs[0]["content"])
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), msgs[-1]["content"])

        try:
            from app.llm.client import get_llm_client
            llm = get_llm_client()
            summary_prompt = (
                "用不超过20个字概括这段对话的主题，只返回概括结果，不要加引号或任何额外说明。\n\n"
                f"用户: {user_msg[:200]}\n"
                f"助手: {asst_msg[:300]}"
            )
            title = llm.chat_sync(
                [{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=50,
            ).strip().strip('"').strip("'").strip("。").strip("，")
            if len(title) < 2:
                title = user_msg[:30]
        except Exception as exc:
            logger.warning("[summarize] LLM failed: %s", exc)
            title = user_msg[:30]

        conv.title = title
        await session.commit()
        return {"conversation_id": conversation_id, "title": title}
