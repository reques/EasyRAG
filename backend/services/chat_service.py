"""对话服务 — 带 DB 持久化的对话管理。"""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from backend.repositories.conversation_repository import (
    ConversationRepository,
    MessageRepository,
)
from backend.storage.postgres.models_conversation import Conversation, Message

logger = get_logger(__name__)


async def create_conversation(
    session: AsyncSession, user_id: uuid.UUID, title: Optional[str] = None
) -> Conversation:
    repo = ConversationRepository(session)
    conv = Conversation(user_id=user_id, title=title or "New Conversation")
    await repo.add(conv)
    return conv


async def add_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    metadata_json: Optional[str] = None,
) -> Message:
    repo = MessageRepository(session)
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata_json=metadata_json,
    )
    await repo.add(msg)

    # Touch conversation updated_at
    conv_repo = ConversationRepository(session)
    conv = await conv_repo.get_by_id(conversation_id)
    if conv:
        from datetime import datetime, timezone
        conv.updated_at = datetime.now(timezone.utc)
        await session.flush()

    # 情景记忆: 消息数达 10 的倍数时增量压缩会话摘要（仅 assistant 落库后触发,
    # 一轮 = user+assistant 两条, 避免每条都算）
    if role == "assistant" and conv:
        try:
            from app.memory.manager import maybe_update_summary
            await maybe_update_summary(session, conversation_id)
        except Exception as exc:
            logger.warning("[chat] summary trigger failed: %s", exc)

    # 语义记忆: 用户消息含触发词时提取事实（偏好/身份/明确要求）
    if role == "user" and conv:
        try:
            from app.memory.manager import extract_and_store_fact
            await extract_and_store_fact(session, conv.user_id, content, conversation_id)
        except Exception as exc:
            logger.warning("[chat] fact extraction failed: %s", exc)

    return msg


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID
) -> Optional[Conversation]:
    repo = ConversationRepository(session)
    return await repo.get_with_messages(conversation_id)


async def get_conversation_history(
    session: AsyncSession, conversation_id: uuid.UUID
) -> List[dict]:
    """获取对话历史，返回 [{"role", "content", "meta"?}] 格式。"""
    import json as _json
    msg_repo = MessageRepository(session)
    msgs = await msg_repo.list_by_conversation(conversation_id)
    out: List[dict] = []
    for m in msgs:
        item: dict = {"role": m.role, "content": m.content}
        raw = getattr(m, "metadata_json", None)
        if raw:
            try:
                item["meta"] = _json.loads(raw)
            except Exception:
                pass
        out.append(item)
    return out


async def get_compressed_history(
    session: AsyncSession, conversation_id: uuid.UUID
) -> List[dict]:
    """情景记忆压缩: 有 summary 的会话返回 [summary system 消息 + 最近 N 轮]。

    长对话不再塞全部历史——摘要承载远期上下文, 最近消息保留细节。
    无 summary 时返回完整历史（行为与 get_conversation_history 一致）。
    """
    from app.memory.manager import RECENT_TURNS_KEPT

    conv_repo = ConversationRepository(session)
    conv = await conv_repo.get_by_id(conversation_id)
    full = await get_conversation_history(session, conversation_id)

    if not conv or not conv.summary:
        return full

    # 只在"压缩真的更短"时才用压缩版（短会话 summary + 最近N轮反而更长, 不值得）
    # 一轮 = user+assistant 两条, 取 2*N 条
    recent = full[-(RECENT_TURNS_KEPT * 2):]
    if len(full) <= RECENT_TURNS_KEPT * 2:
        return full  # 全部历史本就在窗口内, 无需压缩
    out: List[dict] = [{
        "role": "system",
        "content": f"以下是本次对话到目前为止的内容摘要：\n{conv.summary}",
    }]
    out.extend(recent)
    return out


async def list_user_conversations(
    session: AsyncSession, user_id: uuid.UUID
) -> list[Conversation]:
    repo = ConversationRepository(session)
    return list(await repo.list_by_user(user_id))


def _fallback_title(query: str) -> str:
    """LLM 失败时的兜底标题：清理标点/换行后截取，避免生硬的原文直切。"""
    import re
    text = re.sub(r"\s+", " ", query.strip())
    # 循环去掉开头的常见祈使/敬语前缀（可叠加，如"请帮我"）
    for _ in range(3):
        new = re.sub(r"^(请|麻烦|帮我|给我|告诉我|请问|我想知道|我想了解|你能不能|你能)\s*", "", text)
        if new == text:
            break
        text = new
    return text[:24] + ("…" if len(text) > 24 else "") if text else "新对话"


async def generate_conversation_title(query: str, answer: str) -> str:
    """用 LLM 为一轮对话生成语义化标题。

    提供给 LLM 完整的用户输入(截 500 字)与助手回答(截 600 字)，
    要求输出名词性短语而非句子。失败时回退到清理后的关键词截取。
    注意 max_tokens 下限：DeepSeek 等 reasoning 模型 max_tokens 过低
    (≤40) 会把额度耗在思考上导致正文为空, 实测 100 是安全下限。
    """
    try:
        from app.llm.client import get_llm_client
        llm = get_llm_client()
        prompt = (
            "为以下对话生成一个简短的会话标题，用于显示在对话列表侧边栏。\n"
            "要求：\n"
            "1. 提炼对话的**主题**而非复述原文，用名词性短语（如「民法典人格权解读」「设备异常处理流程」）\n"
            "2. 不超过 16 个字，不要标点结尾，不要引号\n"
            "3. 只输出标题本身，不要任何解释\n\n"
            f"用户提问：{query.strip()[:500]}\n"
            f"助手回答：{answer.strip()[:600]}"
        )
        title = (
            await llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                # DeepSeek 等 reasoning 模型把额度耗在思考上会导致正文为空;
                # 输入越长 reasoning 消耗越大, 低 max_tokens 必现空响应。
                # 标题任务输出极短但 reasoning 不可控, 给足 256 保底。
                max_tokens=256,
            )
        ).strip().strip('"').strip("'").strip("。").strip("，").strip("、")
        # LLM 返回过长（没按指令）或为空 → 兜底
        if 2 <= len(title) <= 30:
            return title
        logger.warning("[title] LLM title out of range (%d chars), fallback", len(title))
        return _fallback_title(query)
    except Exception as exc:
        logger.warning("[title] generation failed, fallback: %s", exc)
        return _fallback_title(query)
