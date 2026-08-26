"""对话服务 — 带 DB 持久化的对话管理。"""

from __future__ import annotations

import asyncio
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

# 内存级后台任务注册表：持引用防止 asyncio task 被 GC 中途回收
_background_tasks: set = set()


def _spawn_background(coro) -> bool:
    """把协程作为后台任务调度（持有引用防 GC）。无运行中事件循环时跳过。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("[chat] no running event loop, background task skipped")
        return False
    task = loop.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return True


async def _extract_fact_background(
    user_id: uuid.UUID, content: str, conversation_id: uuid.UUID
) -> None:
    """后台事实提取：独立 session，失败不影响主链路（语义记忆写路径）。"""
    try:
        from backend.storage.postgres.manager import get_session
        from app.memory.manager import extract_and_store_fact

        async with get_session() as s:
            await extract_and_store_fact(s, user_id, content, conversation_id)
            await s.commit()
    except Exception as exc:
        logger.warning("[chat] background fact extraction failed: %s", exc)


async def create_conversation(
    session: AsyncSession, user_id: uuid.UUID, title: Optional[str] = None
) -> Conversation:
    repo = ConversationRepository(session)
    conv = Conversation(user_id=user_id, title=title or "New Conversation")
    await repo.add(conv)
    return conv


async def delete_message(
    session: AsyncSession, message_id: uuid.UUID
) -> bool:
    """删除指定消息（2026-08-21：被终止的对话轮整轮不入历史时使用）。

    调用方负责 commit（本函数只 flush，与其他 service 函数一致）。
    """
    repo = MessageRepository(session)
    msg = await repo.get_by_id(message_id)
    if msg is None:
        return False
    await repo.delete(msg)
    return True


async def add_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    metadata_json: Optional[str] = None,
    image: Optional[str] = None,
) -> Message:
    repo = MessageRepository(session)
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        image=image,
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

    # 语义记忆: 用户消息含触发词时后台提取事实（偏好/身份/明确要求）。
    # 2026-08-15: 从同步执行改为后台任务 —— 不再阻塞 agent 运行 / SSE 开流，
    # 且提取走 fast tier + 内容去重（见 manager.extract_and_store_fact）。
    if role == "user" and conv:
        try:
            from app.memory.manager import should_extract_fact
            if should_extract_fact(content):
                _spawn_background(
                    _extract_fact_background(conv.user_id, content, conversation_id)
                )
        except Exception as exc:
            logger.warning("[chat] fact extraction trigger failed: %s", exc)

    return msg


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID
) -> Optional[Conversation]:
    repo = ConversationRepository(session)
    return await repo.get_with_messages(conversation_id)


async def get_conversation_history(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    """获取对话历史，返回 [{"role", "content", "created_at", "meta"?}] 格式。

    limit/offset 显式窗口参数：默认最多 100 条（UI 展示上限），调用方可按需
    覆盖——上下文注入用真实尾部窗口（见 get_compressed_history），
    避免"最早 100 条"式的隐式截断。
    """
    import json as _json
    msg_repo = MessageRepository(session)
    msgs = await msg_repo.list_by_conversation(
        conversation_id, limit=limit, offset=offset
    )
    out: List[dict] = []
    for m in msgs:
        item: dict = {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "image": getattr(m, "image", None),
            "created_at": m.created_at.isoformat() if m.created_at else "",
        }
        raw = getattr(m, "metadata_json", None)
        if raw:
            try:
                item["meta"] = _json.loads(raw)
            except Exception:
                pass
        out.append(item)
    return out


async def _count_conversation_messages(
    session: AsyncSession, conversation_id: uuid.UUID
) -> int:
    from sqlalchemy import func, select

    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(Message)
                .where(Message.conversation_id == conversation_id)
            )
        ).scalar_one()
    )


def decide_history_window(
    count: int, has_summary: bool, window: int, cap: int
) -> dict:
    """纯逻辑：给定消息总数/是否有摘要，决定上下文窗口策略。

    返回 {"mode", "limit", "offset"}：
      - full       全部历史都在窗口内（<= window），原样返回
      - compressed 有摘要且超窗口 → 摘要 + 最近 window 条（真实尾部）
      - cap_tail   无摘要且超窗口 → 最近 min(cap, count) 条（显式上限兜底，
                    避免"摘要长期失败 + 超长会话"撑爆上下文窗口）
    """
    if count <= window:
        return {"mode": "full", "limit": count, "offset": 0}
    if has_summary:
        return {"mode": "compressed", "limit": window, "offset": count - window}
    tail = min(cap, count)
    return {"mode": "cap_tail", "limit": tail, "offset": count - tail}


async def get_compressed_history(
    session: AsyncSession, conversation_id: uuid.UUID
) -> List[dict]:
    """情景记忆压缩: 有 summary 的会话返回 [summary system 消息 + 最近 N 轮]。

    长对话不再塞全部历史——摘要承载远期上下文, 最近消息保留细节。
    无 summary 时返回显式上限（HISTORY_CONTEXT_MAX_MESSAGES，默认 100）内的
    完整历史；超出上限取真实尾部并记日志（避免隐式截断 / 无界增长）。
    """
    from app.memory.manager import RECENT_TURNS_KEPT
    from app.core.config import get_settings

    cfg = get_settings()
    conv_repo = ConversationRepository(session)
    conv = await conv_repo.get_by_id(conversation_id)
    count = await _count_conversation_messages(session, conversation_id)
    window = RECENT_TURNS_KEPT * 2
    plan = decide_history_window(
        count,
        bool(conv and conv.summary),
        window,
        cfg.HISTORY_CONTEXT_MAX_MESSAGES,
    )

    if plan["mode"] == "compressed":
        recent = await get_conversation_history(
            session, conversation_id, limit=plan["limit"], offset=plan["offset"]
        )
        return [{
            "role": "system",
            "content": f"以下是本次对话到目前为止的内容摘要：\n{conv.summary}",
        }] + recent

    if plan["mode"] == "cap_tail":
        logger.warning(
            "[chat] long conversation (%d msgs) without summary, "
            "injecting last %d messages (HISTORY_CONTEXT_MAX_MESSAGES)",
            count, plan["limit"],
        )
    return await get_conversation_history(
        session, conversation_id, limit=plan["limit"], offset=plan["offset"]
    )


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


async def delete_conversation(
    session: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    """删除整个会话（级联删除其所有消息）。验证会话归属当前用户。

    Message.conversation_id 外键带 ondelete="CASCADE"，且 ORM relationship
    带 cascade="all, delete-orphan"，删除 Conversation 即级联删除消息。
    """
    conv = await get_conversation(session, conversation_id)
    if not conv or conv.user_id != user_id:
        return False

    await session.delete(conv)
    await session.commit()
    return True
