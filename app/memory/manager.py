"""分层记忆管理 — 情景记忆（会话摘要）+ 语义记忆（用户事实）。

工作记忆即 Agent 执行的中间状态（dynamic/deep 路径为 LangGraph messages state；旧 AgentState 已随 single 固定管线退役），不在此模块。
本模块负责跨轮次/跨会话的持久记忆：
  - 情景记忆: 会话级增量摘要（消息数达到阈值时压缩, 注入 prompt 替代全部历史）
  - 语义记忆: 用户级 facts（规则触发存储 + 注入, LLM 自动提取留后续）

可靠性设计（2026-08-15 修复"摘要失败丢段"）：
  以 conversations.last_summarized_message_id 记录上次成功折叠的位置（含）。
  每次折叠只处理该位置之后的新消息；LLM 失败时不推进指针，下次触发重试
  同一段 —— 中间段消息永远不会从摘要里丢失（旧实现只看"最后 10 条"，
  某次压缩失败后失败点之前的新消息会永久蒸发）。
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from backend.storage.postgres.models_conversation import Conversation, Message
from backend.storage.postgres.models_memory import UserFact

logger = get_logger(__name__)

# 距上次成功摘要以来新增消息数达到该值即触发压缩（user+assistant 都计数）
SUMMARY_INTERVAL = 10
# 注入 prompt 时保留最近 N 轮原始消息（配合 summary）
RECENT_TURNS_KEPT = 10
# 单次折叠最多处理的新消息数（防超长 prompt；超出部分下次继续，不丢弃）
SUMMARY_FOLD_BATCH = 20


# ── 情景记忆：会话摘要 ─────────────────────────────────────────────────────

async def maybe_update_summary(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> bool:
    """距上次成功摘要以来新增消息数达到 SUMMARY_INTERVAL 时, 增量压缩会话摘要。

    增量策略: 旧 summary + 自 last_summarized_message_id 之后的新消息
    （单次最多 SUMMARY_FOLD_BATCH 条）→ LLM 压缩成新 summary。
    失败时不推进 last_summarized_message_id, 下次触发重试同一段, 不丢消息。
    返回是否实际执行了压缩（含失败重试）。失败静默记日志, 不阻塞对话主链路。
    """
    conv = (
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
    ).scalar_one_or_none()
    if not conv:
        return False

    # 自上次成功折叠点之后的新消息（升序；指针为空时从第一条开始）
    last_id = conv.last_summarized_message_id or 0
    pending = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.id > last_id)
            .order_by(Message.id.asc())
        )
    ).scalars().all()
    if len(pending) < SUMMARY_INTERVAL:
        return False

    fold = pending[:SUMMARY_FOLD_BATCH]
    try:
        from app.llm.client import get_llm_client
        llm = get_llm_client(tier="fast")
        # 新增消息（自上次摘要之后）的文本
        new_text = "\n".join(f"{m.role}: {m.content[:200]}" for m in fold)
        prompt = (
            "把以下对话内容压缩成一段简洁的会话摘要（不超过 150 字），"
            "保留关键话题、用户意图和重要结论，丢弃寒暄和冗余细节。\n"
        )
        if conv.summary:
            prompt += f"\n已有摘要（需融合）:\n{conv.summary}\n"
        prompt += f"\n新增对话:\n{new_text}\n\n只输出摘要本身，不要任何解释。"
        summary = (await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=256,
        )).strip()
        if summary:
            conv.summary = summary
            # 关键：只有成功才推进折叠断点（含本次最后一条已折叠消息）
            conv.last_summarized_message_id = fold[-1].id
            await session.flush()
            logger.info(
                "[memory] summary updated for conv %s (folded msgs %d..%d, %d pending left)",
                conversation_id, fold[0].id, fold[-1].id, len(pending) - len(fold),
            )
            return True
    except Exception as exc:
        logger.warning("[memory] summary update failed: %s", exc)
    return False


# ── 语义记忆：用户事实 ─────────────────────────────────────────────────────

async def add_user_fact(
    session: AsyncSession,
    user_id: uuid.UUID,
    fact: str,
    source_conversation_id: Optional[uuid.UUID] = None,
) -> UserFact:
    """存储一条用户事实（语义记忆）。内容去重：同一用户已存在相同事实时直接返回旧记录。"""
    fact_text = fact.strip()
    if not fact_text:
        raise ValueError("fact must not be empty")
    existing = (
        await session.execute(
            select(UserFact).where(
                UserFact.user_id == user_id,
                UserFact.fact == fact_text,
            )
        )
    ).scalars().first()
    if existing is not None:
        logger.info("[memory] user fact duplicate skipped for %s: %s", user_id, fact_text[:40])
        return existing
    record = UserFact(
        user_id=user_id,
        fact=fact_text,
        source_conversation_id=source_conversation_id,
    )
    session.add(record)
    await session.flush()
    logger.info("[memory] user fact added for %s: %s", user_id, fact_text[:40])
    return record


async def get_user_facts(
    session: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 20,
) -> List[str]:
    """查询用户 facts（按时间倒序取最近 limit 条），供注入 prompt。"""
    rows = (
        await session.execute(
            select(UserFact)
            .where(UserFact.user_id == user_id)
            .order_by(UserFact.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return [r.fact for r in rows]


# 规则触发关键词：用户消息含这些时, 尝试提取事实存入语义记忆。
# 2026-08-15 收紧：去掉过宽的"以后"（"以后再说吧"等无事实可提的常态表达），
# 其余保持覆盖（身份/偏好/明确要求）。提取在后台任务执行 + fast tier，
# 误触发只产生一次廉价调用，且 LLM 对无事实内容输出 NONE 不入库。
FACT_TRIGGER_KEYWORDS = ("记住", "我喜欢", "我是", "叫我", "我的", "偏好")


def should_extract_fact(query: str) -> bool:
    """规则判断: 用户消息是否包含值得存入语义记忆的信息。"""
    q = query.strip()
    return any(kw in q for kw in FACT_TRIGGER_KEYWORDS)


async def extract_and_store_fact(
    session: AsyncSession,
    user_id: uuid.UUID,
    query: str,
    conversation_id: Optional[uuid.UUID] = None,
) -> bool:
    """规则触发时, 用 LLM 从用户消息提取事实并存入（含去重）。失败静默。"""
    if not should_extract_fact(query):
        return False
    try:
        from app.llm.client import get_llm_client
        llm = get_llm_client(tier="fast")
        prompt = (
            "从以下用户消息中提取一条值得跨会话记住的事实（用户偏好/身份/明确要求）。\n"
            "要求：一句话陈述, 不超过 30 字。如果没有值得记住的事实, 只输出 \"NONE\"。\n\n"
            f"用户消息: {query.strip()[:300]}"
        )
        fact = (await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=100,
        )).strip().strip('"').strip("'")
        if fact and fact != "NONE" and len(fact) <= 60:
            await add_user_fact(session, user_id, fact, conversation_id)
            return True
    except Exception as exc:
        logger.warning("[memory] fact extraction failed: %s", exc)
    return False
