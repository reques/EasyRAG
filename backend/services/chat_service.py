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

    return msg


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID
) -> Optional[Conversation]:
    repo = ConversationRepository(session)
    return await repo.get_with_messages(conversation_id)


async def get_conversation_history(
    session: AsyncSession, conversation_id: uuid.UUID
) -> List[dict]:
    """获取对话历史，返回 [{"role": ..., "content": ...}] 格式。"""
    msg_repo = MessageRepository(session)
    msgs = await msg_repo.list_by_conversation(conversation_id)
    return [{"role": m.role, "content": m.content} for m in msgs]


async def list_user_conversations(
    session: AsyncSession, user_id: uuid.UUID
) -> list[Conversation]:
    repo = ConversationRepository(session)
    return list(await repo.list_by_user(user_id))
