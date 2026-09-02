"""对话仓库。"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.repositories.base import BaseRepository
from backend.storage.postgres.models_conversation import Conversation, Message


class ConversationRepository(BaseRepository[Conversation]):
    model = Conversation

    async def get_with_messages(self, id: uuid.UUID) -> Optional[Conversation]:
        stmt = (
            select(Conversation)
            .where(Conversation.id == id)
            .options(selectinload(Conversation.messages))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> Sequence[Conversation]:
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def list_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """按时间正序取消息，支持显式 limit/offset 窗口。

        offset 用于取真实尾部（上下文注入场景），避免"limit 100 截到最早
        100 条"导致最近对话反而丢失。
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
