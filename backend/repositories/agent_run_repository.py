"""Repositories for persistent multi-agent runs."""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.repositories.base import BaseRepository
from backend.storage.postgres.models_agent_run import AgentRun, Run, Task


class RunRepository(BaseRepository[Run]):
    model = Run

    async def get_detail_for_user(
        self, run_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[Run]:
        stmt = (
            select(Run)
            .where(Run.id == run_id, Run.user_id == user_id)
            .options(selectinload(Run.tasks), selectinload(Run.agent_runs))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_conversation_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> Sequence[Run]:
        stmt = (
            select(Run)
            .where(
                Run.conversation_id == conversation_id,
                Run.user_id == user_id,
            )
            .options(selectinload(Run.tasks), selectinload(Run.agent_runs))
            .order_by(Run.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class TaskRepository(BaseRepository[Task]):
    model = Task

    async def get_by_key(self, run_id: uuid.UUID, task_key: str) -> Optional[Task]:
        stmt = select(Task).where(Task.run_id == run_id, Task.task_key == task_key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_run(self, run_id: uuid.UUID) -> Sequence[Task]:
        stmt = select(Task).where(Task.run_id == run_id).order_by(Task.position)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class AgentRunRepository(BaseRepository[AgentRun]):
    model = AgentRun

    async def get_by_task(self, task_id: uuid.UUID) -> Optional[AgentRun]:
        stmt = select(AgentRun).where(AgentRun.task_id == task_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

