"""用户长期记忆管理 API。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.memory.manager import (
    DuplicateUserFactError,
    delete_episode_memory,
    delete_user_fact,
    list_episode_records,
    list_user_fact_records,
    update_user_fact,
)
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_user import User

router = APIRouter(prefix="/memory", tags=["memory"])


class UserFactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    fact: str
    source_conversation_id: Optional[uuid.UUID] = None
    created_at: datetime


class UserFactUpdate(BaseModel):
    fact: str = Field(..., min_length=1, max_length=500)

    @field_validator("fact")
    @classmethod
    def strip_fact(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("fact must not be blank")
        return value


class EpisodeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_conversation_id: Optional[uuid.UUID] = None
    source_message_id: Optional[int] = None
    title: str
    goal: str
    outcome: str
    summary: str
    lessons: Optional[str] = None
    unfinished: Optional[str] = None
    created_at: datetime


@router.get("/facts", response_model=list[UserFactResponse])
async def list_facts(
    limit: int = Query(default=100, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    """列出当前用户的长期事实记忆。"""
    async with get_session() as session:
        return await list_user_fact_records(session, current_user.id, limit=limit)


@router.patch("/facts/{fact_id}", response_model=UserFactResponse)
async def update_fact(
    fact_id: uuid.UUID,
    payload: UserFactUpdate,
    current_user: User = Depends(get_current_user),
):
    """更新当前用户的一条事实记忆。"""
    async with get_session() as session:
        try:
            record = await update_user_fact(
                session, current_user.id, fact_id, payload.fact
            )
        except DuplicateUserFactError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory fact not found",
            )
        await session.commit()
        await session.refresh(record)
        return record


@router.delete("/facts/{fact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_fact(
    fact_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """删除当前用户的一条事实记忆。"""
    async with get_session() as session:
        deleted = await delete_user_fact(session, current_user.id, fact_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory fact not found",
            )
        await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/episodes", response_model=list[EpisodeResponse])
async def list_episodes(
    limit: int = Query(default=100, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    """列出当前用户可管理的情景记忆。"""
    async with get_session() as session:
        return await list_episode_records(session, current_user.id, limit=limit)


@router.delete("/episodes/{episode_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_episode(
    episode_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """删除当前用户的一条情景记忆。"""
    async with get_session() as session:
        deleted = await delete_episode_memory(session, current_user.id, episode_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Episode memory not found")
        await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
