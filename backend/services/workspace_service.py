"""Workspace organization without changing conversation/execution identities."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, select

from backend.storage.postgres.models_conversation import Conversation, Message, Workspace, WorkspaceSection
from backend.storage.postgres.models_chat_stream import ChatStream


def conversation_view(conv: Conversation, *, has_messages: bool = False) -> dict:
    return {
        "id": str(conv.id), "session_id": str(conv.id), "title": conv.title,
        "workspace_id": str(conv.workspace_id) if conv.workspace_id else None,
        "has_messages": has_messages,
        "archived_at": conv.archived_at.isoformat() if conv.archived_at else None,
        "created_at": conv.created_at.isoformat() if conv.created_at else "",
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
    }


async def conversations_with_messages(session, identifiers) -> set:
    if not identifiers:
        return set()
    return set((await session.execute(select(Message.conversation_id).where(
        Message.conversation_id.in_(identifiers)
    ).distinct())).scalars().all())


def workspace_view(workspace, count=0):
    return {"id": str(workspace.id), "name": workspace.name,
            "is_pinned": workspace.is_pinned, "local_path": workspace.local_path,
            "section_id": str(workspace.section_id) if workspace.section_id else None,
            "conversation_count": count}


async def owned_section(session, section_id, user_id, *, lock=False):
    stmt = select(WorkspaceSection).where(WorkspaceSection.id == section_id, WorkspaceSection.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update()
    section = (await session.execute(stmt)).scalar_one_or_none()
    if section is None:
        raise HTTPException(404, "Section not found")
    return section


async def owned_workspace(session, workspace_id: uuid.UUID, user_id: uuid.UUID, *, lock=False):
    stmt = select(Workspace).where(Workspace.id == workspace_id, Workspace.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update()
    workspace = (await session.execute(stmt)).scalar_one_or_none()
    if workspace is None:
        raise HTTPException(404, "Workspace not found")
    return workspace


async def list_workspaces(session, user_id: uuid.UUID) -> list[dict]:
    rows = (await session.execute(
        select(Workspace, func.count(Conversation.id))
        .outerjoin(Conversation, (Conversation.workspace_id == Workspace.id) & (Conversation.user_id == user_id)
                   & Conversation.archived_at.is_(None) & Conversation.messages.any())
        .where(Workspace.user_id == user_id).group_by(Workspace.id)
        .order_by(Workspace.is_pinned.desc(), Workspace.created_at.asc(), Workspace.id.asc())
    )).all()
    return [workspace_view(w, count) for w, count in rows]


async def create_session(session, user_id, title, workspace_id):
    # All membership writes lock the target workspace before the conversation.
    # Removal uses the same workspace lock, avoiding FK failures during races.
    if workspace_id is not None:
        await owned_workspace(session, workspace_id, user_id, lock=True)
    conv = Conversation(user_id=user_id, title=title, workspace_id=workspace_id)
    session.add(conv)
    await session.flush()
    return conv


async def update_session(session, user_id, conversation_id, changes):
    if changes.get("workspace_id") is not None:
        await owned_workspace(session, changes["workspace_id"], user_id, lock=True)
    conv = (await session.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .with_for_update()
    )).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    for key, value in changes.items():
        setattr(conv, key, value)
    await session.flush()
    await session.refresh(conv)
    return conv


async def archive_sessions(session, user_id, *, workspace_id=None, conversation_id=None):
    if workspace_id is not None:
        await owned_workspace(session, workspace_id, user_id, lock=True)
    stmt = select(Conversation).where(Conversation.user_id == user_id, Conversation.archived_at.is_(None))
    if workspace_id is not None:
        stmt = stmt.where(Conversation.workspace_id == workspace_id, Conversation.messages.any())
    else:
        stmt = stmt.where(Conversation.id == conversation_id)
    convs = (await session.execute(stmt.order_by(Conversation.id).with_for_update())).scalars().all()
    if conversation_id is not None and not convs:
        raise HTTPException(404, "Conversation not found")
    ids = [conv.id for conv in convs]
    if ids and (await session.execute(select(ChatStream.id).where(
        ChatStream.conversation_id.in_(ids), ChatStream.status.in_(("running", "cancel_requested"))
    ).limit(1))).scalar_one_or_none():
        raise HTTPException(409, "请先停止正在运行的任务，再归档聊天。")
    for conv in convs:
        conv.archived_at = datetime.now(timezone.utc)
    await session.flush()
    return ids
