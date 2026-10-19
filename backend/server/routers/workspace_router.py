"""Authenticated workspace and conversation organization endpoints."""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select

from backend.server.utils.auth_middleware import get_current_user
from backend.services import workspace_service as service
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_conversation import Conversation, Workspace, WorkspaceSection
from backend.storage.postgres.models_user import User

router = APIRouter(prefix="/chat", tags=["workspaces"])
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


class WorkspaceName(BaseModel):
    name: Name


class WorkspaceCreate(WorkspaceName):
    local_path: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1024)] | None = None


class WorkspaceUpdate(BaseModel):
    name: Name | None = None
    is_pinned: bool | None = None
    local_path: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1024)] | None = None
    section_id: uuid.UUID | None = None


class ConversationCreate(BaseModel):
    title: Title = "新对话"
    workspace_id: uuid.UUID | None = None


class ConversationUpdate(BaseModel):
    title: Title | None = None
    workspace_id: uuid.UUID | None = None


@router.get("/workspaces")
async def list_workspaces(user: User = Depends(get_current_user)):
    async with get_session() as session:
        return await service.list_workspaces(session, user.id)


@router.post("/workspaces", status_code=201)
async def create_workspace(body: WorkspaceCreate, user: User = Depends(get_current_user)):
    async with get_session() as session:
        workspace = Workspace(user_id=user.id, name=body.name, local_path=body.local_path or None)
        session.add(workspace)
        await session.commit()
        return service.workspace_view(workspace)


@router.patch("/workspaces/{workspace_id}")
async def rename_workspace(workspace_id: uuid.UUID, body: WorkspaceUpdate, user: User = Depends(get_current_user)):
    changes = body.model_dump(exclude_unset=True)
    if any(key in changes and changes[key] is None for key in ("name", "is_pinned")):
        raise HTTPException(422, "name and is_pinned cannot be null")
    async with get_session() as session:
        if changes.get("section_id") is not None:
            await service.owned_section(session, changes["section_id"], user.id, lock=True)
        workspace = await service.owned_workspace(session, workspace_id, user.id, lock=True)
        for key, value in changes.items():
            setattr(workspace, key, value or None if key == "local_path" else value)
        await session.commit()
        return service.workspace_view(workspace)


@router.get("/workspaces/{workspace_id}")
async def get_workspace(workspace_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        return service.workspace_view(await service.owned_workspace(session, workspace_id, user.id))


@router.get("/workspace-sections")
async def list_sections(user: User = Depends(get_current_user)):
    async with get_session() as session:
        rows = (await session.execute(select(WorkspaceSection).where(
            WorkspaceSection.user_id == user.id
        ).order_by(WorkspaceSection.created_at, WorkspaceSection.id))).scalars().all()
        return [{"id": str(row.id), "name": row.name} for row in rows]


@router.post("/workspace-sections", status_code=201)
async def create_section(body: WorkspaceName, user: User = Depends(get_current_user)):
    async with get_session() as session:
        row = WorkspaceSection(user_id=user.id, name=body.name)
        session.add(row)
        await session.commit()
        return {"id": str(row.id), "name": row.name}


@router.patch("/workspace-sections/{section_id}")
async def rename_section(section_id: uuid.UUID, body: WorkspaceName, user: User = Depends(get_current_user)):
    async with get_session() as session:
        row = await service.owned_section(session, section_id, user.id, lock=True)
        row.name = body.name
        await session.commit()
        return {"id": str(row.id), "name": row.name}


@router.delete("/workspace-sections/{section_id}")
async def remove_section(section_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        row = await service.owned_section(session, section_id, user.id, lock=True)
        await session.delete(row)
        await session.commit()
        return {"deleted": True, "workspaces_preserved": True}


@router.post("/workspaces/{workspace_id}/archive")
async def archive_workspace(workspace_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        ids = await service.archive_sessions(session, user.id, workspace_id=workspace_id)
        await session.commit()
        return {"archived_count": len(ids), "conversation_ids": [str(id) for id in ids]}


@router.delete("/workspaces/{workspace_id}")
async def remove_workspace(workspace_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        workspace = await service.owned_workspace(session, workspace_id, user.id, lock=True)
        # FK SET NULL retains messages, task journals, checkpoints and traces.
        await session.delete(workspace)
        await session.commit()
        return {"id": str(workspace_id), "deleted": True, "conversations_preserved": True}


@router.post("/conversations", status_code=201)
async def create_conversation(body: ConversationCreate, user: User = Depends(get_current_user)):
    async with get_session() as session:
        conv = await service.create_session(session, user.id, body.title, body.workspace_id)
        await session.commit()
        populated = await service.conversations_with_messages(session, [conv.id])
        return service.conversation_view(conv, has_messages=conv.id in populated)


@router.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: uuid.UUID, body: ConversationUpdate, user: User = Depends(get_current_user)):
    changes = body.model_dump(exclude_unset=True)
    if "title" in changes and changes["title"] is None:
        raise HTTPException(422, "title cannot be null")
    async with get_session() as session:
        conv = await service.update_session(session, user.id, conversation_id, changes)
        await session.commit()
        populated = await service.conversations_with_messages(session, [conv.id])
        return service.conversation_view(conv, has_messages=conv.id in populated)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        conv = (await session.execute(select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user.id,
        ))).scalar_one_or_none()
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        populated = await service.conversations_with_messages(session, [conv.id])
        return service.conversation_view(conv, has_messages=conv.id in populated)


@router.post("/conversations/{conversation_id}/archive")
async def archive_conversation(conversation_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        await service.archive_sessions(session, user.id, conversation_id=conversation_id)
        await session.commit()
        return {"archived": True}


@router.post("/conversations/{conversation_id}/restore")
async def restore_conversation(conversation_id: uuid.UUID, user: User = Depends(get_current_user)):
    async with get_session() as session:
        conv = (await session.execute(select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user.id,
        ).with_for_update())).scalar_one_or_none()
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        conv.archived_at = None
        await session.commit()
        await session.refresh(conv)
        populated = await service.conversations_with_messages(session, [conv.id])
        return service.conversation_view(conv, has_messages=conv.id in populated)
