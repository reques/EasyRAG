"""Owner-scoped agent management; global presets are read-only."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_user import User
from backend.services.agent_config_service import AgentUpsert, list_agents, save_agent, delete_agent

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("")
async def catalog(user: User = Depends(get_current_user)):
    from app.tools.registry import get_tool_registry
    async with get_session() as session:
        items = await list_agents(session, user.id)
    tools = [dict(name=tool.name, description=tool.description[:240], available=tool.is_available())
             for tool in get_tool_registry().list_all(available_only=False)]
    return dict(agents=items, tools=tools)


@router.post("", status_code=201)
async def create(req: AgentUpsert, user: User = Depends(get_current_user)):
    async with get_session() as session:
        try:
            result = await save_agent(session, user.id, req)
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(409, "智能体标识已存在") from exc
    return result


@router.put("/{agent_id}")
async def update(agent_id: str, req: AgentUpsert, user: User = Depends(get_current_user)):
    async with get_session() as session:
        result = await save_agent(session, user.id, req, agent_id)
        await session.commit()
    return result


@router.delete("/{agent_id}", status_code=204)
async def remove(agent_id: str, user: User = Depends(get_current_user)):
    async with get_session() as session:
        await delete_agent(session, user.id, agent_id)
        await session.commit()
