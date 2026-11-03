"""Management and admission resolution for existing/personal agent definitions."""
from __future__ import annotations

import json
import uuid
from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.agents.configuration import AgentConfiguration
from backend.storage.postgres.models_agent_config import AgentConfig


class AgentUpsert(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    slug: str = Field(default="", max_length=64, pattern=r"^$|^[a-z][a-z0-9-]*$")
    kind: Literal["main", "subagent"] = "main"
    description: str = Field(default="", max_length=500)
    mode: Literal["dynamic", "deepagents"] = "dynamic"
    system_prompt: str = Field(default="", max_length=20000)
    model_id: str | None = Field(default=None, max_length=64)
    tool_names: list[str] | None = Field(default=None, max_length=128)
    subagent_ids: list[str] = Field(default_factory=list, max_length=32)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value):
        if not value.strip():
            raise ValueError("智能体名称不能为空")
        return value.strip()


def builtin_agents():
    from app.agents.deep.subagents import get_subagents, resolve_tool_spec
    children = get_subagents()
    defaults = [
        dict(id="default-chatbot", slug="default-chatbot", name="智能助手", kind="main",
             description="基础对话助手，按需检索知识库、调用工具，回答日常问题。", mode="dynamic", tool_names=None,
             system_prompt="", subagent_ids=[]),
        dict(id="deep-research", slug="deep-research", name="深度研究", kind="main",
             description="面向复杂研究任务，规划拆解、委派子智能体，核验并综合研究结论。", mode="deepagents", tool_names=None,
             system_prompt="", subagent_ids=["builtin:" + child.name for child in children]),
    ]
    labels = {"research-agent": "研究助理", "coding-agent": "代码助手"}
    defaults.extend(dict(id="builtin:" + child.name, slug=child.name, name=labels.get(child.name, child.name),
                         kind="subagent", description=child.description, mode="dynamic", tool_names=list(resolve_tool_spec(child.tools)),
                         system_prompt=child.system_prompt, subagent_ids=[]) for child in children)
    return [dict(item, model_id=None, enabled=True, source="builtin", can_edit=False) for item in defaults]


def serialize(record):
    return dict(json.loads(record.config_json), id=record.public_id, slug=record.slug, name=record.name,
                kind=record.kind, description=record.description, enabled=record.enabled,
                source="personal", can_edit=True)


async def personal_records(session, owner_id):
    return (await session.execute(select(AgentConfig).where(AgentConfig.owner_id == owner_id)
                                  .order_by(AgentConfig.created_at))).scalars().all()


async def list_agents(session, owner_id):
    return builtin_agents() + [serialize(row) for row in await personal_records(session, owner_id)]


async def owned_record(session, owner_id, agent_id):
    try:
        record_id = uuid.UUID(agent_id.removeprefix("agent:")) if agent_id.startswith("agent:") else None
    except ValueError:
        record_id = None
    row = await session.get(AgentConfig, record_id) if record_id else None
    if row is None or row.owner_id != owner_id:
        raise HTTPException(404, "个人智能体不存在或无权访问")
    return row


async def _validate_model(session, owner_id, model_id):
    if not model_id:
        return
    if model_id.startswith("custom:"):
        from backend.repositories.model_config_repository import CustomModelConfigRepository
        if await CustomModelConfigRepository(session).get_by_public_id(owner_id, model_id) is None:
            raise HTTPException(404, "所选模型不存在或无权访问")
    else:
        from app.llm.models import get_chat_model_profile, UnknownChatModelError
        try:
            get_chat_model_profile(model_id, require_available=False)
        except UnknownChatModelError as exc:
            raise HTTPException(400, "所选模型不存在") from exc


async def save_agent(session, owner_id, req, agent_id=None):
    catalog = await list_agents(session, owner_id)
    by_id = {item["id"]: item for item in catalog}
    record = await owned_record(session, owner_id, agent_id) if agent_id else None
    if record and req.kind != record.kind:
        raise HTTPException(400, "已有智能体不能切换主/子类型，请复制后创建")
    slug = record.slug if record else req.slug or "personal-" + uuid.uuid4().hex[:12]
    if any(item["slug"] == slug and item["id"] != agent_id for item in catalog):
        raise HTTPException(409, "智能体标识已存在")
    if req.kind == "subagent" and (req.mode != "dynamic" or req.subagent_ids):
        raise HTTPException(400, "子智能体独立执行任务，不可再配置子智能体")
    if req.mode != "deepagents" and req.subagent_ids:
        raise HTTPException(400, "只有深度研究模式可以委派子智能体")
    for child_id in req.subagent_ids:
        child = by_id.get(child_id)
        if child is None or child["kind"] != "subagent":
            raise HTTPException(400, "只能关联当前用户可访问的子智能体")
    await _validate_model(session, owner_id, req.model_id)
    if req.tool_names is not None:
        from app.tools.registry import get_tool_registry
        known = set(get_tool_registry().list_names(available_only=False))
        # Existing file presets may use capability specs. Personal definitions
        # select actual catalog tools, rather than introducing wildcard grants.
        if set(req.tool_names) - known:
            raise HTTPException(400, "工具列表包含未注册工具，请刷新可用工具")
    config = dict(mode=req.mode, system_prompt=req.system_prompt, model_id=req.model_id or None,
                  tool_names=list(dict.fromkeys(req.tool_names)) if req.tool_names is not None else None,
                  subagent_ids=list(dict.fromkeys(req.subagent_ids)))
    if record is None:
        record = AgentConfig(owner_id=owner_id, slug=slug, name=req.name, kind=req.kind,
                             description=req.description, enabled=req.enabled, config_json=json.dumps(config, ensure_ascii=False))
        session.add(record)
    else:
        record.name, record.description, record.enabled = req.name, req.description, req.enabled
        record.config_json = json.dumps(config, ensure_ascii=False)
    await session.flush()
    return serialize(record)


async def delete_agent(session, owner_id, agent_id):
    row = await owned_record(session, owner_id, agent_id)
    for item in await personal_records(session, owner_id):
        if item.id != row.id and agent_id in json.loads(item.config_json).get("subagent_ids", []):
            raise HTTPException(409, f"该子智能体仍被「{item.name}」使用，请先解除关联")
    await session.delete(row)


async def model_in_use(session, owner_id, model_id):
    return any(json.loads(row.config_json).get("model_id") == model_id
               for row in await personal_records(session, owner_id))


async def resolve_agent(session, owner_id, agent_id, *, snapshot=None):
    if not agent_id:
        return None
    catalog = {item["id"]: item for item in await list_agents(session, owner_id)}
    current = catalog.get(agent_id)
    if current is None:
        raise HTTPException(404, "智能体不存在或无权访问")
    if not current["enabled"]:
        raise HTTPException(409, "该智能体已停用，请选择其他智能体")
    item = snapshot if snapshot and snapshot.get("id") == agent_id else current
    from app.agents.deep.subagents import SubAgentConfig
    from backend.server.routers.chat_router import _resolve_request_model
    children, models, child_snapshots = [], {}, []
    candidates = item.get("children") if snapshot else None
    if candidates is None:
        candidates = [catalog[child] for child in item["subagent_ids"] if child in catalog]
    for child in candidates:
        authorized = catalog.get(child["id"])
        if authorized is None or authorized["kind"] != "subagent":
            raise HTTPException(409, "子智能体已移除，请重新配置后再运行")
        if not authorized["enabled"]:
            continue
        children.append(SubAgentConfig(name=child["slug"], description=child["description"],
                         system_prompt=child["system_prompt"], tools=tuple(child["tool_names"] if child["tool_names"] is not None else ("*",))))
        if child["model_id"]:
            models[child["slug"]] = await _resolve_request_model(child["model_id"], owner_id, session)
        child_snapshots.append(child)
    snapshot_value = {**item, "children": child_snapshots}
    return AgentConfiguration(id=item["id"], name=item["name"], mode=item["mode"],
        system_prompt=item["system_prompt"], tool_names=tuple(item["tool_names"]) if item["tool_names"] is not None else None,
        subagents=tuple(children), child_models=models, snapshot=snapshot_value)
