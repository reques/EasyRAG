"""对话路由 — 带 DB 持久化的对话 API。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.services.observability_service import admission_fields
from app.llm.models import (
    ChatModelProfile,
    ChatModelUnavailableError,
    UnknownChatModelError,
    get_chat_model_profile,
    list_chat_model_profiles,
)
from app.skills.loader import SkillDefinition
from app.skills.registry import list_builtin_skills, merge_available_skills
from app.skills.runtime import SkillRuntimeContext, resolve_dependency_closure
from backend.services.chat_service import (
    create_conversation,
    add_message,
    count_conversation_messages,
    count_folded_messages,
    get_compressed_history,
    get_conversation_history,
    list_user_conversations,
    get_conversation,
    generate_conversation_title,
    schedule_memory_evaluation,
)
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user, get_admin_user
from backend.storage.postgres.models_user import User
from backend.repositories.knowledge_repository import KnowledgeBaseRepository
from backend.repositories.model_config_repository import CustomModelConfigRepository
from backend.services.model_config_service import (
    ModelConfigValidationError,
    encrypt_api_key,
    profile_from_custom_model,
    validate_and_normalize_base_url,
)
from backend.storage.postgres.models_model_config import CustomModelConfig
from backend.repositories.skill_config_repository import CustomSkillConfigRepository
from backend.services.skill_config_service import (
    SkillConfigValidationError,
    definition_from_record,
    delete_personal_skill_files,
    slugify,
    validate_slug,
    write_personal_skill,
)
from backend.storage.postgres.models_skill_config import CustomSkillConfig
from backend.storage.postgres.models_chat_stream import ChatStream
from backend.services import chat_stream_service as streams

logger = get_logger(__name__)
cfg = get_settings()
router = APIRouter(prefix="/chat", tags=["chat"])


async def _load_knowledge_scope(
    session: AsyncSession, user_id: uuid.UUID
) -> tuple[list[str], list[dict[str, Any]]]:
    """Load retrieval IDs and display metadata from the same owner-scoped query."""
    catalog = await KnowledgeBaseRepository(session).list_catalog_by_owner(user_id)
    return [item["id"] for item in catalog], catalog


# ── Request / Response ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    request_id: Optional[uuid.UUID] = None
    conversation_id: Optional[str] = None  # None = 创建新会话
    model_id: Optional[str] = Field(default=None, max_length=64)
    # 2026-09-04 Skill 重构：语义从"注入这些 Skill 的指令"改为"本次请求可用的
    # Skill 范围"。上限由 SKILLS_MAX_SELECTED 控制（默认 10；渐进式披露下首轮
    # 只给摘要行，放宽不会撑爆上下文）。运行时上限校验在 _resolve_request_skills。
    skill_ids: list[str] = Field(default_factory=list, max_length=32)
    # 2026-08-21：深度研究开关 — 用户显式选择时强制走 DeepAgents 工作流
    # （主 Agent + SubAgent），不依赖全局 AGENT_MODE 配置
    deep_research: bool = False
    # 2026-08-25：图片输入。前端粘贴/上传后以 data URL（data:image/...;base64,...）
    # 形式随对话请求发出。后端据此裁决：所选模型支持多模态则直读，否则 OCR 转文字。
    image: Optional[str] = Field(default=None, description="图片 data URL，可选")
    resume_checkpoint: bool = False
    """复用最后一条用户消息，从 pending LangGraph checkpoint 继续执行。"""


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    run_id: str = ""
    model_id: str = ""
    model_name: str = ""
    intent: str = ""
    steps: list[str | dict[str, Any]] = Field(default_factory=list)
    sources: list[dict] = []
    elapsed_seconds: float = 0.0
    skills: list[dict] = Field(default_factory=list)
    resumed: bool = False
    trace_id: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)
    context_usage: dict[str, int] = Field(default_factory=dict)


async def _persist_trace_safely(
    result: dict[str, Any],
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    source_message_id: int,
    mode: str,
    model_id: str,
    goal: str,
) -> str:
    """Best-effort trace persistence; observability must not break the answer."""
    if not result.get("events"):
        from app.observability.context import execution_context
        context = execution_context()
        return context.local_trace_id if context else ""
    try:
        from backend.services.trace_service import persist_execution_trace

        return await persist_execution_trace(
            get_session,
            conversation_id=conversation_id,
            user_id=user_id,
            source_message_id=source_message_id,
            events=result["events"],
            mode=mode,
            model_id=model_id,
            goal=goal,
        )
    except Exception as exc:
        logger.warning("[chat] execution trace persist failed: %s", exc)
        return ""


class ResumeCheckpointRequest(BaseModel):
    request_id: Optional[uuid.UUID] = None
    model_id: Optional[str] = Field(default=None, max_length=64)
    skill_ids: list[str] = Field(default_factory=list, max_length=32)
    deep_research: Optional[bool] = None


class ConversationSummary(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


class ChatModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    available: bool
    is_default: bool
    source: str = "builtin"
    provider_type: str = "cloud"
    can_delete: bool = False
    supports_vision: bool = False


class ChatModelListResponse(BaseModel):
    default_model_id: str
    models: list[ChatModelInfo]
    # 上下文占用指示器的分母（Settings.CHAT_CONTEXT_WINDOW），全局单值。
    context_window: int = 0


class CustomModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    provider_name: str = Field(default="", max_length=80)
    provider_type: Literal["local", "cloud"]
    base_url: str = Field(..., min_length=1, max_length=512)
    model_name: str = Field(..., min_length=1, max_length=160)
    api_key: str = Field(default="", max_length=8192)
    requires_api_key: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # 是否支持图片（多模态）输入：用户添加自定义模型时勾选
    supports_vision: bool = False


def _last_resumable_user_message(conv, query: str):
    """校验恢复请求必须对应当前会话最后一条尚未回答的用户消息。"""
    messages = list(conv.messages or [])
    last = messages[-1] if messages else None
    if last is None or last.role != "user" or last.content != query:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="最后一条用户消息与恢复请求不一致，无法恢复工作记忆",
        )
    return last


class ChatSkillInfo(BaseModel):
    """Skill 列表项。**不含正文** —— 渐进式披露：正文只经 read_skill 工具
    （模型侧）或 ``GET /skills/{slug}/content``（编辑器侧）获取。"""

    id: str  # = slug（兼容前端既有 selectedSkillIds 字段名）
    slug: str
    name: str
    description: str
    tool_names: list[str] = Field(default_factory=list)
    skill_dependencies: list[str] = Field(default_factory=list)
    category: str = "通用"
    icon: str = "sparkles"
    source: str = "builtin"  # builtin | personal
    can_edit: bool = False
    body_available: bool = False


class ChatSkillContent(BaseModel):
    """单个 Skill 的完整正文（配置弹窗编辑用）。"""

    slug: str
    name: str
    description: str
    body: str
    tool_names: list[str] = Field(default_factory=list)
    category: str = "自定义"
    icon: str = "sparkles"
    can_edit: bool = False


class ChatSkillToolInfo(BaseModel):
    name: str
    description: str
    available: bool
    # 公共工具不受 Skill 门控（kb_search / calculator / datetime_tool）
    public: bool = False


class ChatSkillListResponse(BaseModel):
    max_selected: int = 10
    skills: list[ChatSkillInfo]
    tools: list[ChatSkillToolInfo]


class CustomSkillUpsert(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)
    # SKILL.md 正文（旧字段名保留，前端表单不用改）
    instructions: str = Field(..., min_length=1, max_length=20000)
    tool_names: list[str] = Field(default_factory=list, max_length=8)
    category: str = Field(default="自定义", max_length=32)
    icon: str = Field(default="sparkles", max_length=32)
    # 目录名；省略时由 name 生成（中文名回退为 skill-<uuid8>）
    slug: Optional[str] = Field(default=None, max_length=128)


async def _resolve_request_model(
    model_id: Optional[str], user_id: uuid.UUID, session: AsyncSession
) -> ChatModelProfile:
    """Validate the public model selection before any message is persisted."""
    if model_id and model_id.startswith("custom:"):
        record = await CustomModelConfigRepository(session).get_by_public_id(
            user_id, model_id
        )
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="自定义模型不存在或无权访问",
            )
        try:
            profile = profile_from_custom_model(record)
        except ModelConfigValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        if not profile.available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"模型 {profile.name} 配置不完整，请重新配置",
            )
        return profile

    try:
        return get_chat_model_profile(model_id)
    except (UnknownChatModelError, ChatModelUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


async def _accessible_skills(
    user_id: uuid.UUID, session: AsyncSession
) -> list[SkillDefinition]:
    """用户可访问的全部 Skill = 内置（磁盘）+ 自己的个人 Skill（索引 → 磁盘）。

    索引行存在但磁盘文件缺失时跳过并告警 —— 给用户一个点开是空的 Skill 比
    少一项更糟。同 slug 时个人版本覆盖内置（对齐 Yuxi）。
    """
    records = await CustomSkillConfigRepository(session).list_by_owner(user_id)
    personal: list[SkillDefinition] = []
    for record in records:
        definition = definition_from_record(record, owner_id=user_id)
        if definition is None:
            logger.warning(
                "[chat/skills] skipped personal skill %s (file missing)", record.slug
            )
            continue
        personal.append(definition)

    return merge_available_skills(personal)


async def _resolve_request_skills(
    skill_ids: list[str], user_id: uuid.UUID, session: AsyncSession
) -> list[SkillDefinition]:
    """把请求的 slug 列表解析为**有效集合**（含 skill_dependencies 闭包）。

    语义变更（2026-09-04）：返回值不再是"要注入的指令"，而是"本次请求可用的
    Skill 范围"。正文是否进 prompt、工具是否解锁，由 SkillsMiddleware 按
    ``activated_skills`` 逐轮裁决。

    依赖闭包只在用户可访问集合内展开 —— 因此不会扩大用户权限（对齐 Yuxi）。
    """
    normalized = list(
        dict.fromkeys(
            skill_id.strip().lower() for skill_id in skill_ids if skill_id.strip()
        )
    )
    max_selected = cfg.SKILLS_MAX_SELECTED
    if len(normalized) > max_selected:
        raise HTTPException(
            status_code=400, detail=f"每次最多选择 {max_selected} 个 Skill"
        )
    if not normalized:
        return []

    available = await _accessible_skills(user_id, session)
    catalog = {skill.slug: skill for skill in available}
    for slug in normalized:
        if len(slug) > 128:
            raise HTTPException(status_code=400, detail="Skill slug 格式不正确")
        if slug not in catalog:
            raise HTTPException(
                status_code=404, detail=f"Skill {slug} 不存在或无权访问"
            )

    closure = resolve_dependency_closure(normalized, catalog)
    return [catalog[slug] for slug in closure]


def _selected_skill_payload(
    definitions: list[SkillDefinition],
) -> list[dict[str, str]]:
    """随消息落库与 SSE 返回的 Skill 快照（历史消息展示用）。"""
    return [{"id": d.slug, "slug": d.slug, "name": d.name} for d in definitions]


def _normalize_custom_skill_request(
    req: CustomSkillUpsert, *, existing_slug: Optional[str] = None
) -> dict[str, Any]:
    """校验并归一化个人 Skill 的写入请求（DB 索引字段 + 磁盘内容字段分离）。

    ``existing_slug``：更新时沿用原 slug —— 改 slug 等于换目录，会让历史消息
    里的 Skill 快照指向不存在的标识，因此更新不允许改 slug。
    """
    name = req.name.strip()
    body = req.instructions.strip()
    if not name or not body:
        raise HTTPException(status_code=400, detail="Skill 名称和工作指令不能为空")

    try:
        if existing_slug:
            slug = existing_slug
        elif req.slug:
            slug = validate_slug(req.slug)
        else:
            slug = slugify(name, fallback=f"skill-{uuid.uuid4().hex[:8]}")
    except SkillConfigValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "index": {
            "slug": slug,
            "name": name,
            "description": req.description.strip(),
            "category": req.category.strip() or "自定义",
            "icon": req.icon.strip() or "sparkles",
        },
        "content": {
            "slug": slug,
            "name": name,
            "description": req.description.strip(),
            "body": body,
            "tool_names": list(req.tool_names),
            "category": req.category.strip() or "自定义",
            "icon": req.icon.strip() or "sparkles",
        },
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/skills", response_model=ChatSkillListResponse)
async def list_chat_skills(
    current_user: User = Depends(get_current_user),
):
    """Return built-in and owner-scoped personal Skills plus tool metadata.

    响应**不含 Skill 正文**（渐进式披露）：编辑器需要正文时单独请求
    ``GET /skills/{slug}/content``。
    """
    async with get_session() as session:
        definitions = await _accessible_skills(current_user.id, session)

    from app.tools.registry import get_tool_registry

    registry = get_tool_registry()
    tools = [
        ChatSkillToolInfo(
            name=tool.name,
            description=tool.description,
            available=tool.is_available(),
            public=bool((tool.metadata or {}).get("public")),
        )
        for tool in registry.list_all(available_only=False)
    ]
    return ChatSkillListResponse(
        max_selected=cfg.SKILLS_MAX_SELECTED,
        skills=[ChatSkillInfo(**d.to_public_dict()) for d in definitions],
        tools=tools,
    )


@router.get("/skills/{slug}/content", response_model=ChatSkillContent)
async def get_chat_skill_content(
    slug: str,
    current_user: User = Depends(get_current_user),
):
    """Return one Skill's full SKILL.md body (配置弹窗编辑 / 内置只读查看)。"""
    async with get_session() as session:
        definitions = await _accessible_skills(current_user.id, session)
    target = next((d for d in definitions if d.slug == slug.strip().lower()), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Skill 不存在或无权访问")
    return ChatSkillContent(
        slug=target.slug,
        name=target.name,
        description=target.description,
        body=target.body,
        tool_names=list(target.tool_dependencies),
        category=target.category,
        icon=target.icon,
        can_edit=target.can_edit,
    )


@router.post(
    "/skills",
    response_model=ChatSkillInfo,
    status_code=status.HTTP_201_CREATED,
)
async def create_custom_chat_skill(
    req: CustomSkillUpsert,
    current_user: User = Depends(get_current_user),
):
    """创建个人 Skill：先写磁盘 SKILL.md，再落 DB 索引行。

    顺序见 ``skill_config_service`` 模块文档 —— 反过来会在文件写失败时留下
    一条指向空目录的索引行。
    """
    values = _normalize_custom_skill_request(req)
    index_values, content_values = values["index"], values["content"]

    if any(s.slug == index_values["slug"] for s in list_builtin_skills()):
        raise HTTPException(
            status_code=409,
            detail=f"slug {index_values['slug']} 与内置 Skill 冲突，请换一个",
        )

    async with get_session() as session:
        repo = CustomSkillConfigRepository(session)
        if await repo.get_by_name(current_user.id, index_values["name"]):
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill")
        if await repo.get_by_slug(current_user.id, index_values["slug"]):
            raise HTTPException(status_code=409, detail="已存在相同 slug 的 Skill")

        try:
            definition = write_personal_skill(
                owner_id=current_user.id, **content_values
            )
        except SkillConfigValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        record = CustomSkillConfig(owner_id=current_user.id, **index_values)
        try:
            await repo.add(record)
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            # 索引行没落成 → 回滚磁盘，避免留下无索引的孤儿目录
            delete_personal_skill_files(current_user.id, index_values["slug"])
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill") from exc
    return ChatSkillInfo(**definition.to_public_dict())


@router.put("/skills/{slug}", response_model=ChatSkillInfo)
async def update_custom_chat_skill(
    slug: str,
    req: CustomSkillUpsert,
    current_user: User = Depends(get_current_user),
):
    """更新个人 Skill。slug 不可改 —— 换 slug 等于换目录，会让历史消息里的
    Skill 快照指向不存在的标识。"""
    async with get_session() as session:
        repo = CustomSkillConfigRepository(session)
        record = await repo.get_by_slug(current_user.id, slug)
        if record is None:
            raise HTTPException(status_code=404, detail="自定义 Skill 不存在或无权访问")

        values = _normalize_custom_skill_request(req, existing_slug=record.slug)
        index_values, content_values = values["index"], values["content"]

        duplicate = await repo.get_by_name(current_user.id, index_values["name"])
        if duplicate is not None and duplicate.id != record.id:
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill")

        try:
            definition = write_personal_skill(
                owner_id=current_user.id, **content_values
            )
        except SkillConfigValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for key, value in index_values.items():
            setattr(record, key, value)
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill") from exc
    return ChatSkillInfo(**definition.to_public_dict())


@router.delete("/skills/{slug}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_chat_skill(
    slug: str,
    current_user: User = Depends(get_current_user),
):
    """删除个人 Skill：先删 DB 索引，再删目录。

    文件删除失败只 warn —— 留下的孤儿目录由
    ``prune_orphan_directories`` 兜底，不阻塞用户操作。
    """
    async with get_session() as session:
        repo = CustomSkillConfigRepository(session)
        record = await repo.get_by_slug(current_user.id, slug)
        if record is None:
            raise HTTPException(status_code=404, detail="自定义 Skill 不存在或无权访问")
        target_slug = record.slug
        await repo.delete(record)
        await session.commit()
    delete_personal_skill_files(current_user.id, target_slug)
    return None


@router.get("/models", response_model=ChatModelListResponse)
async def list_chat_models(
    current_user: User = Depends(get_current_user),
):
    """Return safe model metadata for the conversation model selector."""
    profiles = list_chat_model_profiles()
    async with get_session() as session:
        records = await CustomModelConfigRepository(session).list_by_owner(
            current_user.id
        )
    for record in records:
        try:
            profiles.append(profile_from_custom_model(record))
        except ModelConfigValidationError:
            profiles.append(ChatModelProfile(
                id=record.public_id,
                name=record.name,
                provider=record.provider_name,
                provider_type=record.provider_type,
                source="custom",
                base_url=record.base_url,
                api_key="",
                model=record.model_name,
                temperature=record.temperature,
                requires_api_key=True,
                supports_vision=bool(record.supports_vision),
            ))
    return ChatModelListResponse(
        default_model_id=cfg.LLM_DEFAULT_MODEL_ID,
        context_window=cfg.CHAT_CONTEXT_WINDOW,
        models=[
            ChatModelInfo(
                **profile.to_public_dict(
                    default_model_id=cfg.LLM_DEFAULT_MODEL_ID
                )
            )
            for profile in profiles
        ],
    )


@router.post("/models", response_model=ChatModelInfo, status_code=status.HTTP_201_CREATED)
async def create_custom_chat_model(
    req: CustomModelCreate,
    current_user: User = Depends(get_current_user),
):
    """Persist an owner-scoped local/cloud OpenAI-compatible model profile."""
    name = req.name.strip()
    model_name = req.model_name.strip()
    provider_name = req.provider_name.strip() or (
        "本地模型" if req.provider_type == "local" else "自定义云端"
    )
    if not name or not model_name:
        raise HTTPException(status_code=400, detail="模型名称和模型 ID 不能为空")
    if req.requires_api_key and not req.api_key.strip():
        raise HTTPException(status_code=400, detail="该模型配置要求填写 API Key")
    try:
        base_url = validate_and_normalize_base_url(req.base_url, req.provider_type)
        encrypted_key = encrypt_api_key(req.api_key)
    except ModelConfigValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with get_session() as session:
        repo = CustomModelConfigRepository(session)
        if await repo.get_by_name(current_user.id, name):
            raise HTTPException(status_code=409, detail="已存在同名自定义模型")
        record = CustomModelConfig(
            owner_id=current_user.id,
            name=name,
            provider_name=provider_name,
            provider_type=req.provider_type,
            base_url=base_url,
            model_name=model_name,
            api_key_encrypted=encrypted_key,
            requires_api_key=req.requires_api_key,
            temperature=req.temperature,
            supports_vision=bool(req.supports_vision),
        )
        try:
            await repo.add(record)
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(status_code=409, detail="已存在同名自定义模型") from exc

    profile = profile_from_custom_model(record)
    return ChatModelInfo(
        **profile.to_public_dict(default_model_id=cfg.LLM_DEFAULT_MODEL_ID)
    )


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_chat_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
):
    """Delete one user-owned custom model; built-in profiles are immutable."""
    async with get_session() as session:
        repo = CustomModelConfigRepository(session)
        record = await repo.get_by_public_id(current_user.id, model_id)
        if record is None:
            raise HTTPException(status_code=404, detail="自定义模型不存在或无权访问")
        await repo.delete(record)
        await session.commit()
    from app.llm.client import evict_chat_model_clients

    evict_chat_model_clients(model_id)
    return None

@router.post("/send", response_model=ChatResponse)
async def send_message(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """Use the same durable execution and cancellation boundary as SSE clients."""
    response = await send_message_stream(req, current_user)
    async for frame in response.body_iterator:
        for line in frame.splitlines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event.get("type") == "done":
                return ChatResponse(
                    **{key: value for key, value in event.items() if key in ChatResponse.model_fields and key not in ("answer", "conversation_id")},
                    conversation_id=response.headers["X-Conversation-Id"], answer=event.get("content", ""),
                )
            if event.get("type") in ("error", "cancelled"):
                raise HTTPException(409, event.get("detail") or "Execution interrupted")
    raise HTTPException(409, "Execution ended without a result")



def _stream_response(run, after=0):
    return StreamingResponse(
        streams.read_events(run.id, run.user_id, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "X-Chat-Stream-Id": str(run.id), "X-Conversation-Id": str(run.conversation_id)},
    )


async def _require_idle_conversation(session, conversation_id):
    from sqlalchemy import select
    from backend.storage.postgres.models_conversation import Conversation

    while True:
        # All mutators share admission's conversation lock. Checking status
        # without it would allow deletion/compaction to race a newly admitted run.
        locked = (await session.execute(select(Conversation.id).where(
            Conversation.id == conversation_id
        ).with_for_update())).scalar_one_or_none()
        if locked is None:
            raise HTTPException(404, "Conversation not found")
        active = (await session.execute(select(ChatStream).where(
            ChatStream.conversation_id == conversation_id, ChatStream.status.in_(streams.ACTIVE)
        ))).scalar_one_or_none()
        if active is None:
            return
        await streams.recover_stale(session, active)
        if active.status in streams.ACTIVE:
            raise HTTPException(409, {"message": "会话任务仍在执行，请重新连接或先取消。", "stream_id": str(active.id)})
        # Stale recovery commits its terminal event and releases the lock. Acquire
        # it again and recheck before mutating the conversation.


@router.get("/observability")
async def observability_capabilities(current_user: User = Depends(get_current_user)):
    return {"can_view": bool(current_user.is_superuser or current_user.role == "admin")}


@router.get("/streams/{stream_id}/observability")
async def execution_observability(stream_id: uuid.UUID, current_user: User = Depends(get_admin_user)):
    from backend.services.observability_service import navigation
    async with get_session() as session:
        # Being an operator does not grant cross-user chat access.
        run = await streams.owned_run(session, stream_id, current_user.id)
        return navigation(run)


@router.get("/streams/{stream_id}/events")
async def stream_events(stream_id: uuid.UUID, after: int = Query(0, ge=0), current_user: User = Depends(get_current_user)):
    async with get_session() as session:
        run = await streams.owned_run(session, stream_id, current_user.id)
        if after > run.last_sequence:
            raise HTTPException(400, "Event cursor is ahead of the stream")
        return _stream_response(run, after)


@router.post("/streams/{stream_id}/cancel")
async def cancel_stream(stream_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    return await streams.request_cancel(stream_id, current_user.id)


@router.get("/conversations/{conversation_id}/stream")
async def conversation_stream(conversation_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    async with get_session() as session:
        conv = await get_conversation(session, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(404, "Conversation not found")
        run = await streams.latest_stream(session, conversation_id, current_user.id)
        if run:
            await streams.recover_stale(session, run)
        return {"stream": streams.view(run) if run else None}


@router.post("/stream")
async def send_message_stream(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """流式对话 — SSE 逐 token 推送 Agent 回复, 结束时推送引用块。

    事件序列:
      data: {"type": "conversation_id", "conversation_id": "..."}
      data: {"type": "delta", "content": "<增量文本>"}   (多次)
      data: {"type": "done", "sources": [...], "intent": "...", "elapsed_seconds": 1.23}
      data: {"type": "error", "detail": "..."}           (仅出错时)

    设计: 检索(同步)用 run_in_executor 跑, 生成用 LLM chat_stream 流式,
    最终答案 + 引用落库与 /chat/send 保持一致。
    """
    import asyncio
    from sqlalchemy import select
    from backend.storage.postgres.models_conversation import Conversation

    stream_id = req.request_id or uuid.uuid4()
    request_fingerprint = streams.fingerprint(req.model_dump(mode="json"))

    async with get_session() as session:
        existing = await session.get(ChatStream, stream_id)
        if existing is not None:
            if existing.user_id != current_user.id:
                raise HTTPException(404, "Chat execution not found")
            if existing.fingerprint != request_fingerprint:
                raise HTTPException(409, "request_id was already used for a different request")
            return _stream_response(existing)
        selected_model = await _resolve_request_model(
            req.model_id, current_user.id, session
        )
        selected_skills = await _resolve_request_skills(
            req.skill_ids, current_user.id, session
        )
        skill_payload = _selected_skill_payload(selected_skills)
        # 获取或创建会话
        if req.conversation_id:
            conv = await get_conversation(session, uuid.UUID(req.conversation_id))
            if not conv or conv.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            conv_id = conv.id
            is_new = False
            await session.execute(select(Conversation.id).where(Conversation.id == conv_id).with_for_update())
            # A concurrent retry may have waited for the first reservation's
            # conversation lock after its initial request-ID lookup missed it.
            existing = await session.get(ChatStream, stream_id)
            if existing is not None:
                if existing.user_id != current_user.id:
                    raise HTTPException(404, "Chat execution not found")
                if existing.fingerprint != request_fingerprint:
                    raise HTTPException(409, "request_id was already used for a different request")
                return _stream_response(existing)
            await _require_idle_conversation(session, conv_id)
        else:
            conv = await create_conversation(session, current_user.id)
            conv_id = conv.id
            is_new = True

        previous = None
        if req.resume_checkpoint:
            # Completion may have committed while admission waited on the lock.
            await session.refresh(conv, attribute_names=["messages"])
            user_message = _last_resumable_user_message(conv, req.query)
            previous = await streams.latest_stream(session, conv_id, current_user.id)
            if previous and previous.source_message_id == user_message.id and previous.status == "cancelled":
                raise HTTPException(409, "任务已主动取消，请发送新消息开始新任务。")
        else:
            user_message = await add_message(
                session,
                conv_id,
                "user",
                req.query,
                image=req.image,
                metadata_json=json.dumps({
                    "skills": skill_payload,
                    "model_id": selected_model.id,
                    "deep_research": req.deep_research,
                }, ensure_ascii=False),
            )
        user_message_id = user_message.id
        stream_run = ChatStream(
            id=stream_id, user_id=current_user.id, conversation_id=conv_id,
            source_message_id=user_message_id, fingerprint=request_fingerprint,
            owner_token=uuid.uuid4().hex, status="running", last_sequence=0,
            **admission_fields(previous),
        )
        db_history = await get_compressed_history(session, conv_id)
        knowledge_base_ids, knowledge_catalog = await _load_knowledge_scope(
            session, current_user.id
        )

        session.add(stream_run)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing = await session.get(ChatStream, stream_id)
            if existing is not None and existing.user_id == current_user.id and existing.fingerprint == request_fingerprint:
                return _stream_response(existing)
            raise HTTPException(409, "Conversation already has an active execution")

    async def _event_gen_inner(execution):
        from app.services.agent_service import get_agent_service
        from dataclasses import replace
        from app.agents.events import public_event

        # ── 图片裁决（deep / dynamic 共用）──────────────────────────────────────────
        # OCR 回退只做一次：所选模型不支持多模态时，把图片转成文字拼进查询；
        # 多模态模型则保留 image_for_context 直读，query 不变。
        effective_query = req.query
        image_for_context: Optional[str] = None
        if req.image:
            if selected_model.supports_vision:
                image_for_context = req.image
            else:
                try:
                    from app.ocr.engine import ocr_image_to_text
                    _ocr_text = (await ocr_image_to_text(req.image)).strip()
                    effective_query = (
                        f"{req.query}\n\n[图片识别内容（OCR）]:\n{_ocr_text}"
                        if _ocr_text
                        else f"{req.query}\n\n[图片无法识别为文字，已忽略图片内容]"
                    )
                except Exception as _ocr_exc:
                    logger.warning("[chat/stream] OCR 失败，降级为无图：%s", _ocr_exc)
                    effective_query = f"{req.query}\n\n[图片识别服务不可用，已忽略图片内容]"

        from app.agents.context import ChatContext

        _chat_ctx = ChatContext.from_request(
            effective_query,
            thread_id=str(conv_id),
            user_id=current_user.id,
            model_id=selected_model.id,
            skill_ids=[d.slug for d in selected_skills],
            knowledge_base_ids=knowledge_base_ids,
            knowledge_catalog=knowledge_catalog,
            image_data=image_for_context,
            history=db_history,          # ← 关键：传入 DB 历史
            input_message_id=str(user_message_id),
            resume_checkpoint=req.resume_checkpoint,
            deep_research=req.deep_research,
        )
        _chat_ctx.execution_id = str(execution.id)
        _chat_ctx.local_trace_id = stream_run.local_trace_id or ""
        _chat_ctx.observability_trace_id = stream_run.observability_trace_id or ""


        agent = get_agent_service()
        agent_mode = agent.select_agent_mode(effective_query, _chat_ctx)
        use_deep = agent_mode == "deepagents"
        execution.agent_mode = agent_mode
        execution.model_id = selected_model.id
        from app.observability.context import current_observation, execution_context
        obs_context = execution_context()
        if obs_context:
            obs_context.metadata.update(agentMode=agent_mode, modelProfileId=selected_model.id)
            current_observation().update(metadata=obs_context.metadata)
        stream_context = replace(_chat_ctx, cancel_event=execution.cancel_event, deep_research=use_deep)
        loop = asyncio.get_running_loop()

        def _sse(payload: dict) -> str:
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        yield _sse({
            "type": "conversation_id",
            "conversation_id": str(conv_id),
            # run_id 在委派持久化完成后才可知，随 done 事件下发
            "run_id": "",
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
            "agent_mode": agent_mode,
            "resume_requested": req.resume_checkpoint,
        })

        if use_deep:
            # ── DeepAgents 路径：主 Agent + task → SubAgent（同步 + 状态透传）─
            import queue as _q
            from app.agents.progress import ProgressProjector
            from app.agents.events import public_event

            status_queue: "_q.Queue" = _q.Queue()
            deep_progress_summaries: list[dict] = []
            # 阶段 6：收集本轮 status（思考/工具/委派步骤），随 done + meta 持久化
            collected_steps: list[dict] = []
            progress_projector = ProgressProjector()

            def _deep_status(step: str, detail: str = ""):
                try:
                    progress = progress_projector.feed(step, detail)
                    if progress:
                        progress["created_at"] = time.time()
                        deep_progress_summaries.append(dict(progress))
                        status_queue.put({"type": "progress_summary", **progress})
                except Exception:
                    pass

            # 阶段 2：on_step 在 executor 线程经 use_request_context 铺设前
            # 不属于 _chat_ctx（不能跨线程携带 closure），先挂到线程局部副本。
            from dataclasses import replace

            _deep_ctx = replace(stream_context, on_step=_deep_status)

            # 统一事件流 → orchestrator 时代 SSE 协议（阶段 5）：委派树/工具时间线
            # 由前端现有任务面板与 AgentActivity 直接消费，无需新组件。
            # 映射规则见 delegation_service.bridge_delegation_event（可单测）。
            from backend.services.delegation_service import bridge_delegation_event

            _panel_sent = {"v": False}

            def _bridge_event(ev: dict) -> None:
                try:
                    status_queue.put({
                        "type": "agent_event",
                        "event": public_event(ev),
                    })
                    for payload in bridge_delegation_event(ev):
                        if payload["type"] == "status":
                            collected_steps.append({
                                "step": payload.get("step", ""),
                                "detail": payload.get("detail", ""),
                                "task_id": payload.get("task_id", ""),
                            })
                        if payload["type"] == "sub_tasks":
                            _panel_sent["v"] = True
                        elif (
                            payload["type"] == "status"
                            and payload.get("step") == "task_started"
                            and not _panel_sent["v"]
                        ):
                            # 单任务委派（task 工具）无 spawn_start：自造单任务清单，
                            # 且必须先于 task_started 下发（前端面板需要先初始化）
                            _panel_sent["v"] = True
                            status_queue.put({"type": "sub_tasks", "tasks": [{
                                "task_id": payload.get("task_id", ""),
                                "goal": (ev.get("content") or "")[:200],
                                "worker_hint": ev.get("subagent_type", ""),
                            }]})
                        elif payload["type"] == "progress_summary":
                            payload["created_at"] = time.time()
                        status_queue.put(payload)
                except Exception:
                    pass

            def _run_deep_in_thread():
                from app.agents.request_context import use_request_context

                # 公开流式契约：event_sink 由 AgentService.run_stream 统一进入，
                # 路由层不再触及 _run_deep 这类私有执行路径。
                with use_request_context(_deep_ctx, skill_definitions=selected_skills, model_profile=selected_model):
                    return agent.run_stream(
                        _deep_ctx.query, context=_deep_ctx, event_sink=_bridge_event
                    )

            # 注意：此处不能用 `start` 命名 —— _event_gen_inner 内任何赋值都会
            # 把 start 变成局部变量，遮蔽外层函数的 start（单 Agent 分支的
            # elapsed 计算依赖它），导致 UnboundLocalError（实测 bug）
            deep_start = time.perf_counter()
            from contextvars import copy_context
            deep_future = loop.run_in_executor(None, copy_context().run, _run_deep_in_thread)
            execution.workers.append(deep_future)
            while True:
                try:
                    ev = await loop.run_in_executor(
                        None, status_queue.get, True, 0.1
                    )
                except _q.Empty:
                    if deep_future.done():
                        break
                    continue
                if isinstance(ev, dict) and ev.get("type"):
                    # progress_summary / sub_tasks / status / worker_output（统一事件流桥接）
                    if ev.get("type") == "progress_summary":
                        deep_progress_summaries.append({
                            k: ev.get(k)
                            for k in ("id", "sequence", "phase", "status", "text", "created_at")
                        })
                    yield _sse(ev)
            try:
                deep_result = deep_future.result()
            except Exception as exc:
                logger.error("[chat/stream] deepagents error: %s", exc)
                warning_progress = progress_projector.feed("fallback", str(exc))
                if warning_progress:
                    warning_progress["created_at"] = time.time()
                    deep_progress_summaries.append(dict(warning_progress))
                    yield _sse({"type": "progress_summary", **warning_progress})
                deep_result = {
                    "final_answer": f"处理请求时发生错误: {exc}",
                    "steps": [],
                    "is_fallback": True,
                }
            answer = deep_result.get("final_answer", "") or ""
            # 阶段 5：委派执行落库（best-effort；无委派事件自动跳过）
            delegation_run_id = ""
            if deep_result.get("events"):
                from backend.services.delegation_service import persist_delegation

                try:
                    delegation_run_id = await persist_delegation(
                        get_session,
                        conversation_id=conv_id,
                        user_id=current_user.id,
                        events=deep_result.get("events"),
                        goal=req.query,
                        model_id=selected_model.id,
                        source_message_id=user_message_id,
                    ) or ""
                except Exception as exc:
                    logger.warning("[chat/stream] delegation persist failed: %s", exc)
            if delegation_run_id:
                current_observation().update(metadata={"delegationRunId": delegation_run_id})
            trace_id = await _persist_trace_safely(
                deep_result,
                conversation_id=conv_id,
                user_id=current_user.id,
                source_message_id=user_message_id,
                mode="deepagents",
                model_id=selected_model.id,
                goal=effective_query,
            )
            # 阶段 6：把本轮实时收到的 status（思考/工具/委派步骤）与 artifact
            # 交付物随 done 事件与 meta 持久化，刷新会话后仍可恢复完整轨迹。
            step_objs: list[dict] = collected_steps
            deep_artifacts: list[dict] = deep_result.get("artifacts") or []
            elapsed = round(time.perf_counter() - deep_start, 3)

            meta = {
                "intent": "deepagents",
                "agent_mode": agent_mode,
                "steps": step_objs,
                "artifacts": deep_artifacts,
                "progress_summaries": deep_progress_summaries,
                "model_id": selected_model.id,
                "model_name": selected_model.name,
                "skills": skill_payload,
                "trace_id": trace_id,
                "token_usage": deep_result.get("token_usage") or {},
                "context_usage": deep_result.get("context_usage") or {},
            }
            if answer:
                yield _sse({"type": "delta", "content": answer})
            done_payload = {
                "type": "done",
                "content": answer,
                "sources": [],
                "intent": "deepagents",
                "agent_mode": agent_mode,
                "run_id": delegation_run_id,
                "steps": step_objs,
                "artifacts": deep_artifacts,
                "progress_summaries": deep_progress_summaries,
                "elapsed_seconds": elapsed,
                "resumed": bool(deep_result.get("resumed")),
                "model_id": selected_model.id,
                "model_name": selected_model.name,
                "skills": skill_payload,
                "trace_id": trace_id,
                "token_usage": deep_result.get("token_usage") or {},
                "context_usage": deep_result.get("context_usage") or {},
            }
            await streams.commit_result(execution, answer, meta, done_payload)
            schedule_memory_evaluation(
                current_user.id, req.query, conv_id, assistant_content=answer,
                source_message_id=user_message_id,
                execution={"mode": "deepagents", "steps": step_objs, "artifacts": deep_artifacts, "error": deep_result.get("error_message")},
            )

            # 新会话标题后台生成
            if is_new and answer:
                async def _gen_title_deep():
                    try:
                        title = await generate_conversation_title(req.query, answer)
                        async with get_session() as session:
                            c = await get_conversation(session, conv_id)
                            if c:
                                c.title = title
                                await session.commit()
                    except Exception as exc:
                        logger.warning("[chat/stream] deep title gen failed: %s", exc)

                from app.observability.context import detached_context
                asyncio.get_event_loop().create_task(_gen_title_deep(), context=detached_context())
            return

        # ── 轻量动态 Agent 路径（auto 普通问题 / AGENT_MODE=dynamic）──
        # 在 executor 线程跑动态 Agent（模型通过函数调用自行决定
        # 调工具 / 检索 / 直接回答）；通过安全队列把每一步
        # 状态实时桥接回事件循环，SSE 逐步推给前端。
        from app.agents.progress import ProgressProjector
        from app.agents.response_stream import ProgressEventCollector
        import queue as _dyn_queue
        _dyn_status_queue: "_dyn_queue.Queue" = _dyn_queue.Queue()
        _dyn_events = ProgressEventCollector()
        _dyn_collected_steps = _dyn_events.steps
        _dyn_collected_artifacts = _dyn_events.artifacts
        _dyn_progress: list[dict] = []
        _dyn_projector = ProgressProjector()

        def _dyn_status(step: str, detail: str = ""):
            _dyn_status_queue.put(_dyn_events.step(step, detail))
            try:
                progress = _dyn_projector.feed(step, detail)
                if progress:
                    progress["created_at"] = time.time()
                    _dyn_progress.append(dict(progress))
                    _dyn_status_queue.put({"type": "progress_summary", **progress})
            except Exception:
                pass

        def _dyn_artifact(ev: dict):
            # 行动摘要按 id 合并落库；SSE 仍逐段发送。正文不进入过程历史。
            _dyn_status_queue.put(_dyn_events.artifact(ev))

        # 阶段 2：on_step/on_artifact 在 executor 线程经 use_request_context 铺设
        # 前不属于 _chat_ctx（不能跨线程携带 closure），故在 _run_dynamic_in_thread
        # 内把回调挂到线程局部的副本上，由门面写入传播层。
        from dataclasses import replace

        _dyn_ctx = replace(stream_context, on_step=_dyn_status, on_artifact=_dyn_artifact)

        def _run_dynamic_in_thread():
            from app.agents.request_context import use_request_context

            def _stream_event(event: dict) -> None:
                _dyn_status_queue.put({
                    "type": "agent_event",
                    "event": public_event(event),
                })

            # 与 deep 分支同构：公开流式契约，不再调用私有 _run_dynamic。
            with use_request_context(_dyn_ctx, skill_definitions=selected_skills, model_profile=selected_model):
                return agent.run_stream(
                    _dyn_ctx.query, context=_dyn_ctx, event_sink=_stream_event
                )

        _dyn_start = time.perf_counter()
        from contextvars import copy_context
        _dyn_future = loop.run_in_executor(None, copy_context().run, _run_dynamic_in_thread)
        execution.workers.append(_dyn_future)
        while True:
            try:
                ev = await loop.run_in_executor(None, _dyn_status_queue.get, True, 0.1)
            except Exception:
                ev = None
            if ev is None:
                if _dyn_future.done():
                    break
                continue
            yield _sse(ev)
        try:
            dyn_result = _dyn_future.result()
        except Exception as exc:
            logger.error("[chat/stream] dynamic agent error: %s", exc)
            dyn_result = {
                "final_answer": f"\u5904\u7406\u8bf7\u6c42\u65f6\u53d1\u751f\u9519\u8bef: {exc}",
                "steps": [],
                "sources": [],
                "is_fallback": True,
            }
        answer = dyn_result.get("final_answer", "") or ""
        elapsed = round(time.perf_counter() - _dyn_start, 3)

        # 委派持久化（best-effort；无事件自动跳过）
        delegation_run_id = ""
        if dyn_result.get("events"):
            from backend.services.delegation_service import persist_delegation

            try:
                delegation_run_id = await persist_delegation(
                    get_session,
                    conversation_id=conv_id,
                    user_id=current_user.id,
                    events=dyn_result.get("events"),
                    goal=req.query,
                    model_id=selected_model.id,
                    source_message_id=user_message_id,
                ) or ""
            except Exception as exc:
                logger.warning("[chat/stream] dynamic delegation persist failed: %s", exc)
        if delegation_run_id:
            current_observation().update(metadata={"delegationRunId": delegation_run_id})
        trace_id = await _persist_trace_safely(
            dyn_result,
            conversation_id=conv_id,
            user_id=current_user.id,
            source_message_id=user_message_id,
            mode="dynamic",
            model_id=selected_model.id,
            goal=effective_query,
        )

        meta = {
            "intent": "dynamic",
            "agent_mode": agent_mode,
            "sources": dyn_result.get("sources") or [],
            "steps": _dyn_collected_steps,
            "progress_summaries": _dyn_progress,
            "artifacts": _dyn_collected_artifacts,
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
            "trace_id": trace_id,
            "token_usage": dyn_result.get("token_usage") or {},
            "context_usage": dyn_result.get("context_usage") or {},
        }
        if answer and not _dyn_events.answer_streamed:
            yield _sse({"type": "delta", "content": answer})
        done_payload = {
            "type": "done",
            "content": answer,
            "sources": dyn_result.get("sources") or [],
            "intent": "dynamic",
            "agent_mode": agent_mode,
            "run_id": delegation_run_id,
            "steps": _dyn_collected_steps,
            "artifacts": _dyn_collected_artifacts,
            "progress_summaries": _dyn_progress,
            "elapsed_seconds": elapsed,
            "resumed": bool(dyn_result.get("resumed")),
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
            "trace_id": trace_id,
            "token_usage": dyn_result.get("token_usage") or {},
            "context_usage": dyn_result.get("context_usage") or {},
        }
        await streams.commit_result(execution, answer, meta, done_payload)
        schedule_memory_evaluation(
            current_user.id, req.query, conv_id, assistant_content=answer,
            source_message_id=user_message_id,
            execution={"mode": "dynamic", "steps": _dyn_collected_steps, "artifacts": _dyn_collected_artifacts, "error": dyn_result.get("error_message")},
        )

        # 新会话标题后台生成
        if is_new and answer:
            async def _gen_title_dynamic():
                try:
                    title = await generate_conversation_title(req.query, answer)
                    async with get_session() as session:
                        c = await get_conversation(session, conv_id)
                        if c:
                            c.title = title
                            await session.commit()
                except Exception as exc:
                    logger.warning("[chat/stream] dynamic title gen failed: %s", exc)

            from app.observability.context import detached_context
            asyncio.get_event_loop().create_task(_gen_title_dynamic(), context=detached_context())
        return

    streams.start_execution(stream_run, _event_gen_inner)
    return _stream_response(stream_run)


@router.post("/conversations/{conversation_id}/resume")
async def resume_conversation_checkpoint(
    conversation_id: uuid.UUID,
    req: ResumeCheckpointRequest,
    current_user: User = Depends(get_current_user),
):
    """恢复最后一条尚未完成的用户消息，从 pending checkpoint 继续 SSE。"""
    async with get_session() as session:
        conv = await get_conversation(session, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        active = await streams.latest_stream(session, conversation_id, current_user.id)
        if active:
            await streams.recover_stale(session, active)
            if active.status in streams.ACTIVE or (req.request_id and active.id == req.request_id):
                return _stream_response(active)
        messages = list(conv.messages or [])
        last = messages[-1] if messages else None
        if last is None or last.role != "user":
            raise HTTPException(
                status_code=409,
                detail="当前会话没有尚未完成的用户消息",
            )
        query = last.content
        image = last.image
        try:
            saved_request = json.loads(last.metadata_json or "{}")
        except (TypeError, ValueError):
            saved_request = {}
        saved_skills = saved_request.get("skills") or []
        saved_skill_ids = [
            str(item.get("id") or "") for item in saved_skills
            if isinstance(item, dict) and item.get("id")
        ]
    return await send_message_stream(
        ChatRequest(
            request_id=req.request_id,
            query=query,
            conversation_id=str(conversation_id),
            model_id=req.model_id or saved_request.get("model_id"),
            skill_ids=req.skill_ids or saved_skill_ids,
            deep_research=(
                req.deep_research
                if req.deep_research is not None
                else bool(saved_request.get("deep_research"))
            ),
            image=image,
            resume_checkpoint=True,
        ),
        current_user,
    )


@router.get("/runs/{run_id}")
async def get_multi_agent_run(
    run_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """Return one owner-scoped multi-agent run with tasks and worker runs."""
    from backend.repositories.agent_run_repository import RunRepository
    from backend.services.agent_run_service import serialize_run

    async with get_session() as session:
        run = await RunRepository(session).get_detail_for_user(
            run_id, current_user.id
        )
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return serialize_run(run)


@router.get("/traces/{trace_id}")
async def get_agent_trace(
    trace_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return a persisted execution trace owned by the current user."""
    from backend.services.trace_service import get_execution_trace

    async with get_session() as session:
        trace = await get_execution_trace(session, trace_id, current_user.id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace


@router.get("/conversations/{conversation_id}/traces")
async def list_conversation_traces(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """List persisted execution traces for one owner-scoped conversation."""
    from backend.services.trace_service import list_execution_traces

    async with get_session() as session:
        conv = await get_conversation(session, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        traces = await list_execution_traces(session, conversation_id, current_user.id)
        return {"conversation_id": str(conversation_id), "traces": traces}


@router.get("/conversations/{conversation_id}/runs")
async def list_conversation_runs(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """List durable multi-agent runs for one owner-scoped conversation."""
    from backend.repositories.agent_run_repository import RunRepository
    from backend.services.agent_run_service import serialize_run

    async with get_session() as session:
        conv = await get_conversation(session, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        runs = await RunRepository(session).list_by_conversation_for_user(
            conversation_id, current_user.id
        )
        return {
            "conversation_id": str(conversation_id),
            "runs": [serialize_run(run) for run in runs],
        }


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    current_user: User = Depends(get_current_user),
):
    """列出当前用户的所有会话。"""
    async with get_session() as session:
        convs = await list_user_conversations(session, current_user.id)
        return [
            ConversationSummary(
                id=str(c.id),
                title=c.title,
                created_at=c.created_at.isoformat() if c.created_at else "",
                updated_at=c.updated_at.isoformat() if c.updated_at else "",
            )
            for c in convs
        ]


@router.get("/conversations/{conversation_id}/history")
async def get_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """获取指定会话的对话历史。"""
    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = await get_conversation_history(session, uuid.UUID(conversation_id))
        return {"conversation_id": conversation_id, "messages": messages}


@router.post("/conversations/{conversation_id}/summarize")
async def summarize_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """用 LLM 为已有对话生成摘要标题。"""
    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msgs = await get_conversation_history(session, uuid.UUID(conversation_id))
        if len(msgs) < 2:
            raise HTTPException(status_code=400, detail="Conversation too short to summarize")

        # 取前 2 轮对话生成标题（复用统一的语义化标题函数）
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), msgs[0]["content"])
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), msgs[-1]["content"])

        title = await generate_conversation_title(user_msg, asst_msg)
        conv.title = title
        await session.commit()
        return {"conversation_id": conversation_id, "title": title}


@router.get("/conversations/{conversation_id}/context")
async def get_context_state(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """当前上下文窗口状态：摘要 + 折叠水位，供 UI 展示"哪些历史已被压缩"。"""
    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        from app.core.config import get_settings
        from app.memory.manager import RECENT_TURNS_KEPT

        message_count = await count_conversation_messages(
            session, uuid.UUID(conversation_id)
        )
        folded = await count_folded_messages(
            session, uuid.UUID(conversation_id),
            getattr(conv, "last_summarized_message_id", None),
        )
        cfg = get_settings()
        return {
            "conversation_id": conversation_id,
            "has_summary": bool(conv.summary),
            "summary": conv.summary or "",
            "message_count": message_count,
            "folded_messages": folded,
            "kept_window": RECENT_TURNS_KEPT * 2,
            "compactable": max(0, message_count - cfg.COMPACT_KEEP_TURNS * 2 - folded),
        }


@router.post("/conversations/{conversation_id}/compact")
async def compact_conversation_endpoint(
    conversation_id: str,
    keep_turns: Optional[int] = None,
    current_user: User = Depends(get_current_user),
):
    """压缩上下文窗口（``/compact``）：把保留窗口之外的历史折叠进会话摘要。

    之后每一轮对话只注入「摘要 + 最近若干轮原文」，长对话不再线性膨胀。
    ``keep_turns`` 可显式指定"保留最近 N 轮原文"（越小压得越狠，1..50）。
    """
    from app.core.config import get_settings
    from app.memory.manager import compact_conversation, RECENT_TURNS_KEPT

    # 显式入参校验（避免为 Query 约束额外引入依赖）
    if keep_turns is not None:
        keep_turns = max(1, min(int(keep_turns), 50))

    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await _require_idle_conversation(session, conv.id)

        try:
            result = await compact_conversation(
                session, uuid.UUID(conversation_id), keep_turns=keep_turns
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        await session.commit()

        settings = get_settings()
        effective_keep = int(keep_turns or settings.COMPACT_KEEP_TURNS)
        window = effective_keep * 2
        result["keep_turns"] = effective_keep
        result["keep_window"] = window
        result["auto_keep_window"] = RECENT_TURNS_KEPT * 2

        if result.get("compacted"):
            if result.get("complete"):
                result["message"] = (
                    f"已折叠 {result.get('newly_folded', 0)} 条历史消息，"
                    f"后续只注入摘要 + 最近 {window} 条原文。"
                )
            else:
                result["message"] = (
                    f"已折叠 {result.get('newly_folded', 0)} 条，历史较长——"
                    f"再执行一次 /compact 可继续压缩。"
                )
        elif result.get("message_count", 0) <= window:
            result["message"] = (
                f"当前对话共 {result.get('message_count', 0)} 条消息，"
                f"还没超过保留窗口（{window} 条），无需压缩。"
            )
        else:
            # 诚实说明：不是"对话不够长"，而是自动摘要已覆盖更早的历史
            tail = max(0, int(result.get("message_count", 0)) - int(result.get("folded_messages", 0)))
            result["message"] = (
                f"更早的历史已由摘要折叠（覆盖 {result.get('folded_messages', 0)} 条），"
                f"当前注入的是「摘要 + 最近 {tail} 条原文」。"
                f"想进一步收缩可用 /compact {max(1, tail // 4)}（保留更少原文）。"
            )
        return result


@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """删除整个会话及其所有消息（级联删除）。验证会话归属当前用户。"""
    from backend.services.chat_service import delete_conversation

    async with get_session() as session:
        conv = await get_conversation(session, uuid.UUID(conversation_id))
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(404, "Conversation not found")
        await _require_idle_conversation(session, conv.id)
        deleted = await delete_conversation(
            session, uuid.UUID(conversation_id), current_user.id
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
    return {"conversation_id": conversation_id, "deleted": True}
