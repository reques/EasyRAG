"""对话路由 — 带 DB 持久化的对话 API。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from app.llm.models import (
    ChatModelProfile,
    ChatModelUnavailableError,
    UnknownChatModelError,
    get_chat_model_profile,
    list_chat_model_profiles,
)
from app.skills.catalog import SkillProfile, get_builtin_skill, list_builtin_skills
from app.skills.context import SkillRuntimeContext
from backend.services.chat_service import (
    create_conversation,
    add_message,
    get_compressed_history,
    get_conversation_history,
    list_user_conversations,
    get_conversation,
    generate_conversation_title,
)
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user
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
    encode_tool_names,
    profile_from_custom_skill,
)
from backend.storage.postgres.models_skill_config import CustomSkillConfig

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
    conversation_id: Optional[str] = None  # None = 创建新会话
    model_id: Optional[str] = Field(default=None, max_length=64)
    skill_ids: list[str] = Field(default_factory=list, max_length=3)
    # 2026-08-21：深度研究开关 — 用户显式选择时强制走 DeepAgents 工作流
    # （主 Agent + SubAgent），不依赖全局 AGENT_MODE 配置
    deep_research: bool = False
    # 2026-08-25：图片输入。前端粘贴/上传后以 data URL（data:image/...;base64,...）
    # 形式随对话请求发出。后端据此裁决：所选模型支持多模态则直读，否则 OCR 转文字。
    image: Optional[str] = Field(default=None, description="图片 data URL，可选")


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    run_id: str = ""
    model_id: str = ""
    model_name: str = ""
    intent: str = ""
    steps: list[str] = []
    sources: list[dict] = []
    elapsed_seconds: float = 0.0
    skills: list[dict] = Field(default_factory=list)


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


class ChatSkillInfo(BaseModel):
    id: str
    name: str
    description: str
    instructions: str
    tool_names: list[str] = Field(default_factory=list)
    category: str = "通用"
    icon: str = "sparkles"
    source: str = "builtin"
    can_edit: bool = False


class ChatSkillToolInfo(BaseModel):
    name: str
    description: str
    available: bool


class ChatSkillListResponse(BaseModel):
    max_selected: int = 3
    skills: list[ChatSkillInfo]
    tools: list[ChatSkillToolInfo]


class CustomSkillUpsert(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)
    instructions: str = Field(..., min_length=1, max_length=6000)
    tool_names: list[str] = Field(default_factory=list, max_length=8)
    category: str = Field(default="自定义", max_length=32)
    icon: str = Field(default="sparkles", max_length=32)


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


async def _resolve_request_skills(
    skill_ids: list[str], user_id: uuid.UUID, session: AsyncSession
) -> list[SkillProfile]:
    """Resolve selected built-in/custom Skills with owner checks."""
    normalized = list(dict.fromkeys(skill_id.strip() for skill_id in skill_ids if skill_id.strip()))
    if len(normalized) > 3:
        raise HTTPException(status_code=400, detail="每次最多选择 3 个 Skill")

    repo = CustomSkillConfigRepository(session)
    profiles: list[SkillProfile] = []
    for skill_id in normalized:
        if len(skill_id) > 64:
            raise HTTPException(status_code=400, detail="Skill ID 格式不正确")
        profile = get_builtin_skill(skill_id)
        if profile is None and skill_id.startswith("custom:"):
            record = await repo.get_by_public_id(user_id, skill_id)
            if record is not None:
                try:
                    profile = profile_from_custom_skill(record)
                except SkillConfigValidationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Skill {skill_id} 不存在或无权访问",
            )
        profiles.append(profile)

    if sum(len(profile.instructions) for profile in profiles) > 12000:
        raise HTTPException(status_code=400, detail="所选 Skill 指令总长度超过限制")
    return profiles


def _selected_skill_payload(
    profiles: list[SkillProfile],
) -> list[dict[str, str]]:
    return [{"id": profile.id, "name": profile.name} for profile in profiles]


def _normalize_custom_skill_request(
    req: CustomSkillUpsert,
) -> dict[str, Any]:
    name = req.name.strip()
    instructions = req.instructions.strip()
    if not name or not instructions:
        raise HTTPException(status_code=400, detail="Skill 名称和工作指令不能为空")
    try:
        tool_names_json = encode_tool_names(req.tool_names)
    except SkillConfigValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "name": name,
        "description": req.description.strip(),
        "instructions": instructions,
        "tool_names_json": tool_names_json,
        "category": req.category.strip() or "自定义",
        "icon": req.icon.strip() or "sparkles",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/skills", response_model=ChatSkillListResponse)
async def list_chat_skills(
    current_user: User = Depends(get_current_user),
):
    """Return built-in and owner-scoped custom Skills plus tool metadata."""
    profiles = list_builtin_skills()
    async with get_session() as session:
        records = await CustomSkillConfigRepository(session).list_by_owner(
            current_user.id
        )
    for record in records:
        try:
            profiles.append(profile_from_custom_skill(record))
        except SkillConfigValidationError:
            logger.warning("[chat/skills] skipped invalid custom Skill %s", record.id)

    from app.tools.registry import get_tool_registry

    registry = get_tool_registry()
    tools = [
        ChatSkillToolInfo(
            name=tool.name,
            description=tool.description,
            available=tool.is_available(),
        )
        for tool in registry.list_all(available_only=False)
    ]
    return ChatSkillListResponse(
        skills=[ChatSkillInfo(**profile.to_public_dict()) for profile in profiles],
        tools=tools,
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
    values = _normalize_custom_skill_request(req)
    async with get_session() as session:
        repo = CustomSkillConfigRepository(session)
        if await repo.get_by_name(current_user.id, values["name"]):
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill")
        record = CustomSkillConfig(owner_id=current_user.id, **values)
        try:
            await repo.add(record)
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill") from exc
    return ChatSkillInfo(**profile_from_custom_skill(record).to_public_dict())


@router.put("/skills/{skill_id}", response_model=ChatSkillInfo)
async def update_custom_chat_skill(
    skill_id: str,
    req: CustomSkillUpsert,
    current_user: User = Depends(get_current_user),
):
    values = _normalize_custom_skill_request(req)
    async with get_session() as session:
        repo = CustomSkillConfigRepository(session)
        record = await repo.get_by_public_id(current_user.id, skill_id)
        if record is None:
            raise HTTPException(status_code=404, detail="自定义 Skill 不存在或无权访问")
        duplicate = await repo.get_by_name(current_user.id, values["name"])
        if duplicate is not None and duplicate.id != record.id:
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill")
        for key, value in values.items():
            setattr(record, key, value)
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(status_code=409, detail="已存在同名自定义 Skill") from exc
    return ChatSkillInfo(**profile_from_custom_skill(record).to_public_dict())


@router.delete("/skills/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_chat_skill(
    skill_id: str,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        repo = CustomSkillConfigRepository(session)
        record = await repo.get_by_public_id(current_user.id, skill_id)
        if record is None:
            raise HTTPException(status_code=404, detail="自定义 Skill 不存在或无权访问")
        await repo.delete(record)
        await session.commit()
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
    """发送消息并获取 Agent 回复（持久化对话历史）。"""
    start = time.perf_counter()

    async with get_session() as session:
        selected_model = await _resolve_request_model(
            req.model_id, current_user.id, session
        )
        selected_skills = await _resolve_request_skills(
            req.skill_ids, current_user.id, session
        )
        skill_context = SkillRuntimeContext.from_profiles(selected_skills)
        skill_payload = _selected_skill_payload(selected_skills)
        # 获取或创建会话
        conv_id = None
        is_new = False
        if req.conversation_id:
            conv = await get_conversation(session, uuid.UUID(req.conversation_id))
            if not conv or conv.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found",
                )
            conv_id = conv.id
        else:
            conv = await create_conversation(session, current_user.id)
            conv_id = conv.id
            is_new = True

        # 保存用户消息
        user_message = await add_message(
            session,
            conv_id,
            "user",
            req.query,
            image=req.image,
            metadata_json=json.dumps({"skills": skill_payload}, ensure_ascii=False),
        )
        await session.commit()
        user_message_id = user_message.id

        # 加载对话历史（情景记忆压缩：有 summary 时 = 摘要+最近N轮，否则完整历史）
        db_history = await get_compressed_history(session, conv_id)
        knowledge_base_ids, knowledge_catalog = await _load_knowledge_scope(
            session, current_user.id
        )

    # ── 图片裁决（OCR 回退只做一次）────────────────────────────────────────────
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
                logger.warning("[chat/send] OCR 失败，降级为无图：%s", _ocr_exc)
                effective_query = f"{req.query}\n\n[图片识别服务不可用，已忽略图片内容]"

    # =====================================================================
    # 调用 LangGraph Agent，传入 DB 中的对话历史。
    # AGENT_MODE=multi 作为 deepagents 的兼容别名，由 AgentService 内部路由。
    # =====================================================================
    result: dict[str, Any] = {}
    try:
        from app.services.agent_service import get_agent_service
        from app.llm.client import use_chat_model
        from app.skills.context import use_skill_context

        agent = get_agent_service()
        with use_chat_model(selected_model), use_skill_context(skill_context):
            result = agent.run(
                query=effective_query,
                session_id=str(conv_id),
                history=db_history,          # ← 关键：传入 DB 历史
                user_id=current_user.id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
                image_data=image_for_context,
            )
        answer = result.get("final_answer", "")
    except Exception as exc:
        logger.error("[chat/send] agent error: %s", exc)
        answer = f"处理请求时发生错误: {exc}"

    # 委派持久化（best-effort，复用 Run/Task/AgentRun 表）；无委派事件时自动跳过。
    delegation_run_id = ""
    if result.get("events"):
        from backend.services.delegation_service import persist_delegation

        try:
            delegation_run_id = await persist_delegation(
                get_session,
                conversation_id=conv_id,
                user_id=current_user.id,
                events=result.get("events"),
                goal=effective_query,
                model_id=selected_model.id,
                source_message_id=user_message_id,
            ) or ""
        except Exception as exc:
            logger.warning("[chat/send] delegation persist failed: %s", exc)

    # 兜底：Agent 返回空答案时，用 LLM 直接生成（跳过检索）
    if not answer.strip():
        logger.warning("[chat/send] agent returned empty answer, fallback to direct LLM")
        try:
            from app.llm.client import get_llm_client
            llm = get_llm_client(profile=selected_model)
            from app.services.knowledge_catalog import format_knowledge_catalog

            fallback_messages = [
                {
                    "role": "system",
                    "content": format_knowledge_catalog(knowledge_catalog),
                },
                {
                    "role": "user",
                    "content": (
                    f"请简要回答以下问题（200字以内）：\n\n{req.query}\n\n"
                    "如果问题涉及法律条款，请引用具体法条编号。"
                    ),
                },
            ]
            if skill_context.active:
                fallback_messages.insert(0, {
                    "role": "system",
                    "content": skill_context.render_prompt(),
                })
            fallback_answer = llm.chat_sync(fallback_messages)
            if fallback_answer and fallback_answer.strip():
                answer = fallback_answer
                logger.info("[chat/send] direct LLM fallback succeeded (%d chars)", len(fallback_answer))
            else:
                answer = "抱歉，模型暂时无法生成回答，请稍后重试或简化问题。"
        except Exception as fb_exc:
            logger.error("[chat/send] direct LLM fallback failed: %s", fb_exc)
            answer = "抱歉，处理请求时遇到问题，请稍后重试。"

    # 自动设置会话标题（首次对话时用 LLM 生成语义摘要）
    if is_new and req.query.strip() and answer.strip():
        title = await generate_conversation_title(req.query, answer)
        async with get_session() as session:
            conv = await get_conversation(session, conv_id)
            if conv:
                conv.title = title
                await session.commit()

    elapsed = round(time.perf_counter() - start, 3)

    # 保存助手回复
    async with get_session() as session:
        meta = json.dumps({
            "intent": result.get("intent", ""),
            "run_id": delegation_run_id,
            "steps": result.get("steps", []),
            "sources": result.get("sources", []),
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
        }, ensure_ascii=False)
        await add_message(session, conv_id, "assistant", answer, metadata_json=meta)
        await session.commit()

    return ChatResponse(
        conversation_id=str(conv_id),
        answer=answer,
        run_id=delegation_run_id,
        model_id=selected_model.id,
        model_name=selected_model.name,
        intent=result.get("intent", ""),
        steps=result.get("steps", []),
        sources=result.get("sources", []),
        elapsed_seconds=elapsed,
        skills=skill_payload,
    )


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
    from fastapi.responses import StreamingResponse

    start = time.perf_counter()

    async with get_session() as session:
        selected_model = await _resolve_request_model(
            req.model_id, current_user.id, session
        )
        selected_skills = await _resolve_request_skills(
            req.skill_ids, current_user.id, session
        )
        skill_context = SkillRuntimeContext.from_profiles(selected_skills)
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
        else:
            conv = await create_conversation(session, current_user.id)
            conv_id = conv.id
            is_new = True

        user_message = await add_message(
            session,
            conv_id,
            "user",
            req.query,
            image=req.image,
            metadata_json=json.dumps({"skills": skill_payload}, ensure_ascii=False),
        )
        await session.commit()
        user_message_id = user_message.id
        db_history = await get_compressed_history(session, conv_id)
        knowledge_base_ids, knowledge_catalog = await _load_knowledge_scope(
            session, current_user.id
        )

    # ── 图片裁决（deep / single 共用）──────────────────────────────────────────
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

    from app.services.agent_service import AgentService

    # 深度研究开关（按请求选择）→ 强制 DeepAgents；全局 AGENT_MODE=deepagents 同样启用。
    # AGENT_MODE=multi 作为 deepagents 的兼容别名；auto 仍按请求判断是否多智能体——
    # 均路由到 DeepAgents（2026-08-26 阶段 5，Orchestrator 已退役）。
    use_deep = (
        cfg.AGENT_MODE == "deepagents"
        or cfg.AGENT_MODE == "multi"
        or bool(req.deep_research)
        or (
            cfg.AGENT_MODE == "auto"
            and AgentService._should_use_multi(effective_query, db_history)
        )
    )
    # 通过 use_dynamic 新增轻量动态 Agent（auto 普通问题 / AGENT_MODE=dynamic）
    # 模型每轮自行决定直接回答 / 检索 / 调工具，简单问题不走复杂流程。
    use_dynamic = (
        not use_deep
        and cfg.AGENT_MODE != "single"
        and cfg.AGENT_MODE in ("dynamic", "auto")
    )
    agent_mode = "deepagents" if use_deep else ("dynamic" if use_dynamic else "single")

    async def _event_gen_inner():
        from app.services.agent_service import get_agent_service
        from app.llm.client import get_llm_client, use_chat_model
        from app.skills.context import use_skill_context

        loop = asyncio.get_event_loop()
        agent = get_agent_service()

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
        })

        if use_deep:
            # ── DeepAgents 路径：主 Agent + task → SubAgent（同步 + 状态透传）─
            import queue as _q
            from app.agents.progress import ProgressProjector
            from app.agents.events import use_event_sink

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

            # 统一事件流 → orchestrator 时代 SSE 协议（阶段 5）：委派树/工具时间线
            # 由前端现有任务面板与 AgentActivity 直接消费，无需新组件。
            # 映射规则见 delegation_service.bridge_delegation_event（可单测）。
            from backend.services.delegation_service import bridge_delegation_event

            _panel_sent = {"v": False}

            def _bridge_event(ev: dict) -> None:
                try:
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
                with use_chat_model(selected_model), use_skill_context(skill_context), \
                        use_event_sink(_bridge_event):
                    return agent._run_deep(
                        req.query,
                        session_id=str(conv_id),
                        history=db_history,
                        user_id=current_user.id,
                        knowledge_base_ids=knowledge_base_ids,
                        knowledge_catalog=knowledge_catalog,
                        on_step=_deep_status,
                        # 阶段 6：artifact（推理 / 工具调用 / 工具返回 / 检索片段）
                        # 经统一事件流 sink（_bridge_event → bridge_delegation_event）
                        # 实时下发到当前会话，无需 on_artifact 回调二次透传。
                        on_artifact=None,
                    )

            # 注意：此处不能用 `start` 命名 —— _event_gen_inner 内任何赋值都会
            # 把 start 变成局部变量，遮蔽外层函数的 start（单 Agent 分支的
            # elapsed 计算依赖它），导致 UnboundLocalError（实测 bug）
            deep_start = time.perf_counter()
            deep_future = loop.run_in_executor(None, _run_deep_in_thread)
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
            # 阶段 6：把本轮实时收到的 status（思考/工具/委派步骤）与 artifact
            # 交付物随 done 事件与 meta 持久化，刷新会话后仍可恢复完整轨迹。
            step_objs: list[dict] = collected_steps
            deep_artifacts: list[dict] = deep_result.get("artifacts") or []
            elapsed = round(time.perf_counter() - deep_start, 3)

            # 落库（与单 Agent 分支一致的 metadata 结构）
            try:
                async with get_session() as session:
                    meta = {
                        "intent": "deepagents",
                        "agent_mode": agent_mode,
                        "steps": step_objs,
                        "artifacts": deep_artifacts,
                        "progress_summaries": deep_progress_summaries,
                        "model_id": selected_model.id,
                        "model_name": selected_model.name,
                        "skills": skill_payload,
                    }
                    await add_message(
                        session, conv_id, "assistant", answer,
                        metadata_json=json.dumps(meta, ensure_ascii=False)
                    )
                    await session.commit()
            except Exception as exc:
                logger.warning("[chat/stream] deepagents persist failed: %s", exc)

            yield _sse({
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
                "model_id": selected_model.id,
                "model_name": selected_model.name,
                "skills": skill_payload,
            })

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

                asyncio.get_event_loop().create_task(_gen_title_deep())
            return

        # ── 轻量动态 Agent 路径（auto 普通问题 / AGENT_MODE=dynamic）──
        # 在 executor 线程跑动态 Agent（模型通过函数调用自行决定
        # 调工具 / 检索 / 直接回答）；通过安全队列把每一步
        # 状态实时桥接回事件循环，SSE 逐步推给前端。
        from app.agents.progress import ProgressProjector
        import queue as _dyn_queue
        _dyn_status_queue: "_dyn_queue.Queue" = _dyn_queue.Queue()
        _dyn_collected_steps: list[dict] = []
        _dyn_collected_artifacts: list[dict] = []
        _dyn_progress: list[dict] = []
        _dyn_projector = ProgressProjector()

        def _dyn_status(step: str, detail: str = ""):
            ev = {"step": step, "detail": detail}
            _dyn_collected_steps.append(ev)
            _dyn_status_queue.put({"type": "status", **ev})
            try:
                progress = _dyn_projector.feed(step, detail)
                if progress:
                    progress["created_at"] = time.time()
                    _dyn_progress.append(dict(progress))
                    _dyn_status_queue.put({"type": "progress_summary", **progress})
            except Exception:
                pass

        def _dyn_artifact(ev: dict):
            _dyn_collected_artifacts.append(dict(ev))
            _dyn_status_queue.put({"type": "artifact", **ev})

        def _run_dynamic_in_thread():
            with use_chat_model(selected_model), use_skill_context(skill_context):
                return agent._run_dynamic(
                    effective_query,
                    session_id=str(conv_id),
                    history=db_history,
                    user_id=current_user.id,
                    knowledge_base_ids=knowledge_base_ids,
                    knowledge_catalog=knowledge_catalog,
                    image_data=image_for_context,
                    on_step=_dyn_status,
                    on_artifact=_dyn_artifact,
                )

        _dyn_start = time.perf_counter()
        _dyn_future = loop.run_in_executor(None, _run_dynamic_in_thread)
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

        # 落库助手回复（含步骤/进度/中间产出）
        try:
            async with get_session() as session:
                meta = json.dumps({
                    "intent": "dynamic",
                    "agent_mode": agent_mode,
                    "sources": dyn_result.get("sources") or [],
                    "steps": _dyn_collected_steps,
                    "progress_summaries": _dyn_progress,
                    "artifacts": _dyn_collected_artifacts,
                    "model_id": selected_model.id,
                    "model_name": selected_model.name,
                    "skills": skill_payload,
                }, ensure_ascii=False)
                await add_message(session, conv_id, "assistant", answer, metadata_json=meta)
                await session.commit()
        except Exception as exc:
            logger.warning("[chat/stream] dynamic persist failed: %s", exc)

        yield _sse({
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
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
        })

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

            asyncio.get_event_loop().create_task(_gen_title_dynamic())
        return

        # ── 单 Agent 路径（带实时思考过程透出）─────────────────────────────────
        # 1. 同步编排(检索/工具)在 executor 线程跑, 通过线程安全队列把每一步的
        #    状态实时桥接回事件循环, SSE 逐步推给前端 — 不再是黑盒等待。
        import queue as _queue
        step_queue: "_queue.Queue" = _queue.Queue()
        _SENTINEL = object()
        # 收集本轮全部状态步骤，随 meta 落库（历史加载时恢复思考过程）
        collected_steps: list[dict] = []
        # 收集本轮中间产出（检索片段/工具结果/思维链），随 meta 落库
        collected_artifacts: list[dict] = []
        from app.agents.progress import ProgressProjector
        single_projector = ProgressProjector()
        # 收集本轮工作日志（进度摘要），随 meta 落库，历史回放时恢复
        single_progress: list[dict] = []

        def _on_step(step: str, detail: str = ""):
            ev = {"step": step, "detail": detail}
            collected_steps.append(ev)
            step_queue.put({"type": "status", **ev})
            try:
                progress = single_projector.feed(step, detail)
                if progress:
                    progress["created_at"] = time.time()
                    single_progress.append(dict(progress))
                    step_queue.put({"type": "progress_summary", **progress})
            except Exception:
                pass

        def _on_artifact(ev: dict):
            collected_artifacts.append(dict(ev))
            step_queue.put({"type": "artifact", **ev})

        def _prepare():
            try:
                with use_chat_model(selected_model), use_skill_context(skill_context):
                    return agent.prepare_context(
                        effective_query,
                        db_history,
                        user_id=current_user.id,
                        knowledge_base_ids=knowledge_base_ids,
                        knowledge_catalog=knowledge_catalog,
                        image_data=image_for_context,
                        on_step=_on_step,
                        on_artifact=_on_artifact,
                    )
            finally:
                step_queue.put(_SENTINEL)

        prepare_future = loop.run_in_executor(None, _prepare)

        # 边等 prepare 边 drain 队列, 实时推状态
        ctx = None
        prepare_error: Optional[Exception] = None
        while True:
            try:
                ev = await loop.run_in_executor(None, step_queue.get, True, 0.1)
            except Exception:
                ev = None  # queue.Empty 超时 → 检查 future 是否完成
            if ev is _SENTINEL:
                break
            if ev is not None:
                yield _sse(ev)
            if prepare_future.done() and step_queue.empty():
                break
        # drain 残留
        while not step_queue.empty():
            ev = step_queue.get_nowait()
            if ev is not _SENTINEL:
                yield _sse(ev)

        try:
            ctx = prepare_future.result()
        except Exception as exc:
            prepare_error = exc

        if prepare_error is not None or ctx is None:
            logger.error("[chat/stream] prepare_context error: %s", prepare_error)
            yield _sse({"type": "error", "detail": f"检索失败: {prepare_error}"})
            return

        # 1b. 主协程里 async 反查 file_id 并回填到引用(executor 线程里
        #     asyncio.run 会与主线程 async engine 冲突, 故在此统一补齐)。
        try:
            from app.graph.nodes import lookup_file_ids_async
            pairs = [
                (s.get("knowledge_base_id", ""), s.get("title", ""))
                for s in ctx["sources"]
                if s.get("type") in ("kb", "knowledge_graph")
            ]
            fid_map = await lookup_file_ids_async(pairs)
            for s in ctx["sources"]:
                key = (s.get("knowledge_base_id", ""), s.get("title", ""))
                if key in fid_map:
                    s["file_id"] = fid_map[key]
        except Exception as exc:
            logger.warning("[chat/stream] file_id backfill failed: %s", exc)

        # 2. 流式生成（含思维链实时透出 + 空响应兜底）
        answer_parts: list[str] = []
        _think_buf: list[str] = []        # 思考增量缓冲（合并后落库，避免碎片化）
        _think_id = f"think-{conv_id}"    # 流式思考的事件 id（前端按 id 追加内容）
        try:
            llm = get_llm_client(profile=selected_model)
            async for ev in llm.chat_stream_events(ctx["messages"]):
                if ev["type"] == "thought":
                    # DeepSeek 类模型的 reasoning_content：实时推给前端展示思考过程
                    _think_buf.append(ev["text"])
                    yield _sse({
                        "type": "artifact",
                        "id": _think_id,
                        "kind": "thought",
                        "stage": "generate",
                        "title": "思考",
                        "content": ev["text"],
                        "streaming": True,
                    })
                else:
                    answer_parts.append(ev["text"])
                    yield _sse({"type": "delta", "content": ev["text"]})
        except Exception as exc:
            logger.error("[chat/stream] generation error: %s", exc)
            yield _sse({"type": "error", "detail": f"生成失败: {exc}"})
            return

        # 思考流结束标记（前端把"思考"卡片标记为完成）
        if _think_buf:
            yield _sse({
                "type": "artifact",
                "id": _think_id,
                "kind": "thought",
                "stage": "generate",
                "streaming": False,
            })
            collected_artifacts.append({
                "kind": "thought",
                "stage": "generate",
                "title": "思考",
                "content": "".join(_think_buf)[:2000],
            })

        # 兜底：流式返回空时用同步调用重试（API 偶发空响应，尤其法律类内容）
        if not answer_parts:
            logger.warning("[chat/stream] stream returned 0 tokens, falling back to sync")
            try:
                fallback_answer = await loop.run_in_executor(
                    None, llm.chat_sync, ctx["messages"]
                )
            except Exception as fb_exc:
                logger.error("[chat/stream] sync fallback also failed: %s", fb_exc)
                yield _sse({"type": "error", "detail": "模型未返回有效回答，请重试"})
                return

            if fallback_answer and fallback_answer.strip():
                answer_parts = [fallback_answer]
                yield _sse({"type": "delta", "content": fallback_answer})
                logger.info("[chat/stream] sync fallback succeeded (%d chars)", len(fallback_answer))
            else:
                logger.warning("[chat/stream] sync fallback also returned empty")
                yield _sse({"type": "error", "detail": "模型未返回有效回答，请尝试简化问题后重试"})
                return

        answer = "".join(answer_parts).strip()
        elapsed = round(time.perf_counter() - start, 3)

        # 生成完成 → 收尾状态事件：前端据此把"生成回答"步骤标记为完成
        _gen_done = {"step": "generate_done", "detail": f"回答生成完成（{len(answer)} 字符）"}
        collected_steps.append(_gen_done)
        yield _sse({"type": "status", **_gen_done})
        _gen_progress = single_projector.feed("generate_done", _gen_done["detail"])
        if _gen_progress:
            _gen_progress["created_at"] = time.time()
            single_progress.append(dict(_gen_progress))
            yield _sse({"type": "progress_summary", **_gen_progress})

        # 3. 落库助手回复(含引用)
        try:
            async with get_session() as session:
                meta = json.dumps({
                    "intent": ctx["intent"],
                    "agent_mode": agent_mode,
                    "sources": ctx["sources"],
                    "steps": collected_steps,
                    "progress_summaries": single_progress,
                    "artifacts": collected_artifacts,
                    "model_id": selected_model.id,
                    "model_name": selected_model.name,
                    "skills": skill_payload,
                }, ensure_ascii=False)
                await add_message(session, conv_id, "assistant", answer, metadata_json=meta)
                await session.commit()
        except Exception as exc:
            logger.warning("[chat/stream] persist answer failed: %s", exc)

        yield _sse({
            "type": "done",
            "sources": ctx["sources"],
            "intent": ctx["intent"],
            "agent_mode": agent_mode,
            "steps": collected_steps,
            "progress_summaries": single_progress,
            "artifacts": collected_artifacts,
            "elapsed_seconds": elapsed,
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
        })

        # 4. 新会话标题生成 — 在 done 之后的后台协程里做，不阻塞 SSE 流。
        #    LLM 生成语义化标题(非原文截取)，前端下次轮询会话列表时即可见。
        if is_new and answer:
            async def _gen_title():
                try:
                    title = await generate_conversation_title(req.query, answer)
                    async with get_session() as session:
                        c = await get_conversation(session, conv_id)
                        if c:
                            c.title = title
                            await session.commit()
                    logger.info("[chat/stream] title generated: %s", title)
                except Exception as exc:
                    logger.warning("[chat/stream] title gen failed: %s", exc)

            asyncio.get_event_loop().create_task(_gen_title())

    async def event_gen():
        """Ensure an interrupted stream cannot leave a run permanently active,
        and a terminated turn is NOT persisted to history.

        2026-08-21：用户停止生成 / 客户端断开 → 本轮不入历史：
        - assistant 消息在生成器内保存，流中断后不会执行（天然不落库）；
        - 用户消息在开流前已保存（user_message_id），终止时删除；
        - 若是本轮新建的会话（is_new），整会话删除，避免留下空壳。
        """
        completed_normally = False
        try:
            async for chunk in _event_gen_inner():
                yield chunk
            completed_normally = True
        finally:
            if not completed_normally:
                # 注意：此处处于任务取消状态（CancelledError 处理中），
                # 直接 await 会立即再次抛出 CancelledError —— 必须用
                # asyncio.shield 让清理协程在后台独立完成。
                async def _cleanup_terminated_turn():
                    try:
                        async with get_session() as session:
                            from backend.services.chat_service import (
                                delete_conversation,
                                delete_message,
                            )

                            if is_new:
                                await delete_conversation(
                                    session, conv_id, current_user.id
                                )
                            else:
                                await delete_message(session, user_message_id)
                                await session.commit()
                        logger.info(
                            "[chat/stream] turn terminated, not persisted "
                            "(conv=%s new=%s)", conv_id, is_new
                        )
                    except Exception as exc:
                        logger.warning(
                            "[chat/stream] terminated-turn cleanup failed: %s", exc
                        )

                cleanup_task = asyncio.create_task(_cleanup_terminated_turn())
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    # shield keeps the cleanup coroutine alive on the app loop.
                    pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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


@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """删除整个会话及其所有消息（级联删除）。验证会话归属当前用户。"""
    from backend.services.chat_service import delete_conversation

    async with get_session() as session:
        deleted = await delete_conversation(
            session, uuid.UUID(conversation_id), current_user.id
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
    return {"conversation_id": conversation_id, "deleted": True}
