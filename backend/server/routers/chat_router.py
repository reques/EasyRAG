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
            metadata_json=json.dumps({"skills": skill_payload}, ensure_ascii=False),
        )
        await session.commit()
        user_message_id = user_message.id

        # 加载对话历史（情景记忆压缩：有 summary 时 = 摘要+最近N轮，否则完整历史）
        db_history = await get_compressed_history(session, conv_id)
        knowledge_base_ids, knowledge_catalog = await _load_knowledge_scope(
            session, current_user.id
        )

    from app.services.agent_service import AgentService

    use_multi = cfg.AGENT_MODE == "multi" or (
        cfg.AGENT_MODE == "auto"
        and AgentService._should_use_multi(req.query, db_history)
    )
    multi_run_id: Optional[uuid.UUID] = None
    if use_multi:
        from backend.services.agent_run_service import create_run

        async with get_session() as session:
            run = await create_run(
                session,
                conversation_id=conv_id,
                user_id=current_user.id,
                source_message_id=user_message_id,
                goal=req.query,
                model_id=selected_model.id,
            )
            await session.commit()
            multi_run_id = run.id

    # =====================================================================
    # 调用 LangGraph Agent，传入 DB 中的对话历史
    # =====================================================================
    result: dict[str, Any] = {}
    try:
        from app.services.agent_service import get_agent_service
        from app.llm.client import use_chat_model
        from app.skills.context import use_skill_context

        agent = get_agent_service()
        with use_chat_model(selected_model), use_skill_context(skill_context):
            result = agent.run(
                query=req.query,
                session_id=str(conv_id),
                history=db_history,          # ← 关键：传入 DB 历史
                user_id=current_user.id,
                knowledge_base_ids=knowledge_base_ids,
                knowledge_catalog=knowledge_catalog,
            )
        answer = result.get("final_answer", "")
    except Exception as exc:
        logger.error("[chat/send] agent error: %s", exc)
        answer = f"处理请求时发生错误: {exc}"

    if multi_run_id:
        from backend.services.agent_run_service import (
            create_tasks as persist_tasks,
            finalize_run,
            finish_task,
            start_pending_tasks,
        )

        try:
            async with get_session() as session:
                task_details = result.get("task_details", [])
                if result.get("intent") == "multi_agent" and task_details:
                    await persist_tasks(
                        session, multi_run_id, task_details, selected_model.id
                    )
                    await start_pending_tasks(session, multi_run_id)
                    worker_reports = result.get("worker_reports", [])
                    for report in worker_reports:
                        await finish_task(
                            session,
                            multi_run_id,
                            report.get("task_id", ""),
                            worker_status=report.get("status", "error"),
                            output_summary=report.get("summary", ""),
                            error_summary=report.get("error", ""),
                        )
                    all_failed = bool(worker_reports) and not any(
                        report.get("status") in {"done", "done_with_concerns"}
                        for report in worker_reports
                    )
                    await finalize_run(
                        session,
                        multi_run_id,
                        status="failed" if all_failed else "completed",
                        execution_mode=result.get("execution_mode", ""),
                        error_summary="所有子任务执行失败" if all_failed else "",
                    )
                elif result:
                    await finalize_run(
                        session,
                        multi_run_id,
                        status="completed",
                        execution_mode="degenerate",
                    )
                else:
                    await finalize_run(
                        session,
                        multi_run_id,
                        status="failed",
                        error_summary="多智能体执行未返回结果",
                    )
                await session.commit()
        except Exception as exc:
            logger.warning("[chat/send] run %s persist failed: %s", multi_run_id, exc)

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
            "run_id": str(multi_run_id) if multi_run_id else "",
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
        run_id=str(multi_run_id) if multi_run_id else "",
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
            metadata_json=json.dumps({"skills": skill_payload}, ensure_ascii=False),
        )
        await session.commit()
        user_message_id = user_message.id
        db_history = await get_compressed_history(session, conv_id)
        knowledge_base_ids, knowledge_catalog = await _load_knowledge_scope(
            session, current_user.id
        )

    # Decide and persist the multi-agent run before opening the stream so the
    # first event can expose a durable run_id to the client.
    from app.services.agent_service import AgentService

    use_multi = cfg.AGENT_MODE == "multi" or (
        cfg.AGENT_MODE == "auto"
        and AgentService._should_use_multi(req.query, db_history)
    )
    multi_run_id: Optional[uuid.UUID] = None
    if use_multi:
        from backend.services.agent_run_service import create_run

        async with get_session() as session:
            run = await create_run(
                session,
                conversation_id=conv_id,
                user_id=current_user.id,
                source_message_id=user_message_id,
                goal=req.query,
                model_id=selected_model.id,
            )
            await session.commit()
            multi_run_id = run.id

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
            "run_id": str(multi_run_id) if multi_run_id else "",
            "model_id": selected_model.id,
            "model_name": selected_model.name,
            "skills": skill_payload,
        })

        if use_multi:
            # ── 多智能体路径：Orchestrator + 状态实时透传 ────────────────────
            try:
                from app.agents.orchestrator import get_orchestrator
                from backend.services.agent_run_service import (
                    create_tasks as persist_tasks,
                    finalize_run,
                    finish_task,
                    start_task,
                )

                orchestrator = get_orchestrator()

                # 状态实时透传：orchestrator 在 executor 线程跑，通过线程安全队列
                # 把每一步状态桥接回事件循环, SSE 逐步推给前端 — 复杂任务不再黑盒等待。
                import queue as _q

                status_queue: "_q.Queue" = _q.Queue()
                status_list: list[dict] = []  # 完整状态列表，供落库
                _ORCH_SENTINEL = object()

                def _on_status(step: str, detail: str, task_id: str = ""):
                    ev = {"step": step, "detail": detail, "task_id": task_id}
                    status_list.append(ev)
                    status_queue.put({"type": "status", **ev})

                def _on_tasks(tasks: list):
                    # 拆解完成 → 推送待办清单，前端渲染侧边任务进度面板
                    status_queue.put({"type": "sub_tasks", "tasks": tasks})

                # 收集 worker 中间产出，落库时与汇总结果拼接成完整答案
                worker_outputs: list[dict] = []

                def _on_worker_done(report):
                    # Worker 完成 → 推送子任务产出（边执行边输出中间结果）
                    content = report.detail or report.summary or ""
                    if report.status == "error":
                        content = f"⚠️ 子任务 {report.task_id} 执行失败：{report.error or '未知错误'}"
                    if not content.strip():
                        content = f"（子任务 {report.task_id} 无产出）"
                    worker_outputs.append({
                        "task_id": report.task_id,
                        "worker": report.worker_name,
                        "content": content,
                    })
                    status_queue.put({
                        "type": "worker_output",
                        "task_id": report.task_id,
                        "worker": report.worker_name,
                        "status": report.status,
                        "error": report.error or "",
                        "content": content,
                    })

                async def _prepare_orchestrator_event(ev: dict) -> dict:
                    """Persist one lifecycle transition, then expose it to SSE."""
                    ev_type = ev.get("type", "status")
                    try:
                        async with get_session() as session:
                            if ev_type == "sub_tasks":
                                await persist_tasks(
                                    session,
                                    multi_run_id,
                                    ev.get("tasks", []),
                                    selected_model.id,
                                )
                            elif ev_type == "worker_output":
                                await finish_task(
                                    session,
                                    multi_run_id,
                                    ev.get("task_id", ""),
                                    worker_status=ev.get("status", "error"),
                                    output_summary=ev.get("content", ""),
                                    error_summary=ev.get("error", ""),
                                )
                            elif ev.get("step") == "task_started":
                                await start_task(
                                    session,
                                    multi_run_id,
                                    ev.get("task_id", ""),
                                )
                            await session.commit()
                    except Exception as exc:
                        logger.warning(
                            "[chat/stream] run %s lifecycle persist failed: %s",
                            multi_run_id,
                            exc,
                        )

                    if ev_type == "worker_output":
                        return {
                            "type": "worker_output",
                            "run_id": str(multi_run_id),
                            "task_id": ev["task_id"],
                            "worker": ev["worker"],
                            "status": ev.get("status", ""),
                            "content": ev["content"],
                        }
                    if ev_type == "sub_tasks":
                        return {
                            "type": "sub_tasks",
                            "run_id": str(multi_run_id),
                            "tasks": ev["tasks"],
                        }
                    if ev.get("step") == "tool_call":
                        return {
                            "type": "tool_call",
                            "run_id": str(multi_run_id),
                            "task_id": ev.get("task_id", ""),
                            "detail": ev.get("detail", ""),
                        }
                    return {
                        "type": "status",
                        "run_id": str(multi_run_id),
                        "task_id": ev.get("task_id", ""),
                        "step": ev["step"],
                        "detail": ev["detail"],
                    }

                # 在 executor 里跑 orchestrator（同步 LLM 调用）
                # return_synthesize_payload=True：synthesize 交给主事件循环流式生成
                def _run_orch():
                    try:
                        with use_chat_model(selected_model), use_skill_context(skill_context):
                            return orchestrator.run(
                                req.query,
                                history=db_history,
                                status_callback=_on_status,
                                worker_done_callback=_on_worker_done,
                                tasks_callback=_on_tasks,
                                return_synthesize_payload=True,
                                knowledge_base_ids=knowledge_base_ids,
                                knowledge_catalog=knowledge_catalog,
                            )
                    finally:
                        status_queue.put(_ORCH_SENTINEL)

                orch_future = loop.run_in_executor(None, _run_orch)

                # 边等 orchestrator 边 drain 队列, 实时推状态和子任务产出
                result = None
                while True:
                    try:
                        ev = await loop.run_in_executor(None, status_queue.get, True, 0.1)
                    except Exception:
                        ev = None  # queue.Empty 超时 → 检查 future
                    if ev is _ORCH_SENTINEL:
                        break
                    if ev is not None:
                        yield _sse(await _prepare_orchestrator_event(ev))
                    if orch_future.done() and status_queue.empty():
                        break
                # drain 残留
                while not status_queue.empty():
                    ev = status_queue.get_nowait()
                    if ev is _ORCH_SENTINEL:
                        continue
                    yield _sse(await _prepare_orchestrator_event(ev))

                try:
                    result = orch_future.result()
                except Exception as exc:
                    logger.error("[chat/stream] orchestrator future error: %s", exc)
                    async with get_session() as session:
                        await finalize_run(
                            session,
                            multi_run_id,
                            status="failed",
                            error_summary=str(exc),
                        )
                        await session.commit()
                    yield _sse({"type": "status", "step": "fallback", "detail": "多智能体失败，回退单 Agent"})
                    result = None

                # 拆解器判定单一意图 → 回退单 Agent 快速路径（走下面的 single 分支）
                if result and result.get("degenerate_to_single"):
                    async with get_session() as session:
                        await finalize_run(
                            session,
                            multi_run_id,
                            status="completed",
                            execution_mode="degenerate",
                        )
                        await session.commit()
                    yield _sse({"type": "status", "step": "fallback", "detail": "单一意图，走快速路径"})
                elif result:
                    # ── 流式汇总：在主事件循环里用 chat_stream 逐 token 整合 ──
                    payload = result.get("synthesize_payload")
                    answer = ""  # 汇总部分
                    if payload:
                        reports = payload["reports"]
                        # 多任务：LLM 流式整合；单任务成功：直接用该任务产出（已推过，避免重复）
                        ok_reports = [r for r in reports if r.ok()]
                        if len(ok_reports) == 1 and not payload["final_inst"]:
                            answer = ok_reports[0].detail or ok_reports[0].summary
                            if answer:
                                yield _sse({"type": "delta", "content": answer})
                        elif not ok_reports:
                            answer = "所有子任务执行失败，无法生成回答。"
                            yield _sse({"type": "delta", "content": answer})
                        else:
                            combined = "\n\n".join(
                                f"## {r.task_id} ({r.worker_name})\n{r.detail or r.summary}"
                                for r in ok_reports
                            )
                            prompt = (
                                f"用户原始查询：{payload['query']}\n\n"
                                f"各子任务产出：\n{combined}\n\n"
                                f"汇总要求：{payload['final_inst'] or '综合各子任务结果，给出完整、连贯的回答。'}"
                            )
                            if skill_context.active:
                                prompt = skill_context.render_prompt() + "\n\n" + prompt
                            try:
                                llm = get_llm_client(profile=selected_model)
                                parts: list[str] = []
                                async for chunk in llm.chat_stream(
                                    [{"role": "user", "content": prompt}]
                                ):
                                    parts.append(chunk)
                                    yield _sse({"type": "delta", "content": chunk})
                                answer = "".join(parts).strip()
                                if not answer:
                                    # 流式整合空响应 → 回退用 combined 作为答案
                                    logger.warning("[chat/stream] synthesize stream empty, fallback combined")
                                    answer = combined
                                    yield _sse({"type": "delta", "content": combined})
                            except Exception as exc:
                                logger.error("[chat/stream] synthesize stream failed: %s", exc)
                                answer = combined
                                yield _sse({"type": "delta", "content": combined})
                    else:
                        # 无 payload（兼容旧路径，理论上 return_synthesize_payload=True 时不会到这）
                        answer = result.get("final_answer", "")
                        if answer:
                            yield _sse({"type": "delta", "content": answer})

                    # 子任务产出作为独立过程数据保存；聊天正文只持久化最终回答。
                    full_answer = answer

                    elapsed = round(time.perf_counter() - start, 3)

                    # 落库
                    try:
                        async with get_session() as session:
                            meta = json.dumps({
                                "intent": result.get("intent", "multi_agent"),
                                "run_id": str(multi_run_id),
                                "worker_outputs": worker_outputs,
                                "sources": result.get("sources", []),
                                # status_list 是 {step, detail} 对象数组，前端可直接渲染；
                                # result["steps"] 是 orchestrator 内部字符串日志，格式不兼容
                                "steps": status_list,
                                "execution_mode": result.get("execution_mode", ""),
                                "model_id": selected_model.id,
                                "model_name": selected_model.name,
                                "skills": skill_payload,
                            }, ensure_ascii=False)
                            await add_message(session, conv_id, "assistant", full_answer, metadata_json=meta)
                            await session.commit()
                    except Exception as exc:
                        logger.warning("[chat/stream] multi persist failed: %s", exc)

                    run_status = "completed"
                    if payload and not any(report.ok() for report in payload["reports"]):
                        run_status = "failed"
                    async with get_session() as session:
                        await finalize_run(
                            session,
                            multi_run_id,
                            status=run_status,
                            execution_mode=result.get("execution_mode", ""),
                            error_summary=(
                                "所有子任务执行失败" if run_status == "failed" else ""
                            ),
                        )
                        await session.commit()

                    yield _sse({
                        "type": "done",
                        "run_id": str(multi_run_id),
                        "sources": result.get("sources", []),
                        "intent": result.get("intent", "multi_agent"),
                        "steps": status_list,
                        "elapsed_seconds": elapsed,
                        "execution_mode": result.get("execution_mode", ""),
                        "model_id": selected_model.id,
                        "model_name": selected_model.name,
                        "skills": skill_payload,
                    })

                    # 新会话标题后台生成
                    if is_new and answer:
                        async def _gen_title_multi():
                            try:
                                title = await generate_conversation_title(req.query, answer)
                                async with get_session() as session:
                                    c = await get_conversation(session, conv_id)
                                    if c:
                                        c.title = title
                                        await session.commit()
                            except Exception as exc:
                                logger.warning("[chat/stream] title gen failed: %s", exc)

                        asyncio.get_event_loop().create_task(_gen_title_multi())

                    return
            except Exception as exc:
                logger.error("[chat/stream] multi-agent error, fallback single: %s", exc)
                try:
                    from backend.services.agent_run_service import finalize_run

                    async with get_session() as session:
                        await finalize_run(
                            session,
                            multi_run_id,
                            status="failed",
                            error_summary=str(exc),
                        )
                        await session.commit()
                except Exception as persist_exc:
                    logger.warning(
                        "[chat/stream] failed to finalize run %s: %s",
                        multi_run_id,
                        persist_exc,
                    )
                yield _sse({"type": "status", "step": "fallback", "detail": "多智能体失败，回退单 Agent"})
                # 继续走下面的 single 路径

        # ── 单 Agent 路径（带实时思考过程透出）─────────────────────────────────
        # 1. 同步编排(检索/工具)在 executor 线程跑, 通过线程安全队列把每一步的
        #    状态实时桥接回事件循环, SSE 逐步推给前端 — 不再是黑盒等待。
        import queue as _queue
        step_queue: "_queue.Queue" = _queue.Queue()
        _SENTINEL = object()
        # 收集本轮全部状态步骤，随 meta 落库（历史加载时恢复思考过程）
        collected_steps: list[dict] = []

        def _on_step(step: str, detail: str = ""):
            ev = {"step": step, "detail": detail}
            collected_steps.append(ev)
            step_queue.put({"type": "status", **ev})

        def _prepare():
            try:
                with use_chat_model(selected_model), use_skill_context(skill_context):
                    return agent.prepare_context(
                        req.query,
                        db_history,
                        user_id=current_user.id,
                        knowledge_base_ids=knowledge_base_ids,
                        knowledge_catalog=knowledge_catalog,
                        on_step=_on_step,
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

        # 2. 流式生成（含空响应兜底）
        answer_parts: list[str] = []
        try:
            llm = get_llm_client(profile=selected_model)
            async for delta in llm.chat_stream(ctx["messages"]):
                answer_parts.append(delta)
                yield _sse({"type": "delta", "content": delta})
        except Exception as exc:
            logger.error("[chat/stream] generation error: %s", exc)
            yield _sse({"type": "error", "detail": f"生成失败: {exc}"})
            return

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

        # 3. 落库助手回复(含引用)
        try:
            async with get_session() as session:
                meta = json.dumps({
                    "intent": ctx["intent"],
                    "sources": ctx["sources"],
                    "steps": collected_steps,
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
            "steps": collected_steps,
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
        """Ensure an interrupted stream cannot leave a run permanently active."""
        completed_normally = False
        try:
            async for chunk in _event_gen_inner():
                yield chunk
            completed_normally = True
        finally:
            if multi_run_id:
                async def _close_unfinished_run():
                    from backend.services.agent_run_service import finalize_run_if_active

                    async with get_session() as session:
                        changed = await finalize_run_if_active(
                            session,
                            multi_run_id,
                            status="failed" if completed_normally else "cancelled",
                            error_summary=(
                                "流式响应结束但运行未写入终态"
                                if completed_normally
                                else "客户端在运行完成前断开连接"
                            ),
                        )
                        if changed:
                            await session.commit()

                close_task = asyncio.create_task(_close_unfinished_run())
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    # shield keeps the database finalizer alive on the app loop.
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
