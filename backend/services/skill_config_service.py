"""Validation and conversion for user-defined Skills."""
from __future__ import annotations

import json
from typing import Iterable

from app.skills.catalog import SkillProfile
from app.tools.registry import get_tool_registry
from backend.storage.postgres.models_skill_config import CustomSkillConfig


class SkillConfigValidationError(ValueError):
    pass


def validate_tool_names(tool_names: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(name).strip() for name in tool_names if str(name).strip()))
    if len(normalized) > 8:
        raise SkillConfigValidationError("一个 Skill 最多配置 8 个工具")
    known = set(get_tool_registry().list_names(available_only=False))
    unknown = [name for name in normalized if name not in known]
    if unknown:
        raise SkillConfigValidationError(
            "包含未注册工具：" + "、".join(unknown)
        )
    return normalized


def encode_tool_names(tool_names: Iterable[str]) -> str:
    return json.dumps(list(validate_tool_names(tool_names)), ensure_ascii=False)


def decode_tool_names(raw: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError) as exc:
        raise SkillConfigValidationError("Skill 工具配置损坏") from exc
    if not isinstance(value, list):
        raise SkillConfigValidationError("Skill 工具配置必须是列表")
    return validate_tool_names(value)


def profile_from_custom_skill(record: CustomSkillConfig) -> SkillProfile:
    return SkillProfile(
        id=record.public_id,
        name=record.name,
        description=record.description,
        instructions=record.instructions,
        tool_names=decode_tool_names(record.tool_names_json),
        category=record.category,
        icon=record.icon,
        source="custom",
    )
