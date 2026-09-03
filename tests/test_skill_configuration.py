"""Skill catalog, request scoping and tool-permission regression tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.exceptions import ToolExecutionError
from app.skills.catalog import SkillProfile, get_builtin_skill, list_builtin_skills
from app.skills.context import SkillRuntimeContext, use_skill_context
from app.tools.registry import ToolDefinition, ToolRegistry
from backend.server.routers.chat_router import ChatRequest
from backend.services.skill_config_service import (
    SkillConfigValidationError,
    validate_tool_names,
)


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=name, fn=lambda: name)


def test_builtin_skill_catalog_has_safe_basics():
    skills = list_builtin_skills()
    assert len(skills) >= 5
    assert len({skill.id for skill in skills}) == len(skills)
    assert get_builtin_skill("builtin:knowledge-research") is not None
    assert get_builtin_skill("builtin:web-research").tool_names == ("web_search",)


def test_skill_runtime_merges_prompts_and_tool_permissions():
    runtime = SkillRuntimeContext.from_profiles([
        SkillProfile(
            id="one",
            name="研究",
            description="",
            instructions="先核验来源",
            tool_names=("web_search",),
        ),
        SkillProfile(
            id="two",
            name="计算",
            description="",
            instructions="展示计算过程",
            tool_names=("calculator", "web_search"),
        ),
    ])
    assert runtime.allowed_tool_names == ("web_search", "calculator")
    assert runtime.allows_tool("calculator")
    assert not runtime.allows_tool("datetime_tool")
    assert "先核验来源" in runtime.render_prompt()
    assert "展示计算过程" in runtime.render_prompt()


def test_selected_skills_filter_schema_and_block_execution():
    registry = ToolRegistry()
    registry.register(_tool("web_search"))
    registry.register(_tool("calculator"))
    selected = SkillProfile(
        id="selected",
        name="联网",
        description="",
        instructions="使用联网搜索",
        tool_names=("web_search",),
    )
    with use_skill_context([selected]):
        assert registry.list_names() == ["web_search"]
        assert registry.invoke("web_search") == "web_search"
        with pytest.raises(ToolExecutionError, match="not allowed"):
            registry.invoke("calculator")
    assert set(registry.list_names()) == {"web_search", "calculator"}


def test_custom_skill_tool_names_must_be_registered():
    assert validate_tool_names(["calculator", "calculator"]) == ("calculator",)
    with pytest.raises(SkillConfigValidationError, match="未注册工具"):
        validate_tool_names(["definitely_missing_tool"])


def test_chat_request_limits_skill_selection():
    request = ChatRequest(query="hello", skill_ids=["a", "b", "c"])
    assert request.skill_ids == ["a", "b", "c"]
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", skill_ids=["a", "b", "c", "d"])


def test_skill_repository_is_owner_scoped():
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "backend/repositories/skill_config_repository.py"
    ).read_text(encoding="utf-8")
    assert "CustomSkillConfig.owner_id == owner_id" in source
    assert "CustomSkillConfig.id == record_id" in source
