"""Skill 系统回归测试 — 文件加载 / 三层集合 / 渐进式披露门控 / middleware。

2026-09-04 重构后的契约（参照 Yuxi，见
``docs/plans/2026-09-04-skill-management-refactor-yuxi.md``）：
Skill 来自磁盘 SKILL.md，用户勾选定"有效集合"，模型 ``read_skill`` 后才
解锁工具。
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.exceptions import ToolExecutionError
from app.skills.loader import (
    SkillDefinition,
    SkillLoadError,
    load_skill_directory,
    parse_skill_markdown,
    render_skill_markdown,
)
from app.skills.registry import list_builtin_skills, validate_builtin_dependencies
from app.skills.runtime import (
    SkillRuntimeContext,
    resolve_dependency_closure,
    use_skill_context,
)
from app.tools.registry import ToolDefinition, ToolRegistry
from backend.server.routers.chat_router import ChatRequest
from backend.services.skill_config_service import (
    SkillConfigValidationError,
    slugify,
    validate_slug,
    validate_tool_names,
)


def _tool(name: str, *, public: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name, description=name, fn=lambda: name,
        metadata={"public": True} if public else {},
    )


def _definition(slug: str, *, tools=(), deps=(), body="指令正文") -> SkillDefinition:
    return SkillDefinition(
        slug=slug, name=slug, description=f"{slug} 描述", body=body,
        tool_dependencies=tuple(tools), skill_dependencies=tuple(deps),
    )


def _write_skill(root: Path, slug: str, text: str) -> Path:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return directory


# ── loader：frontmatter 解析与校验 ────────────────────────────────────────

def test_parse_skill_markdown_extracts_frontmatter_and_body():
    definition = parse_skill_markdown("""\
---
name: 联网研究
slug: web-research
description: 联网检索并核验
category: 研究
icon: globe
tool_dependencies: [web_search]
---

## 何时使用

需要时效信息时。
""")
    assert definition.slug == "web-research"
    assert definition.name == "联网研究"
    assert definition.tool_dependencies == ("web_search",)
    assert definition.category == "研究"
    assert "## 何时使用" in definition.body
    # 渐进式披露：正文不进面向前端的序列化
    assert "body" not in definition.to_public_dict()
    assert definition.to_public_dict()["body_available"] is True


def test_slug_defaults_to_name_only_when_name_is_slug_shaped():
    ok = parse_skill_markdown("---\nname: my-skill\ndescription: d\n---\n\nbody\n")
    assert ok.slug == "my-skill"
    # 中文 name 省略 slug → 校验不通过（对齐 Yuxi）
    with pytest.raises(SkillLoadError, match="省略 slug"):
        parse_skill_markdown("---\nname: 联网研究\ndescription: d\n---\n\nbody\n")


@pytest.mark.parametrize("text,match", [
    ("no frontmatter here", "frontmatter"),
    ("---\ndescription: d\n---\n\nbody\n", "name"),
    ("---\nname: a\n---\n\nbody\n", "description"),
    ("---\nname: a\ndescription: d\nslug: Bad_Slug\n---\n\nbody\n", "非法"),
    ("---\nname: a\ndescription: d\nskill_dependencies: [a]\n---\n\nbody\n", "自身"),
])
def test_parse_rejects_invalid_frontmatter(text, match):
    with pytest.raises(SkillLoadError, match=match):
        parse_skill_markdown(text)


def test_tool_dependencies_limited_to_eight():
    tools = ", ".join(f"t{i}" for i in range(9))
    with pytest.raises(SkillLoadError, match="tool_dependencies"):
        parse_skill_markdown(
            f"---\nname: a\ndescription: d\ntool_dependencies: [{tools}]\n---\n\nb\n"
        )


def test_render_and_parse_round_trip():
    """渲染的 SKILL.md 必须能被解析回等价定义（保存路径依赖这个往返）。"""
    text = render_skill_markdown(
        name="我的技能", slug="my-skill", description="用途说明",
        body="## 工作方式\n\n照做。", tool_dependencies=["calculator"],
        category="自定义", icon="wand",
    )
    back = parse_skill_markdown(text, source="personal", owner_id="u1")
    assert (back.slug, back.name, back.description) == ("my-skill", "我的技能", "用途说明")
    assert back.tool_dependencies == ("calculator",)
    assert back.icon == "wand"
    assert back.can_edit is True


def test_load_skill_directory_requires_skill_md(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SkillLoadError, match="SKILL.md"):
        load_skill_directory(tmp_path / "empty")
    directory = _write_skill(tmp_path, "ok-skill", """
        ---
        name: ok-skill
        description: d
        ---

        body
    """)
    assert load_skill_directory(directory).slug == "ok-skill"


# ── registry：内置 Skill 目录 ─────────────────────────────────────────────

def test_builtin_skills_load_from_disk():
    skills = list_builtin_skills()
    slugs = {s.slug for s in skills}
    assert {"knowledge-research", "web-research", "data-analysis",
            "professional-writing", "legal-analysis"} <= slugs
    assert all(s.source == "builtin" and not s.can_edit for s in skills)
    assert all(s.body for s in skills), "内置 Skill 必须有正文"
    by_slug = {s.slug: s for s in skills}
    assert by_slug["web-research"].tool_dependencies == ("web_search",)
    assert by_slug["legal-analysis"].skill_dependencies == ("knowledge-research",)


def test_builtin_dependencies_all_resolvable():
    assert validate_builtin_dependencies() == []


# ── runtime：依赖闭包 ─────────────────────────────────────────────────────

def test_dependency_closure_expands_and_preserves_order():
    catalog = {
        "a": _definition("a", deps=("b",)),
        "b": _definition("b", deps=("c",)),
        "c": _definition("c"),
        "unrelated": _definition("unrelated"),
    }
    assert resolve_dependency_closure(["a"], catalog) == ("a", "b", "c")


def test_dependency_closure_handles_cycles():
    catalog = {"a": _definition("a", deps=("b",)), "b": _definition("b", deps=("a",))}
    assert resolve_dependency_closure(["a"], catalog) == ("a", "b")


def test_dependency_closure_cannot_escape_catalog():
    """闭包只在可访问集合内展开 —— 不能借依赖扩大用户权限（对齐 Yuxi）。"""
    catalog = {"a": _definition("a", deps=("secret",))}
    assert resolve_dependency_closure(["a"], catalog) == ("a",)


def test_dependency_closure_respects_depth_limit():
    catalog = {str(i): _definition(str(i), deps=(str(i + 1),)) for i in range(6)}
    catalog["6"] = _definition("6")
    assert resolve_dependency_closure(["0"], catalog, max_depth=2) == ("0", "1", "2")


def test_from_selection_expands_closure():
    runtime = SkillRuntimeContext.from_selection(
        ["legal-analysis"], list_builtin_skills()
    )
    assert runtime.effective_slugs == ("legal-analysis", "knowledge-research")


def test_preload_must_be_within_effective_set():
    skills = list_builtin_skills()
    runtime = SkillRuntimeContext.from_selection(
        ["web-research"], skills, preload=["web-research"]
    )
    assert runtime.activated_slugs == ("web-research",)
    assert runtime.unlocked_tool_names == ("web_search",)
    with pytest.raises(ValueError, match="preload_skills"):
        SkillRuntimeContext.from_selection(
            ["web-research"], skills, preload=["data-analysis"]
        )


# ── runtime：渐进式披露门控 ───────────────────────────────────────────────

def test_no_skill_selected_allows_every_tool():
    """未启用 Skill → 保持项目既有的自动工具行为。"""
    runtime = SkillRuntimeContext()
    assert runtime.active is False
    assert runtime.allows_tool("anything")


def test_tools_stay_locked_until_skill_is_read():
    runtime = SkillRuntimeContext.from_definitions([
        _definition("research", tools=("web_search",)),
    ])
    assert runtime.allows_tool("web_search") is False
    assert runtime.allows_tool("read_skill") is True, "激活入口必须永远可用"

    runtime.activate("research")
    assert runtime.allows_tool("web_search") is True
    assert runtime.unlocked_tool_names == ("web_search",)
    assert runtime.gated_tool_names == ()


def test_activating_one_skill_does_not_unlock_another():
    runtime = SkillRuntimeContext.from_definitions([
        _definition("research", tools=("web_search",)),
        _definition("writing", tools=("text_tool",)),
    ])
    runtime.activate("research")
    assert runtime.allows_tool("web_search") is True
    assert runtime.allows_tool("text_tool") is False


def test_activate_outside_effective_set_is_ignored():
    runtime = SkillRuntimeContext.from_definitions([_definition("a", tools=("t",))])
    assert runtime.activate("not-in-set") is None
    assert runtime.activated_slugs == ()


def test_public_tools_bypass_the_gate():
    """公共工具不受 Skill 门控 —— 否则首轮连知识库都查不了（规划 §2.2）。"""
    registry = ToolRegistry()
    registry.register(_tool("kb_search", public=True))
    registry.register(_tool("web_search"))
    runtime = SkillRuntimeContext.from_definitions([_definition("writing", tools=("text_tool",))])
    with use_skill_context(runtime):
        assert registry.invoke("kb_search") == "kb_search"
        with pytest.raises(ToolExecutionError, match="not allowed"):
            registry.invoke("web_search")


def test_registry_list_views_are_never_gated():
    """列表/构建视图不受 Skill 门控（2026-09-04 回归修复）。

    进程级缓存的 Agent（build_main_agent / build_dynamic_agent /
    build_subagent）在构建时消费 ``list_all`` —— 构建发生在首个请求的
    上下文里，若列表视图套用渐进式披露门控，首轮激活集为空，未激活工具
    会被从缓存 Agent 里永久剔除，read_skill 解锁的只是不存在的工具。
    门控只属于 invoke（可见性在 middleware 的 wrap_model_call）。
    """
    registry = ToolRegistry()
    registry.register(_tool("kb_search", public=True))
    registry.register(_tool("web_search"))
    runtime = SkillRuntimeContext.from_definitions([_definition("writing", tools=("text_tool",))])
    with use_skill_context(runtime):
        assert set(registry.list_names()) == {"kb_search", "web_search"}
        assert {t.name for t in registry.list_all()} == {"kb_search", "web_search"}
        assert {s["function"]["name"] for s in registry.to_llm_schema()} == {
            "kb_search", "web_search",
        }
    # 未激活任何 Skill 的请求里同样全量可见（故障现场：构建期列表被清空）
    with use_skill_context(SkillRuntimeContext.from_definitions([
        _definition("writing", tools=("text_tool",)),
    ])):
        assert "web_search" in registry.list_names()
        assert "web_search" in registry.to_react_prompt()


def test_public_flag_wins_over_skill_declaration():
    """Skill 在 tool_dependencies 里声明公共工具 ≠ 给它加锁。

    真实场景：knowledge-research 声明 kb_search（说明它会用到），但 kb_search
    是平台基础能力。若声明能加锁，勾选 legal-analysis（依赖 knowledge-research）
    会让模型在读 Skill 之前连知识库都查不了 —— 正是 public 要解决的问题。
    """
    registry = ToolRegistry()
    registry.register(_tool("kb_search", public=True))
    runtime = SkillRuntimeContext.from_definitions([
        _definition("research", tools=("kb_search", "web_search")),
    ])
    with use_skill_context(runtime):
        assert runtime.allows_tool("kb_search") is True, "公共工具优先于门控"
        assert runtime.allows_tool("web_search") is False
        assert registry.invoke("kb_search") == "kb_search"
    # 提示文本也不该把公共工具算成"激活后才可用"
    assert runtime.gated_tool_names == ("web_search",)
    assert runtime.gated_tools_of(runtime.effective[0]) == ("web_search",)
    assert "kb_search" not in runtime.render_prompt()


def test_gated_tool_blocked_in_registry_until_activated():
    """invoke 层门控（ContextVar 侧）：子 Agent 线程 / graph 节点 / MCP 桥接。

    注意 list_names / list_all 不再反映门控（构建视图必须稳定全量，见
    ``test_registry_list_views_are_never_gated``），这里只验证 invoke。"""
    registry = ToolRegistry()
    registry.register(_tool("web_search"))
    registry.register(_tool("calculator", public=True))
    runtime = SkillRuntimeContext.from_definitions([_definition("research", tools=("web_search",))])

    with use_skill_context(runtime):
        assert registry.invoke("calculator") == "calculator"
        with pytest.raises(ToolExecutionError, match="not allowed"):
            registry.invoke("web_search")
        runtime.activate("research")
        assert registry.invoke("web_search") == "web_search"
    # 退出上下文后恢复无门控状态
    assert registry.invoke("web_search") == "web_search"


def test_sync_activated_only_accepts_effective_slugs():
    runtime = SkillRuntimeContext.from_definitions([_definition("a", tools=("t",))])
    runtime.sync_activated(["a", "b-not-effective"])
    assert runtime.activated_slugs == ("a",)


# ── runtime：prompt 渲染 ──────────────────────────────────────────────────

def test_prompt_shows_summary_before_activation_and_body_after():
    runtime = SkillRuntimeContext.from_definitions([
        _definition("research", tools=("web_search",), body="核验来源后再下结论"),
    ])
    pending_prompt = runtime.render_prompt()
    assert "尚未读取" in pending_prompt
    assert "research 描述" in pending_prompt
    assert "核验来源后再下结论" not in pending_prompt, "未激活不得泄露正文"

    runtime.activate("research")
    active_prompt = runtime.render_prompt()
    assert "已读取的 Skill 指令" in active_prompt
    assert "核验来源后再下结论" in active_prompt
    assert "尚未读取" not in active_prompt


def test_eager_prompt_expands_all_bodies():
    """非 Agent 路径（意图识别 / 兜底生成）无 read_skill 循环，用 eager 展开。"""
    runtime = SkillRuntimeContext.from_definitions([
        _definition("research", body="核验来源"),
        _definition("writing", body="先定受众"),
    ])
    prompt = runtime.render_prompt(eager=True)
    assert "核验来源" in prompt and "先定受众" in prompt
    assert "尚未读取" not in prompt


def test_empty_runtime_renders_nothing():
    assert SkillRuntimeContext().render_prompt() == ""


# ── 个人 Skill 校验 ───────────────────────────────────────────────────────

def test_custom_skill_tool_names_must_be_registered():
    assert validate_tool_names(["calculator", "calculator"]) == ("calculator",)
    with pytest.raises(SkillConfigValidationError, match="未注册工具"):
        validate_tool_names(["definitely_missing_tool"])


def test_custom_skill_tool_names_limited():
    with pytest.raises(SkillConfigValidationError, match="最多"):
        validate_tool_names([f"tool_{i}" for i in range(9)])


@pytest.mark.parametrize("bad", ["", "Bad_Slug", "has space", "../escape", "a--b"])
def test_validate_slug_rejects_unsafe_values(bad):
    with pytest.raises(SkillConfigValidationError):
        validate_slug(bad)


def test_slugify_falls_back_for_non_ascii_names():
    assert slugify("My Great Skill") == "my-great-skill"
    assert slugify("联网研究", fallback="skill-abc123") == "skill-abc123"


# ── 请求契约与仓储隔离 ────────────────────────────────────────────────────

def test_chat_request_accepts_more_skills_than_before():
    """渐进式披露下上限放宽（旧上限 3）；运行时上限由 SKILLS_MAX_SELECTED 校验。"""
    request = ChatRequest(query="hello", skill_ids=["a", "b", "c", "d", "e"])
    assert request.skill_ids == ["a", "b", "c", "d", "e"]
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", skill_ids=[str(i) for i in range(33)])


def test_skill_repository_is_owner_scoped():
    source = (
        Path(__file__).parents[1] / "backend/repositories/skill_config_repository.py"
    ).read_text(encoding="utf-8")
    assert "CustomSkillConfig.owner_id == owner_id" in source
    assert "CustomSkillConfig.slug == normalized" in source


def test_personal_dir_rejects_path_traversal():
    from app.skills.registry import personal_dir

    with pytest.raises(SkillLoadError):
        personal_dir("../../etc", "ok-slug")
    with pytest.raises(SkillLoadError):
        personal_dir("owner", "../escape")
