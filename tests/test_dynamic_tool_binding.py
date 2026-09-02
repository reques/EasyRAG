"""阶段 2：SubAgent 动态工具绑定 — */except:/@tag 解析、缓存指纹、动态收窄。

用本地 ToolRegistry + monkeypatch 隔离全局配置，不构建真实模型。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.agents.deep.subagents as sa
from app.tools.registry import ToolDefinition, ToolRegistry


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="web_search", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["联网搜索"], "tags": ["search", "web"]},
    ))
    reg.register(ToolDefinition(
        name="kb_search", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["知识库"], "tags": ["search", "kb"]},
    ))
    reg.register(ToolDefinition(
        name="calculator", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["计算"], "tags": ["math"]},
    ))
    reg.register(ToolDefinition(
        name="text_tool", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["字数"], "tags": ["text"]},
    ))
    return reg


@pytest.fixture
def fake_registry(monkeypatch):
    reg = _registry()
    monkeypatch.setattr("app.tools.registry.get_tool_registry", lambda: reg)
    return reg


# ── tools 声明解析（* / except: / @tag）────────────────────────────────────


def test_resolve_plain_names(fake_registry):
    assert sa.resolve_tool_spec(("web_search", "calculator")) == (
        "calculator", "web_search",
    )


def test_resolve_unregistered_name_ignored(fake_registry):
    """未注册名称静默忽略（与旧行为一致，不抛错）。"""
    assert sa.resolve_tool_spec(("web_search", "no_such_tool")) == ("web_search",)


def test_resolve_star_all(fake_registry):
    assert sa.resolve_tool_spec(("*",)) == (
        "calculator", "kb_search", "text_tool", "web_search",
    )


def test_resolve_star_with_exclude(fake_registry):
    assert sa.resolve_tool_spec(("*", "except:web_search")) == (
        "calculator", "kb_search", "text_tool",
    )


def test_resolve_tag(fake_registry):
    """@search → 所有 tags 含 search 的工具。"""
    assert sa.resolve_tool_spec(("@search",)) == ("kb_search", "web_search")


def test_resolve_tag_with_exclude(fake_registry):
    assert sa.resolve_tool_spec(("@search", "except:web_search")) == ("kb_search",)


def test_resolve_union_of_tags_and_names(fake_registry):
    assert sa.resolve_tool_spec(("@math", "text_tool")) == ("calculator", "text_tool")


def test_resolve_empty(fake_registry):
    assert sa.resolve_tool_spec(()) == ()
    assert sa.resolve_tool_spec(("", "  ")) == ()


# ── 缓存指纹：工具集变化时重建 ────────────────────────────────────────────


def test_build_cache_reuses_same_tool_fingerprint(fake_registry, monkeypatch):
    builds = {"n": 0}

    def _fake_create(**kwargs):
        builds["n"] += 1
        return object()

    monkeypatch.setattr("langchain.agents.create_agent", _fake_create)
    monkeypatch.setattr("app.agents.deep.llm.get_langchain_model", lambda: object())
    sa._subagent_cache.clear()

    cfg = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                            tools=("web_search",))
    first = sa.build_subagent(cfg)
    second = sa.build_subagent(cfg)
    assert first is second and builds["n"] == 1

    # 工具集变化（同名的另一个配置）→ 指纹不同 → 重建
    cfg2 = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                             tools=("web_search", "calculator"))
    third = sa.build_subagent(cfg2)
    assert third is not first and builds["n"] == 2

    # 同工具集的另一种声明写法 → 解析指纹相同 → 复用（不重建）
    cfg3 = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                             tools=("@web",))  # @web 解析为 ("web_search",)
    assert sa.build_subagent(cfg3) is first and builds["n"] == 2
    sa._subagent_cache.clear()


# ── 执行时动态收窄（DEEP_DYNAMIC_TOOLS）───────────────────────────────────


def test_dynamic_narrow_disabled_by_default(fake_registry, monkeypatch):
    monkeypatch.setattr("app.agents.deep.subagents.get_settings",
                        lambda: SimpleNamespace(DEEP_DYNAMIC_TOOLS=False))
    cfg = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                            tools=("*",))
    assert sa._maybe_narrow_tools_by_task(cfg, "联网搜索") is cfg


def test_dynamic_narrow_intersects_with_config(fake_registry, monkeypatch):
    """开启后按任务描述 discover，与配置取交集（只收窄）。"""
    monkeypatch.setattr("app.agents.deep.subagents.get_settings",
                        lambda: SimpleNamespace(DEEP_DYNAMIC_TOOLS=True))
    cfg = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                            tools=("*",))
    narrowed = sa._maybe_narrow_tools_by_task(cfg, "帮我联网搜索")
    assert narrowed is not cfg
    assert narrowed.tools == ("web_search",)


def test_dynamic_narrow_never_expands_config(fake_registry, monkeypatch):
    """discover 命中的工具超出配置白名单 → 不能放大，仍受白名单约束。"""
    monkeypatch.setattr("app.agents.deep.subagents.get_settings",
                        lambda: SimpleNamespace(DEEP_DYNAMIC_TOOLS=True))
    cfg = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                            tools=("calculator",))
    narrowed = sa._maybe_narrow_tools_by_task(cfg, "联网搜索 知识库")
    # 交集为空 → 保留原配置（避免无工具可用）
    assert narrowed is cfg


def test_dynamic_narrow_no_discovery_keeps_config(fake_registry, monkeypatch):
    monkeypatch.setattr("app.agents.deep.subagents.get_settings",
                        lambda: SimpleNamespace(DEEP_DYNAMIC_TOOLS=True))
    cfg = sa.SubAgentConfig(name="a", description="", system_prompt="p",
                            tools=("*",))
    assert sa._maybe_narrow_tools_by_task(cfg, "无关的闲聊") is cfg


# ── 越权错误附可用工具清单 ────────────────────────────────────────────────


def test_unknown_tool_error_lists_available(fake_registry):
    from app.core.exceptions import ToolNotFoundError

    with pytest.raises(ToolNotFoundError) as ei:
        fake_registry.invoke("no_such_tool")
    msg = str(ei.value)
    assert "Available tools" in msg
    assert "web_search" in msg and "calculator" in msg
