"""阶段 2：工具发现（registry.discover）— 元数据匹配 / 计分排序 / limit / 权限。

全部使用本地构造的 ToolRegistry（不依赖全局配置与外部服务）。
"""
from __future__ import annotations

from app.skills.catalog import SkillProfile
from app.skills.context import use_skill_context
from app.tools.registry import ToolDefinition, ToolRegistry


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="web_search", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["联网搜索", "新闻", "最新"], "tags": ["search", "web"]},
    ))
    reg.register(ToolDefinition(
        name="kb_search", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["知识库", "内部资料"], "tags": ["search", "kb"]},
    ))
    reg.register(ToolDefinition(
        name="calculator", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["计算", "算术", "算一下"], "tags": ["math"]},
    ))
    reg.register(ToolDefinition(
        name="text_tool", description="d", fn=lambda **kw: "ok",
        metadata={"scenarios": ["字数", "文本处理"], "tags": ["text"]},
    ))
    return reg


# ── 场景关键词匹配 ────────────────────────────────────────────────────────


def test_discover_matches_scenario_phrases():
    reg = _registry()
    out = reg.discover("帮我联网搜索最新的AI新闻")
    names = [t.name for t in out]
    assert "web_search" in names
    assert names[0] == "web_search"  # 3 条 scenario 命中，得分最高排最前
    assert "calculator" not in names


def test_discover_matches_multiple_tools():
    reg = _registry()
    out = reg.discover("先在知识库里查内部资料，再算一下成本")
    names = [t.name for t in out]
    assert "kb_search" in names
    assert "calculator" in names
    assert "web_search" not in names


def test_discover_no_match_returns_empty():
    reg = _registry()
    assert reg.discover("随便聊聊人生") == []


def test_discover_empty_description_returns_empty():
    reg = _registry()
    assert reg.discover("") == []
    assert reg.discover("   ") == []


# ── @tag 显式提及与单词匹配 ───────────────────────────────────────────────


def test_discover_explicit_tag_mention_scores_high():
    reg = _registry()
    # @math 显式提及（+3）→ calculator 置顶，即使描述里没有"计算"
    out = reg.discover("请用 @math 处理这个表达式")
    assert out and out[0].name == "calculator"


def test_discover_tag_word_boundary():
    reg = _registry()
    # 'math' 作为独立单词命中（+1）；'mathematics' 的子串不应命中
    assert [t.name for t in reg.discover("use math here")] == ["calculator"]
    assert reg.discover("mathematics") == []


# ── limit 截断 ────────────────────────────────────────────────────────────


def test_discover_respects_limit():
    reg = _registry()
    out = reg.discover("联网搜索 知识库 计算 字数", limit=2)
    assert len(out) == 2


# ── 权限：发现结果只能收窄不能放大 ─────────────────────────────────────────


def test_discover_honours_skill_whitelist():
    """skills 白名单只放行 calculator → discover 不应返回被禁工具。"""
    reg = _registry()
    profile = SkillProfile(
        id="test:math-only", name="仅计算", description="", instructions="",
        tool_names=("calculator",),
    )
    with use_skill_context([profile]):
        out = reg.discover("联网搜索 计算")
    assert [t.name for t in out] == ["calculator"]


def test_discover_ignores_unavailable_tools():
    """check_fn 失败的工具不参与发现。"""
    reg = _registry()
    reg.register(ToolDefinition(
        name="broken", description="d", fn=lambda **kw: "ok",
        check_fn=lambda: False,
        metadata={"scenarios": ["联网搜索"], "tags": []},
    ))
    out = reg.discover("联网搜索")
    assert "broken" not in [t.name for t in out]
