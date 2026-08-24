"""DeepAgents 配置可达性（S2）测试。

2026-08-21 修复：
1. AGENT_MODE Literal 增加 "deepagents"（此前代码判断了该值但类型定义没有）；
2. 声明 DEEP_SUBAGENTS_FILE / DEEP_MAIN_RECURSION_LIMIT / DEEP_SUBAGENT_RECURSION_LIMIT
   （此前 DEEP_SUBAGENTS_FILE 未声明，pydantic-settings 读不到 env → 外部
   SubAgent 覆盖实际失效）；
3. 外部文件加载健壮化：不存在/坏格式回退内置并告警，无效条目跳过。

全部用 mock / 临时文件，不调用真实 LLM。
"""
from __future__ import annotations

import json

import pytest

from app.core.config import Settings


# ── 配置声明 ──────────────────────────────────────────────────────────────
def test_agent_mode_literal_accepts_deepagents():
    cfg = Settings(_env_file=None)
    assert cfg.AGENT_MODE in ("auto", "single", "multi", "deepagents")
    # Literal 校验：deepagents 是合法取值
    assert Settings(_env_file=None, AGENT_MODE="deepagents").AGENT_MODE == "deepagents"


def test_deep_config_fields_declared():
    cfg = Settings(_env_file=None)
    assert cfg.DEEP_SUBAGENTS_FILE == ""
    assert cfg.DEEP_MAIN_RECURSION_LIMIT == 20
    assert cfg.DEEP_SUBAGENT_RECURSION_LIMIT == 20
    # 环境变量可覆盖（此前 DEEP_SUBAGENTS_FILE 未声明 → 覆盖失效）
    cfg2 = Settings(_env_file=None, DEEP_SUBAGENTS_FILE="subagents.yaml")
    assert cfg2.DEEP_SUBAGENTS_FILE == "subagents.yaml"


# ── 外部 SubAgent 文件加载健壮化 ─────────────────────────────────────────
def test_load_subagents_file_empty_path():
    from app.agents.deep.subagents import _load_subagents_file

    assert _load_subagents_file("") is None
    assert _load_subagents_file(None) is None


def test_load_subagents_file_missing_falls_back(tmp_path):
    from app.agents.deep.subagents import _load_subagents_file

    missing = tmp_path / "nope.yaml"
    assert _load_subagents_file(str(missing)) is None


def test_load_subagents_file_valid_json(tmp_path):
    from app.agents.deep.subagents import _load_subagents_file

    f = tmp_path / "subs.json"
    f.write_text(json.dumps({
        "subagents": [
            {"name": "x-agent", "description": "d", "system_prompt": "p",
             "tools": ["web_search"]},
        ]
    }, ensure_ascii=False), encoding="utf-8")
    items = _load_subagents_file(str(f))
    assert items and items[0]["name"] == "x-agent"


def test_load_subagents_file_bad_json_falls_back(tmp_path):
    from app.agents.deep.subagents import _load_subagents_file

    f = tmp_path / "bad.json"
    f.write_text("{ not valid json", encoding="utf-8")
    assert _load_subagents_file(str(f)) is None


def test_load_subagents_file_skips_unnamed_entries(tmp_path):
    from app.agents.deep.subagents import _load_subagents_file

    f = tmp_path / "mixed.json"
    f.write_text(json.dumps({
        "subagents": [
            {"name": "ok-agent", "description": "d", "system_prompt": "p", "tools": []},
            {"description": "no name"},
            "not a dict",
        ]
    }, ensure_ascii=False), encoding="utf-8")
    items = _load_subagents_file(str(f))
    assert items and len(items) == 1
    assert items[0]["name"] == "ok-agent"


def test_load_subagents_uses_configured_file(monkeypatch, tmp_path):
    from app.agents.deep import subagents as sub_mod

    f = tmp_path / "subs.json"
    f.write_text(json.dumps({
        "subagents": [
            {"name": "custom-agent", "description": "自定义", "system_prompt": "p",
             "tools": ["web_search", "kb_search"]},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    class _FakeCfg:
        DEEP_SUBAGENTS_FILE = str(f)

    monkeypatch.setattr(sub_mod, "get_settings", lambda: _FakeCfg())
    configs = sub_mod.load_subagents()
    assert len(configs) == 1
    assert configs[0].name == "custom-agent"
    assert configs[0].tools == ("web_search", "kb_search")


def test_load_subagents_falls_back_to_builtin(monkeypatch):
    from app.agents.deep.subagents import DEFAULT_SUBAGENTS, load_subagents

    class _FakeCfg:
        DEEP_SUBAGENTS_FILE = ""  # 未配置 → 内置

    monkeypatch.setattr("app.agents.deep.subagents.get_settings", lambda: _FakeCfg())
    configs = load_subagents()
    assert len(configs) == len(DEFAULT_SUBAGENTS)


# ── recursion_limit 接入配置 ──────────────────────────────────────────────
def test_build_task_tool_default_uses_config(monkeypatch):
    import app.agents.deep.task_tool as tt_mod

    calls: dict = {}

    def _fake_run(cfg, desc, model=None, recursion_limit=None):
        calls["recursion_limit"] = recursion_limit
        return "子智能体完成"

    # task_tool 顶部 import 了 run_subagent → patch task_tool 模块内的名字
    monkeypatch.setattr("app.agents.deep.task_tool.run_subagent", _fake_run)

    class _FakeCfg:
        DEEP_SUBAGENT_RECURSION_LIMIT = 5

    monkeypatch.setattr("app.core.config.get_settings", lambda: _FakeCfg())

    tool = tt_mod.build_task_tool(model=object())  # recursion_limit=None → 读配置
    out = tool.invoke({"description": "查一下", "subagent_type": "research-agent"})
    assert calls["recursion_limit"] == 5
    assert "子智能体完成" in out


def test_build_task_tool_explicit_limit_wins(monkeypatch):
    import app.agents.deep.task_tool as tt_mod

    calls: dict = {}

    def _fake_run(cfg, desc, model=None, recursion_limit=None):
        calls["recursion_limit"] = recursion_limit
        return "ok"

    monkeypatch.setattr("app.agents.deep.task_tool.run_subagent", _fake_run)

    tool = tt_mod.build_task_tool(model=object(), recursion_limit=7)
    tool.invoke({"description": "x", "subagent_type": "research-agent"})
    assert calls["recursion_limit"] == 7
