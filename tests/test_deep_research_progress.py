"""High-level Deep Research progress summaries.

These tests intentionally assert that client-facing progress never contains raw
reasoning or tool arguments. The detailed execution stream remains internal.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _progress_module():
    """Load the zero-dependency module without importing DeepAgents adapters."""

    path = Path(__file__).parents[1] / "app" / "agents" / "deep" / "progress.py"
    spec = importlib.util.spec_from_file_location("deep_research_progress", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_projector_emits_natural_research_lifecycle():
    DeepResearchProgressProjector = _progress_module().DeepResearchProgressProjector

    projector = DeepResearchProgressProjector()

    planning = projector.feed("understand", "private planning text")
    retrieval = projector.feed("retrieve", "知识库命中 4 条")
    delegation = projector.feed(
        "tool",
        '调用 task {"description": "private task", "subagent_type": "research-agent"}',
    )
    searching = projector.feed(
        "research-agent/tool",
        '调用 web_search {"query": "secret raw query"}',
    )
    checked = projector.feed(
        "research-agent/tool_done",
        "工具返回: full private result body",
    )
    writing = projector.feed("generate", "主 Agent 生成回答中...")
    completed = projector.feed("generate_done", "回答完成（1200 字符）")

    assert [
        planning["phase"],
        retrieval["phase"],
        delegation["phase"],
        searching["phase"],
        checked["phase"],
        writing["phase"],
        completed["phase"],
    ] == ["planning", "retrieval", "delegation", "search", "analysis", "synthesis", "complete"]
    assert planning["status"] == "running"
    assert completed["status"] == "completed"
    assert all(item["text"] for item in (planning, retrieval, searching, checked, writing, completed))

    visible_text = "\n".join(
        item["text"] for item in (planning, retrieval, delegation, searching, checked, writing, completed)
    )
    assert "secret raw query" not in visible_text
    assert "full private result body" not in visible_text
    assert "private planning text" not in visible_text
    assert "private task" not in visible_text


def test_projector_reports_recoverable_problem_without_raw_error():
    DeepResearchProgressProjector = _progress_module().DeepResearchProgressProjector

    projector = DeepResearchProgressProjector()
    event = projector.feed(
        "retrieve_done",
        "检索失败: connection refused at internal-host:1234",
    )

    assert event["phase"] == "warning"
    assert event["status"] == "warning"
    assert "替代" in event["text"]
    assert "internal-host" not in event["text"]


def test_projector_ignores_reasoning_and_suppresses_consecutive_duplicates():
    DeepResearchProgressProjector = _progress_module().DeepResearchProgressProjector

    projector = DeepResearchProgressProjector()

    assert projector.feed("agent_reasoning", "hidden chain of thought") is None
    assert projector.feed("research-agent/reason", "hidden subagent reasoning") is None

    first = projector.feed("research-agent/generate", "子智能体生成回答中...")
    duplicate = projector.feed("research-agent/generate", "子智能体生成回答中...")
    assert first is not None
    assert duplicate is None
