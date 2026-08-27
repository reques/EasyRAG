"""阶段 5b 测试 — DeepAgents 委派持久化（事件流 → Run/Task/AgentRun 落库）。"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import List

import pytest

from backend.services import agent_run_service
from backend.services.delegation_service import (
    bridge_delegation_event,
    extract_delegation_from_events,
    persist_delegation,
)


def _ev(kind, stage, **extra):
    return {"kind": kind, "stage": stage, "content": extra.pop("content", ""), **extra}


def _spawn_events():
    """一次 spawn_tasks 的完整委派事件序列（2 成功 1 失败 1 跳过）。"""
    return [
        _ev("tool", "tool_start", content="noise"),  # 非委派事件应被忽略
        _ev("delegation", "spawn_start", task_keys=["a", "b", "c", "d"]),
        _ev("delegation", "task_start", task_key="a", subagent_type="research-agent",
            content="subagent=research-agent do a"),
        _ev("delegation", "task_start", task_key="b", subagent_type="coding-agent",
            content="subagent=coding-agent do b"),
        _ev("delegation", "task_end", task_key="a", content="answer a"),
        _ev("delegation", "task_error", task_key="b", content="boom"),
        _ev("delegation", "task_skip", task_key="c", content="依赖任务 b 未成功"),
        _ev("delegation", "task_start", task_key="d", subagent_type="research-agent",
            content="subagent=research-agent do d"),
        _ev("delegation", "task_end", task_key="d", content="answer d"),
        _ev("delegation", "spawn_end", succeeded=2, total=4),
    ]


# ── extract_delegation_from_events ─────────────────────────────────────────


def test_extract_returns_none_without_delegation_events():
    assert extract_delegation_from_events(None) is None
    assert extract_delegation_from_events([]) is None
    assert extract_delegation_from_events(
        [_ev("tool", "tool_start"), _ev("step", "reasoning")]
    ) is None


def test_extract_spawn_flow_collects_tasks_and_statuses():
    summary = extract_delegation_from_events(_spawn_events())
    assert summary is not None
    assert [t["task_id"] for t in summary["tasks"]] == ["a", "b", "c", "d"]
    assert summary["tasks"][0]["worker_hint"] == "research-agent"
    assert summary["tasks"][1]["worker_hint"] == "coding-agent"
    assert summary["statuses"] == {
        "a": "ok", "b": "failed", "c": "skipped", "d": "ok",
    }
    assert summary["outputs"]["a"] == "answer a"
    assert summary["errors"]["b"] == "boom"
    assert summary["run_status"] == "completed"  # 部分成功 → completed
    assert summary["error_summary"] == ""


def test_extract_all_failed_marks_run_failed():
    events = [
        _ev("delegation", "spawn_start", task_keys=["a"]),
        _ev("delegation", "task_start", task_key="a", subagent_type="research-agent"),
        _ev("delegation", "task_error", task_key="a", content="boom"),
        _ev("delegation", "spawn_end", succeeded=0, total=1),
    ]
    summary = extract_delegation_from_events(events)
    assert summary["run_status"] == "failed"
    assert summary["error_summary"] == "全部委派任务失败"


def test_extract_single_task_delegation():
    """task 工具的单任务委派：仅 task_start（task_tool 内部吞异常）。"""
    events = [
        _ev("delegation", "task_start", task_key="research-agent",
            subagent_type="research-agent", content="do it"),
        _ev("delegation", "task_end", task_key="research-agent", content="done"),
    ]
    summary = extract_delegation_from_events(events)
    assert summary is not None
    assert len(summary["tasks"]) == 1
    assert summary["run_status"] == "completed"


# ── persist_delegation ──────────────────────────────────────────────────────


class _FakeSession:
    def __init__(self):
        self.committed = False

    async def commit(self):
        self.committed = True


@pytest.fixture
def fake_ars(monkeypatch):
    """记录对 agent_run_service 各函数的调用。"""
    calls = {
        "create_run": [], "create_tasks": [], "finish_task": [], "finalize_run": [],
    }
    run_id = uuid.uuid4()

    async def create_run(session, **kwargs):
        calls["create_run"].append(kwargs)

        class _Run:
            id = run_id

        return _Run()

    async def create_tasks(session, rid, tasks, model_id):
        calls["create_tasks"].append((rid, tasks, model_id))
        return []

    async def finish_task(session, rid, key, **kwargs):
        calls["finish_task"].append((rid, key, kwargs))

    async def finalize_run(session, rid, **kwargs):
        calls["finalize_run"].append((rid, kwargs))

    monkeypatch.setattr(agent_run_service, "create_run", create_run)
    monkeypatch.setattr(agent_run_service, "create_tasks", create_tasks)
    monkeypatch.setattr(agent_run_service, "finish_task", finish_task)
    monkeypatch.setattr(agent_run_service, "finalize_run", finalize_run)
    return calls, run_id


def _session_factory(session):
    @asynccontextmanager
    async def _cm():
        yield session

    return _cm


@pytest.mark.asyncio
async def test_persist_delegation_writes_run_tasks_and_finalize(fake_ars):
    calls, run_id = fake_ars
    session = _FakeSession()
    conv_id, user_id = uuid.uuid4(), uuid.uuid4()

    out = await persist_delegation(
        _session_factory(session),
        conversation_id=conv_id,
        user_id=user_id,
        events=_spawn_events(),
        goal="综合任务",
        model_id="m-1",
        source_message_id=7,
    )
    assert out == str(run_id)
    assert session.committed

    run_kwargs = calls["create_run"][0]
    assert run_kwargs["conversation_id"] == conv_id
    assert run_kwargs["user_id"] == user_id
    assert run_kwargs["goal"] == "综合任务"
    assert run_kwargs["model_id"] == "m-1"
    assert run_kwargs["source_message_id"] == 7

    rid, tasks, model_id = calls["create_tasks"][0]
    assert rid == run_id and model_id == "m-1"
    assert [t["task_id"] for t in tasks] == ["a", "b", "c", "d"]

    finished = {key: kwargs["worker_status"] for _, key, kwargs in calls["finish_task"]}
    assert finished == {"a": "done", "b": "error", "c": "skipped", "d": "done"}

    rid, fin_kwargs = calls["finalize_run"][0]
    assert rid == run_id
    assert fin_kwargs["status"] == "completed"
    assert fin_kwargs["execution_mode"] == "deepagents"


@pytest.mark.asyncio
async def test_persist_skips_without_delegation_events(fake_ars):
    calls, _ = fake_ars
    out = await persist_delegation(
        _session_factory(_FakeSession()),
        conversation_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        events=[_ev("tool", "tool_start")],
        goal="x",
    )
    assert out is None
    assert calls["create_run"] == []


@pytest.mark.asyncio
async def test_persist_swallows_errors(fake_ars, monkeypatch):
    _, _ = fake_ars

    async def boom(session, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(agent_run_service, "create_run", boom)
    out = await persist_delegation(
        _session_factory(_FakeSession()),
        conversation_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        events=_spawn_events(),
        goal="x",
    )
    assert out is None  # best-effort：异常不外抛


# ── bridge_delegation_event（统一事件流 → SSE 协议）────────────────────


def test_bridge_spawn_start_emits_sub_tasks():
    payloads = bridge_delegation_event(_ev(
        "delegation", "spawn_start",
        task_keys=["a", "b"],
        tasks=[
            {"task_id": "a", "goal": "do a", "worker_hint": "research-agent"},
            {"task_id": "b", "goal": "do b", "worker_hint": "coding-agent"},
        ],
    ))
    assert payloads == [{"type": "sub_tasks", "tasks": [
        {"task_id": "a", "goal": "do a", "worker_hint": "research-agent"},
        {"task_id": "b", "goal": "do b", "worker_hint": "coding-agent"},
    ]}]


def test_bridge_task_lifecycle_maps_to_panel_events():
    started = bridge_delegation_event(_ev(
        "delegation", "task_start", task_key="a",
        subagent_type="research-agent", content="do a",
    ))
    assert started[0]["type"] == "status"
    assert started[0]["step"] == "task_started"
    assert started[0]["task_id"] == "a"

    ended = bridge_delegation_event(_ev(
        "delegation", "task_end", task_key="a",
        subagent_type="research-agent", content="answer",
    ))[0]
    assert (ended["type"], ended["status"], ended["task_id"]) == (
        "worker_output", "done", "a",
    )

    failed = bridge_delegation_event(_ev(
        "delegation", "task_error", task_key="b", content="boom",
    ))[0]
    assert failed["status"] == "error" and failed["error"] == "boom"

    skipped = bridge_delegation_event(_ev(
        "delegation", "task_skip", task_key="c", content="依赖未成功",
    ))[0]
    assert skipped["status"] == "skipped"


def test_bridge_tool_events_map_to_status_timeline():
    span = {"span": "spawn/a"}
    start = bridge_delegation_event(_ev("tool", "tool_start", **span))[0]
    assert (start["step"], start["task_id"]) == ("tool", "a")
    end = bridge_delegation_event(_ev("tool", "tool_end", **span))[0]
    assert end["step"] == "tool_done"
    err = bridge_delegation_event(_ev("tool", "tool_error", **span))[0]
    assert err["step"] == "fallback"
    prog = bridge_delegation_event(_ev("tool", "progress", **span))[0]
    assert prog["type"] == "progress_summary" and prog["phase"] == "tool"


def test_bridge_artifact_events_map_to_artifact_and_tool_call():
    # 子任务 span 内的工具动作：artifact 载荷 + 任务面板 tool_call 时间线
    payloads = bridge_delegation_event(_ev(
        "artifact", "research-agent/tool", artifact_kind="tool",
        content='{"query": "x"}', span="subagent/research-agent",
    ))
    assert [p["type"] for p in payloads] == ["artifact", "tool_call"]
    assert payloads[0]["kind"] == "tool"
    assert payloads[0]["streaming"] is False
    assert payloads[1]["task_id"] == "research-agent"
    assert '{"query": "x"}' in payloads[1]["detail"]

    # spawn span 内的工具返回：tool_call 落到对应任务键
    ret = bridge_delegation_event(_ev(
        "artifact", "task-b/tool", artifact_kind="tool_result",
        content="tool result body", span="spawn/task-b",
    ))
    assert ret[0]["kind"] == "tool_result"
    assert ret[1]["task_id"] == "task-b"

    # 主 span 的推理/检索 artifact：仅 artifact 载荷，不追加 tool_call
    plain = bridge_delegation_event(_ev(
        "artifact", "reason", artifact_kind="thought",
        content="thinking", span="main",
    ))
    assert len(plain) == 1
    assert plain[0]["type"] == "artifact" and plain[0]["kind"] == "thought"


def test_bridge_ignores_unrelated_events():
    assert bridge_delegation_event(_ev("step", "reasoning")) == []
    assert bridge_delegation_event(_ev("blackboard", "post")) == []
    assert bridge_delegation_event(_ev("delegation", "spawn_end", succeeded=2)) == []
