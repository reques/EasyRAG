"""阶段 3：spawn_tasks — 拓扑排序 / 环检测 / 层级并发 / 依赖注入 / 部分失败聚合。

用 monkeypatch 替换 run_subagent，不调用真实 LLM。
"""
from __future__ import annotations

import threading
import time

import pytest

import app.agents.deep.planner as planner
from app.agents.deep.blackboard import Blackboard
from app.agents.deep.planner import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    TaskSpec,
    parse_task_inputs,
    run_spawn_tasks,
    spawn_tasks_payload,
    validate_dag,
)
from app.core.exceptions import PlanningError


@pytest.fixture(autouse=True)
def _clean_breaker():
    """spawn 失败也计入 S5 熔断：隔离跨测试状态。"""
    import app.agents.deep.task_tool as tt

    tt.reset_task_breaker()
    yield
    tt.reset_task_breaker()


# ── 校验与分层 ────────────────────────────────────────────────────────────


def _spec(key, deps=()):
    return TaskSpec(key=key, description=f"d-{key}",
                    subagent_type="research-agent", depends_on=deps)


def test_validate_dag_layers():
    layers = validate_dag([
        _spec("A"), _spec("B", ("A",)), _spec("C", ("A",)),
        _spec("D", ("B", "C")),
    ])
    assert layers == [["A"], ["B", "C"], ["D"]]


def test_validate_dag_cycle_raises():
    with pytest.raises(PlanningError, match="环"):
        validate_dag([_spec("A", ("B",)), _spec("B", ("A",))])


def test_validate_dag_duplicate_key_raises():
    with pytest.raises(PlanningError, match="重复"):
        validate_dag([_spec("A"), _spec("A")])


def test_validate_dag_external_dep_allowed():
    """依赖指向本批之外 → 视为已就绪的外部前置（revise_plan 追加场景）。"""
    layers = validate_dag([_spec("A", ("prev_round_done",))])
    assert layers == [["A"]]


def test_validate_dag_self_dep_raises():
    with pytest.raises(PlanningError, match="依赖自己"):
        validate_dag([_spec("A", ("A",))])


def test_validate_dag_empty_raises():
    with pytest.raises(PlanningError):
        validate_dag([])


def test_parse_task_inputs_default_keys_and_validation():
    specs = parse_task_inputs([
        {"description": "a", "subagent_type": "research-agent"},
        {"description": "b", "subagent_type": "coding-agent",
         "depends_on": ["task_1"]},
    ])
    assert specs[0].key == "task_1"
    assert specs[1].key == "task_2"
    assert specs[1].depends_on == ("task_1",)
    with pytest.raises(PlanningError, match="description"):
        parse_task_inputs([{"subagent_type": "x"}])
    with pytest.raises(PlanningError, match="subagent_type"):
        parse_task_inputs([{"description": "d"}])


# ── 执行：依赖注入 / 黑板 / 事件 / 上下文重放 ───────────────────────────


def test_execution_runs_in_dependency_order_with_injection(monkeypatch):
    """依赖任务的产出注入后续任务描述；执行顺序符合拓扑。"""
    seen = []          # (key, description)
    outputs = {"A": "A 的调研结论", "B": "B 的结果"}

    def _fake_run(cfg, description, model=None, recursion_limit=None):
        # 从注入文本反查是哪个任务（description 以 "d-<key>" 开头）
        key = description.split()[0].replace("d-", "")
        seen.append((key, description))
        return outputs[key]

    monkeypatch.setattr("app.agents.deep.subagents.run_subagent", _fake_run)
    board = Blackboard()
    results = run_spawn_tasks(
        [_spec("A"), _spec("B", ("A",))], board=board,
    )
    assert results["A"].status == STATUS_OK
    assert results["B"].status == STATUS_OK
    # 顺序：A 先
    assert [k for k, _ in seen] == ["A", "B"]
    # B 的描述里注入了 A 的产出摘要
    desc_b = dict(seen)["B"]
    assert "依赖任务产出" in desc_b
    assert "A 的调研结论" in desc_b
    # 产出落黑板（结构化两级）
    assert board.get("A").summary == "A 的调研结论"
    assert board.get("A").data == "A 的调研结论"


def test_same_layer_runs_concurrently(monkeypatch):
    """同层无依赖任务并发执行（墙钟时间显著小于串行）。"""
    def _fake_run(cfg, description, model=None, recursion_limit=None):
        time.sleep(0.15)
        return "ok"

    monkeypatch.setattr("app.agents.deep.subagents.run_subagent", _fake_run)
    started = time.perf_counter()
    results = run_spawn_tasks([_spec("A"), _spec("B"), _spec("C")])
    elapsed = time.perf_counter() - started
    assert all(r.status == STATUS_OK for r in results.values())
    assert elapsed < 0.4  # 串行需 ≥0.45s


def test_partial_failure_skips_dependents(monkeypatch):
    """A 失败 → 依赖它的 B 跳过；无依赖的 C 正常；聚合文本标注状态。"""
    def _fake_run(cfg, description, model=None, recursion_limit=None):
        if description.startswith("d-A"):
            raise RuntimeError("A 崩了")
        return f"done:{description[:3]}"

    monkeypatch.setattr("app.agents.deep.subagents.run_subagent", _fake_run)
    specs = [_spec("A"), _spec("B", ("A",)), _spec("C")]
    results = run_spawn_tasks(specs)
    assert results["A"].status == STATUS_FAILED
    assert results["B"].status == STATUS_SKIPPED
    assert "A" in results["B"].error
    assert results["C"].status == STATUS_OK

    text = planner.aggregate_results(results, [s.key for s in specs])
    assert "成功 1 个" in text
    assert "[A] ❌ 失败" in text
    assert "[B] ⏭️ 跳过" in text
    assert "[C] ✅ 成功" in text


def test_unknown_subagent_fails_task(monkeypatch):
    specs = [TaskSpec(key="A", description="d", subagent_type="ghost-agent")]
    results = run_spawn_tasks(specs)
    assert results["A"].status == STATUS_FAILED
    assert "未知子智能体" in results["A"].error


def test_events_emitted_with_spawn_span_and_context(monkeypatch):
    """事件进请求 trace（并发上下文重放生效），子任务以 spawn/<key> span 标识。"""
    from app.agents.events import get_trace, use_request_trace

    trace_seen = {}

    def _fake_run(cfg, description, model=None, recursion_limit=None):
        trace_seen[description[:3]] = get_trace()  # 工作线程内可见请求 trace
        return "ok"

    monkeypatch.setattr("app.agents.deep.subagents.run_subagent", _fake_run)
    with use_request_trace(session_id="sess-spawn") as rt:
        run_spawn_tasks([_spec("A"), _spec("B", ("A",))])

    # 工作线程内重放成功：能看到同一 trace
    assert trace_seen["d-A"] is not None
    assert trace_seen["d-A"].trace_id == rt.trace.trace_id
    assert trace_seen["d-A"].session_id == "sess-spawn"

    stages = [ev["stage"] for ev in rt.events if ev["kind"] == "delegation"]
    assert "spawn_start" in stages
    assert stages.count("task_start") == 2
    assert stages.count("task_end") == 2
    assert "spawn_end" in stages
    # 子任务事件带 spawn/<key> span
    spans = {ev["span"] for ev in rt.events if ev["stage"] == "task_start"}
    assert "spawn/A" in spans and "spawn/B" in spans
    # 黑板写通知事件也在流中
    assert any(ev["kind"] == "blackboard" and ev["stage"] == "post"
               for ev in rt.events)


# ── 工具入口（校验失败返回文本，供主 Agent 修正）────────────────────────


def test_payload_validation_error_returns_text(monkeypatch):
    out = spawn_tasks_payload([
        {"description": "a", "subagent_type": "research-agent",
         "depends_on": ["task_2"]},
        {"description": "b", "subagent_type": "research-agent",
         "depends_on": ["task_1"]},
    ])
    assert "校验失败" in out and "环" in out


def test_payload_end_to_end(monkeypatch):
    monkeypatch.setattr(
        "app.agents.deep.subagents.run_subagent",
        lambda cfg, d, model=None, recursion_limit=None: f"done:{d[:4]}",
    )
    out = spawn_tasks_payload([
        {"key": "a", "description": "第一步", "subagent_type": "research-agent"},
        {"key": "b", "description": "第二步", "subagent_type": "research-agent",
         "depends_on": ["a"]},
    ])
    assert "成功 2 个" in out
    assert "[a] ✅ 成功" in out and "[b] ✅ 成功" in out
