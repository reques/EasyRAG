"""阶段 4：动态规划 — revise_plan 追加/取消/已完成保护 + 结构化尾部解析与回退。

用 monkeypatch 替换 run_subagent，不调用真实 LLM。
"""
from __future__ import annotations

import pytest

import app.agents.deep.planner as planner
from app.agents.deep.planner import (
    STATUS_CANCELLED,
    STATUS_OK,
    revise_plan_payload,
    spawn_tasks_payload,
)
from app.agents.deep.subagents import parse_result_tail
from app.agents.events import use_request_trace


@pytest.fixture(autouse=True)
def _clean_state():
    import app.agents.deep.task_tool as tt

    planner.reset_plans()
    tt.reset_task_breaker()
    yield
    planner.reset_plans()
    tt.reset_task_breaker()


def _spawn_two(monkeypatch, session="conv-plan"):
    """spawn 两个任务：A 成功、B 失败（供后续修订）。返回 trace 内调用。"""
    def _fake_run(cfg, description, model=None, recursion_limit=None):
        if description.startswith("fail"):
            raise RuntimeError("故意失败")
        return f"done:{description[:10]}"

    monkeypatch.setattr("app.agents.deep.subagents.run_subagent", _fake_run)
    with use_request_trace(session_id=session):
        spawn_tasks_payload([
            {"key": "A", "description": "第一步", "subagent_type": "research-agent"},
            {"key": "B", "description": "fail me", "subagent_type": "research-agent"},
        ])


# ── 结构化尾部解析 ────────────────────────────────────────────────────────


def test_parse_result_tail_valid_json():
    text = (
        "研究结论：要点1、要点2。\n"
        '{"status": "completed", "concerns": "数据来源未交叉验证", '
        '"suggested_followup": "建议补充第二信源"}'
    )
    tail = parse_result_tail(text)
    assert tail["status"] == "completed"
    assert tail["concerns"] == "数据来源未交叉验证"
    assert tail["suggested_followup"] == "建议补充第二信源"


def test_parse_result_tail_in_code_fence():
    """尾块被 ``` 包裹也能解析（容忍模型格式偏差）。"""
    text = '结论\n```json\n{"status": "partial", "concerns": "", "suggested_followup": "x"}\n```'
    assert parse_result_tail(text)["status"] == "partial"


def test_parse_result_tail_fallback_plain_text():
    """无 JSON 尾块 → status=unknown，raw 保留尾部原文（不抛错）。"""
    tail = parse_result_tail("只有纯文本结论，没有尾块。")
    assert tail["status"] == "unknown"
    assert "纯文本结论" in tail["raw"]


def test_parse_result_tail_empty():
    tail = parse_result_tail("")
    assert tail["status"] == "unknown" and tail["raw"] == ""


def test_subagent_prompt_includes_tail_convention(monkeypatch):
    """build_subagent 把尾部约定追加进子 Agent system prompt（外部配置也生效）。"""
    import app.agents.deep.subagents as sa

    captured = {}
    monkeypatch.setattr(
        "langchain.agents.create_agent",
        lambda **kw: captured.update(kw) or object(),
    )
    cfg = sa.SubAgentConfig(name="x", description="", system_prompt="原始指令")
    sa.build_subagent(cfg, model=object())
    assert captured["system_prompt"].startswith("原始指令")
    assert '"status"' in captured["system_prompt"] and "suggested_followup" in captured["system_prompt"]


# ── revise_plan：无计划 / 取消 / 已完成保护 ──────────────────────────────


def test_revise_without_plan_prompts_spawn(monkeypatch):
    with use_request_trace(session_id="conv-empty"):
        out = revise_plan_payload([{"action": "cancel", "key": "A"}])
    assert "spawn_tasks" in out


def test_cancel_completed_task_rejected(monkeypatch):
    _spawn_two(monkeypatch)
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([{"action": "cancel", "key": "A"}])
    assert "不可撤销" in out
    state = planner.get_plan_state("conv-plan")
    assert state.results["A"].status == STATUS_OK  # 未被改动


def test_cancel_failed_task_succeeds(monkeypatch):
    _spawn_two(monkeypatch)
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([{"action": "cancel", "key": "B"}])
    assert "[B] 已取消" in out
    state = planner.get_plan_state("conv-plan")
    assert state.results["B"].status == STATUS_CANCELLED


# ── revise_plan：追加 / 细化 ─────────────────────────────────────────────


def test_add_task_runs_immediately(monkeypatch):
    _spawn_two(monkeypatch)
    monkeypatch.setattr(
        "app.agents.deep.subagents.run_subagent",
        lambda cfg, d, model=None, recursion_limit=None: "追加任务的结果",
    )
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([{
            "action": "add", "key": "C",
            "description": "补充调研", "subagent_type": "research-agent",
            "depends_on": ["A"],
        }])
    assert "[C] 已加入计划并执行" in out
    assert "追加任务的结果" in out
    state = planner.get_plan_state("conv-plan")
    assert state.results["C"].status == STATUS_OK


def test_add_task_depends_on_failed_rejected(monkeypatch):
    _spawn_two(monkeypatch)
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([{
            "action": "add", "key": "C",
            "description": "x", "subagent_type": "research-agent",
            "depends_on": ["B"],  # B 失败
        }])
    assert "追加失败" in out and "B" in out


def test_add_duplicate_key_rejected(monkeypatch):
    _spawn_two(monkeypatch)
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([{
            "action": "add", "key": "A",
            "description": "x", "subagent_type": "research-agent",
        }])
    assert "已存在" in out


def test_refine_reruns_with_new_description(monkeypatch):
    _spawn_two(monkeypatch)
    seen = []
    monkeypatch.setattr(
        "app.agents.deep.subagents.run_subagent",
        lambda cfg, d, model=None, recursion_limit=None: seen.append(d) or "细化后的结果",
    )
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([
            {"action": "refine", "key": "B", "description": "换个方式重试"},
        ])
    assert "[B] 已用新描述重新执行" in out
    assert "细化后的结果" in out
    assert seen and seen[0].startswith("换个方式重试")
    state = planner.get_plan_state("conv-plan")
    assert state.results["B"].status == STATUS_OK


def test_unknown_action_reported(monkeypatch):
    _spawn_two(monkeypatch)
    with use_request_trace(session_id="conv-plan"):
        out = revise_plan_payload([{"action": "explode", "key": "A"}])
    assert "未知 action" in out


# ── 计划状态按会话隔离 ───────────────────────────────────────────────────


def test_plan_state_scoped_by_session(monkeypatch):
    _spawn_two(monkeypatch, session="conv-1")
    assert planner.get_plan_state("conv-1") is not None
    assert planner.get_plan_state("conv-2") is None
    with use_request_trace(session_id="conv-2"):
        out = revise_plan_payload([{"action": "cancel", "key": "A"}])
    assert "没有可修订的计划" in out


# ── spawn 结果携带结构化尾部（聚合文本展示）──────────────────────────────


def test_spawn_aggregate_shows_tail_notes(monkeypatch):
    tail_json = ('{"status": "completed", "concerns": "需要核实价格", '
                 '"suggested_followup": "建议再查一次"}')
    monkeypatch.setattr(
        "app.agents.deep.subagents.run_subagent",
        lambda cfg, d, model=None, recursion_limit=None: f"结论正文。\n{tail_json}",
    )
    with use_request_trace(session_id="conv-tail"):
        out = spawn_tasks_payload([
            {"key": "A", "description": "查价格", "subagent_type": "research-agent"},
        ])
    assert "遗留关注: 需要核实价格" in out
    assert "建议后续: 建议再查一次" in out
