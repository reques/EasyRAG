"""DeepAgents 批量委派 + 拓扑调度（阶段 3）— ``spawn_tasks`` 工具。

主 Agent 用 ``spawn_tasks(tasks=[{description, subagent_type, depends_on}])``
一次声明多个子任务及其依赖关系（DAG），调度器负责：

- 校验：key 唯一、依赖存在、无自依赖、无环（Kahn 拓扑排序，环 → 报错回主 Agent）
- 分层并发：同层任务用 ThreadPoolExecutor 并发执行 ``run_subagent``，
  层间串行等待（依赖就绪才开下一层）
- 依赖注入：依赖任务的产出（黑板 artifact 摘要）注入后续任务描述
- 部分失败聚合：失败任务的下游标记 skipped，其余继续；聚合文本回主 Agent
- 上下文重放：并发执行统一经 ``events.snapshot_request_context()`` +
  ``run_with_request_context()``（trace/事件日志、skills 白名单、KB 授权、
  chat model 选择全部重放），事件以 ``spawn/<key>`` span 上报
- 熔断复用：子任务失败同样计入 task_tool 的 S5 熔断（同口径防护）

产出物同时写入结构化黑板（``deep/blackboard.py``），生命周期 = 本次调用。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.agents.deep.blackboard import Blackboard
from app.core.exceptions import PlanningError
from app.core.logger import get_logger

logger = get_logger(__name__)

# 同层并发上限（单进程部署，避免子 Agent 过多挤占 LLM 配额）
DEFAULT_MAX_WORKERS = 4

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_CANCELLED = "cancelled"  # 阶段 4：revise_plan 取消（仅未成功的任务）


@dataclass(frozen=True)
class TaskSpec:
    """spawn_tasks 的单个任务声明。"""

    key: str
    description: str
    subagent_type: str
    depends_on: Tuple[str, ...] = ()


@dataclass
class TaskResult:
    key: str
    status: str
    output: str = ""
    error: str = ""
    # 阶段 4：结构化尾部（parse_result_tail），status/concerns/suggested_followup
    tail: Dict[str, str] = field(default_factory=dict)


# ── DAG 校验与分层 ────────────────────────────────────────────────────────


def validate_dag(specs: List[TaskSpec]) -> List[List[str]]:
    """校验任务集合法且无环，返回按依赖分层的 key 列表（层内可并发）。

    依赖引用允许指向本批之外（如 revise_plan 追加时引用上一轮已完成任务，
    视为已就绪的外部前置）；自依赖/环/重复 key 仍拒绝。

    Raises:
        PlanningError: 空任务集 / key 重复或非法 / 自依赖 / 有环。
    """
    if not specs:
        raise PlanningError("spawn_tasks 至少需要一个任务")
    keys: List[str] = []
    seen = set()
    for s in specs:
        key = (s.key or "").strip()
        if not key:
            raise PlanningError("任务 key 不能为空")
        if key in seen:
            raise PlanningError(f"任务 key 重复: {key}")
        seen.add(key)
        keys.append(key)
        if key in s.depends_on:
            raise PlanningError(f"任务 {key} 不能依赖自己")
    internal = set(keys)
    spec_map = {s.key: s for s in specs}
    remaining = set(keys)
    layers: List[List[str]] = []
    while remaining:
        # 外部前置（不在本批）视为已就绪；环仅会在内部依赖间形成，故能检测
        ready = sorted(
            k for k in remaining
            if all(
                dep not in remaining
                for dep in spec_map[k].depends_on
                if dep in internal
            )
        )
        if not ready:
            raise PlanningError(
                f"任务依赖存在环，无法调度: {', '.join(sorted(remaining))}"
            )
        layers.append(ready)
        remaining -= set(ready)
    return layers


# ── 输入解析 ──────────────────────────────────────────────────────


def parse_task_inputs(tasks: List[Dict[str, Any]]) -> List[TaskSpec]:
    """把工具入参（list of dict）转成 TaskSpec，字段缺失给明确错误。"""
    specs: List[TaskSpec] = []
    for i, item in enumerate(tasks or []):
        if not isinstance(item, dict):
            raise PlanningError(f"tasks[{i}] 必须是对象")
        desc = str(item.get("description") or "").strip()
        subagent = str(item.get("subagent_type") or "").strip()
        key = str(item.get("key") or f"task_{i + 1}").strip()
        if not desc:
            raise PlanningError(f"tasks[{i}] 缺少 description")
        if not subagent:
            raise PlanningError(f"tasks[{i}] 缺少 subagent_type")
        deps = tuple(str(d) for d in (item.get("depends_on") or []))
        specs.append(TaskSpec(
            key=key, description=desc, subagent_type=subagent, depends_on=deps,
        ))
    return specs


# ── 调度执行 ──────────────────────────────────────────────────────────────


def _run_one_task(
    spec: TaskSpec,
    board: Blackboard,
    dep_injection: str,
    model=None,
    recursion_limit: int = 20,
) -> TaskResult:
    """在工作线程内执行单个子任务（调用方已重放请求上下文）。"""
    from app.agents.deep.subagents import get_subagent_config, run_subagent
    from app.agents.deep.task_tool import _breaker_check, _breaker_record
    from app.agents.events import emit, use_span
    from app.observability.tracing import trace_span

    cfg = get_subagent_config(spec.subagent_type)
    if cfg is None:
        from app.agents.deep.subagents import get_subagents

        available = ", ".join(c.name for c in get_subagents())
        return TaskResult(
            key=spec.key, status=STATUS_FAILED,
            error=f"未知子智能体类型 '{spec.subagent_type}'，可用: {available}",
        )
    tripped = _breaker_check(spec.subagent_type)
    if tripped:
        return TaskResult(key=spec.key, status=STATUS_FAILED, error=tripped)

    description = spec.description
    if dep_injection:
        description = (
            f"{spec.description}\n\n【依赖任务产出（供参考）】\n{dep_injection}"
        )
    # use_span：本任务的所有事件（含黑板写通知）以 spawn/<key> 标识
    with use_span(f"spawn/{spec.key}"), trace_span(
        f"spawn_task.{spec.key}", task_key=spec.key,
        subagent=spec.subagent_type,
    ):
        emit(
            "delegation", "task_start", f"spawn {spec.key}",
            f"subagent={spec.subagent_type} {description[:200]}",
            task_key=spec.key, subagent_type=spec.subagent_type,
        )
        try:
            from app.agents.deep.observe import (
                get_task_observers,
                use_subagent_observers,
            )

            on_step, on_artifact = get_task_observers() or (None, None)
            if on_step is None and on_artifact is None:
                output = run_subagent(
                    cfg, description, model=model, recursion_limit=recursion_limit
                )
            else:
                with use_subagent_observers(on_step, on_artifact):
                    output = run_subagent(
                        cfg, description, model=model,
                        recursion_limit=recursion_limit,
                    )
            _breaker_record(spec.subagent_type, True)
            board.post(
                spec.key, producer=spec.subagent_type, summary=output, data=output,
            )
            from app.agents.deep.subagents import parse_result_tail

            tail = parse_result_tail(str(output))
            emit(
                "delegation", "task_end", f"spawn {spec.key} 完成",
                str(output)[:200],
                task_key=spec.key, subagent_type=spec.subagent_type,
                tail_status=tail.get("status", "unknown"),
            )
            return TaskResult(
                key=spec.key, status=STATUS_OK, output=str(output), tail=tail,
            )
        except Exception as exc:
            _breaker_record(spec.subagent_type, False)
            logger.warning(
                "[spawn_tasks] %s (%s) failed: %s",
                spec.key, spec.subagent_type, exc,
            )
            emit(
                "delegation", "task_error", f"spawn {spec.key} 失败", str(exc)[:200],
                task_key=spec.key, subagent_type=spec.subagent_type,
            )
            return TaskResult(key=spec.key, status=STATUS_FAILED, error=str(exc))


def run_spawn_tasks(
    specs: List[TaskSpec],
    model=None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    board: Optional[Blackboard] = None,
) -> Dict[str, TaskResult]:
    """按拓扑层级调度执行全部任务，返回 key → TaskResult。

    层内并发（各自独立上下文快照），层间串行；依赖失败/跳过的任务其下游
    标记 skipped（不执行）。
    """
    from app.core.config import get_settings
    from app.observability.tracing import trace_span

    layers = validate_dag(specs)
    board = board or Blackboard()
    spec_map = {s.key: s for s in specs}
    recursion_limit = get_settings().DEEP_SUBAGENT_RECURSION_LIMIT

    with trace_span("spawn_tasks", tasks=len(specs)):
        return _run_spawn_layers(specs, layers, board, spec_map, model, max_workers,
                                 recursion_limit)


def _run_spawn_layers(
    specs: List[TaskSpec],
    layers: List[List[str]],
    board: Blackboard,
    spec_map: Dict[str, TaskSpec],
    model,
    max_workers: int,
    recursion_limit: int,
) -> Dict[str, TaskResult]:
    """拓扑分层执行（从 run_spawn_tasks 拆出，便于包遥测 span）。"""
    from app.agents.events import emit, run_with_request_context, snapshot_request_context

    results: Dict[str, TaskResult] = {}

    emit(
        "delegation", "spawn_start", f"批量委派 {len(specs)} 个任务",
        "; ".join(f"{k}<-({','.join(spec_map[k].depends_on)})" for k in spec_map),
        task_keys=list(spec_map.keys()), layers=layers,
        # 任务清单（阶段 5：前端侧边面板/委派落库复用同一结构）
        tasks=[
            {"task_id": s.key, "goal": s.description[:200],
             "worker_hint": s.subagent_type}
            for s in specs
        ],
    )
    for layer in layers:
        runnable = []
        for key in layer:
            spec = spec_map[key]
            bad_dep = next(
                (d for d in spec.depends_on
                 if results.get(d) and results[d].status != STATUS_OK),
                None,
            )
            if bad_dep:
                results[key] = TaskResult(
                    key=key, status=STATUS_SKIPPED,
                    error=f"依赖任务 {bad_dep} 未成功（{results[bad_dep].status}）",
                )
                emit(
                    "delegation", "task_skip", f"spawn {key} 跳过",
                    results[key].error, task_key=key,
                )
            else:
                runnable.append(spec)
        if not runnable:
            continue
        # 调度点为每个并发任务各捕获一份上下文快照（同一 Context 不能多线程并发 enter）
        snapshots = [snapshot_request_context() for _ in runnable]
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(runnable)),
            thread_name_prefix="spawn",
        ) as pool:
            futures = []
            for spec, snap in zip(runnable, snapshots):
                dep_injection = board.render_for_injection(list(spec.depends_on))
                futures.append(pool.submit(
                    run_with_request_context, snap, _run_one_task,
                    spec, board, dep_injection, model, recursion_limit,
                ))
            wait(futures)
            # futures 与 runnable 顺序对应；worker 只返回结果，主线程写入
            for spec, fut in zip(runnable, futures):
                results[spec.key] = fut.result()

    ok = sum(1 for r in results.values() if r.status == STATUS_OK)
    emit(
        "delegation", "spawn_end", f"批量委派完成 {ok}/{len(specs)}", "",
        succeeded=ok, total=len(specs),
    )
    return results


def aggregate_results(results: Dict[str, TaskResult], spec_order: List[str]) -> str:
    """把各任务结果聚合为单条文本（作为 ToolMessage 回主 Agent）。"""
    order = [k for k in spec_order if k in results]
    ok = sum(1 for r in results.values() if r.status == STATUS_OK)
    lines = [f"批量委派完成：共 {len(order)} 个任务，成功 {ok} 个。"]
    for key in order:
        r = results[key]
        if r.status == STATUS_OK:
            tail_note = ""
            if r.tail.get("concerns"):
                tail_note += f"\n遗留关注: {r.tail['concerns'][:150]}"
            if r.tail.get("suggested_followup"):
                tail_note += f"\n建议后续: {r.tail['suggested_followup'][:150]}"
            lines.append(f"[{key}] ✅ 成功\n{r.output[:600]}{tail_note}")
        elif r.status == STATUS_FAILED:
            lines.append(f"[{key}] ❌ 失败：{r.error[:200]}")
        elif r.status == STATUS_CANCELLED:
            lines.append(f"[{key}] 🚫 已取消")
        else:
            lines.append(f"[{key}] ⏭️ 跳过：{r.error[:200]}")
    return "\n\n".join(lines)


# ── 计划状态与动态修订（2026-08-26 阶段 4）──────────────────────────
# spawn_tasks 同步执行完毕后，把声明与结果记入会话级 PlanState，供
# revise_plan 追加/取消/细化。已完成（ok）的任务不可撤销；取消仅作用于
# 未成功的任务；追加/细化会立即执行新任务并合并回计划。


@dataclass
class PlanState:
    spec_map: Dict[str, TaskSpec]
    results: Dict[str, TaskResult]
    order: List[str]
    ts: float = field(default_factory=time.time)


# 会话级计划状态（单进程；同一会话新一轮 spawn_tasks 覆盖旧计划）
_plans: Dict[str, PlanState] = {}


def reset_plans() -> None:
    """清空计划状态（测试用）。"""
    _plans.clear()


def _plan_key() -> str:
    from app.agents.events import get_trace

    trace = get_trace()
    if trace:
        return trace.session_id or trace.trace_id
    return ""


def get_plan_state(key: Optional[str] = None) -> Optional[PlanState]:
    return _plans.get(key if key is not None else _plan_key())


def _save_plan(specs: List[TaskSpec], results: Dict[str, TaskResult]) -> None:
    _plans[_plan_key()] = PlanState(
        spec_map={s.key: s for s in specs},
        results=results,
        order=[s.key for s in specs],
    )


REVISE_PLAN_TOOL_DESCRIPTION = """修订当前委派计划（在 spawn_tasks 执行后使用）。

参数 actions 为修订动作列表，每项含 action 字段：
- "add": 追加新任务并立即执行（需 key 唯一、description、subagent_type；
  depends_on 只能引用已成功的任务）
- "refine": 细化重发已存在任务（用新 description 重新执行，需 key + description）
- "cancel": 取消未成功的任务（已完成的不可撤销，需 key）

返回：修订说明 + 新执行任务的聚合结果。无现行计划时返回提示（先用 spawn_tasks）。
"""


def revise_plan_payload(actions: List[Dict[str, Any]], model=None) -> str:
    """revise_plan 工具入口：在现行计划上追加/取消/细化，新任务立即执行。"""
    from app.agents.events import emit

    state = get_plan_state()
    if state is None:
        return "当前没有可修订的计划：请先用 spawn_tasks 声明任务。"
    lines: List[str] = []
    to_run: List[TaskSpec] = []
    for i, act in enumerate(actions or []):
        if not isinstance(act, dict):
            lines.append(f"动作[{i}] 格式错误（需对象）")
            continue
        action = str(act.get("action") or "").strip()
        key = str(act.get("key") or "").strip()
        if action == "cancel":
            r = state.results.get(key)
            if r is None:
                lines.append(f"[{key}] 取消失败：任务不存在")
            elif r.status == STATUS_OK:
                lines.append(f"[{key}] 无法取消：任务已完成，不可撤销")
            elif r.status == STATUS_CANCELLED:
                lines.append(f"[{key}] 已取消过，无需重复")
            else:
                r.status = STATUS_CANCELLED
                r.error = "用户取消"
                lines.append(f"[{key}] 已取消")
        elif action == "add":
            desc = str(act.get("description") or "").strip()
            subagent = str(act.get("subagent_type") or "").strip()
            if not key or not desc or not subagent:
                lines.append(f"动作[{i}] add 需要 key、description、subagent_type")
                continue
            if key in state.spec_map:
                lines.append(f"[{key}] 追加失败：key 已存在（改用 refine）")
                continue
            deps = tuple(str(d) for d in (act.get("depends_on") or []))
            bad = next(
                (d for d in deps
                 if state.results.get(d) is None
                 or state.results[d].status != STATUS_OK),
                None,
            )
            if bad is not None:
                lines.append(f"[{key}] 追加失败：依赖 {bad} 未成功完成")
                continue
            spec = TaskSpec(key=key, description=desc, subagent_type=subagent,
                            depends_on=deps)
            state.spec_map[key] = spec
            state.order.append(key)
            to_run.append(spec)
            lines.append(f"[{key}] 已加入计划并执行")
        elif action == "refine":
            desc = str(act.get("description") or "").strip()
            if not key or not desc:
                lines.append(f"动作[{i}] refine 需要 key 和 description")
                continue
            old = state.spec_map.get(key)
            if old is None:
                lines.append(f"[{key}] 细化失败：任务不存在")
                continue
            spec = TaskSpec(key=key, description=desc,
                            subagent_type=old.subagent_type, depends_on=())
            state.spec_map[key] = spec
            to_run.append(spec)
            lines.append(f"[{key}] 已用新描述重新执行")
        else:
            lines.append(f"动作[{i}] 未知 action: {action!r}（可用 add/refine/cancel）")

    if to_run:
        new_results = run_spawn_tasks(to_run, model=model)
        state.results.update(new_results)
        lines.append("")
        lines.append(aggregate_results(
            new_results, [s.key for s in to_run]
        ))
    emit(
        "plan", "revise", f"计划修订 {len(to_run)} 项执行", "; ".join(lines[:6])[:300],
        executed=[s.key for s in to_run],
    )
    return "\n".join(lines)


# ── 工具入口 ──────────────────────────────────────────────────────────────

SPAWN_TASKS_TOOL_DESCRIPTION = """批量委派多个子任务给子智能体并行执行（支持依赖关系）。

适用于复杂多域任务：先规划拆解为多个子任务，用 depends_on 表达先后依赖，
调度器按拓扑层级并发执行并把依赖任务的产出注入后续任务。简单单任务请用 `task`。

参数：
- tasks: 任务列表，每项包含
  - key: 任务标识（可选，缺省 task_1/task_2...）
  - description: 任务描述（必填，写清目标/背景/期望输出）
  - subagent_type: 子智能体名称（必填，见下方可用列表）
  - depends_on: 依赖的任务 key 列表（可选，依赖任务先执行且产出会注入本任务）

返回：各任务状态与产出的聚合文本。校验失败（如依赖成环）会直接返回错误说明。
"""


def spawn_tasks_payload(tasks: List[Dict[str, Any]], model=None) -> str:
    """spawn_tasks 工具的同步入口（校验失败返回错误文本，供主 Agent 修正）。"""
    try:
        specs = parse_task_inputs(tasks)
        validate_dag(specs)  # 提前校验：环/重复等在调度前就给出明确错误
    except PlanningError as exc:
        logger.warning("[spawn_tasks] validation failed: %s", exc)
        return f"批量委派校验失败：{exc}。请修正任务声明（依赖不能成环、key 不能重复）后重试。"
    results = run_spawn_tasks(specs, model=model)
    _save_plan(specs, results)  # 阶段 4：供 revise_plan 追加/取消/细化
    return aggregate_results(results, [s.key for s in specs])


def build_spawn_tasks_tool(model=None) -> Any:
    """构建 ``spawn_tasks`` StructuredTool。model: 测试 mock 注入点。"""
    from langchain_core.tools import StructuredTool

    from app.agents.deep.subagents import subagents_prompt

    def _spawn(tasks: List[Dict[str, Any]]) -> str:
        return spawn_tasks_payload(tasks, model=model)

    return StructuredTool.from_function(
        func=_spawn,
        name="spawn_tasks",
        description=(
            SPAWN_TASKS_TOOL_DESCRIPTION + "\n可用 subagent_type:\n" + subagents_prompt()
        ),
        infer_schema=True,
    )


def build_revise_plan_tool(model=None) -> Any:
    """构建 ``revise_plan`` StructuredTool（阶段 4：动态规划）。"""
    from langchain_core.tools import StructuredTool

    def _revise(actions: List[Dict[str, Any]]) -> str:
        return revise_plan_payload(actions, model=model)

    return StructuredTool.from_function(
        func=_revise,
        name="revise_plan",
        description=REVISE_PLAN_TOOL_DESCRIPTION,
        infer_schema=True,
    )
