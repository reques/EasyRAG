"""DeepAgents 委派持久化（2026-08-26，阶段 5）。

把统一事件流（``app/agents/events.py``）中的委派事件解析并落库，复用
orchestrator 时代的 Run/Task/AgentRun 三表（``agent_run_service``）：

- ``extract_delegation_from_events``：从事件列表提取任务清单与终态
  （spawn_start 建任务 → task_end/task_error/task_skip 收口；
  无委派事件返回 None）
- ``persist_delegation``：best-effort 落库（create_run + create_tasks +
  finish_task + finalize_run），任何异常只记日志不阻塞主流程

事件口径见 ``app/agents/deep/planner.py``（delegation 族）与
``app/agents/deep/task_tool.py``（单任务委派同样落库为 1 任务 Run）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# 委派事件状态 → agent_run 落库状态（finish_task 按 worker_status 映射）
_STATUS_MAP = {
    "ok": "done",
    "failed": "error",
    "skipped": "skipped",
    "cancelled": "cancelled",
}


def extract_delegation_from_events(
    events: Optional[List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """从统一事件流提取委派执行摘要；无委派事件返回 None。

    返回::

        {
            "goal": 建议 Run.goal（首个任务描述），
            "tasks": [{"task_id", "goal", "worker_hint"}, ...]（首次出现序）,
            "statuses": {task_id: ok|failed|skipped|cancelled},
            "outputs": {task_id: 输出摘要},
            "errors": {task_id: 错误摘要},
            "run_status": "completed" | "failed",
            "error_summary": str,
        }

    终态判定：存在 spawn_end 且成功数 > 0 → completed（部分失败视为完成，
    主 Agent 会继续兜底）；全部失败或未收到 spawn_end → failed。
    单任务 ``task`` 委派（仅 task_start）按 completed 记（task_tool 内部
    已把异常转成返回文本，不产生 task_error）。
    """
    tasks: List[Dict[str, Any]] = []
    seen: set = set()
    statuses: Dict[str, str] = {}
    outputs: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    spawn_started = False
    spawn_ended = False
    succeeded = 0
    total = 0

    for ev in events or []:
        if ev.get("kind") != "delegation":
            continue
        stage = ev.get("stage", "")
        key = ev.get("task_key", "")
        if stage == "spawn_start":
            spawn_started = True
            total = int(ev.get("total") or len(ev.get("task_keys") or []) or 0)
        elif stage == "spawn_end":
            spawn_ended = True
            succeeded = int(ev.get("succeeded") or 0)
        elif stage in ("task_start",) and key and key not in seen:
            seen.add(key)
            tasks.append({
                "task_id": key,
                "goal": (ev.get("content") or "")[:400],
                "worker_hint": ev.get("subagent_type") or "subagent",
            })
        elif stage == "task_end" and key:
            statuses[key] = "ok"
            outputs[key] = (ev.get("content") or "")[:300]
        elif stage == "task_error" and key:
            statuses[key] = "failed"
            errors[key] = (ev.get("content") or "")[:300]
        elif stage == "task_skip" and key:
            # 被跳过的任务无 task_start（调度前即标记），也要登记以保留 DAG 全貌
            if key not in seen:
                seen.add(key)
                tasks.append({
                    "task_id": key,
                    "goal": (ev.get("content") or "")[:400],
                    "worker_hint": ev.get("subagent_type") or "scheduler",
                })
            statuses[key] = "skipped"
            errors[key] = (ev.get("content") or "")[:300]

    if not tasks:
        return None
    if not spawn_started and not spawn_ended:
        # task 工具的单任务委派：只有 task_start（无 spawn 事件）
        run_status = "completed"
        error_summary = ""
    else:
        ok_count = succeeded if spawn_ended else sum(
            1 for s in statuses.values() if s == "ok"
        )
        run_status = "completed" if ok_count > 0 else "failed"
        error_summary = "" if run_status == "completed" else "全部委派任务失败"

    goal = tasks[0]["goal"][:200] if tasks else "委派任务"
    return {
        "goal": goal,
        "tasks": tasks,
        "statuses": statuses,
        "outputs": outputs,
        "errors": errors,
        "run_status": run_status,
        "error_summary": error_summary,
    }


async def persist_delegation(
    session_factory,
    *,
    conversation_id,
    user_id,
    events: Optional[List[Dict[str, Any]]],
    goal: str = "",
    model_id: str = "",
    source_message_id: Optional[int] = None,
) -> Optional[str]:
    """把委派执行落库（best-effort）。返回 run_id 字符串，无委派/失败返回 None。

    session_factory: async contextmanager，产出 AsyncSession（路由层注入
    ``get_session``，测试可注入 fake）。conversation_id/user_id 缺失或非
    UUID 时由调用方跳过（不在此兜底构造假身份）。
    """
    from backend.services import agent_run_service as ars

    summary = extract_delegation_from_events(events)
    if summary is None:
        return None
    try:
        async with session_factory() as session:
            run = await ars.create_run(
                session,
                conversation_id=conversation_id,
                user_id=user_id,
                goal=goal or summary["goal"],
                model_id=model_id,
                source_message_id=source_message_id,
            )
            await ars.create_tasks(
                session, run.id, summary["tasks"], model_id
            )
            for task in summary["tasks"]:
                key = task["task_id"]
                status = summary["statuses"].get(key)
                if status is None:
                    continue  # 只有 task_start 未收口（如进程中断），保持 pending
                await ars.finish_task(
                    session,
                    run.id,
                    key,
                    worker_status=_STATUS_MAP.get(status, "error"),
                    output_summary=summary["outputs"].get(key, ""),
                    error_summary=summary["errors"].get(key, ""),
                )
            await ars.finalize_run(
                session,
                run.id,
                status=summary["run_status"],
                execution_mode="deepagents",
                error_summary=summary["error_summary"],
            )
            await session.commit()
        return str(run.id)
    except Exception as exc:
        logger.warning("[delegation_persist] best-effort persist failed: %s", exc)
        return None


# ── 统一事件流 → orchestrator 时代 SSE 协议（阶段 5，前端面板复用）───────────
# 纯函数映射：把一条统一事件翻译成零或多条 SSE 载荷（sub_tasks / status /
# worker_output / progress_summary），由 chat/stream 的 deep 路径经事件 sink
# 推送，前端既有任务面板与 AgentActivity 直接消费（委派树 + 工具时间线）。


def bridge_delegation_event(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把一条统一事件流事件映射为 SSE 载荷列表；无关事件返回空列表。

    单任务委派（task 工具）无 spawn_start：调用方在首次收到 task_start
    时应自造单任务 ``sub_tasks`` 清单（面板状态在调用方维护）。

    artifact 族（阶段 6，2026-08-27）：推理 / 工具调用 / 工具返回 / 检索片段
    以 ``artifact`` SSE 载荷实时下发到当前会话；若事件发生在子任务 span
    内且属于工具动作，则同时追加 ``tool_call`` 载荷，供任务面板工具时间线消费。
    """
    kind = event.get("kind", "")
    stage = event.get("stage", "")
    span = event.get("span", "") or ""
    span_task = span.split("/", 1)[1] if "/" in span else ""

    if kind == "delegation":
        if stage == "spawn_start":
            tasks = event.get("tasks") or []
            if tasks:
                return [{"type": "sub_tasks", "tasks": tasks}]
            return []
        if stage == "task_start":
            return [{
                "type": "status", "step": "task_started",
                "detail": event.get("title", ""),
                "task_id": event.get("task_key", ""),
            }]
        if stage in ("task_end", "task_error", "task_skip"):
            return [{
                "type": "worker_output",
                "task_id": event.get("task_key", ""),
                "worker": event.get("subagent_type", ""),
                "status": (
                    "error" if stage == "task_error"
                    else "skipped" if stage == "task_skip"
                    else "done"
                ),
                "error": event.get("content", "") if stage == "task_error" else "",
                "content": event.get("content", ""),
                "summary": (event.get("content") or "")[:200],
            }]
        return []

    if kind == "tool":
        if stage == "tool_start":
            return [{
                "type": "status", "step": "tool",
                "detail": event.get("title", ""), "task_id": span_task,
            }]
        if stage == "tool_end":
            return [{
                "type": "status", "step": "tool_done",
                "detail": event.get("title", ""), "task_id": span_task,
            }]
        if stage == "tool_error":
            return [{
                "type": "status", "step": "fallback",
                "detail": event.get("title", ""), "task_id": span_task,
            }]
        if stage == "progress":
            return [{
                "type": "progress_summary",
                "id": f"prog-{event.get('ts', 0)}-{span_task}",
                "phase": "tool", "status": "running",
                "text": event.get("title", ""),
            }]

    if kind == "artifact":
        artifact_kind = event.get("artifact_kind") or "info"
        title = str(event.get("title", "") or "")[:80]
        content = str(event.get("content", "") or "")
        payloads: List[Dict[str, Any]] = [{
            "type": "artifact",
            "kind": artifact_kind,
            "stage": event.get("stage", ""),
            "title": title,
            "content": content,
            "streaming": False,
        }]
        # 子任务内部的工具调用 → 任务面板工具时间线（与 worker_output 卡互补）
        if artifact_kind in ("tool", "tool_result", "delegate") and span_task:
            detail = " ".join((content or title).split())[:120]
            payloads.append({
                "type": "tool_call",
                "task_id": span_task,
                "detail": detail,
            })
        return payloads
    return []
