"""Lifecycle operations for persistent multi-agent runs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.agent_run_repository import (
    AgentRunRepository,
    RunRepository,
    TaskRepository,
)
from backend.storage.postgres.models_agent_run import AgentRun, Run, Task


TERMINAL_TASK_STATUSES = {"completed", "failed", "blocked", "cancelled"}
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def serialize_run(run: Run) -> dict[str, Any]:
    """Return the stable public representation used by run query APIs."""
    task_key_by_id = {task.id: task.task_key for task in run.tasks}
    return {
        "id": str(run.id),
        "conversation_id": str(run.conversation_id),
        "status": run.status,
        "mode": run.mode,
        "goal": run.goal,
        "model_id": run.model_id or "",
        "execution_mode": run.execution_mode or "",
        "progress_completed": run.progress_completed,
        "progress_total": run.progress_total,
        "error_summary": run.error_summary or "",
        "started_at": _iso(run.started_at),
        "completed_at": _iso(run.completed_at),
        "created_at": _iso(run.created_at),
        "updated_at": _iso(run.updated_at),
        "tasks": [
            {
                "id": str(task.id),
                "task_id": task.task_key,
                "goal": task.goal,
                "worker_hint": task.worker_hint or "",
                "status": task.status,
                "position": task.position,
                "error_summary": task.error_summary or "",
                "started_at": _iso(task.started_at),
                "completed_at": _iso(task.completed_at),
            }
            for task in run.tasks
        ],
        "agent_runs": [
            {
                "id": str(agent_run.id),
                "task_id": task_key_by_id.get(agent_run.task_id, ""),
                "worker_name": agent_run.worker_name,
                "model_id": agent_run.model_id or "",
                "status": agent_run.status,
                "output_summary": agent_run.output_summary or "",
                "error_summary": agent_run.error_summary or "",
                "tool_call_count": agent_run.tool_call_count,
                "started_at": _iso(agent_run.started_at),
                "completed_at": _iso(agent_run.completed_at),
            }
            for agent_run in run.agent_runs
        ],
    }


async def create_run(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    goal: str,
    model_id: str,
    source_message_id: Optional[int] = None,
) -> Run:
    run = Run(
        conversation_id=conversation_id,
        user_id=user_id,
        source_message_id=source_message_id,
        goal=goal,
        model_id=model_id,
        status="running",
    )
    await RunRepository(session).add(run)
    return run


async def create_tasks(
    session: AsyncSession,
    run_id: uuid.UUID,
    tasks: Iterable[dict[str, Any]],
    model_id: str,
) -> list[Task]:
    """Create the decomposed tasks and their initial worker executions."""
    task_repo = TaskRepository(session)
    agent_repo = AgentRunRepository(session)
    records: list[Task] = []
    for position, item in enumerate(tasks):
        record = Task(
            run_id=run_id,
            task_key=str(item.get("task_id") or f"task-{position + 1}"),
            goal=str(item.get("goal") or ""),
            worker_hint=str(item.get("worker_hint") or "rag"),
            status="pending",
            position=position,
        )
        await task_repo.add(record)
        await agent_repo.add(AgentRun(
            run_id=run_id,
            task_id=record.id,
            worker_name=record.worker_hint or "rag",
            model_id=model_id,
            status="pending",
        ))
        records.append(record)

    run = await RunRepository(session).get_by_id(run_id)
    if run:
        run.progress_total = len(records)
        run.progress_completed = 0
        await session.flush()
    return records


async def start_pending_tasks(session: AsyncSession, run_id: uuid.UUID) -> None:
    now = datetime.now(timezone.utc)
    tasks = await TaskRepository(session).list_by_run(run_id)
    agent_repo = AgentRunRepository(session)
    for task in tasks:
        if task.status != "pending":
            continue
        task.status = "running"
        task.started_at = now
        agent_run = await agent_repo.get_by_task(task.id)
        if agent_run and agent_run.status == "pending":
            agent_run.status = "running"
            agent_run.started_at = now
    await session.flush()


async def start_task(
    session: AsyncSession, run_id: uuid.UUID, task_key: str
) -> None:
    """Mark one task and its worker run as active."""
    task = await TaskRepository(session).get_by_key(run_id, task_key)
    if not task or task.status != "pending":
        return
    now = datetime.now(timezone.utc)
    task.status = "running"
    task.started_at = now
    agent_run = await AgentRunRepository(session).get_by_task(task.id)
    if agent_run and agent_run.status == "pending":
        agent_run.status = "running"
        agent_run.started_at = now
    await session.flush()


async def finish_task(
    session: AsyncSession,
    run_id: uuid.UUID,
    task_key: str,
    *,
    worker_status: str,
    output_summary: str = "",
    error_summary: str = "",
) -> None:
    task = await TaskRepository(session).get_by_key(run_id, task_key)
    if not task:
        return

    now = datetime.now(timezone.utc)
    successful = worker_status in {"done", "done_with_concerns", "completed"}
    task.status = "completed" if successful else (
        worker_status if worker_status in TERMINAL_TASK_STATUSES else "failed"
    )
    task.error_summary = error_summary or None
    task.started_at = task.started_at or now
    task.completed_at = now

    agent_run = await AgentRunRepository(session).get_by_task(task.id)
    if agent_run:
        agent_run.status = task.status
        agent_run.output_summary = output_summary[:12000] or None
        agent_run.error_summary = error_summary or None
        agent_run.started_at = agent_run.started_at or now
        agent_run.completed_at = now

    tasks = await TaskRepository(session).list_by_run(run_id)
    run = await RunRepository(session).get_by_id(run_id)
    if run:
        run.progress_total = len(tasks)
        run.progress_completed = sum(
            1 for item in tasks if item.status in TERMINAL_TASK_STATUSES
        )
    await session.flush()


async def finalize_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    *,
    status: str,
    execution_mode: str = "",
    error_summary: str = "",
) -> None:
    if status not in TERMINAL_RUN_STATUSES:
        raise ValueError(f"invalid terminal run status: {status}")
    run = await RunRepository(session).get_by_id(run_id)
    if not run:
        return
    tasks = await TaskRepository(session).list_by_run(run_id)
    run.status = status
    run.execution_mode = execution_mode or run.execution_mode
    run.progress_total = len(tasks)
    run.progress_completed = sum(
        1 for task in tasks if task.status in TERMINAL_TASK_STATUSES
    )
    run.error_summary = error_summary or None
    run.completed_at = datetime.now(timezone.utc)
    await session.flush()


async def finalize_run_if_active(
    session: AsyncSession,
    run_id: uuid.UUID,
    *,
    status: str,
    error_summary: str = "",
) -> bool:
    """Finalize a run only when another path has not already closed it."""
    run = await RunRepository(session).get_by_id(run_id)
    if not run or run.status in TERMINAL_RUN_STATUSES:
        return False
    await finalize_run(
        session,
        run_id,
        status=status,
        error_summary=error_summary,
    )
    return True
