"""Unit coverage for the persistent multi-agent run contract."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from backend.services.agent_run_service import finalize_run, serialize_run
from backend.storage.postgres.models_agent_run import AgentRun, Run, Task


def _ids():
    return {
        "user": uuid.uuid4(),
        "conversation": uuid.uuid4(),
        "run": uuid.uuid4(),
        "task": uuid.uuid4(),
        "agent": uuid.uuid4(),
    }


def test_run_schema_has_durable_ownership_and_task_identity():
    run_columns = Run.__table__.columns
    task_columns = Task.__table__.columns
    agent_columns = AgentRun.__table__.columns

    assert {"conversation_id", "user_id", "source_message_id"} <= set(run_columns.keys())
    assert {"status", "progress_completed", "progress_total"} <= set(run_columns.keys())
    assert {"run_id", "task_key", "worker_hint", "status"} <= set(task_columns.keys())
    assert {"run_id", "task_id", "worker_name", "status"} <= set(agent_columns.keys())
    assert any(
        constraint.name == "uq_run_task_key"
        for constraint in Task.__table__.constraints
    )


def test_serialize_run_exposes_tasks_and_worker_runs_without_owner_id():
    ids = _ids()
    now = datetime.now(timezone.utc)
    run = Run(
        id=ids["run"],
        conversation_id=ids["conversation"],
        user_id=ids["user"],
        goal="分析项目并生成报告",
        status="completed",
        mode="multi_agent",
        model_id="test-model",
        execution_mode="parallel",
        progress_completed=1,
        progress_total=1,
        started_at=now,
        completed_at=now,
        created_at=now,
        updated_at=now,
    )
    task = Task(
        id=ids["task"],
        run_id=ids["run"],
        task_key="task-1",
        goal="分析项目",
        worker_hint="rag",
        status="completed",
        position=0,
        started_at=now,
        completed_at=now,
        created_at=now,
        updated_at=now,
    )
    agent_run = AgentRun(
        id=ids["agent"],
        run_id=ids["run"],
        task_id=ids["task"],
        worker_name="rag",
        model_id="test-model",
        status="completed",
        output_summary="已完成",
        tool_call_count=2,
        started_at=now,
        completed_at=now,
        created_at=now,
        updated_at=now,
    )
    run.tasks = [task]
    run.agent_runs = [agent_run]

    payload = serialize_run(run)

    assert payload["id"] == str(ids["run"])
    assert payload["tasks"][0]["task_id"] == "task-1"
    assert payload["agent_runs"][0]["task_id"] == "task-1"
    assert payload["agent_runs"][0]["tool_call_count"] == 2
    assert "user_id" not in payload


@pytest.mark.asyncio
async def test_finalize_run_rejects_non_terminal_status_before_db_access():
    with pytest.raises(ValueError, match="invalid terminal run status"):
        await finalize_run(None, uuid.uuid4(), status="running")
