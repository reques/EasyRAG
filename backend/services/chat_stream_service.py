"""A durable SSE journal with detached producers and explicit cancellation.

PostgreSQL is authoritative. HTTP clients only read this journal; disconnecting
one reader never closes its producer. Each active producer renews an ownership
heartbeat. Process loss becomes an interrupted run, not a guessed success.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
import hashlib
import json
from threading import Event
import uuid

import anyio
from fastapi import HTTPException
from sqlalchemy import select

from app.agents.cancellation import AgentRunCancelled
from app.core.logger import get_logger
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_chat_stream import ChatStream, ChatStreamEvent

logger = get_logger(__name__)
ACTIVE = ("running", "cancel_requested")
TERMINAL = ("completed", "cancelled", "failed", "interrupted")
HEARTBEAT_SECONDS = 5
STALE_SECONDS = 90


def fingerprint(payload: dict) -> str:
    data = {key: value for key, value in payload.items() if key != "request_id"}
    if data.get("workspace_id") is None:
        data.pop("workspace_id", None)  # Preserve fingerprints of requests admitted before workspaces.
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def view(run: ChatStream) -> dict:
    return {
        "id": str(run.id), "conversation_id": str(run.conversation_id),
        "status": run.status, "last_sequence": run.last_sequence,
        "source_message_id": run.source_message_id,
        "recoverable": run.status == "interrupted",
    }


def encode(payload: dict) -> str:
    return f"id: {payload['sequence']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def append_in_session(session, run: ChatStream, payload: dict, terminal: str | None = None):
    run.last_sequence += 1
    payload = {**payload, "stream_id": str(run.id), "sequence": run.last_sequence}
    session.add(ChatStreamEvent(stream_id=run.id, sequence=run.last_sequence, payload_json=json.dumps(payload, ensure_ascii=False)))
    run.heartbeat_at = datetime.now(timezone.utc)
    if terminal:
        run.status = terminal
        run.finished_at = run.heartbeat_at
    return payload


async def owned_run(session, stream_id, user_id, *, lock=False):
    stmt = select(ChatStream).where(ChatStream.id == stream_id, ChatStream.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update()
    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, "Chat execution not found")
    return run


async def writable_run(session, stream_id, owner_token):
    run = (await session.execute(select(ChatStream).where(ChatStream.id == stream_id).with_for_update())).scalar_one_or_none()
    if run is None or run.owner_token != owner_token or run.status != "running":
        raise AgentRunCancelled()
    return run


async def latest_stream(session, conversation_id, user_id):
    return (await session.execute(
        select(ChatStream).where(ChatStream.conversation_id == conversation_id, ChatStream.user_id == user_id)
        .order_by(ChatStream.created_at.desc(), ChatStream.id.desc()).limit(1)
    )).scalar_one_or_none()


async def recover_stale(session, run):
    if run.status not in ACTIVE:
        return
    heartbeat = run.heartbeat_at
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    if heartbeat < datetime.now(timezone.utc) - timedelta(seconds=STALE_SECONDS):
        # Re-lock/re-read: another worker may have renewed it since the first read.
        await session.refresh(run, with_for_update=True)
        heartbeat = run.heartbeat_at.replace(tzinfo=timezone.utc) if run.heartbeat_at.tzinfo is None else run.heartbeat_at
        if run.status in ACTIVE and heartbeat < datetime.now(timezone.utc) - timedelta(seconds=STALE_SECONDS):
            terminal = "cancelled" if run.status == "cancel_requested" else "interrupted"
            append_in_session(session, run, {
                "type": "cancelled" if terminal == "cancelled" else "error",
                "detail": "执行进程已中断，可从断点恢复。" if terminal == "interrupted" else "任务已取消。",
                "recoverable": terminal == "interrupted",
            }, terminal)
            await session.commit()
            from types import SimpleNamespace
            from backend.services.observability_service import sync_execution_trace
            await sync_execution_trace(SimpleNamespace(id=run.id, user_id=run.user_id, owner_token=run.owner_token))


class StreamExecution:
    def __init__(self, stream_id, owner_token):
        self.id = stream_id
        self.owner_token = owner_token
        self.cancel_event = Event()
        self.interrupted = False
        self.task = None
        self.workers = []


_executions: dict[uuid.UUID, StreamExecution] = {}


async def _append(execution, payload, terminal=None):
    async with get_session() as session:
        run = await owned_run(session, execution.id, execution.user_id, lock=True)
        if run.owner_token != execution.owner_token or run.status in TERMINAL:
            execution.interrupted = run.status != "cancelled"
            execution.cancel_event.set()
            return
        if run.status == "cancel_requested" and terminal is None:
            execution.cancel_event.set()
            # Discard late output, but drain the executor before acknowledging stop.
            return
        if run.status == "cancel_requested" and terminal:
            terminal = "cancelled"
            payload = {"type": "cancelled", "detail": "已停止生成", "recoverable": False}
        append_in_session(session, run, payload, terminal)
        await session.commit()


async def _watch(execution):
    tick = 0
    while True:
        async with get_session() as session:
            run = await owned_run(session, execution.id, execution.user_id, lock=True)
            if run.owner_token != execution.owner_token or run.status in TERMINAL:
                execution.interrupted = run.status != "cancelled"
                execution.cancel_event.set()
                return
            if run.status == "cancel_requested":
                execution.cancel_event.set()
            if tick % (HEARTBEAT_SECONDS * 2) == 0:
                run.heartbeat_at = datetime.now(timezone.utc)
                await session.commit()
        tick += 1
        await asyncio.sleep(0.5)


async def _watch_safely(execution):
    try:
        await _watch(execution)
    except Exception:
        # A producer that can no longer renew ownership must stop further work.
        logger.exception("[chat-stream] heartbeat failed stream=%s", execution.id)
        execution.interrupted = True
        execution.cancel_event.set()


async def _produce(execution, factory: Callable[[], AsyncIterator[str]]):
    try:
        await _produce_traced(execution, factory)
    finally:
        _executions.pop(execution.id, None)


async def _produce_traced(execution, factory: Callable[[], AsyncIterator[str]]):
    from app.agents.events import use_request_trace
    from app.observability.context import ExecutionContext, execution_scope
    from backend.services.observability_service import sync_execution_trace
    async with get_session() as session:
        run = await owned_run(session, execution.id, execution.user_id)
        previous = await session.get(ChatStream, run.resumed_from_stream_id) if run.resumed_from_stream_id else None
        context = ExecutionContext(
            execution_id=str(run.id), trace_id=run.observability_trace_id or uuid.uuid4().hex,
            local_trace_id=run.local_trace_id or "", user_id=str(run.user_id),
            session_id=str(run.conversation_id), sampled=bool(run.observability_sampled),
            metadata={"resumedFromTraceId": previous.observability_trace_id if previous else None,
                      "resumedFromStreamId": str(previous.id) if previous else None},
        )
    with execution_scope(context) as root, use_request_trace(
        execution.conversation_id, trace_id=context.local_trace_id
    ) as local:
        await sync_execution_trace(execution)
        try:
            await _produce_inner(execution, factory)
        finally:
            # Local events remain available even when cancellation skipped the
            # normal AgentService result. No export/flush is awaited here.
            await sync_execution_trace(execution, local.events)
            try:
                async with get_session() as session:
                    final = await owned_run(session, execution.id, execution.user_id)
                    status = final.status if final.status in TERMINAL else "interrupted"
                if status != "completed":
                    root.error(status=status)
                root.end(status)
            except Exception as exc:
                root.error(exc, status="interrupted")


async def _produce_inner(execution, factory: Callable[[], AsyncIterator[str]]):
    watcher = asyncio.create_task(_watch_safely(execution))
    source = factory()
    try:
        async for frame in source:
            payload = next((json.loads(line[5:].strip()) for line in frame.splitlines() if line.startswith("data:")), None)
            if payload:
                await _append(execution, payload)
        async with get_session() as session:
            run = await owned_run(session, execution.id, execution.user_id)
            finished = run.status in TERMINAL
        if not finished:
            raise RuntimeError("Execution ended before its result was committed")
    except AgentRunCancelled:
        async with get_session() as session:
            run = await owned_run(session, execution.id, execution.user_id)
            explicit = run.owner_token == execution.owner_token and run.status == "cancel_requested"
        if explicit:
            from app.agents.checkpointing import delete_checkpoint_thread
            # Keep the active reservation until cleanup finishes, so a new turn
            # cannot create a checkpoint that this cleanup would then delete.
            await asyncio.to_thread(delete_checkpoint_thread, execution.conversation_id)
            await _append(execution, {"type": "cancelled", "detail": "已停止生成", "recoverable": False}, "cancelled")
        else:
            # Losing a lease is an interruption. An obsolete producer must never
            # remove checkpoints that a replacement may already be using.
            await _append(execution, {"type": "error", "detail": "执行已中断，任务已保留，可从断点恢复。", "recoverable": True}, "interrupted")
    except asyncio.CancelledError:
        execution.cancel_event.set()
        execution.interrupted = True
        await _append(execution, {"type": "error", "detail": "服务中断，可从断点恢复。", "recoverable": True}, "interrupted")
    except Exception:
        from app.observability.context import record_error
        record_error(reason="execution_or_commit_failed")
        logger.exception("[chat-stream] producer failed stream=%s", execution.id)
        execution.cancel_event.set()
        # A journal failure may occur while the synchronous executor still runs.
        # Stop and drain it before exposing a resumable terminal state.
        if execution.workers:
            await asyncio.gather(*execution.workers, return_exceptions=True)
        await _append(execution, {"type": "error", "detail": "任务执行或保存失败，可从断点恢复。", "recoverable": True}, "interrupted")
    finally:
        execution.cancel_event.set()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await source.aclose()


async def commit_result(execution, answer, metadata, payload):
    from app.observability.context import observation_scope, current_observation
    parent = current_observation()
    try:
        with observation_scope("result.commit"):
            await _commit_result_inner(execution, answer, metadata, payload)
    except Exception:
        if parent:
            parent.update(metadata={"failureStage": "result.commit"})
        raise


async def _commit_result_inner(execution, answer, metadata, payload):
    """Answer and terminal event commit together; cancellation competes on this row."""
    from backend.services.chat_service import add_message
    if execution.cancel_event.is_set():
        raise AgentRunCancelled()
    async with get_session() as session:
        await add_message(
            session, uuid.UUID(execution.conversation_id), "assistant", answer,
            metadata_json=json.dumps({**metadata, "stream_id": str(execution.id)}, ensure_ascii=False),
        )
        run = await writable_run(session, execution.id, execution.owner_token)
        if execution.cancel_event.is_set():
            raise AgentRunCancelled()
        append_in_session(session, run, payload, "completed")
        await session.commit()


def start_execution(run, factory):
    execution = StreamExecution(run.id, run.owner_token)
    execution.user_id = run.user_id
    execution.conversation_id = str(run.conversation_id)
    _executions[run.id] = execution
    execution.task = asyncio.create_task(_produce(execution, lambda: factory(execution)))
    def report_failure(task):
        if not task.cancelled() and task.exception() is not None:
            # If PostgreSQL itself is unavailable, stale heartbeat recovery will
            # provide the terminal state when storage returns.
            logger.error("[chat-stream] journal unavailable stream=%s: %s", execution.id, task.exception())
    execution.task.add_done_callback(report_failure)
    return execution


async def request_cancel(stream_id, user_id):
    async with get_session() as session:
        run = await owned_run(session, stream_id, user_id, lock=True)
        await recover_stale(session, run)
        if run.status == "running":
            run.status = "cancel_requested"
            await session.commit()
        result = view(run)
    execution = _executions.get(stream_id)
    if execution and result["status"] == "cancel_requested":
        execution.cancel_event.set()
    return result


async def read_events(stream_id, user_id, after=0):
    cursor = after
    idle = 0
    while True:
        # Starlette cancels a disconnected reader with an AnyIO cancel scope.
        # Finish the short read transaction and release its connection before
        # observing that cancellation, keeping the producer's pool healthy.
        with anyio.CancelScope(shield=True):
            async with get_session() as session:
                run = await owned_run(session, stream_id, user_id)
                await recover_stale(session, run)
                records = (await session.execute(
                    select(ChatStreamEvent).where(ChatStreamEvent.stream_id == stream_id, ChatStreamEvent.sequence > cursor)
                    .order_by(ChatStreamEvent.sequence).limit(256)
                )).scalars().all()
                terminal = run.status in TERMINAL
                last_sequence = run.last_sequence
        for record in records:
            cursor = record.sequence
            yield encode(json.loads(record.payload_json))
        if terminal and cursor >= last_sequence:
            return
        idle = 0 if records else idle + 1
        if idle and idle % 100 == 0:
            yield ": keepalive\n\n"
        await asyncio.sleep(0.1)


async def shutdown_streams():
    executions = list(_executions.values())
    for execution in executions:
        execution.interrupted = True
        execution.cancel_event.set()
    if executions:
        _, pending = await asyncio.wait([item.task for item in executions], timeout=3)
        for task in pending:
            task.cancel()
        await asyncio.gather(*(item.task for item in executions), return_exceptions=True)
