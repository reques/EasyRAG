"""Native observability: every read is scoped to the authenticated data owner."""
from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.server.utils.auth_middleware import get_current_user
from backend.services import user_observability as service
from backend.services.observation_reader import display_content, read_observations
from backend.storage.postgres.manager import get_session
from backend.storage.postgres.models_user import User

router = APIRouter(prefix="/observability", tags=["observability"])


def filters(window: Literal["7d", "30d", "all"] = "7d", q: str = Query("", max_length=200),
            status: str = Query("", max_length=32), model: str = Query("", max_length=160),
            agent: str = Query("", max_length=80), conversation_id: uuid.UUID | None = None):
    return dict(window=window, q=q.strip(), status=status, model=model, agent=agent, conversation_id=conversation_id)


@router.get("/overview")
async def overview(current_user: User = Depends(get_current_user), selected: dict = Depends(filters)):
    async with get_session() as session:
        return await service.overview(session, current_user.id, **selected)


@router.get("/traces")
async def traces(current_user: User = Depends(get_current_user), selected: dict = Depends(filters),
                 page: int = Query(1, ge=1, le=100000), page_size: int = Query(20, ge=1, le=100)):
    async with get_session() as session:
        return await service.list_traces(session, current_user.id, page=page, page_size=page_size, **selected)


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str, current_user: User = Depends(get_current_user)):
    async with get_session() as session:
        return await service.trace_detail(session, trace_id, current_user.id)


@router.get("/traces/{trace_id}/observations")
async def observations(trace_id: str, current_user: User = Depends(get_current_user)):
    async with get_session() as session:
        detail = await service.trace_detail(session, trace_id, current_user.id)
    return await read_observations(detail["observability_trace_id"], sampled=detail["sampled"],
                                   started_at=detail["started_at"], completed_at=detail["completed_at"], user_id=current_user.id)


class JudgeRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=160)
    request_id: uuid.UUID


@router.post("/traces/{trace_id}/judge")
async def judge(trace_id: str, req: JudgeRequest, current_user: User = Depends(get_current_user)):
    from backend.services.turn_judge import judge_trace
    return await judge_trace(get_session, trace_id, current_user.id, req.model_id, str(req.request_id))


@router.get("/evaluations/{run_id}/items/{item_index}")
async def evaluation_item(run_id: uuid.UUID, item_index: int, current_user: User = Depends(get_current_user)):
    from backend.services.evaluation_service import get_run
    async with get_session() as session:
        run = await get_run(session, run_id, current_user.id)
        if not run:
            raise HTTPException(404, "评测运行不存在或无权访问")
        metrics = service.json_object(run.metrics_json)
        details = metrics.get("details") or []
        if item_index < 0 or item_index >= len(details):
            raise HTTPException(404, "评测样例不存在")
        item = details[item_index]
        link = item.get("observability") or {}
        trace_id = link.get("trace_id")
        # Canonical native experiments derive each trace from run UUID + index.
        if trace_id and (link.get("experiment_id") != str(run_id) or
                         trace_id != uuid.uuid5(run_id, str(item_index)).hex):
            raise HTTPException(409, "评测观测关联无效")
        scores = await service.scores_for_trace(session, trace_id, current_user.id)
        payload = {"run_id": str(run_id), "run_name": run.name, "item_index": item_index,
                   "question": display_content(item.get("question")), "case": display_content(item), "scores": scores}
        started_at = run.started_at or run.created_at
        completed_at = run.finished_at or run.created_at
        # Legacy runs saved both timestamps after evaluation. Keep the remote
        # query bounded while allowing their earlier observations to be found.
        if not metrics.get("execution_started_at") and started_at:
            started_at -= timedelta(days=1)
    return {**payload, "remote": await read_observations(trace_id, started_at=started_at,
                                                       completed_at=completed_at, user_id=current_user.id)}
