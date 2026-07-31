"""评估路由（阶段 2D）— 检索质量评估的命名运行与对比。"""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.evaluation_service import run_evaluation, save_run, list_runs, get_run
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.models_user import User

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvalCaseIn(BaseModel):
    query: str = Field(..., min_length=1)
    expected_source: str = Field(..., min_length=1)


class EvalRunRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    cases: List[EvalCaseIn] = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    kb_id: Optional[str] = None


class EvalRunSummary(BaseModel):
    id: str
    name: str
    top_k: int
    query_count: int
    hit_rate: float
    mrr: float
    avg_score: float
    created_at: str


class EvalRunDetail(EvalRunSummary):
    details: list


def _to_summary(r) -> EvalRunSummary:
    return EvalRunSummary(
        id=str(r.id), name=r.name, top_k=r.top_k, query_count=r.query_count,
        hit_rate=r.hit_rate, mrr=r.mrr, avg_score=r.avg_score,
        created_at=r.created_at.isoformat() if r.created_at else "",
    )


@router.post("/runs", response_model=EvalRunDetail, status_code=status.HTTP_201_CREATED)
async def create_run(
    req: EvalRunRequest,
    current_user: User = Depends(get_current_user),
):
    """执行一次评估并保存为命名运行（同步逐条检索，评估集大时较慢）。"""
    cases = [(c.query, c.expected_source) for c in req.cases]
    metrics = run_evaluation(cases, top_k=req.top_k)

    async with get_session() as session:
        run = await save_run(
            session,
            name=req.name,
            metrics=metrics,
            top_k=req.top_k,
            kb_id=uuid.UUID(req.kb_id) if req.kb_id else None,
        )
        await session.commit()
        detail = json.loads(run.metrics_json or "{}").get("details", [])
        return EvalRunDetail(**_to_summary(run).model_dump(), details=detail)


@router.get("/runs", response_model=list[EvalRunSummary])
async def list_all_runs(current_user: User = Depends(get_current_user)):
    """列出全部评估运行（对比视图数据源）。"""
    async with get_session() as session:
        return [_to_summary(r) for r in await list_runs(session)]


@router.get("/runs/{run_id}", response_model=EvalRunDetail)
async def get_run_detail(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    """单次运行明细（逐条 query 命中情况）。"""
    async with get_session() as session:
        run = await get_run(session, uuid.UUID(run_id))
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        detail = json.loads(run.metrics_json or "{}").get("details", [])
        return EvalRunDetail(**_to_summary(run).model_dump(), details=detail)
