"""评估路由（阶段 2D）— 检索质量评估的命名运行与对比。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.services.evaluation_service import (
    EvaluationCase,
    run_evaluation,
    save_run,
    list_runs,
    get_run,
)
from backend.repositories.knowledge_repository import (
    KnowledgeBaseRepository,
    KnowledgeFileRepository,
)
from backend.storage.postgres.manager import get_session
from backend.server.utils.auth_middleware import get_current_user
from backend.storage.postgres.models_user import User

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvalCaseIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    expected_file_id: uuid.UUID
    expected_chunk_id: str = Field(..., min_length=1, max_length=256)
    reference_answer: str = Field(..., min_length=1, max_length=100_000)

    @field_validator("question", "expected_chunk_id", "reference_answer")
    @classmethod
    def strip_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class EvalRunRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    cases: List[EvalCaseIn] = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    kb_id: uuid.UUID


class EvalRunSummary(BaseModel):
    id: str
    name: str
    knowledge_base_id: str
    top_k: int
    query_count: int
    hit_rate: float
    mrr: float
    hit_rate_at_k: float
    mrr_at_k: float
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    avg_score: float
    created_at: str


class EvalRunDetail(EvalRunSummary):
    metrics: dict[str, Any]
    details: list


def _metrics_payload(run) -> dict[str, Any]:
    try:
        payload = json.loads(run.metrics_json or "{}")
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        return {}


def _to_summary(r) -> EvalRunSummary:
    metrics = _metrics_payload(r)
    return EvalRunSummary(
        id=str(r.id), name=r.name,
        knowledge_base_id=str(r.knowledge_base_id),
        top_k=r.top_k, query_count=r.query_count,
        hit_rate=r.hit_rate, mrr=r.mrr,
        hit_rate_at_k=metrics.get("hit_rate_at_k", r.hit_rate),
        mrr_at_k=metrics.get("mrr_at_k", r.mrr),
        recall_at_k=metrics.get("recall_at_k", r.hit_rate),
        precision_at_k=metrics.get("precision_at_k", 0.0),
        ndcg_at_k=metrics.get("ndcg_at_k", 0.0),
        avg_score=r.avg_score,
        created_at=r.created_at.isoformat() if r.created_at else "",
    )


@router.post("/runs", response_model=EvalRunDetail, status_code=status.HTTP_201_CREATED)
async def create_run(
    req: EvalRunRequest,
    current_user: User = Depends(get_current_user),
):
    """执行一次评估并保存为命名运行（同步逐条检索，评估集大时较慢）。"""
    # 权限检查使用短会话，避免评估期间长时间占用数据库连接。
    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(req.kb_id)
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        file_repo = KnowledgeFileRepository(session)
        expected_file_ids = list({case.expected_file_id for case in req.cases})
        expected_files = await file_repo.list_by_ids_for_kb(
            req.kb_id,
            expected_file_ids,
        )
        files_by_id = {file.id: file for file in expected_files}
        missing_file_ids = [
            str(file_id)
            for file_id in expected_file_ids
            if file_id not in files_by_id
        ]
        if missing_file_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "Expected files must belong to the selected knowledge base",
                    "file_ids": missing_file_ids,
                },
            )

        cases = [
            EvaluationCase(
                question=case.question,
                expected_file_id=str(case.expected_file_id),
                expected_chunk_id=case.expected_chunk_id,
                reference_answer=case.reference_answer,
                expected_source=files_by_id[case.expected_file_id].filename,
            )
            for case in req.cases
        ]

    metrics = await asyncio.to_thread(
        run_evaluation,
        cases,
        req.top_k,
        knowledge_base_id=req.kb_id,
    )

    async with get_session() as session:
        # 评估可能耗时较长，保存前再次校验，防止期间知识库被删除
        # 或所有权发生变化。
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(req.kb_id)
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        run = await save_run(
            session,
            name=req.name,
            metrics=metrics,
            top_k=req.top_k,
            kb_id=req.kb_id,
        )
        await session.commit()
        metrics = _metrics_payload(run)
        detail = metrics.pop("details", [])
        return EvalRunDetail(
            **_to_summary(run).model_dump(),
            metrics=metrics,
            details=detail,
        )


@router.get("/runs", response_model=list[EvalRunSummary])
async def list_all_runs(current_user: User = Depends(get_current_user)):
    """列出当前用户知识库下的评估运行（对比视图数据源）。"""
    async with get_session() as session:
        return [
            _to_summary(r)
            for r in await list_runs(session, current_user.id)
        ]


@router.get("/runs/{run_id}", response_model=EvalRunDetail)
async def get_run_detail(
    run_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """单次运行明细（逐条 query 命中情况）。"""
    async with get_session() as session:
        run = await get_run(session, run_id, current_user.id)
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        metrics = _metrics_payload(run)
        detail = metrics.pop("details", [])
        return EvalRunDetail(
            **_to_summary(run).model_dump(),
            metrics=metrics,
            details=detail,
        )
