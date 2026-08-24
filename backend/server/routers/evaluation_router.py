"""评估路由（规范化评测体系）— 数据集 / 运行 / 报告。

接口分层：
- /evaluation/datasets       评测数据集（Golden Set）CRUD 与导入导出
- /evaluation/runs           执行并保存命名运行、历史对比、Markdown 报告
- /evaluation/chunk-candidates   golden set 标注辅助：给定问题+文件返回候选 chunk
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from backend.services.evaluation_service import (
    EvaluationCase,
    run_evaluation,
    save_run,
    list_runs,
    get_run,
)
from backend.services.evaluation_datasets import (
    export_dataset_json,
    save_dataset,
    list_datasets,
    get_dataset,
    delete_dataset,
)
from backend.services.evaluation_report import build_markdown_report
from backend.services.ragas_evaluator import SUPPORTED_RAGAS_METRICS
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
    # question-specific 相关 chunk 集：黄金标注应精确到条文/chunk 级，
    # 而不是把整份文件当作相关集（参考 RAGAs reference_contexts 语义）。
    expected_chunk_ids: List[str] = Field(default_factory=list, max_length=32)
    expected_chunk_id: Optional[str] = Field(default=None, max_length=256)
    reference_answer: str = Field(default="", max_length=100_000)
    expect_miss: bool = Field(default=False)

    @field_validator("question")
    @classmethod
    def strip_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("expected_chunk_id")
    @classmethod
    def strip_optional_chunk_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("expected_chunk_ids")
    @classmethod
    def clean_chunk_ids(cls, value: List[str]) -> List[str]:
        seen: List[str] = []
        for item in value:
            item = str(item).strip()
            if item and item not in seen:
                seen.append(item)
        return seen

    @field_validator("reference_answer")
    @classmethod
    def strip_optional_answer(cls, value: str) -> str:
        return value.strip()


class EvalRunRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    cases: List[EvalCaseIn] = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    kb_id: uuid.UUID
    dataset_id: Optional[uuid.UUID] = Field(default=None)
    # 按运行覆盖全局 RAGAS_METRICS，便于同一数据集跑不同指标集做对比。
    ragas_metrics: Optional[List[str]] = Field(default=None, max_length=16)

    @field_validator("ragas_metrics")
    @classmethod
    def validate_ragas_metrics(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = []
        for item in value:
            item = str(item).strip()
            if item and item not in cleaned:
                cleaned.append(item)
        unknown = sorted(set(cleaned) - SUPPORTED_RAGAS_METRICS)
        if unknown:
            raise ValueError(f"Unsupported RAGAs metrics: {', '.join(unknown)}")
        return cleaned


class DatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    kb_id: uuid.UUID
    description: str = Field(default="", max_length=512)
    cases: List[EvalCaseIn] = Field(..., min_length=1, max_length=1000)


class ChunkCandidatesRequest(BaseModel):
    kb_id: uuid.UUID
    file_id: uuid.UUID
    question: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=8, ge=1, le=30)


class ChunkCandidateOut(BaseModel):
    chunk_id: str
    snippet: str
    score: float


class EvalRunSummary(BaseModel):
    id: str
    name: str
    knowledge_base_id: str
    dataset_id: Optional[str] = None
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
    ragas_status: Optional[str] = None
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
    ragas = metrics.get("ragas") or {}
    return EvalRunSummary(
        id=str(r.id), name=r.name,
        knowledge_base_id=str(r.knowledge_base_id),
        dataset_id=str(r.dataset_id) if getattr(r, "dataset_id", None) else None,
        top_k=r.top_k, query_count=r.query_count,
        hit_rate=r.hit_rate, mrr=r.mrr,
        hit_rate_at_k=metrics.get("hit_rate_at_k", r.hit_rate),
        mrr_at_k=metrics.get("mrr_at_k", r.mrr),
        recall_at_k=metrics.get("recall_at_k", r.hit_rate),
        precision_at_k=metrics.get("precision_at_k", 0.0),
        ndcg_at_k=metrics.get("ndcg_at_k", 0.0),
        avg_score=r.avg_score,
        ragas_status=ragas.get("status"),
        created_at=r.created_at.isoformat() if r.created_at else "",
    )


# ── 运行 ──────────────────────────────────────────────────────────────────────


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
                expected_chunk_ids=tuple(case.expected_chunk_ids),
                expected_chunk_id=case.expected_chunk_id,
                reference_answer=case.reference_answer or "",
                expected_source=files_by_id[case.expected_file_id].filename,
                expect_miss=case.expect_miss,
            )
            for case in req.cases
        ]

    metrics = await asyncio.to_thread(
        run_evaluation,
        cases,
        req.top_k,
        knowledge_base_id=req.kb_id,
        ragas_metrics=req.ragas_metrics,
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
            dataset_id=req.dataset_id,
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


@router.get("/runs/{run_id}/report", response_class=PlainTextResponse)
async def get_run_report(
    run_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """导出单次运行的 Markdown 评测报告。"""
    async with get_session() as session:
        run = await get_run(session, run_id, current_user.id)
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        kb_name = ""
        if run.knowledge_base_id:
            kb_repo = KnowledgeBaseRepository(session)
            kb = await kb_repo.get_by_id(run.knowledge_base_id)
            kb_name = kb.name if kb else ""
        metrics = _metrics_payload(run)
        return build_markdown_report(
            run_name=run.name,
            created_at=run.created_at.isoformat() if run.created_at else "",
            knowledge_base_name=kb_name,
            metrics=metrics,
            ragas=metrics.get("ragas"),
        )


# ── 数据集 ────────────────────────────────────────────────────────────────────


@router.post(
    "/datasets",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset(
    req: DatasetCreateRequest,
    current_user: User = Depends(get_current_user),
):
    """保存评测数据集（同名覆盖并递增 version）。"""
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
                expected_chunk_ids=tuple(case.expected_chunk_ids),
                expected_chunk_id=case.expected_chunk_id,
                reference_answer=case.reference_answer or "",
                expected_source=files_by_id[case.expected_file_id].filename,
                expect_miss=case.expect_miss,
            )
            for case in req.cases
        ]
        dataset = await save_dataset(
            session,
            name=req.name,
            kb_id=req.kb_id,
            description=req.description,
            cases=cases,
        )
        await session.commit()
        return export_dataset_json(dataset)


@router.get("/datasets", response_model=list[dict])
async def list_all_datasets(current_user: User = Depends(get_current_user)):
    """列出当前用户可访问的评测数据集（不含 cases 明细，仅元信息）。"""
    async with get_session() as session:
        datasets = await list_datasets(session, current_user.id)
        return [
            {
                "id": str(d.id),
                "name": d.name,
                "knowledge_base_id": (
                    str(d.knowledge_base_id) if d.knowledge_base_id else None
                ),
                "description": d.description,
                "case_count": d.case_count,
                "version": d.version,
                "created_at": d.created_at.isoformat() if d.created_at else "",
                "updated_at": d.updated_at.isoformat() if d.updated_at else "",
            }
            for d in datasets
        ]


@router.get("/datasets/{dataset_id}", response_model=dict)
async def get_dataset_detail(
    dataset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """读取数据集详情（含 cases，可直接用于发起运行或二次编辑）。"""
    async with get_session() as session:
        dataset = await get_dataset(session, dataset_id, current_user.id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        return export_dataset_json(dataset)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        deleted = await delete_dataset(session, dataset_id, current_user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        await session.commit()


# ── Golden set 标注辅助 ───────────────────────────────────────────────────────


@router.post("/chunk-candidates", response_model=list[ChunkCandidateOut])
async def chunk_candidates(
    req: ChunkCandidatesRequest,
    current_user: User = Depends(get_current_user),
):
    """给定问题+目标文件，返回该文件内被检索到的候选 chunk。

    用于 golden set 标注：人工从候选里勾选「真正回答该问题所需的
    那几条 chunk」作为 expected_chunk_ids，而不是整文件兜底。
    """
    async with get_session() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = await kb_repo.get_by_id(req.kb_id)
        if not kb or kb.owner_id != current_user.id:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        file_repo = KnowledgeFileRepository(session)
        files = await file_repo.list_by_ids_for_kb(req.kb_id, [req.file_id])
        if not files:
            raise HTTPException(status_code=404, detail="File not found")
        source = files[0].filename

    def _collect() -> list[dict]:
        from app.rag.retriever import get_document_chunk_id, get_retriever

        retriever = get_retriever()
        docs = retriever.retrieve(
            req.question,
            top_k=req.top_k,
            knowledge_base_ids=[str(req.kb_id)],
        )
        candidates: list[dict] = []
        seen: set[str] = set()
        for doc in docs:
            metadata = doc.get("metadata") or {}
            if str(metadata.get("source") or "") != source:
                continue
            chunk_id = get_document_chunk_id(
                str(req.kb_id),
                str(doc.get("content") or ""),
                metadata,
            )
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            candidates.append({
                "chunk_id": chunk_id,
                "snippet": str(doc.get("content") or "")[:200],
                "score": float(metadata.get("score", 0.0)),
            })
        return candidates

    return await asyncio.to_thread(_collect)