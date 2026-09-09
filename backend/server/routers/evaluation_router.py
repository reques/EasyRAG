"""评估路由（规范化评测体系）— 数据集 / 运行 / 报告。

接口分层：
- /evaluation/datasets       评测数据集（Golden Set）CRUD 与导入导出
- /evaluation/runs           执行并保存命名运行、历史对比、Markdown 报告
- /evaluation/chunk-candidates   golden set 标注辅助：给定问题+文件返回候选 chunk
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import uuid
from typing import Any, List, Literal, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sqlalchemy.exc import IntegrityError

from app.core.config import get_settings
from backend.services.evaluation_benchmarks import (
    assess_benchmark,
    build_benchmark_snapshot,
    create_benchmark,
    delete_benchmark,
    duplicate_benchmark,
    export_benchmark_json,
    get_benchmark,
    list_benchmarks,
    update_benchmark,
)
from backend.services.evaluation_service import (
    EvaluationCase,
    run_evaluation,
    save_run,
    list_runs,
    get_run,
)
from backend.services.evaluation_datasets import (
    deserialize_cases,
    export_dataset_json,
    save_dataset,
    list_datasets,
    get_dataset,
    delete_dataset,
    update_dataset,
)
from backend.services.evaluation_import import MAX_IMPORT_BYTES, parse_dataset_import
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
cfg = get_settings()


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


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=512)
    cases: Optional[List[EvalCaseIn]] = Field(
        default=None,
        min_length=1,
        max_length=1000,
    )

    @model_validator(mode="after")
    def require_change(self):
        if not self.model_fields_set:
            raise ValueError("At least one field must be supplied")
        return self

    @field_validator("name")
    @classmethod
    def clean_optional_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value


class BenchmarkMetricRule(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class BenchmarkMetrics(BaseModel):
    recall_at_k: BenchmarkMetricRule = Field(
        default_factory=lambda: BenchmarkMetricRule(enabled=True, threshold=0.8)
    )
    mrr_at_k: BenchmarkMetricRule = Field(
        default_factory=lambda: BenchmarkMetricRule(enabled=True, threshold=0.7)
    )
    context_relevance: BenchmarkMetricRule = Field(
        default_factory=lambda: BenchmarkMetricRule(enabled=False, threshold=0.75)
    )
    faithfulness: BenchmarkMetricRule = Field(
        default_factory=lambda: BenchmarkMetricRule(enabled=False, threshold=0.9)
    )

    @model_validator(mode="after")
    def require_enabled_metric(self):
        if not any(rule.enabled for rule in (
            self.recall_at_k,
            self.mrr_at_k,
            self.context_relevance,
            self.faithfulness,
        )):
            raise ValueError("At least one metric must be enabled")
        return self


class BenchmarkRetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    mode: Literal["basic", "enhanced"] = "basic"


class BenchmarkCreateRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    dataset_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=512)
    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)
    retrieval_config: BenchmarkRetrievalConfig = Field(
        default_factory=BenchmarkRetrievalConfig
    )
    status: Literal["draft", "active", "archived"] = "active"

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value


class BenchmarkUpdateRequest(BaseModel):
    dataset_id: Optional[uuid.UUID] = None
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=512)
    metrics: Optional[BenchmarkMetrics] = None
    retrieval_config: Optional[BenchmarkRetrievalConfig] = None
    status: Optional[Literal["draft", "active", "archived"]] = None

    @model_validator(mode="after")
    def require_change(self):
        if not self.model_fields_set:
            raise ValueError("At least one field must be supplied")
        return self

    @field_validator("name")
    @classmethod
    def clean_optional_name(cls, value: Optional[str]) -> Optional[str]:
        return DatasetUpdateRequest.clean_optional_name(value)


class BenchmarkDuplicateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_optional_name(cls, value: Optional[str]) -> Optional[str]:
        return DatasetUpdateRequest.clean_optional_name(value)


class BenchmarkRunRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_optional_name(cls, value: Optional[str]) -> Optional[str]:
        return DatasetUpdateRequest.clean_optional_name(value)


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


async def _require_owned_knowledge_base(session, kb_id: uuid.UUID, owner_id: uuid.UUID):
    kb = await KnowledgeBaseRepository(session).get_by_id(kb_id)
    if not kb or kb.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


async def _require_dataset_for_knowledge_base(
    session,
    dataset_id: uuid.UUID,
    kb_id: uuid.UUID,
    owner_id: uuid.UUID,
):
    dataset = await get_dataset(session, dataset_id, owner_id)
    if not dataset or dataset.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")
    return dataset


async def _prepare_evaluation_cases(session, kb_id: uuid.UUID, raw_cases) -> list[EvaluationCase]:
    file_ids: list[uuid.UUID] = []
    for case in raw_cases:
        try:
            file_id = uuid.UUID(str(case.expected_file_id))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid expected_file_id: {case.expected_file_id}",
            ) from exc
        if file_id not in file_ids:
            file_ids.append(file_id)

    files = await KnowledgeFileRepository(session).list_by_ids_for_kb(kb_id, file_ids)
    files_by_id = {file.id: file for file in files}
    missing = [str(file_id) for file_id in file_ids if file_id not in files_by_id]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Expected files must belong to the selected knowledge base",
                "file_ids": missing,
            },
        )

    return [
        EvaluationCase(
            question=case.question,
            expected_file_id=str(case.expected_file_id),
            expected_chunk_ids=tuple(case.expected_chunk_ids),
            expected_chunk_id=getattr(case, "expected_chunk_id", None),
            reference_answer=case.reference_answer or "",
            expected_source=files_by_id[uuid.UUID(str(case.expected_file_id))].filename,
            expect_miss=case.expect_miss,
        )
        for case in raw_cases
    ]


async def _resolve_import_cases(session, kb_id: uuid.UUID, parsed: dict[str, Any]):
    raw_cases = parsed.get("cases") or []
    errors = list(parsed.get("errors") or [])
    requested_ids: list[uuid.UUID] = []
    requested_names: list[str] = []
    resolved_ids: list[Optional[uuid.UUID]] = []

    for row_number, raw in enumerate(raw_cases, start=1):
        file_id = None
        if raw.get("expected_file_id"):
            try:
                file_id = uuid.UUID(raw["expected_file_id"])
                requested_ids.append(file_id)
            except ValueError:
                errors.append({
                    "row": row_number,
                    "field": "expected_file_id",
                    "message": "不是有效的 UUID",
                })
        elif raw.get("expected_filename"):
            requested_names.append(raw["expected_filename"])
        resolved_ids.append(file_id)

    file_repo = KnowledgeFileRepository(session)
    files_by_id = {
        item.id: item
        for item in await file_repo.list_by_ids_for_kb(kb_id, requested_ids)
    }
    files_by_name: dict[str, list] = {}
    for item in await file_repo.list_by_filenames_for_kb(kb_id, requested_names):
        files_by_name.setdefault(item.filename, []).append(item)

    valid_cases: list[EvalCaseIn] = []
    valid_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cases):
        row_number = index + 1
        file_id = resolved_ids[index]
        if file_id is not None and file_id not in files_by_id:
            errors.append({
                "row": row_number,
                "field": "expected_file_id",
                "message": "文件不属于当前知识库",
            })
            continue
        if file_id is None and raw.get("expected_filename"):
            matches = files_by_name.get(raw["expected_filename"], [])
            if not matches:
                errors.append({
                    "row": row_number,
                    "field": "expected_filename",
                    "message": "当前知识库中未找到该文件",
                })
                continue
            if len(matches) > 1:
                errors.append({
                    "row": row_number,
                    "field": "expected_filename",
                    "message": "存在同名文件，请改用 expected_file_id",
                })
                continue
            file_id = matches[0].id
        if any(error.get("row") == row_number for error in errors):
            continue
        if file_id is None:
            continue

        try:
            case = EvalCaseIn(
                question=raw.get("question", ""),
                expected_file_id=file_id,
                expected_chunk_ids=raw.get("expected_chunk_ids") or [],
                reference_answer=raw.get("reference_answer") or "",
                expect_miss=bool(raw.get("expect_miss")),
            )
        except ValidationError as exc:
            for item in exc.errors():
                errors.append({
                    "row": row_number,
                    "field": ".".join(str(part) for part in item.get("loc", ())),
                    "message": item.get("msg", "字段无效"),
                })
            continue
        valid_cases.append(case)
        valid_rows.append(case.model_dump(mode="json"))

    return valid_cases, {
        "name": parsed.get("name") or "golden-set",
        "description": parsed.get("description") or "",
        "total_rows": len(raw_cases),
        "valid_count": len(valid_cases),
        "invalid_count": len({
            error.get("row") for error in errors if error.get("row", 0) > 0
        }),
        "cases": valid_rows,
        "errors": errors,
    }


# ── 指标目录与评估基准 ────────────────────────────────────────────────────────


@router.get("/metric-catalog", response_model=list[dict])
async def get_metric_catalog(_current_user: User = Depends(get_current_user)):
    ragas_llm_ready = bool(
        cfg.RAGAS_ENABLED
        and (cfg.RAGAS_LLM_API_KEY or cfg.LLM_API_KEY)
        and (cfg.RAGAS_LLM_MODEL or cfg.LLM_MODEL)
    )
    answer_generation_ready = bool(cfg.LLM_API_KEY and cfg.LLM_MODEL)
    return [
        {
            "id": "recall_at_k",
            "name": "Recall@K",
            "group": "retrieval",
            "available": True,
            "requires": [],
        },
        {
            "id": "mrr_at_k",
            "name": "MRR@K",
            "group": "ranking",
            "available": True,
            "requires": [],
        },
        {
            "id": "context_relevance",
            "name": "上下文相关性",
            "group": "semantic",
            "available": ragas_llm_ready and answer_generation_ready,
            "requires": ["ragas", "llm", "reference_answer"],
            "unavailable_reason": (
                None if ragas_llm_ready else "需要启用 RAGAs 并配置评估 LLM"
            ),
        },
        {
            "id": "faithfulness",
            "name": "回答忠实度",
            "group": "generation",
            "available": ragas_llm_ready,
            "requires": ["answer_generation", "ragas", "llm"],
            "unavailable_reason": (
                None if ragas_llm_ready and answer_generation_ready
                else "需要启用 RAGAs，并配置回答生成与评估 LLM"
            ),
        },
    ]


@router.get("/benchmarks", response_model=list[dict])
async def list_benchmark_endpoints(
    kb_id: Optional[uuid.UUID] = Query(default=None),
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        if kb_id is not None:
            await _require_owned_knowledge_base(session, kb_id, current_user.id)
        items = await list_benchmarks(
            session,
            current_user.id,
            knowledge_base_id=kb_id,
        )
        return [export_benchmark_json(item) for item in items]


@router.post("/benchmarks", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_benchmark_endpoint(
    req: BenchmarkCreateRequest,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        await _require_owned_knowledge_base(
            session,
            req.knowledge_base_id,
            current_user.id,
        )
        await _require_dataset_for_knowledge_base(
            session,
            req.dataset_id,
            req.knowledge_base_id,
            current_user.id,
        )
        try:
            benchmark = await create_benchmark(
                session,
                knowledge_base_id=req.knowledge_base_id,
                dataset_id=req.dataset_id,
                name=req.name,
                description=req.description,
                metrics=req.metrics.model_dump(),
                retrieval_config=req.retrieval_config.model_dump(),
                status=req.status,
            )
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A benchmark with this name already exists",
            ) from exc
        return export_benchmark_json(benchmark)


@router.get("/benchmarks/{benchmark_id}", response_model=dict)
async def get_benchmark_endpoint(
    benchmark_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        benchmark = await get_benchmark(session, benchmark_id, current_user.id)
        if benchmark is None:
            raise HTTPException(status_code=404, detail="Evaluation benchmark not found")
        return export_benchmark_json(benchmark)


@router.patch("/benchmarks/{benchmark_id}", response_model=dict)
async def update_benchmark_endpoint(
    benchmark_id: uuid.UUID,
    req: BenchmarkUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        benchmark = await get_benchmark(session, benchmark_id, current_user.id)
        if benchmark is None:
            raise HTTPException(status_code=404, detail="Evaluation benchmark not found")
        changes = req.model_dump(exclude_unset=True)
        if req.dataset_id is not None:
            await _require_dataset_for_knowledge_base(
                session,
                req.dataset_id,
                benchmark.knowledge_base_id,
                current_user.id,
            )
        try:
            await update_benchmark(session, benchmark, changes=changes)
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A benchmark with this name already exists",
            ) from exc
        return export_benchmark_json(benchmark)


@router.delete(
    "/benchmarks/{benchmark_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_benchmark_endpoint(
    benchmark_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        if not await delete_benchmark(session, benchmark_id, current_user.id):
            raise HTTPException(status_code=404, detail="Evaluation benchmark not found")
        await session.commit()


@router.post(
    "/benchmarks/{benchmark_id}/duplicate",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def duplicate_benchmark_endpoint(
    benchmark_id: uuid.UUID,
    req: BenchmarkDuplicateRequest,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        source = await get_benchmark(session, benchmark_id, current_user.id)
        if source is None:
            raise HTTPException(status_code=404, detail="Evaluation benchmark not found")
        if req.name:
            name = req.name
        else:
            existing = await list_benchmarks(
                session,
                current_user.id,
                knowledge_base_id=source.knowledge_base_id,
            )
            names = {item.name for item in existing}
            base_name = f"{source.name} 副本"
            name = base_name
            suffix = 2
            while name in names:
                name = f"{base_name} {suffix}"
                suffix += 1
        try:
            duplicate = await duplicate_benchmark(session, source, name=name)
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A benchmark with this name already exists",
            ) from exc
        return export_benchmark_json(duplicate)


@router.post(
    "/benchmarks/{benchmark_id}/runs",
    response_model=EvalRunDetail,
    status_code=status.HTTP_201_CREATED,
)
async def run_benchmark_endpoint(
    benchmark_id: uuid.UUID,
    req: BenchmarkRunRequest,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        benchmark = await get_benchmark(session, benchmark_id, current_user.id)
        if benchmark is None:
            raise HTTPException(status_code=404, detail="Evaluation benchmark not found")
        if benchmark.status == "archived":
            raise HTTPException(status_code=409, detail="Archived benchmarks cannot run")
        dataset = await _require_dataset_for_knowledge_base(
            session,
            benchmark.dataset_id,
            benchmark.knowledge_base_id,
            current_user.id,
        )
        benchmark_payload = export_benchmark_json(benchmark)
        metrics_config = benchmark_payload["metrics"]
        retrieval_config = benchmark_payload["retrieval_config"]
        if retrieval_config.get("mode") != "basic":
            raise HTTPException(
                status_code=409,
                detail="Enhanced benchmark execution is not available yet",
            )
        faithfulness_enabled = bool(
            (metrics_config.get("faithfulness") or {}).get("enabled")
        )
        context_enabled = bool(
            (metrics_config.get("context_relevance") or {}).get("enabled")
        )
        if (context_enabled or faithfulness_enabled) and not cfg.RAGAS_ENABLED:
            raise HTTPException(
                status_code=409,
                detail="Semantic metrics require RAGAs to be enabled",
            )
        if faithfulness_enabled and not (cfg.LLM_API_KEY and cfg.LLM_MODEL):
            raise HTTPException(
                status_code=409,
                detail="Faithfulness evaluation requires the main LLM to be configured",
            )
        raw_cases = deserialize_cases(json.loads(dataset.cases_json or "[]"))
        if context_enabled and any(not case.reference_answer for case in raw_cases):
            raise HTTPException(
                status_code=422,
                detail="Context relevance requires reference_answer for every case",
            )
        cases = await _prepare_evaluation_cases(
            session,
            benchmark.knowledge_base_id,
            raw_cases,
        )
        snapshot = build_benchmark_snapshot(benchmark, dataset)

    metrics = await asyncio.to_thread(
        run_evaluation,
        cases,
        int(retrieval_config.get("top_k", 5)),
        knowledge_base_id=benchmark.knowledge_base_id,
        ragas_metrics=(
            (["context_precision"] if context_enabled else [])
            + (["faithfulness"] if faithfulness_enabled else [])
        ) or None,
        score_threshold=float(retrieval_config.get("score_threshold", 0.0)),
        run_ragas=context_enabled or faithfulness_enabled,
        generate_answers=faithfulness_enabled,
    )
    metrics["benchmark_assessment"] = assess_benchmark(metrics_config, metrics)
    run_name = req.name or (
        f"{benchmark.name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    async with get_session() as session:
        benchmark = await get_benchmark(session, benchmark_id, current_user.id)
        if benchmark is None:
            raise HTTPException(status_code=404, detail="Evaluation benchmark not found")
        run = await save_run(
            session,
            name=run_name,
            metrics=metrics,
            top_k=int(retrieval_config.get("top_k", 5)),
            kb_id=benchmark.knowledge_base_id,
            dataset_id=benchmark.dataset_id,
            benchmark_id=benchmark.id,
            benchmark_version=snapshot["benchmark"]["version"],
            benchmark_snapshot=snapshot,
        )
        await session.commit()
        stored_metrics = _metrics_payload(run)
        details = stored_metrics.pop("details", [])
        return EvalRunDetail(
            **_to_summary(run).model_dump(),
            metrics=stored_metrics,
            details=details,
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
async def list_all_datasets(
    kb_id: Optional[uuid.UUID] = Query(default=None),
    current_user: User = Depends(get_current_user),
):
    """列出当前用户可访问的评测数据集（不含 cases 明细，仅元信息）。"""
    async with get_session() as session:
        if kb_id is not None:
            await _require_owned_knowledge_base(session, kb_id, current_user.id)
        datasets = await list_datasets(
            session,
            current_user.id,
            knowledge_base_id=kb_id,
        )
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


@router.post("/datasets/import/preview", response_model=dict)
async def preview_dataset_import(
    file: UploadFile = File(...),
    kb_id: uuid.UUID = Form(...),
    current_user: User = Depends(get_current_user),
):
    content = await file.read(MAX_IMPORT_BYTES + 1)
    try:
        parsed = parse_dataset_import(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with get_session() as session:
        await _require_owned_knowledge_base(session, kb_id, current_user.id)
        _, preview = await _resolve_import_cases(session, kb_id, parsed)
        return preview


@router.post(
    "/datasets/import",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def import_dataset(
    file: UploadFile = File(...),
    kb_id: uuid.UUID = Form(...),
    name: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
):
    content = await file.read(MAX_IMPORT_BYTES + 1)
    try:
        parsed = parse_dataset_import(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with get_session() as session:
        await _require_owned_knowledge_base(session, kb_id, current_user.id)
        case_inputs, preview = await _resolve_import_cases(session, kb_id, parsed)
        if preview["errors"]:
            raise HTTPException(status_code=422, detail=preview)
        dataset_name = (name or preview["name"]).strip()
        if not dataset_name or len(dataset_name) > 128:
            raise HTTPException(status_code=422, detail="Dataset name is invalid")
        dataset_description = (
            description if description is not None else preview["description"]
        ).strip()
        if len(dataset_description) > 512:
            raise HTTPException(status_code=422, detail="Description is too long")
        cases = await _prepare_evaluation_cases(session, kb_id, case_inputs)
        dataset = await save_dataset(
            session,
            name=dataset_name,
            kb_id=kb_id,
            description=dataset_description,
            cases=cases,
        )
        await session.commit()
        return export_dataset_json(dataset)


@router.get("/datasets/{dataset_id}/export", response_class=JSONResponse)
async def export_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        dataset = await get_dataset(session, dataset_id, current_user.id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        return JSONResponse(
            content=export_dataset_json(dataset),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="evaluation-dataset-{dataset.id}.json"'
                )
            },
        )


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


@router.patch("/datasets/{dataset_id}", response_model=dict)
async def update_dataset_endpoint(
    dataset_id: uuid.UUID,
    req: DatasetUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    async with get_session() as session:
        dataset = await get_dataset(session, dataset_id, current_user.id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Evaluation dataset not found")
        changes = req.model_dump(exclude_unset=True, exclude={"cases"})
        if req.cases is not None:
            changes["cases"] = await _prepare_evaluation_cases(
                session,
                dataset.knowledge_base_id,
                req.cases,
            )
        await update_dataset(session, dataset, **changes)
        await session.commit()
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
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is referenced by an evaluation benchmark",
            ) from exc


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
