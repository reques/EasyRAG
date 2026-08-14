"""检索评估服务（阶段 2D）— 量化检索质量。

核心思路：
- 每条测试用例包含问题、预期文件 ID、预期 chunk ID 和参考答案；
- 每条问题走真实 retriever.retrieve，并同时记录文件级与 chunk 级命中；
- 聚合指标：chunk HitRate@k、chunk MRR@k、文件级指标和 avg_score；
- 每次评估以命名运行（EvaluationRun）落库，可横向对比不同分块策略/参数的效果。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.services.ragas_evaluator import (
    RagasEvaluationSample,
    get_ragas_evaluator,
)
from backend.services.retrieval_metrics import calculate_ranking_metrics
from backend.storage.postgres.models_knowledge import EvaluationRun, KnowledgeBase

logger = get_logger(__name__)
cfg = get_settings()

@dataclass(frozen=True)
class EvaluationCase:
    """A reusable retrieval/RAG evaluation example."""

    question: str
    expected_file_id: str
    expected_chunk_id: str
    reference_answer: str
    # Internal lookup value used for legacy stores that do not persist file_id.
    expected_source: str = ""


def run_evaluation(
    cases: List[EvaluationCase],
    top_k: Optional[int] = None,
    *,
    knowledge_base_id: uuid.UUID | str,
) -> Dict[str, Any]:
    """对评估集逐条检索并计算指标（同步，直接调 retriever 单例）。

    主指标为严格 chunk 级 HitRate/MRR/Recall/Precision/nDCG@K；同时
    返回文件级指标和逐条明细。全程不调用 LLM。
    """
    from app.rag.retriever import get_document_chunk_id, get_retriever

    k = top_k or cfg.RETRIEVER_TOP_K
    scoped_kb_id = str(uuid.UUID(str(knowledge_base_id)))
    retriever = get_retriever()

    details: List[Dict[str, Any]] = []
    ragas_samples: List[RagasEvaluationSample] = []
    chunk_metric_rows = []
    file_metric_rows = []
    score_sum = 0.0
    scored = 0

    for case in cases:
        try:
            docs = retriever.retrieve(
                case.question,
                top_k=k,
                knowledge_base_ids=[scoped_kb_id],
            )
        except Exception as exc:
            logger.warning(
                "[eval] retrieve failed for %r: %s",
                case.question[:50],
                exc,
            )
            docs = []

        top_score = (
            (docs[0].get("metadata") or {}).get("score", 0.0)
            if docs
            else 0.0
        )
        retrieved_chunk_ids: List[str] = []
        retrieved_file_ids: List[str] = []
        for d in docs:
            metadata = d.get("metadata") or {}
            source = str(metadata.get("source") or "")
            returned_file_id = str(metadata.get("file_id") or "")
            returned_chunk_id = get_document_chunk_id(
                scoped_kb_id,
                str(d.get("content") or ""),
                metadata,
            )
            # Legacy Milvus collections do not persist file_id. The filename
            # was ownership-checked before evaluation, so an exact source match
            # can safely recover the expected public file ID for local metrics.
            if (
                not returned_file_id
                and case.expected_source
                and source == case.expected_source
            ):
                returned_file_id = case.expected_file_id
            retrieved_file_ids.append(returned_file_id or f"source:{source}")
            retrieved_chunk_ids.append(returned_chunk_id)

        chunk_metrics = calculate_ranking_metrics(
            retrieved_chunk_ids,
            [case.expected_chunk_id],
            k,
        )
        file_metrics = calculate_ranking_metrics(
            retrieved_file_ids,
            [case.expected_file_id],
            k,
        )
        chunk_metric_rows.append(chunk_metrics)
        file_metric_rows.append(file_metrics)
        ragas_samples.append(RagasEvaluationSample(
            question=case.question,
            retrieved_context_ids=retrieved_chunk_ids,
            reference_context_ids=[case.expected_chunk_id],
            retrieved_contexts=[
                str(doc.get("content") or "")
                for doc in docs
            ],
            reference_answer=case.reference_answer,
        ))
        if docs:
            score_sum += float(top_score)
            scored += 1

        details.append({
            "question": case.question,
            "expected_file_id": case.expected_file_id,
            "expected_chunk_id": case.expected_chunk_id,
            "reference_answer": case.reference_answer,
            "file_hit_rank": file_metrics.first_relevant_rank,
            "chunk_hit_rank": chunk_metrics.first_relevant_rank,
            # Backward-compatible alias; the strict chunk match is canonical.
            "hit_rank": chunk_metrics.first_relevant_rank,
            "top_score": round(float(top_score), 4),
            "returned": len(docs),
            "chunk_metrics": chunk_metrics.to_dict(),
            "file_metrics": file_metrics.to_dict(),
        })

    def mean(rows, attribute: str) -> float:
        if not rows:
            return 0.0
        return round(sum(getattr(row, attribute) for row in rows) / len(rows), 4)

    hit_rate_at_k = mean(chunk_metric_rows, "hit_rate_at_k")
    mrr_at_k = mean(chunk_metric_rows, "reciprocal_rank_at_k")
    file_hit_rate_at_k = mean(file_metric_rows, "hit_rate_at_k")
    file_mrr_at_k = mean(file_metric_rows, "reciprocal_rank_at_k")
    result = {
        "metrics_version": "local-v1",
        "k": k,
        "hit_rate_at_k": hit_rate_at_k,
        "mrr_at_k": mrr_at_k,
        "recall_at_k": mean(chunk_metric_rows, "recall_at_k"),
        "precision_at_k": mean(chunk_metric_rows, "precision_at_k"),
        "ndcg_at_k": mean(chunk_metric_rows, "ndcg_at_k"),
        "file_hit_rate_at_k": file_hit_rate_at_k,
        "file_mrr_at_k": file_mrr_at_k,
        "file_recall_at_k": mean(file_metric_rows, "recall_at_k"),
        "file_precision_at_k": mean(file_metric_rows, "precision_at_k"),
        "file_ndcg_at_k": mean(file_metric_rows, "ndcg_at_k"),
        # Compatibility aliases for existing database columns and clients.
        "hit_rate": hit_rate_at_k,
        "mrr": mrr_at_k,
        "file_hit_rate": file_hit_rate_at_k,
        "file_mrr": file_mrr_at_k,
        "avg_score": round(score_sum / scored, 4) if scored else 0.0,
        "details": details,
    }
    if cfg.RAGAS_ENABLED:
        result["ragas"] = get_ragas_evaluator(cfg).evaluate(ragas_samples)
    else:
        result["ragas"] = {
            "status": "disabled",
            "execution_mode": cfg.RAGAS_EXECUTION_MODE,
            "metrics": {},
        }
    return result


async def save_run(
    session: AsyncSession,
    name: str,
    metrics: Dict[str, Any],
    top_k: int,
    kb_id: Optional[uuid.UUID] = None,
) -> EvaluationRun:
    """把一次评估结果落库为命名运行。"""
    run = EvaluationRun(
        name=name,
        knowledge_base_id=kb_id,
        top_k=top_k,
        query_count=len(metrics.get("details", [])),
        hit_rate=metrics["hit_rate"],
        mrr=metrics["mrr"],
        avg_score=metrics["avg_score"],
        metrics_json=json.dumps(metrics, ensure_ascii=False),
    )
    session.add(run)
    await session.flush()
    logger.info(
        "[eval] run '%s' saved: hit_rate=%.2f mrr=%.2f n=%d",
        name, run.hit_rate, run.mrr, run.query_count,
    )
    return run


async def list_runs(
    session: AsyncSession,
    owner_id: uuid.UUID,
) -> List[EvaluationRun]:
    """列出当前用户有权访问的评估运行（按时间倒序）。"""
    stmt = (
        select(EvaluationRun)
        .join(
            KnowledgeBase,
            EvaluationRun.knowledge_base_id == KnowledgeBase.id,
        )
        .where(KnowledgeBase.owner_id == owner_id)
        .order_by(EvaluationRun.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> Optional[EvaluationRun]:
    """按 ID 获取当前用户有权访问的评估运行。"""
    stmt = (
        select(EvaluationRun)
        .join(
            KnowledgeBase,
            EvaluationRun.knowledge_base_id == KnowledgeBase.id,
        )
        .where(
            EvaluationRun.id == run_id,
            KnowledgeBase.owner_id == owner_id,
        )
    )
    return (await session.execute(stmt)).scalar_one_or_none()
