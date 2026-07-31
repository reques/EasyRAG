"""检索评估服务（阶段 2D）— 量化检索质量。

核心思路：
- 评估集 = [(query, expected_source)]，expected_source 为该 query 期望命中的文件名；
- 每条 query 走真实 retriever.retrieve，判定 expected_source 是否出现在 top_k 结果中；
- 聚合指标：HitRate@k（命中率）、MRR@k（平均倒数排名）、avg_score（平均分）；
- 每次评估以命名运行（EvaluationRun）落库，可横向对比不同分块策略/参数的效果。
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logger import get_logger
from backend.storage.postgres.models_knowledge import EvaluationRun

logger = get_logger(__name__)
cfg = get_settings()

# 评估集条目: (query, expected_source)
EvalCase = Tuple[str, str]


def run_evaluation(
    cases: List[EvalCase],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """对评估集逐条检索并计算指标（同步，直接调 retriever 单例）。

    返回 {"hit_rate", "mrr", "avg_score", "details": [...]}。
    """
    from app.rag.retriever import get_retriever

    k = top_k or cfg.RETRIEVER_TOP_K
    retriever = get_retriever()

    details: List[Dict[str, Any]] = []
    hits = 0
    rr_sum = 0.0
    score_sum = 0.0
    scored = 0

    for query, expected in cases:
        try:
            docs = retriever.retrieve(query, top_k=k)
        except Exception as exc:
            logger.warning("[eval] retrieve failed for %r: %s", query[:50], exc)
            docs = []

        hit_rank: Optional[int] = None
        top_score = docs[0]["metadata"].get("score", 0.0) if docs else 0.0
        for rank, d in enumerate(docs, start=1):
            src = d["metadata"].get("source", "")
            if expected == src or expected in src:
                hit_rank = rank
                break

        if hit_rank is not None:
            hits += 1
            rr_sum += 1.0 / hit_rank
        if docs:
            score_sum += top_score
            scored += 1

        details.append({
            "query": query,
            "expected_source": expected,
            "hit_rank": hit_rank,
            "top_score": round(top_score, 4),
            "returned": len(docs),
        })

    n = len(cases)
    return {
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
        "avg_score": round(score_sum / scored, 4) if scored else 0.0,
        "details": details,
    }


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


async def list_runs(session: AsyncSession) -> List[EvaluationRun]:
    """列出全部评估运行（按时间倒序）。"""
    stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
    return list((await session.execute(stmt)).scalars().all())


async def get_run(session: AsyncSession, run_id: uuid.UUID) -> Optional[EvaluationRun]:
    stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
    return (await session.execute(stmt)).scalar_one_or_none()
