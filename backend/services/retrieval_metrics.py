"""Deterministic, model-free metrics for ranked retrieval results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class RankingMetrics:
    """Metrics for one query at a fixed cutoff K."""

    hit_rate_at_k: float
    reciprocal_rank_at_k: float
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    first_relevant_rank: Optional[int]
    relevant_retrieved: int
    relevant_total: int

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_ranking_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> RankingMetrics:
    """Calculate binary-relevance HitRate/MRR/Recall/Precision/nDCG@K.

    Precision uses the conventional fixed ``K`` denominator. Duplicate result
    IDs are credited only once, preventing duplicate chunks from inflating the
    relevance count. Empty relevance sets produce zero-valued metrics.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    ranked = [
        str(value)
        for value in retrieved_ids[:k]
        if value is not None and str(value)
    ]
    relevant = {
        str(value)
        for value in relevant_ids
        if value is not None and str(value)
    }
    if not relevant:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, None, 0, 0)

    first_rank: Optional[int] = None
    credited: set[str] = set()
    dcg = 0.0
    for rank, item_id in enumerate(ranked, start=1):
        if item_id not in relevant or item_id in credited:
            continue
        credited.add(item_id)
        if first_rank is None:
            first_rank = rank
        dcg += 1.0 / math.log2(rank + 1)

    relevant_retrieved = len(credited)
    ideal_count = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return RankingMetrics(
        hit_rate_at_k=1.0 if first_rank is not None else 0.0,
        reciprocal_rank_at_k=(1.0 / first_rank) if first_rank else 0.0,
        recall_at_k=relevant_retrieved / len(relevant),
        precision_at_k=relevant_retrieved / k,
        ndcg_at_k=(dcg / idcg) if idcg else 0.0,
        first_relevant_rank=first_rank,
        relevant_retrieved=relevant_retrieved,
        relevant_total=len(relevant),
    )
