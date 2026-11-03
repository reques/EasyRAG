"""Link persisted local evaluation results to the exact experiment-item roots."""
from __future__ import annotations

import math
from urllib.parse import quote

from app.core.config import get_settings
from app.observability.langfuse_client import valid_base_url
from backend.services.score_delivery_service import enqueue_score


def experiment_navigation(experiment):
    cfg = get_settings()
    base = cfg.EASYRAG_OBSERVABILITY_WEB_BASE_URL
    if not valid_base_url(base) or not cfg.EASYRAG_OBSERVABILITY_PROJECT_ID:
        return None
    return (f"{base.rstrip('/')}/project/{quote(cfg.EASYRAG_OBSERVABILITY_PROJECT_ID, safe='')}"
            f"/experiments/results?baseline={quote(experiment['id'], safe='')}")


async def enqueue_evaluation_scores(session, metrics, user_id):
    ragas_details = {item["index"]: item.get("scores", {}) for item in (metrics.get("ragas") or {}).get("details", [])}
    for index, detail in enumerate(metrics.get("details", [])):
        link = detail.get("observability")
        if not link or detail.get("retrieval_error"):
            continue
        scores = {}
        for key, name in (("hit_rate_at_k", "hit_rate_at_k"), ("reciprocal_rank_at_k", "mrr_at_k"),
                          ("recall_at_k", "recall_at_k"), ("precision_at_k", "precision_at_k"), ("ndcg_at_k", "ndcg_at_k")):
            scores[f"retrieval.{name}"] = (detail.get("chunk_metrics") or {}).get(key)
        if detail.get("expect_miss"):
            scores = {"retrieval.expected_miss": 1 if detail.get("false_positive") == 0 else 0}
        if not detail.get("generation_error"):
            scores.update({f"ragas.{name}": value for name, value in ragas_details.get(index, {}).items()})
        for name, value in scores.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            await enqueue_score(session, score_id=f"eval-{link['experiment_id']}-{index}-{name}",
                                user_id=user_id, payload={"trace_id": link["trace_id"],
                                "observation_id": link["observation_id"], "name": name,
                                "value": value, "data_type": "NUMERIC", "environment": "experiment"})
