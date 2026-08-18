from __future__ import annotations

import math

import pytest

from backend.services.retrieval_metrics import calculate_ranking_metrics


def test_calculates_binary_ranking_metrics_at_k():
    metrics = calculate_ranking_metrics(
        ["noise", "relevant-a", "relevant-b", "outside-cutoff"],
        {"relevant-a", "relevant-b", "relevant-c"},
        k=3,
    )

    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal_dcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.reciprocal_rank_at_k == 0.5
    assert metrics.recall_at_k == pytest.approx(2 / 3)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.ndcg_at_k == pytest.approx(expected_dcg / ideal_dcg)
    assert metrics.first_relevant_rank == 2


def test_cutoff_excludes_relevant_results_below_k():
    metrics = calculate_ranking_metrics(
        ["noise", "relevant"],
        {"relevant"},
        k=1,
    )

    assert metrics.hit_rate_at_k == 0.0
    assert metrics.reciprocal_rank_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.ndcg_at_k == 0.0


def test_duplicate_results_do_not_inflate_relevance_counts():
    metrics = calculate_ranking_metrics(
        ["relevant", "relevant", "noise"],
        {"relevant", "missing"},
        k=3,
    )

    assert metrics.relevant_retrieved == 1
    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == pytest.approx(1 / 3)


def test_empty_relevance_set_returns_zero_metrics():
    metrics = calculate_ranking_metrics(["a"], [], k=3)

    assert metrics.hit_rate_at_k == 0.0
    assert metrics.relevant_total == 0
    assert metrics.first_relevant_rank is None


def test_invalid_k_is_rejected():
    with pytest.raises(ValueError):
        calculate_ranking_metrics([], [], k=0)
