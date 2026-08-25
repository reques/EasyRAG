"""评测数据集与报告 - 规范化评测体系的数据层与报告层测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.services import evaluation_report
from backend.services.evaluation_datasets import (
    deserialize_cases,
    export_dataset_json,
    serialize_cases,
)
from backend.services.evaluation_service import EvaluationCase


def make_case(**overrides) -> EvaluationCase:
    base = dict(
        question="民法典第一千二百六十条讲了什么",
        expected_file_id="22222222-2222-2222-2222-222222222222",
        expected_chunk_ids=("chunk-1", "chunk-2"),
        reference_answer="答",
        expected_source="民法典.pdf",
    )
    base.update(overrides)
    return EvaluationCase(**base)


def test_serialize_deserialize_round_trip():
    cases = [
        make_case(),
        make_case(
            question="负样本",
            expected_chunk_ids=(),
            expect_miss=True,
        ),
    ]
    payload = serialize_cases(cases)
    restored = deserialize_cases(payload)
    # expected_source 是运行时内部字段，不参与持久化，逐字段断言
    assert restored[0].question == cases[0].question
    assert restored[0].expected_file_id == cases[0].expected_file_id
    assert restored[0].expected_chunk_ids == ("chunk-1", "chunk-2")
    assert restored[1].expect_miss is True
    assert restored[1].expected_chunk_ids == ()


def test_deserialize_absorbs_legacy_single_chunk_field():
    payload = [
        {
            "question": "q",
            "expected_file_id": "22222222-2222-2222-2222-222222222222",
            "expected_chunk_id": "legacy-chunk",
        }
    ]
    cases = deserialize_cases(payload)
    assert cases[0].expected_chunk_ids == ("legacy-chunk",)
    assert cases[0].expected_chunk_id == "legacy-chunk"


def test_deserialize_rejects_blank_question():
    with pytest.raises(ValueError):
        deserialize_cases([{"question": "  ", "expected_file_id": "x"}])


def test_export_dataset_json_shape():
    dataset = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="legal-golden-v1",
        knowledge_base_id="22222222-2222-2222-2222-222222222222",
        description="desc",
        case_count=1,
        version=2,
        cases_json=json.dumps(serialize_cases([make_case()]), ensure_ascii=False),
        created_at=None,
        updated_at=None,
    )
    payload = export_dataset_json(dataset)
    assert payload["name"] == "legal-golden-v1"
    assert payload["version"] == 2
    assert payload["cases"][0]["expected_chunk_ids"] == ["chunk-1", "chunk-2"]


def test_build_markdown_report_contains_key_sections():
    metrics = {
        "metrics_version": "local-v2",
        "k": 5,
        "hit_rate_at_k": 1.0,
        "mrr_at_k": 0.8,
        "recall_at_k": 0.4,
        "precision_at_k": 0.2,
        "ndcg_at_k": 0.9,
        "avg_score": 0.7,
        "run_metadata": {"chunk_strategy": "legal", "k": 5},
        "analysis": {
            "missed_count": 1,
            "low_recall_count": 1,
            "false_positive_count": 0,
            "missed": [{"question": "q1", "top_score": 0.3}],
            "low_recall": [{"question": "q2", "recall_at_k": 0.2}],
            "false_positives": [],
        },
        "details": [
            {
                "question": "q1",
                "reference_mode": "chunk_ids",
                "expected_chunk_count": 2,
                "chunk_hit_rank": 1,
                "top_score": 0.9,
                "returned": 5,
            }
        ],
    }
    report = evaluation_report.build_markdown_report(
        run_name="legal-recursive-v2",
        created_at="2026-08-24T10:00:00+08:00",
        knowledge_base_name="法律知识库",
        metrics=metrics,
        ragas={"status": "completed", "metrics": {"id_context_recall": 0.5}},
    )
    assert "RAG 检索评估报告 - legal-recursive-v2" in report
    assert "## 1. 运行环境" in report
    assert "## 2. 确定性检索指标" in report
    assert "## 3. RAGAs 指标" in report
    assert "## 4. 逐条明细" in report
    assert "## 5. 失败分析" in report
    assert "精确 chunk 标注" in report
    assert "id_context_recall" in report