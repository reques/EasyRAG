"""评测数据集（Golden Set）服务 - 持久化、导入导出、与运行关联。

评测数据集是「规范化 RAG 评测」的数据基础：
- 每条用例标注与该问题真正相关的 chunk 集（expected_chunk_ids），
  而不是把整份文件当作相关集，保证 Recall/Precision 语义与 RAGAs 官方口径一致；
- 支持负样本（expect_miss=True）：用于度量检索误报率；
- 数据集可复用、可版本化（同名更新 version+1）、可导入导出 JSON。
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.evaluation_service import EvaluationCase
from backend.storage.postgres.models_knowledge import EvaluationDataset, KnowledgeBase


# ── 序列化 / 反序列化 ──────────────────────────────────────────────────────────


def serialize_cases(cases: List[EvaluationCase]) -> List[Dict[str, Any]]:
    """把 EvaluationCase 序列化为可持久化 / 可导出的 JSON 结构。"""
    rows: List[Dict[str, Any]] = []
    for case in cases:
        rows.append({
            "question": case.question,
            "expected_file_id": case.expected_file_id,
            "expected_chunk_ids": sorted(set(case.expected_chunk_ids)),
            "reference_answer": case.reference_answer,
            "expect_miss": case.expect_miss,
        })
    return rows


def deserialize_cases(payload: Any) -> List[EvaluationCase]:
    """从 JSON 结构还原 EvaluationCase，并做字段校验与规范化。

    兼容旧数据：legacy 单值字段 expected_chunk_id 会被吸收进
    expected_chunk_ids，避免已有评测集在升级后丢失标注。
    """
    if not isinstance(payload, list):
        raise ValueError("cases must be a JSON list")
    cases: List[EvaluationCase] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("each case must be a JSON object")
        question = str(raw.get("question") or "").strip()
        if not question:
            raise ValueError("case.question must not be blank")
        expected_file_id = str(raw.get("expected_file_id") or "").strip()
        if not expected_file_id:
            raise ValueError("case.expected_file_id is required")

        chunk_ids = raw.get("expected_chunk_ids") or []
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        expected_chunk_ids: List[str] = [
            str(cid).strip() for cid in chunk_ids if str(cid).strip()
        ]
        legacy_chunk = str(raw.get("expected_chunk_id") or "").strip()
        if legacy_chunk and legacy_chunk not in expected_chunk_ids:
            expected_chunk_ids.insert(0, legacy_chunk)

        cases.append(EvaluationCase(
            question=question,
            expected_file_id=expected_file_id,
            expected_chunk_ids=tuple(expected_chunk_ids),
            expected_chunk_id=expected_chunk_ids[0] if expected_chunk_ids else None,
            reference_answer=str(raw.get("reference_answer") or "").strip(),
            expect_miss=bool(raw.get("expect_miss", False)),
        ))
    return cases


def export_dataset_json(dataset: EvaluationDataset) -> Dict[str, Any]:
    """把数据集序列化为 API 响应（含 cases 明细）。"""
    cases = deserialize_cases(json.loads(dataset.cases_json or "[]"))
    return {
        "id": str(dataset.id),
        "name": dataset.name,
        "knowledge_base_id": (
            str(dataset.knowledge_base_id)
            if dataset.knowledge_base_id else None
        ),
        "description": dataset.description,
        "case_count": dataset.case_count,
        "version": dataset.version,
        "cases": serialize_cases(cases),
        "created_at": (
            dataset.created_at.isoformat() if dataset.created_at else ""
        ),
        "updated_at": (
            dataset.updated_at.isoformat() if dataset.updated_at else ""
        ),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────


async def save_dataset(
    session: AsyncSession,
    *,
    name: str,
    kb_id: uuid.UUID,
    description: str,
    cases: List[EvaluationCase],
) -> EvaluationDataset:
    """新建或更新同名评测数据集（同名覆盖并递增 version）。"""
    stmt = select(EvaluationDataset).where(
        EvaluationDataset.name == name,
        EvaluationDataset.knowledge_base_id == kb_id,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    payload = json.dumps(serialize_cases(cases), ensure_ascii=False)
    if existing is not None:
        existing.cases_json = payload
        existing.description = description
        existing.case_count = len(cases)
        existing.version = existing.version + 1
        return existing
    dataset = EvaluationDataset(
        name=name,
        knowledge_base_id=kb_id,
        description=description,
        cases_json=payload,
        case_count=len(cases),
        version=1,
    )
    session.add(dataset)
    await session.flush()
    return dataset


async def list_datasets(
    session: AsyncSession,
    owner_id: uuid.UUID,
) -> List[EvaluationDataset]:
    """列出当前用户有权访问的评测数据集（按更新时间倒序）。"""
    stmt = (
        select(EvaluationDataset)
        .join(
            KnowledgeBase,
            EvaluationDataset.knowledge_base_id == KnowledgeBase.id,
        )
        .where(KnowledgeBase.owner_id == owner_id)
        .order_by(EvaluationDataset.updated_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_dataset(
    session: AsyncSession,
    dataset_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> Optional[EvaluationDataset]:
    stmt = (
        select(EvaluationDataset)
        .join(
            KnowledgeBase,
            EvaluationDataset.knowledge_base_id == KnowledgeBase.id,
        )
        .where(
            EvaluationDataset.id == dataset_id,
            KnowledgeBase.owner_id == owner_id,
        )
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def delete_dataset(
    session: AsyncSession,
    dataset_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> bool:
    dataset = await get_dataset(session, dataset_id, owner_id)
    if dataset is None:
        return False
    await session.delete(dataset)
    return True