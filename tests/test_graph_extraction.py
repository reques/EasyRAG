"""Tests for mixed graph extraction: rule-based (legal) + generic NER (zh/en) + LLM fallback."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

from backend.services import graph_service as gs


KB_ID = uuid.uuid4()


def test_extract_chinese_entities_via_jieba():
    ents = gs._extract_entities_generic(
        "借款人应当按照约定的期限返还借款，贷款人可以催告借款人在合理期限内返还。"
    )
    names = [n for n, _, _ in ents]
    assert any("借款" in n for n in names)
    assert any("期限" in n for n in names)


def test_extract_english_entities_and_filter_stopwords():
    ents = gs._extract_entities_generic(
        "Machine Learning uses neural networks and GPU for training large models."
    )
    names = [n for n, _, _ in ents]
    assert "Machine Learning" in names
    assert "GPU" in names
    # jieba 不再把英文停用词（uses/and/for）标成实体
    assert not any(n in ("uses", "and", "for") for n in names)


def test_legal_chunks_extract_concepts_and_structure(monkeypatch):
    chunks = [
        ("[第十二章 借款合同]\n第六百七十五条　【借款人返还借款的期限】 借款人应当按约定期限返还借款。", {"chunk_index": 0}),
        ("[第十二章 借款合同]\n第六百七十六条　【借款合同的内容】 借款合同的内容一般包括借款种类、数额、利率、期限等。", {"chunk_index": 1}),
        ("[第十二章 借款合同]\n第六百七十七条　【借款合同的形式】 借款合同应当采用书面形式。", {"chunk_index": 2}),
    ]
    assert gs._looks_like_legal_chunks(chunks) is True

    llm_calls = []

    async def fake_llm(s, kb_id, chunks, sn, progress_callback=None):
        llm_calls.append(sn)
        # LLM 语义抽取：概念实体 + 语义关系
        return {"entities": 2, "relations": 1}

    monkeypatch.setattr(gs, "_extract_graph_llm", fake_llm)

    result = asyncio.run(gs.extract_graph_from_chunks(MagicMock(), KB_ID, chunks, "law.txt"))
    # 双层抽取：LLM 概念(2+1) + 规则结构(3 条文 + 1 章节 = 4 实体；3 归属 = 3 关系)
    assert llm_calls == ["law.txt"]
    assert result == {"entities": 2 + 4, "relations": 1 + 3}


def test_generic_chunks_extract_llm_concepts(monkeypatch):
    calls = []

    async def fake_llm(s, kb_id, chunks, sn, progress_callback=None):
        calls.append(sn)
        return {"entities": 1, "relations": 0}

    monkeypatch.setattr(gs, "_extract_graph_llm", fake_llm)

    chunks = [("Machine Learning uses neural networks for training.", {"chunk_index": 0})]
    asyncio.run(gs.extract_graph_from_chunks(MagicMock(), KB_ID, chunks, "doc.txt"))
    # 双层抽取：LLM 概念抽取总是被调用
    assert calls == ["doc.txt"]
