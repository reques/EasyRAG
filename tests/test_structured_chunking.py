"""Tests for the generic structure-aware splitter (split_structured).

覆盖：标题层级路径、编号条目、章节词层级、超长 section 滑动窗口二级切分。
通用性：不绑定任何领域（法律/规章/技术文档均可）。
"""
from __future__ import annotations

import pytest

from app.rag.chunker import split_structured


def test_markdown_headings_build_section_paths():
    text = (
        "# 第一章 总则\n"
        "第一段内容。\n"
        "## 第一节 基本规定\n"
        "第二段内容。\n"
    )
    chunks = split_structured(text, chunk_size=200, chunk_overlap=20)
    assert any("[第一章 总则]" in c for c in chunks)
    assert any("[第一章 总则 > 第一节 基本规定]" in c for c in chunks)
    assert any("第一段内容" in c for c in chunks)
    assert any("第二段内容" in c for c in chunks)


def test_numbered_items_keep_body_with_number():
    text = (
        "一、总则\n"
        "总则的说明文字。\n"
        "1. 第一个条目\n"
        "2. 第二个条目\n"
        "三、附则\n"
        "附则说明。\n"
    )
    chunks = split_structured(text, chunk_size=200, chunk_overlap=20)
    # 平级条目各自成块，编号进路径、正文保留
    assert any(c.startswith("[一、]") and "总则的说明文字" in c for c in chunks)
    assert any(c.startswith("[1.]") and "第一个条目" in c for c in chunks)
    assert any(c.startswith("[2.]") and "第二个条目" in c for c in chunks)
    assert any(c.startswith("[三、]") and "附则说明" in c for c in chunks)


def test_chapter_words_keep_hierarchy():
    text = (
        "第一章 基本规定\n"
        "第一条 立法目的内容。\n"
        "第二章 分则\n"
        "第一节 一般规定\n"
        "第三条 原则规定内容。\n"
    )
    chunks = split_structured(text, chunk_size=200, chunk_overlap=20)
    assert any("[第一章 > 第一条]" in c for c in chunks)
    assert any("[第二章 > 第一节 > 第三条]" in c for c in chunks)


def test_overlong_section_falls_back_to_sliding_window():
    body = "内容" * 200  # 400 字，远超 chunk_size
    text = f"# 标题\n{body}\n"
    chunks = split_structured(text, chunk_size=100, chunk_overlap=20)
    # 超长 section 应被滑动窗口切成多块（>1），且每块保留标题前缀
    assert len(chunks) > 1
    assert all("[标题]" in c for c in chunks)
    # 滑动窗口带 overlap：相邻块有重叠文本
    assert chunks[0][-30:] in chunks[1]


def test_plain_text_no_structure_uses_window():
    text = "无结构纯文本。" * 30
    chunks = split_structured(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(not c.startswith("[") for c in chunks)


def test_empty_and_whitespace_return_empty():
    assert split_structured("") == []
    assert split_structured("   \n  ") == []
