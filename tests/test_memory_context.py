"""上下文管理与记忆规则测试（纯逻辑部分）。

覆盖 2026-08-15 P0 修复：
- decide_history_window：full / compressed / cap_tail 三态窗口决策——
  compressed 取真实尾部（修复旧实现"最早 100 条内的伪最近窗口"）；
  无摘要超限时 cap_tail 显式兜底，而不是隐式截断 / 无界增长。
- should_extract_fact：语义记忆触发词（收紧后过宽的"以后"不再误触发）。
"""
from __future__ import annotations

from app.memory.manager import should_extract_fact
from backend.services.chat_service import decide_history_window


# ── decide_history_window ────────────────────────────────────────────────
def test_window_full_when_within_window():
    plan = decide_history_window(count=20, has_summary=True, window=20, cap=100)
    assert plan["mode"] == "full"
    assert plan["limit"] == 20 and plan["offset"] == 0


def test_window_compressed_uses_true_tail():
    # 150 条消息、有摘要 → 摘要 + 最近 20 条（消息 131..150）
    plan = decide_history_window(count=150, has_summary=True, window=20, cap=100)
    assert plan["mode"] == "compressed"
    assert plan["limit"] == 20
    assert plan["offset"] == 130


def test_window_cap_tail_when_no_summary():
    # 150 条、无摘要 → 取最近 100 条（消息 51..150），而不是最早 100 条
    plan = decide_history_window(count=150, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 100
    assert plan["offset"] == 50


def test_window_cap_tail_bounded_by_cap():
    plan = decide_history_window(count=300, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 100 and plan["offset"] == 200


def test_window_no_summary_within_cap_returns_all():
    plan = decide_history_window(count=60, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 60 and plan["offset"] == 0


def test_window_short_conversation_full():
    plan = decide_history_window(count=8, has_summary=False, window=20, cap=100)
    assert plan["mode"] == "full"
    assert plan["limit"] == 8 and plan["offset"] == 0


# ── should_extract_fact（语义记忆触发词）────────────────────────────────
def test_fact_trigger_identity():
    assert should_extract_fact("我是一名律师")
    assert should_extract_fact("叫我小王就行")


def test_fact_trigger_preference():
    assert should_extract_fact("我喜欢简洁的回答")
    assert should_extract_fact("帮我记住这个")
    assert should_extract_fact("我的偏好是英文回复")


def test_fact_no_trigger_normal_question():
    assert not should_extract_fact("民法典第10条是什么")
    assert not should_extract_fact("今天天气怎么样")


def test_fact_no_trigger_broad_word_removed():
    # "以后"过宽（"以后再说吧"无事实可提），已从触发词中移除
    assert not should_extract_fact("这个问题以后再说吧")
    assert not should_extract_fact("我们以后再聊")
