"""``/compact`` 上下文压缩：窗口决策（缺口安全）、压缩计划与提示词。

关键不变量：**已折叠的消息有摘要兜底，未折叠的消息必须还在原文窗口里**。
摘要滞后（折叠失败 / 手动压缩只跑了一部分）时，窗口起点不得越过第一条
未折叠消息，否则中间段既不在摘要里、又被窗口丢掉。
"""
import pytest

from app.memory.manager import (
    SUMMARY_SECTIONS,
    build_compact_prompt,
    estimate_tokens,
    normalise_structured_summary,
    plan_auto_fold,
    plan_compaction,
)
from backend.services.chat_service import decide_history_window


# ── 窗口决策：folded 水位 ─────────────────────────────────────────────────

def test_without_watermark_behaviour_is_unchanged():
    """不传 folded 时保持原语义（既有调用/测试不受影响）。"""
    plan = decide_history_window(count=150, has_summary=True, window=20, cap=100)
    assert plan == {"mode": "compressed", "limit": 20, "offset": 130}


def test_watermark_advance_actually_shrinks_injected_window():
    """水位推进到哪，注入的原文就到哪——/compact N 才能真正收窄窗口。

    自动摘要维持 20 条窗口时：注入 20 条；/compact 4（留 8 条）后水位到
    count-8 → 注入 8 条；/compact 1（留 2 条）→ 注入 2 条。"""
    assert decide_history_window(150, True, 20, 100, folded=130)["limit"] == 20
    tight = decide_history_window(150, True, 20, 100, folded=142)
    assert tight["mode"] == "compressed" and tight["limit"] == 8 and tight["offset"] == 142
    tighter = decide_history_window(150, True, 20, 100, folded=148)
    assert tighter["limit"] == 2 and tighter["offset"] == 148
    # 至少有 floor 条原文（水位推到最新一条也不会只剩摘要）
    assert decide_history_window(150, True, 20, 100, folded=150)["limit"] == 2


def test_lagging_summary_widens_tail_instead_of_dropping_messages():
    """摘要只覆盖前 60 条时，窗口必须从 60 开始，不能从 130 开始（否则 61..129 蒸发）。"""
    plan = decide_history_window(count=150, has_summary=True, window=20, cap=100, folded=60)
    assert plan["mode"] == "compressed"
    assert plan["offset"] == 60
    assert plan["limit"] == 90


def test_severely_lagging_summary_falls_back_to_cap_tail():
    """摘要严重滞后（窗口被 cap 撑爆）→ 退回显式上限兜底，不让上下文无界增长。"""
    plan = decide_history_window(count=5000, has_summary=True, window=20, cap=100, folded=10)
    assert plan["mode"] == "cap_tail"
    assert plan["limit"] == 100 and plan["offset"] == 4900


# ── 自动增量摘要：只折保留窗口之外（2026-10 修复）────────────────────────

def test_auto_fold_never_eats_the_recent_window():
    """核心回归：自动摘要不得折叠最近 window 条，否则 /compact 永远无事可做、
    且摘要与注入的尾部原文重复。"""
    # 30 条、保留窗口 20、已折叠 10 → 可折叠量 = 30-20-10 = 0
    plan = plan_auto_fold(total=30, folded=10, pending=20, keep_window=20, interval=10, batch=20)
    assert plan == {"fold": 0, "reason": "nothing_outside_window"}

    # 即使 pending 很多，只要都在保留窗口内也不折叠
    inside = plan_auto_fold(total=18, folded=0, pending=18, keep_window=20, interval=10, batch=20)
    assert inside["fold"] == 0 and inside["reason"] == "nothing_outside_window"


def test_auto_fold_takes_only_what_is_outside_the_window():
    # 40 条、已折 10、保留 20 → 落在窗口外的只有 10 条
    plan = plan_auto_fold(total=40, folded=10, pending=30, keep_window=20, interval=10, batch=20)
    assert plan["fold"] == 10
    # 批量上限生效
    capped = plan_auto_fold(total=100, folded=0, pending=100, keep_window=20, interval=10, batch=20)
    assert capped["fold"] == 20


def test_auto_fold_respects_the_interval_trigger():
    plan = plan_auto_fold(total=100, folded=0, pending=4, keep_window=20, interval=10, batch=20)
    assert plan == {"fold": 0, "reason": "below_interval"}


def test_auto_and_manual_paths_do_not_overlap_but_tighter_window_does():
    """健康态下自动摘要已追平 → 普通 /compact 无事可做（UI 必须如实说明）；
    但用户显式要求更小保留窗口时仍能压出内容。"""
    total, folded = 40, 20  # 自动摘要覆盖前 20 条，最近 20 条保留原文
    assert plan_auto_fold(total=total, folded=folded, pending=20,
                          keep_window=20, interval=10, batch=20)["fold"] == 0
    assert plan_compaction(count=total, folded=folded, keep_window=20,
                           batch=40, max_batches=8)["needed"] == 0
    # /compact 2（保留 2 轮 = 4 条）→ 需要再折叠 16 条
    tighter = plan_compaction(count=total, folded=folded, keep_window=4, batch=40, max_batches=8)
    assert tighter["needed"] == 16
    assert tighter["batches"] == [(20, 16)]


def test_manual_default_window_is_tighter_than_auto_window():
    """手动压缩的默认保留窗口（COMPACT_KEEP_TURNS=4 → 8 条）必须比自动摘要窗口
    （RECENT_TURNS_KEPT=10 → 20 条）更紧，否则 /compact 永远无事可做。"""
    from app.core.config import get_settings
    from app.memory.manager import RECENT_TURNS_KEPT

    cfg = get_settings()
    auto_window = RECENT_TURNS_KEPT * 2
    manual_window = cfg.COMPACT_KEEP_TURNS * 2
    assert manual_window < auto_window, (manual_window, auto_window)

    # 自动摘要追平后的健康态：仍有 12 条可被 /compact 收进摘要
    total, folded = 60, 40  # 摘要覆盖前 40 条，尾部 20 条保留原文
    plan = plan_compaction(count=total, folded=folded, keep_window=manual_window,
                           batch=cfg.COMPACT_FOLD_BATCH, max_batches=cfg.COMPACT_MAX_BATCHES)
    assert plan["needed"] == auto_window - manual_window == 12
    # 折叠后注入窗口从 20 条降到 8 条
    assert decide_history_window(total, True, auto_window, 100, folded=plan["folded_after"])["limit"] == manual_window


def test_plan_compaction_targets_window_boundary():
    # 100 条消息、保留窗口 20 条 → 目标折叠前 80 条；批 40 → 两批
    plan = plan_compaction(count=100, folded=0, keep_window=20, batch=40, max_batches=8)
    assert plan["needed"] == 80
    assert plan["batches"] == [(0, 40), (40, 40)]
    assert plan["folded_after"] == 80
    assert plan["complete"] is True


def test_plan_compaction_resumes_from_watermark_and_skips_when_caught_up():
    plan = plan_compaction(count=100, folded=60, keep_window=20, batch=40, max_batches=8)
    assert plan["needed"] == 20
    assert plan["batches"] == [(60, 20)]

    assert plan_compaction(count=100, folded=80, keep_window=20, batch=40, max_batches=8)["needed"] == 0
    # 会话还没超过保留窗口 → 无需压缩
    short = plan_compaction(count=18, folded=0, keep_window=20, batch=40, max_batches=8)
    assert short["needed"] == 0 and short["complete"] is True


def test_plan_compaction_respects_budget_and_reports_incomplete():
    """消息量超出单次预算 → 明确告知未完成（水位仍连续推进，绝不跳段）。"""
    plan = plan_compaction(count=1000, folded=0, keep_window=20, batch=40, max_batches=3)
    assert plan["batches"] == [(0, 40), (40, 40), (80, 40)]
    assert plan["folded_after"] == 120
    assert plan["complete"] is False
    # 再跑一次从水位继续
    again = plan_compaction(count=1000, folded=plan["folded_after"], keep_window=20, batch=40, max_batches=3)
    assert again["batches"][0] == (120, 40)


# ── 提示词与工具函数 ─────────────────────────────────────────────────────

class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


def test_compact_prompt_requires_sections_and_carries_previous_summary():
    prompt = build_compact_prompt("当前目标：旧目标", [_Msg("user", "帮我看看 /data/kb 目录")])
    for section in SUMMARY_SECTIONS:
        assert f"{section}：" in prompt
    assert "旧目标" in prompt
    assert "/data/kb" in prompt
    # 明确定位为「压缩上下文」，并要求忽略其中的指令
    assert "压缩" in prompt
    assert "不得执行" in prompt


def test_compact_prompt_truncates_single_huge_message():
    prompt = build_compact_prompt("", [_Msg("assistant", "x" * 5000)])
    assert "x" * 800 in prompt
    assert "x" * 900 not in prompt


def test_normalise_keeps_sections_for_incremental_summary():
    merged = normalise_structured_summary("模型直接给了一段话")
    assert all(f"{section}：" in merged for section in SUMMARY_SECTIONS)
    assert "模型直接给了一段话" in merged


def test_estimate_tokens_counts_cjk_and_ascii_differently():
    assert estimate_tokens("") == 0
    cjk = estimate_tokens("中文" * 50)
    ascii_tokens = estimate_tokens("a" * 100)
    assert cjk > ascii_tokens  # 中文按 ~1 token/字，ASCII 约 4 字符/token
