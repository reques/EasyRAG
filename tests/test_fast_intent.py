"""快速路径意图判定（app/graph/fast_intent.py）规则测试。

覆盖截图反馈的核心场景：简单常识问题（"在餐馆吃坏肚子怎么办"）应判为
direct 直接回答，而不是被误判成联网/工具查询；同时验证计算/时间/问候
的零成本分流，以及涉及实时数据/知识库/追问时安全回退（返回 None）。
"""
from __future__ import annotations

from app.graph.fast_intent import fast_intent_detect

HISTORY = [
    {"role": "user", "content": "今天天气怎么样"},
    {"role": "assistant", "content": "今天晴，气温 25 度。"},
]


def _d(query, history=None):
    return fast_intent_detect(query, history or [])


# ── 截图核心场景：简单常识问题 → direct ────────────────────────────────
def test_food_poisoning_advice_is_direct():
    r = _d("我在一家餐馆吃坏肚子了，该怎么办")
    assert r is not None
    assert r["intent"] == "direct"
    assert r["requires_retrieval"] is False
    assert r["requires_tool"] is False
    assert r["tool_name"] is None
    assert r["use_react"] is False


def test_howto_question_is_direct():
    r = _d("如何做红烧肉")
    assert r is not None and r["intent"] == "direct"


def test_definition_question_is_direct():
    r = _d("什么是光合作用")
    assert r is not None and r["intent"] == "direct"


def test_advice_question_is_direct():
    r = _d("晚上失眠怎么办")
    assert r is not None and r["intent"] == "direct"


# ── 明确计算请求 → tool_use + calculator（跳过 LLM 分类）───────────────
def test_pure_arithmetic_goes_calculator():
    r = _d("1+1")
    assert r is not None
    assert r["intent"] == "tool_use"
    assert r["tool_name"] == "calculator"
    assert r["tool_args"]["expression"] == "1+1"


def test_parenthesized_expression_goes_calculator():
    r = _d("(12+34)*2")
    assert r is not None
    assert r["tool_name"] == "calculator"
    assert r["tool_args"]["expression"] == "(12+34)*2"


def test_natural_language_calc_goes_calculator():
    r = _d("帮我计算 2+2")
    assert r is not None
    assert r["tool_name"] == "calculator"
    assert r["tool_args"]["expression"] == "2+2"


def test_calc_with_equals_goes_calculator():
    r = _d("1+1等于几")
    assert r is not None
    assert r["tool_name"] == "calculator"
    assert r["tool_args"]["expression"] == "1+1"


# ── 明确日期时间请求 → tool_use + datetime_tool ────────────────────────
def test_current_time_goes_datetime():
    r = _d("现在几点")
    assert r is not None
    assert r["intent"] == "tool_use"
    assert r["tool_name"] == "datetime_tool"


def test_current_date_goes_datetime():
    r = _d("今天几号")
    assert r is not None and r["tool_name"] == "datetime_tool"


def test_weekday_goes_datetime():
    r = _d("今天星期几")
    assert r is not None and r["tool_name"] == "datetime_tool"


# ── 问候闲聊 → chitchat（跳过 LLM 分类）────────────────────────────────
def test_greeting_is_chitchat():
    r = _d("你好")
    assert r is not None and r["intent"] == "chitchat"


def test_thanks_is_chitchat():
    r = _d("谢谢")
    assert r is not None and r["intent"] == "chitchat"


# ── 需要实时数据/知识库 → 返回 None，交回 LLM 分类器 ───────────────────
def test_weather_query_falls_back():
    assert _d("帮我查询一下无锡今天的天气") is None


def test_news_query_falls_back():
    assert _d("今天的新闻") is None


def test_stock_query_falls_back():
    assert _d("茅台股票今天多少钱") is None


def test_kb_law_query_falls_back():
    assert _d("民法典里关于合同违约是怎么规定的") is None


def test_kb_manual_query_falls_back():
    assert _d("产品手册里怎么说的") is None


# ── 追问（依赖上文）→ 返回 None，交回 LLM 分类器 ───────────────────────
def test_short_followup_falls_back():
    assert _d("今天呢", HISTORY) is None


def test_pronoun_followup_falls_back():
    assert _d("那明天会下雨吗", HISTORY) is None


def test_fresh_question_with_history_still_direct():
    # 有历史但问题是自包含的新问题 → 仍可走快速路径
    r = _d("如何做红烧肉", HISTORY)
    assert r is not None and r["intent"] == "direct"


# ── 边界：不确定的查询不要误判 ─────────────────────────────────────────
def test_unknown_query_returns_none():
    assert _d("帮我写一个 python 脚本") is None


def test_empty_query_returns_none():
    assert _d("") is None


def test_long_query_falls_back():
    long_q = "请帮我详细分析一下这个项目的架构设计是否合理，并且给出改进建议，最好能结合代码层面说明"
    assert _d(long_q) is None
