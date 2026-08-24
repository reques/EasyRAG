"""轻量规则快速路径：简单问题不经过 LLM 意图分类，直接回答。

背景（2026-08-15）：
用户反馈"我在一家餐馆吃坏肚子了，该怎么办"这类简单常识问题也被意图分类器
判成"联网/工具查询（置信度 75%）"并实际调用了 web_search——思考链路长、
开销大且不合理。本模块提供零成本规则预判，在 LLM 意图分类之前拦截明显
简单的请求：

  1. 明确计算请求（"1+1"、"帮我计算 2+2"）   → tool_use + calculator
  2. 明确日期时间请求（"现在几点"、"今天几号"） → tool_use + datetime_tool
  3. 问候闲聊（"你好"、"谢谢"）               → chitchat
  4. 简单自包含常识/生活/写作问题（"吃坏肚子
     怎么办"、"如何做红烧肉"）               → direct（直接回答，不检索不调工具）

其余（不确定、涉及实时数据/知识库、依赖上文的追问）→ 返回 None，
交回 LLM 意图分类器（其 prompt 也已加 direct 意图防误判）。

设计原则：宁可漏（回退 LLM），不可错（把需要工具/检索的问题判成直接回答）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ── 明确计算触发 ────────────────────────────────────────────────────────
# 只匹配"整体就是算式 / 明确说计算"的请求，避免把常识问题误判成计算。
_CALC_PREFIX_RE = re.compile(r"^(计算|算一下|帮我算|帮我计算|请计算|帮我算一下)\s*")
_CALC_CHARS = r"[\d\s+\-*/%().^]"
_CALC_FULL_RE = re.compile(rf"^{_CALC_CHARS}+$")

# ── 明确日期时间触发（只匹配问"现在几点/今天几号"这类说法）──────────
_DATETIME_HINTS = (
    "几点", "几号", "星期几", "周几", "几月几日",
    "现在的日期", "今天的日期", "现在的时间", "今天的时间",
)

# ── 直接回答问题特征（命中其一且通过守卫即判 direct）─────────────────
_DIRECT_HINTS = (
    "怎么办", "如何", "怎么", "为什么", "什么是", "是什么", "哪些",
    "能不能", "可不可以", "可以吗", "该不该", "要不要", "好吗", "行吗",
    "咋办", "咋", "啥", "介绍一下", "介绍下", "解释一下", "解释下",
    "讲讲", "说一下", "说说", "推荐", "区别", "特点", "好处", "坏处", "作用",
)

# 命中这些关键词 → 可能依赖实时数据/搜索/工具，不判直接回答
_TOOL_GUARD_KEYWORDS = (
    "搜索", "检索", "查询", "查一下", "查查",
    "新闻", "天气", "股票", "股价", "汇率", "油价", "金价", "大盘",
    "指数", "实时", "最新", "排行", "多少钱", "报价", "价格",
)

# 命中这些关键词 → 内容可能在已上传文档/知识库里，不判直接回答
_KB_KEYWORDS = (
    "法律", "法条", "民法典", "刑法", "民法", "合同", "劳动法", "劳动合同",
    "条例", "条款", "制度", "手册", "文档", "知识库", "规范", "流程",
    "规程", "规定", "工伤", "赔偿标准", "商标", "专利", "著作权",
    "安全生产", "行政处罚", "诉讼", "起诉", "仲裁",
)

# ── 问候/闲聊（与直接回答同路，但语义标签不同）────────────────────────
_CHITCHAT_HINTS = (
    "你好", "您好", "嗨", "hello", "hi", "在吗", "早上好", "中午好",
    "下午好", "晚上好", "谢谢", "多谢", "再见", "拜拜", "辛苦了",
    "你是谁", "你能做什么", "你会什么", "你叫什么",
)

# ── 追问特征（只含指代/语气词；"如何/怎么样"是自包含问句开头词，
#    不视为追问标记，避免"如何做红烧肉"这类问题在有历史时被误跳过）───
_FOLLOWUP_MARKERS = ("呢", "那", "它", "他", "她", "这", "还有", "再", "吗")

# 超过该长度的提问不判直接回答（长问题交给 LLM 分类）
_MAX_DIRECT_LEN = 60


def _is_self_contained(query: str, history: List[Dict[str, str]]) -> bool:
    """仅当查询不依赖对话历史时才走快速路径。

    与 nodes._needs_rewrite 的差异：重写（rewrite）对短查询可以一律尝试
    （改写自包含短句是无害 no-op），但快速路径判定必须更精确——只有
    明显带指代/语气词且较短的查询才视为追问，交回 LLM 分类器。
    """
    if not history:
        return True
    q = query.strip()
    if len(q) <= 4:  # 短到只剩语气词（"今天呢""怎么样"）→ 几乎必然是追问
        return False
    return not (any(h in q for h in _FOLLOWUP_MARKERS) and len(q) <= 30)


def _calc_expression(query: str) -> Optional[str]:
    """从查询里提取可安全求值的算式；提取不到返回 None。"""
    q = (
        query.strip()
        .strip("?？")
        .replace("=", "")
        .replace("等于多少", "")
        .replace("等于几", "")
    )
    q = _CALC_PREFIX_RE.sub("", q).strip()
    if _CALC_FULL_RE.match(q):
        return q
    return None


def _datetime_ask(query: str) -> bool:
    q = query.strip().lower()
    return any(h in q for h in _DATETIME_HINTS)


def fast_intent_detect(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """规则快速路径意图判定。

    返回可直接合并进 AgentState 的部分 intent 字典；无法确定时返回 None，
    由调用方回退 LLM 意图分类器。
    """
    history = history or []
    q = query.strip()
    if not q:
        return None
    if not _is_self_contained(q, history):
        return None
    ql = q.lower()

    # 1) 明确计算请求
    if any(k in ql for k in ("计算", "算一下", "等于多少", "等于几")) or _CALC_FULL_RE.match(
        q.strip("?？= ")
    ):
        expr = _calc_expression(q)
        if expr:
            return {
                "intent": "tool_use",
                "intent_confidence": 1.0,
                "requires_retrieval": False,
                "requires_tool": True,
                "tool_name": "calculator",
                "tool_args": {"expression": expr},
                "use_react": False,
            }

    # 2) 明确日期时间请求
    if _datetime_ask(q):
        return {
            "intent": "tool_use",
            "intent_confidence": 1.0,
            "requires_retrieval": False,
            "requires_tool": True,
            "tool_name": "datetime_tool",
            "tool_args": {},
            "use_react": False,
        }

    # 3) 可能依赖实时数据 / 搜索 / 知识库 → 交回 LLM 分类器
    if any(k in ql for k in _TOOL_GUARD_KEYWORDS):
        return None
    if any(k in q for k in _KB_KEYWORDS):
        return None

    # 4) 问候闲聊
    if any(k in ql for k in _CHITCHAT_HINTS):
        return {
            "intent": "chitchat",
            "intent_confidence": 1.0,
            "requires_retrieval": False,
            "requires_tool": False,
            "tool_name": None,
            "tool_args": {},
            "use_react": False,
        }

    # 5) 简单自包含常识/生活/写作问题 → 直接回答（不检索、不调工具）
    if len(q) <= _MAX_DIRECT_LEN and any(k in q for k in _DIRECT_HINTS):
        return {
            "intent": "direct",
            "intent_confidence": 1.0,
            "requires_retrieval": False,
            "requires_tool": False,
            "tool_name": None,
            "tool_args": {},
            "use_react": False,
        }

    return None
