"""
Prompt templates used across the agent workflow.

All templates use Python f-string-style placeholders wrapped in
`{variable}` notation.  Build them with `PromptTemplate.format(**kwargs)`.
"""
from __future__ import annotations


class PromptTemplate:
    """Minimal prompt template supporting `{var}` substitution."""

    def __init__(self, template: str):
        self._template = template

    def format(self, **kwargs) -> str:
        try:
            return self._template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(f"Missing prompt variable: {exc}") from exc

    def __str__(self) -> str:
        return self._template


# ── Query Rewrite (指代消解 / 上下文还原) ────────────────────────────────────

QUERY_REWRITE = PromptTemplate(
    """你是一个查询改写器。根据对话历史，把用户最新这条可能有指代/省略的消息，改写成一个不依赖上下文也能独立理解的完整问题。

对话历史（最近的在前）:
{history}

用户最新消息: {query}

改写规则:
- 如果最新消息已经自包含（能独立看懂），原样返回，不要画蛇添足
- 如果有指代（"今天呢/那明天/它/这个/那里/还有吗"等），结合上文补全主语和宾语
- 保留用户原本的意图和语气，只补全省略信息，不改变诉求
- 输出只有改写后的那一句话，不要任何解释、引号或前后缀

改写结果:
"""
)

# ── Intent Recognition ────────────────────────────────────────────────────────

INTENT_RECOGNITION = PromptTemplate(
    """You are an intent classifier for an AI assistant.

Conversation history (most recent last, may be empty):
{history}

Current user query (already rewritten to be self-contained): {query}

Classify the intent of the CURRENT query. Use the history ONLY to understand
references, not to change the topic — classify what the user is asking NOW.

Return ONLY valid JSON in this exact format:
{{
  "intent": "<one of: knowledge_qa | tool_use | complex_task | direct | chitchat>",
  "confidence": <float 0.0-1.0>,
  "requires_retrieval": <true|false>,
  "requires_tool": <true|false>,
  "tool_name": "<one of the available tools below, or null>",
  "tool_args": {{}},
  "reasoning": "<one sentence why>"
}}

Currently available tools:
{available_tools}

Intent definitions:
- knowledge_qa  : question answerable from a stored knowledge base (laws, docs, manuals)
- tool_use      : needs a live tool — calculation, current time, OR real-time web info
- complex_task  : multi-step task combining retrieval AND tools
- direct        : general-knowledge / life-advice / how-to / common-sense / writing question
                  answerable from the model's own knowledge WITHOUT any tool or
                  knowledge base (e.g. "吃坏肚子怎么办", "如何做红烧肉", "什么是光合作用")
- chitchat      : greeting, thanks, small talk with no factual ask

CRITICAL routing rules:
- 常识/生活建议/健康/科普/做法/写作类问题（如"在餐馆吃坏肚子怎么办""如何做红烧肉"
  "什么是光合作用""帮我写一封辞职信"）→ direct，requires_retrieval=false、
  requires_tool=false。不要因为"可能查得到"就调 web_search。
- Only use web_search for genuinely live/real-time data: weather, news, stock prices,
  exchange rates, currency conversion, current time, today's events. 一般知识性问题不要联网。
- Weather, news, stock prices, exchange rates, "今天/现在/最新" real-time facts → tool_use + web_search, requires_retrieval=false. NEVER route these to knowledge_qa just because a KB exists.
- "today/yesterday/tomorrow" + a topic → almost always real-time → tool_use + web_search.
- Only use knowledge_qa when the user asks about content that plausibly lives in uploaded documents (法律条文, 公司文档, 产品手册).
- For tool_use populate tool_name and tool_args. tool_name MUST be one of the available tools listed above — never invent a tool name.
- When the user explicitly names a tool (e.g. "用 echo 工具..."), use exactly that tool with its correct args.
"""
)

# ── ReAct Reasoning ───────────────────────────────────────────────────────────
# （阶段 0，2026-09-02：TASK_PLANNING 与 REACT_REASONING 随 single 固定管线退役删除；
#  意图分流/任务规划由 dynamic / deepagents 路径的模型函数调用承接。）

# ── Answer Generation (with context) ─────────────────────────────────────────

ANSWER_WITH_CONTEXT = PromptTemplate(
    """You are a knowledgeable assistant. Use the provided context to answer the question accurately.

Question: {query}

Retrieved context:
{context}

Tool result (if any): {tool_result}

Instructions:
- Base your answer primarily on the retrieved context.
- If the context does not contain enough information, say so honestly.
- Be concise, factual, and well-structured.
- Do NOT fabricate information not present in the context.

Answer:
"""
)

# ── Answer Generation (no retrieval context) ─────────────────────────────────

ANSWER_NO_CONTEXT = PromptTemplate(
    """You are a helpful assistant. Answer the following question using your general knowledge.

Question: {query}

Tool result (if any): {tool_result}

Be concise and accurate. If you are uncertain, say so.

Answer:
"""
)

# ── Answer Validation ─────────────────────────────────────────────────────────
# （阶段 0，2026-09-02：ANSWER_VALIDATION 与 FALLBACK_ANSWER 随 single 固定管线
#  退役删除；质量兜底由 dynamic/deep 路径自身的 fallback 与 degraded 逻辑承接。）

# ── Empty Retrieval Recovery ──────────────────────────────────────────────────

EMPTY_RETRIEVAL_RECOVERY = PromptTemplate(
    """No relevant documents were found in the knowledge base for your question.

Question: {query}

I'll answer based on general knowledge, but this may be less accurate.

Answer:
"""
)

# ── 增强检索：结构化上下文答案生成 ──────────────────────────────────────────

ANSWER_WITH_ENHANCED_CONTEXT = PromptTemplate(
    """你是一个知识专家助手。以下是经过多路径检索和图谱分析后的结构化知识，请基于这些信息回答问题。

问题: {query}

{context}

工具结果（如有）: {tool_result}

指令:
- 优先使用知识块中提供的信息，知识块之间用「## 知识块 N」分隔
- 每个知识块内包含相关实体、关系链和支撑原文
- 综合多个知识块的信息回答，注意跨块关联
- 如果信息不足或矛盾，诚实说明
- 回答要结构化且有深度，但不要冗长
- 不要编造知识块中没有的信息

回答:
"""
)

# ── 增强检索：无上下文回退 ───────────────────────────────────────────────────

ANSWER_WITH_ENHANCED_NO_CONTEXT = PromptTemplate(
    """你是一个有帮助的助手。以下问题在知识库中未找到相关信息，请基于你的通用知识回答。

问题: {query}

工具结果（如有）: {tool_result}

请简洁准确，不确定的地方诚实说明。

回答:
"""
)
