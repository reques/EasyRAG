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


# ── Intent Recognition ────────────────────────────────────────────────────────

INTENT_RECOGNITION = PromptTemplate(
    """You are an intent classifier for an AI assistant. Analyse the user query and return a JSON object.

User query: {query}

Return ONLY valid JSON in this exact format:
{{
  "intent": "<one of: knowledge_qa | tool_use | complex_task | chitchat>",
  "confidence": <float 0.0-1.0>,
  "requires_retrieval": <true|false>,
  "requires_tool": <true|false>,
  "tool_name": "<calculator|datetime_tool|text_tool|null>",
  "tool_args": {{}},
  "reasoning": "<one sentence why>"
}}

Intent definitions:
- knowledge_qa  : user asks a factual question answerable from a knowledge base
- tool_use      : user wants a calculation, time lookup, or text processing
- complex_task  : multi-step reasoning combining retrieval and/or tools
- chitchat      : casual greeting or off-topic

For tool_use, populate tool_name and tool_args appropriately.
Example tool_args for calculator: {{"expression": "12 * 34"}}
Example tool_args for datetime_tool: {{"format": "%Y-%m-%d"}}
Example tool_args for text_tool: {{"operation": "word_count", "text": "..."}}
"""
)

# ── Task Planning ─────────────────────────────────────────────────────────────

TASK_PLANNING = PromptTemplate(
    """You are a task planner. Decompose the user request into an ordered list of sub-tasks.

User request: {query}
Detected intent: {intent}

Return ONLY valid JSON:
{{
  "sub_tasks": ["<step 1>", "<step 2>", ...],
  "needs_retrieval": <true|false>,
  "needs_tool": <true|false>
}}

Keep each sub-task concise (one action). Maximum 5 sub-tasks.
"""
)

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

ANSWER_VALIDATION = PromptTemplate(
    """You are a quality checker. Evaluate whether the draft answer adequately addresses the question.

Original question: {query}
Draft answer: {draft_answer}

Return ONLY valid JSON:
{{
  "passed": <true|false>,
  "score": <int 1-10>,
  "feedback": "<one sentence feedback or empty string>"
}}

Criteria:
- Does the answer address the question?
- Is the answer factual and not self-contradictory?
- Is the length appropriate (not too short, not unnecessarily verbose)?
"""
)

# ── Fallback Answer ───────────────────────────────────────────────────────────

FALLBACK_ANSWER = PromptTemplate(
    """I encountered a problem while processing your request.

Your question: {query}
Error: {error}

I cannot provide a complete answer at this time. Please try again or rephrase your question.
"""
)

# ── Empty Retrieval Recovery ──────────────────────────────────────────────────

EMPTY_RETRIEVAL_RECOVERY = PromptTemplate(
    """No relevant documents were found in the knowledge base for your question.

Question: {query}

I'll answer based on general knowledge, but this may be less accurate.

Answer:
"""
)
