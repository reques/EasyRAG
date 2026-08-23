"""RAG Worker — 复用现有向量检索，回答知识库相关问题。"""

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.workers.base import BaseWorker, TaskBrief, WorkerReport
from app.core.logger import get_logger
from app.services.knowledge_catalog import format_knowledge_catalog

logger = get_logger(__name__)


class RagWorker(BaseWorker):
    """知识库问答 Worker：检索 + 生成，白名单 [web_search]。"""

    name = "rag"
    persona = (
        "你是一个知识库问答专家。你的任务是基于提供的上下文回答用户问题。"
        "如果上下文不足以回答，明确说明并给出建议。"
        "回答要准确、简洁，引用上下文时标注来源编号。"
    )
    tool_names = ["web_search"]

    def __init__(self):
        super().__init__()
        self._retriever = None

    @property
    def retriever(self):
        """lazy 初始化 retriever，避免模块导入时连接 Milvus。"""
        if self._retriever is None:
            from app.rag.retriever import get_retriever

            self._retriever = get_retriever()
        return self._retriever

    def run(self, brief: TaskBrief) -> WorkerReport:
        steps = [f"rag_worker 接收任务: {brief.goal[:80]}"]
        try:
            # 1. 向量检索（lazy：mock 测试时可能无 Milvus，失败不阻塞）
            docs = []
            try:
                docs = self.retriever.retrieve(
                    brief.goal,
                    top_k=4,
                    knowledge_base_ids=brief.knowledge_base_ids,
                )
                steps.append(f"检索到 {len(docs)} 条相关文档")
            except Exception as exc:
                steps.append(f"检索失败（继续无上下文生成）: {exc}")

            # 2. 构建上下文
            context = "\n\n".join(
                f"[{i+1}] {d['content']}" for i, d in enumerate(docs)
            )
            if not context:
                context = "（知识库中未找到相关内容）"

            # 3. 组装消息
            messages = [
                {"role": "system", "content": self.persona},
                {
                    "role": "system",
                    "content": format_knowledge_catalog(brief.knowledge_catalog),
                },
            ]
            history_msg = self._history_context_message(brief)
            if history_msg:
                messages.append(history_msg)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"任务目标：{brief.goal}\n\n"
                        f"约束条件：{'; '.join(brief.constraints) or '无'}\n\n"
                        f"参考上下文：\n{context}\n\n"
                        f"请基于上下文完成上述任务目标。"
                    ),
                },
            )

            # 4. LLM 生成
            answer = self._chat(messages, temperature=0.3, max_tokens=8192)
            steps.append(f"LLM 生成完成，回答长度 {len(answer)} 字符")

            return WorkerReport(
                task_id=brief.task_id,
                worker_name=self.name,
                status="done",
                summary=answer[:500],
                detail=answer,
                artifacts={
                    "retrieved_count": len(docs),
                    "sources": [d.get("metadata", {}).get("source", "") for d in docs],
                },
                steps=steps,
            )
        except Exception as exc:
            logger.error("[rag_worker] error: %s", exc)
            return WorkerReport(
                task_id=brief.task_id,
                worker_name=self.name,
                status="error",
                error=str(exc),
                steps=steps + [f"ERROR: {exc}"],
            )
