"""Legal Worker — 法律条文查询与解读。"""

from __future__ import annotations

from app.agents.workers.base import BaseWorker, TaskBrief, WorkerReport
from app.core.logger import get_logger

logger = get_logger(__name__)


class LegalWorker(BaseWorker):
    """法律专家 Worker：法条查询、合规分析，白名单 [web_search]。"""

    name = "legal"
    persona = (
        "你是一个法律专家，擅长中国法律法规查询与解读。"
        "你的回答必须：\n"
        "1. 引用具体法律条文（法律名称+条款号）\n"
        "2. 给出条文原文或准确转述\n"
        "3. 解释条文含义及适用场景\n"
        "4. 如有争议或注意事项，明确提示\n"
        "回答格式：先结论，后条文依据，再解释。"
    )
    tool_names = ["web_search"]

    def run(self, brief: TaskBrief) -> WorkerReport:
        steps = [f"legal_worker 接收任务: {brief.goal[:80]}"]
        try:
            # 黑板上下文（M2）：引用前序任务产出
            board_ctx = self.board_context(exclude_task=brief.task_id)
            board_section = f"\n\n引用任务的产出：\n{board_ctx}" if board_ctx else ""

            messages = [
                {"role": "system", "content": self.persona},
                {
                    "role": "user",
                    "content": (
                        f"任务目标：{brief.goal}\n\n"
                        f"背景信息：{brief.context or '无'}\n\n"
                        f"约束条件：{'; '.join(brief.constraints) or '无'}"
                        f"{board_section}\n\n"
                        f"请完成上述法律相关任务。"
                    ),
                },
            ]

            answer = self._chat(messages, temperature=0.2, max_tokens=8192)
            steps.append(f"LLM 生成完成，回答长度 {len(answer)} 字符")

            return WorkerReport(
                task_id=brief.task_id,
                worker_name=self.name,
                status="done",
                summary=answer[:500],
                detail=answer,
                artifacts={"legal_query": brief.goal},
                steps=steps,
            )
        except Exception as exc:
            logger.error("[legal_worker] error: %s", exc)
            return WorkerReport(
                task_id=brief.task_id,
                worker_name=self.name,
                status="error",
                error=str(exc),
                steps=steps + [f"ERROR: {exc}"],
            )
