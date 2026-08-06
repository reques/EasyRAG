"""Code Worker — 代码生成与解释。"""

from __future__ import annotations

from app.agents.workers.base import BaseWorker, TaskBrief, WorkerReport
from app.core.logger import get_logger

logger = get_logger(__name__)


class CodeWorker(BaseWorker):
    """代码专家 Worker：生成、解释、调试代码，白名单 [calculator, text_tool]。"""

    name = "code"
    persona = (
        "你是一个资深软件工程师，擅长编写清晰、可运行的代码。"
        "你的回答必须：\n"
        "1. 提供完整可运行的代码块（用 ```python 包裹）\n"
        "2. 代码注释说明关键逻辑\n"
        "3. 如有依赖或环境要求，明确说明\n"
        "4. 给出使用示例或测试用例\n"
        "代码风格：遵循 PEP 8，变量命名语义化，函数有类型注解。"
    )
    tool_names = ["calculator", "text_tool"]

    def run(self, brief: TaskBrief) -> WorkerReport:
        steps = [f"code_worker 接收任务: {brief.goal[:80]}"]
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
                        f"请完成上述编程任务。"
                    ),
                },
            ]

            answer = self._chat(messages, temperature=0.3, max_tokens=8192)
            steps.append(f"LLM 生成完成，回答长度 {len(answer)} 字符")

            # 提取代码块
            code_snippets = self._extract_code_snippets(answer)
            steps.append(f"提取到 {len(code_snippets)} 个代码块")

            return WorkerReport(
                task_id=brief.task_id,
                worker_name=self.name,
                status="done",
                summary=answer[:500],
                detail=answer,
                artifacts={
                    "code_snippets": code_snippets,
                    "snippet_count": len(code_snippets),
                },
                steps=steps,
            )
        except Exception as exc:
            logger.error("[code_worker] error: %s", exc)
            return WorkerReport(
                task_id=brief.task_id,
                worker_name=self.name,
                status="error",
                error=str(exc),
                steps=steps + [f"ERROR: {exc}"],
            )
