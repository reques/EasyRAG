"""Worker 基类与数据契约。

借鉴 subagent-driven-development 的「任务简报 brief」思想：
每个子任务一份结构化 TaskBrief（goal/context/constraints/worker_hint），
Worker 执行后返回结构化 WorkerReport（status/summary/detail/artifacts/steps/error）。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TaskBrief:
    """Orchestrator 拆解出的子任务简报，自包含派发单元。"""

    task_id: str
    goal: str
    context: str = ""
    constraints: List[str] = field(default_factory=list)
    worker_hint: str = ""  # rag / legal / code
    knowledge_base_ids: List[str] = field(default_factory=list)
    knowledge_catalog: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkerReport:
    """Worker 执行后的结构化报告。"""

    task_id: str
    worker_name: str
    status: str  # done | done_with_concerns | blocked | error
    summary: str = ""
    detail: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.status in ("done", "done_with_concerns")


class BaseWorker(ABC):
    """专家 Worker 基类。

    属性：
        name: Worker 标识名（用于注册表路由）
        persona: system prompt 人格设定
        tool_names: 允许调用的工具白名单（空列表 = 无工具）
    """

    name: str = ""
    persona: str = ""
    tool_names: List[str] = []

    def __init__(self):
        self._llm = None
        self.blackboard = None  # M2: orchestrator 注入，供 run_with_board 使用
        self.tool_callback = None  # 工具调用钩子 fn(tool_name, args)，orchestrator 注入

    # ── LLM client（lazy，可注入 mock）─────────────────────────────────────
    @property
    def llm(self):
        if self._llm is None:
            from app.llm.client import get_llm_client

            self._llm = get_llm_client()
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    # ── 工具白名单检查 ──────────────────────────────────────────────────────
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """返回白名单内工具的 LLM schema（供 ReAct prompt 使用）。"""
        if not self.tool_names:
            return []
        from app.tools.registry import get_tool_registry

        registry = get_tool_registry()
        return [
            t.to_llm_schema()
            for t in registry.list_tools()
            if t.name in self.tool_names
        ]

    def invoke_tool(self, name: str, **kwargs) -> Any:
        """调用白名单内工具，越权抛 PermissionError。"""
        if name not in self.tool_names:
            raise PermissionError(
                f"Worker '{self.name}' 无权调用工具 '{name}'，白名单: {self.tool_names}"
            )
        if self.tool_callback:
            try:
                self.tool_callback(name, kwargs)
            except Exception:
                pass
        from app.tools.registry import get_tool_registry

        registry = get_tool_registry()
        return registry.invoke(name, **kwargs)

    # ── 黑板消费（M2）───────────────────────────────────────────────────────
    def board_context(self, exclude_task: str = "", max_chars: int = 300) -> str:
        """读取黑板 artifact 列表（排除自身任务），供 prompt 注入。"""
        if self.blackboard is None:
            return ""
        return self.blackboard.render_for_prompt(
            exclude_task=exclude_task, max_chars=max_chars
        )

    # ── 核心执行接口 ────────────────────────────────────────────────────────
    @abstractmethod
    def run(self, brief: TaskBrief) -> WorkerReport:
        """执行子任务，返回结构化报告。子类必须实现。"""

    def run_with_board(self, brief: TaskBrief) -> WorkerReport:
        """包装 run()：成功时自动把产出 post 到黑板（M2）。"""
        report = self.run(brief)
        if report.ok() and self.blackboard is not None:
            self.blackboard.post_artifact(
                key=f"{brief.task_id}:{self.name}",
                task_id=brief.task_id,
                producer=self.name,
                summary=report.summary[:500],
                data=report.artifacts,
                tags=[self.name, "worker_output"],
            )
        return report

    # ── 工具方法 ────────────────────────────────────────────────────────────
    def _extract_code_snippets(self, text: str) -> List[str]:
        """从 LLM 输出中提取 ```code``` 块。"""
        pattern = r"```(?:python|py|bash|sh|javascript|js|java|cpp|c|go|rust|sql)?\n(.*?)```"
        return re.findall(pattern, text, re.DOTALL)

    def _chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """同步 LLM 调用（Worker 内统一走 chat_sync）。"""
        return self.llm.chat_sync(messages, **kwargs)

    def _chat_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """同步 JSON 模式 LLM 调用（自动剥 markdown fence）。"""
        raw = self.llm.chat_sync(messages, **kwargs)
        # 剥 markdown fence
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # 去掉首行 ```json 和尾行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
