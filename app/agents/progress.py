"""将 Agent 执行事件投影为面向用户的简短进度日志（工作日志）。

投影器是刻意的确定性实现：不拷贝模型推理、工具参数、结果正文、主机名或
异常细节，只把真实的生命周期边界翻译成紧凑的进度摘要，适合 SSE 流式输出。
覆盖 单 Agent / 多智能体 / DeepAgents 三条路径；文本为高层概括而非思维链。
"""

from __future__ import annotations

import re
from typing import Any, Optional

_PRIVATE_STEP_NAMES = {"agent_reasoning", "reason", "react", "thought", "tool_call"}
_PROBLEM_MARKERS = ("失败", "错误", "异常", "error", "failed", "timeout", "refused")


def _base_step(step: str) -> str:
    name = str(step or "").rsplit("/", 1)[-1]
    return name[:-5] if name.endswith("_done") else name


def _subagent_label(step: str) -> str:
    if "/" not in step:
        return ""
    name = step.rsplit("/", 1)[0]
    known = {
        "research-agent": "研究助理",
        "coding-agent": "分析助理",
    }
    return known.get(name, "专项助理")


def _has_problem(detail: str) -> bool:
    text = str(detail or "").lower()
    return any(marker in text for marker in _PROBLEM_MARKERS)


def _tool_name(detail: str) -> str:
    match = re.search(r"调用\s*([a-zA-Z0-9_-]+)", str(detail or ""))
    return (match.group(1) if match else "").lower()


class ProgressProjector:
    """有状态的事件到进度映射器，带连续重复抑制。"""

    def __init__(self) -> None:
        self._sequence = 0
        self._last_signature: Optional[tuple[str, str, str]] = None

    def _event(self, phase: str, text: str, status: str = "running") -> Optional[dict[str, Any]]:
        signature = (phase, text, status)
        if signature == self._last_signature:
            return None
        self._last_signature = signature
        self._sequence += 1
        return {
            "id": f"progress-{self._sequence}",
            "sequence": self._sequence,
            "phase": phase,
            "status": status,
            "text": text,
        }

    def feed(self, step: str, detail: str = "") -> Optional[dict[str, Any]]:
        """把一条执行步骤翻译成安全的进度摘要；无价值时返回 None。"""

        step = str(step or "")
        detail = str(detail or "")
        base = _base_step(step)
        is_done = step.rsplit("/", 1)[-1].endswith("_done")
        subagent = _subagent_label(step)

        if base in _PRIVATE_STEP_NAMES:
            return None

        if _has_problem(detail) or base in ("fallback", "degenerate"):
            return self._event(
                "warning",
                "当前步骤遇到问题，正在切换到可用的替代路径，下一步会继续核对已有材料。",
                "warning",
            )

        if base == "understand":
            if is_done:
                return self._event(
                    "planning",
                    "问题已理解并还原完整意图，下一步判断问题类型。",
                    "completed",
                )
            return self._event("planning", "正在理解问题并结合上下文还原完整意图。")

        if base == "intent":
            if is_done:
                return self._event(
                    "planning",
                    "已确定问题类型，选择最合适的处理路径。",
                    "completed",
                )
            return self._event("planning", "正在判断问题类型，选择处理路径。")

        if base == "retrieve":
            count_match = re.search(r"命中\s*(\d+)\s*条", detail)
            if count_match:
                return self._event(
                    "retrieval",
                    f"已完成知识库检索，找到 {count_match.group(1)} 条相关内容，下一步提取关键证据。",
                    "completed",
                )
            if is_done:
                return self._event(
                    "retrieval",
                    "检索已完成，下一步扩大范围或直接进入回答。",
                    "completed",
                )
            return self._event("retrieval", "正在检索知识库中的相关资料。")

        if base == "tool":
            if is_done:
                return self._event(
                    "analysis",
                    "工具执行完成，正在检查结果质量并提炼可用信息。",
                    "completed",
                )
            tool = _tool_name(detail)
            actor = f"{subagent}正在" if subagent else "正在"
            if tool in {"web_search", "search", "tavily_search", "duckduckgo"}:
                return self._event("search", f"{actor}搜索外部资料，完成后会比较不同来源的一致性。")
            if tool == "kb_search":
                return self._event("retrieval", f"{actor}补充检索知识库，查找尚未覆盖的证据。")
            if tool == "task":
                return self._event("delegation", "任务路径已确定，正在派发子任务并行收集信息。")
            return self._event("action", f"{actor}调用辅助工具补充必要信息。")

        if base == "decompose":
            if is_done:
                return self._event(
                    "delegation",
                    "任务拆解完成，下一步派发子任务执行。",
                    "completed",
                )
            return self._event("delegation", "正在把复杂任务拆解为若干子任务。")

        if base == "dispatch":
            if is_done:
                return self._event(
                    "delegation",
                    "子任务派发完成，正在并行收集各方向结果。",
                    "completed",
                )
            return self._event("delegation", "正在派发子任务并行执行。")

        if base == "task_started":
            return self._event(
                "delegation",
                f"{subagent or '子任务'}开始执行，正在推进当前环节。",
            )

        if base == "synthesize":
            if is_done:
                return self._event(
                    "complete",
                    "各方向结果已汇总，完整回答准备就绪。",
                    "completed",
                )
            return self._event("synthesis", "各子任务已完成，正在汇总整合最终回答。")

        if base == "generate":
            if is_done:
                return self._event(
                    "complete",
                    "回答生成完成，结果已就绪。",
                    "completed",
                )
            if subagent:
                return self._event(
                    "analysis",
                    f"{subagent}已完成一轮资料收集，正在整理阶段性发现。",
                    "completed",
                )
            return self._event(
                "synthesis",
                "资料检索和分析已基本完成，正在撰写最终回答。",
            )

        return None
