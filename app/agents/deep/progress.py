"""Project DeepAgents execution events into safe, user-facing progress notes.

The projector is deliberately deterministic and does not copy model reasoning,
tool arguments, result bodies, host names, or exception details. It translates
real lifecycle boundaries into a compact work log suitable for SSE streaming.
"""

from __future__ import annotations

import re
from typing import Any, Optional


_PRIVATE_STEP_NAMES = {"agent_reasoning", "reason", "react", "thought"}
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


def _tool_name(detail: str) -> str:
    match = re.search(r"调用\s*([a-zA-Z0-9_-]+)", str(detail or ""))
    return (match.group(1) if match else "").lower()


def _has_problem(detail: str) -> bool:
    text = str(detail or "").lower()
    return any(marker in text for marker in _PROBLEM_MARKERS)


class DeepResearchProgressProjector:
    """Stateful step-to-progress mapper with consecutive duplicate suppression."""

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
        """Translate one execution step into a safe progress event, if useful."""

        step = str(step or "")
        detail = str(detail or "")
        base = _base_step(step)
        is_done = step.rsplit("/", 1)[-1].endswith("_done")
        subagent = _subagent_label(step)

        if base in _PRIVATE_STEP_NAMES:
            return None

        if _has_problem(detail) or base == "fallback":
            return self._event(
                "warning",
                "当前步骤遇到问题，正在切换到可用的替代路径。下一步会继续核对已有资料，避免中断研究。",
                "warning",
            )

        if base == "understand":
            return self._event(
                "planning",
                "正在拆解研究目标并确定资料范围。下一步会检索知识库和可用的外部来源。",
            )

        if base == "retrieve":
            count_match = re.search(r"命中\s*(\d+)\s*条", detail)
            if count_match:
                return self._event(
                    "retrieval",
                    f"已完成知识库检索，找到 {count_match.group(1)} 条相关内容。下一步会提取关键证据并检查覆盖范围。",
                    "completed",
                )
            if is_done:
                return self._event(
                    "retrieval",
                    "知识库检索已完成，暂未找到直接相关内容。下一步会扩大检索范围并尝试其他来源。",
                    "completed",
                )
            return self._event(
                "retrieval",
                "正在检索知识库中的相关资料。完成后会筛选可用于回答的证据。",
            )

        if base == "tool":
            if is_done:
                return self._event(
                    "analysis",
                    "刚完成一轮信息获取，正在检查结果质量并提炼可用发现。下一步会继续查证或进入汇总。",
                    "completed",
                )
            tool = _tool_name(detail)
            actor = f"{subagent}正在" if subagent else "正在"
            if tool in {"web_search", "search", "tavily_search", "duckduckgo"}:
                return self._event(
                    "search",
                    f"{actor}搜索外部资料，重点补充可核验来源。完成后会比较不同来源的一致性。",
                )
            if tool == "kb_search":
                return self._event(
                    "retrieval",
                    f"{actor}补充检索知识库，查找尚未覆盖的证据。下一步会合并相关片段。",
                )
            if tool == "task":
                return self._event(
                    "delegation",
                    "研究路线已经确定，正在分派专项任务并行收集证据。下一步会持续汇总各项发现。",
                )
            return self._event(
                "action",
                f"{actor}使用辅助工具补充必要信息。完成后会检查结果是否足以支持结论。",
            )

        if base == "generate" and subagent:
            return self._event(
                "analysis",
                f"{subagent}已完成一轮资料收集，正在整理阶段发现。下一步会把结果交给主研究流程汇总。",
                "completed",
            )

        if base == "generate":
            if is_done:
                return self._event(
                    "complete",
                    "研究与整合已经完成，完整结果已准备好。",
                    "completed",
                )
            return self._event(
                "synthesis",
                "资料检索和交叉分析已基本完成，正在整合关键发现并撰写最终回答。",
            )

        return None
