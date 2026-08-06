"""共享黑板（Blackboard）— 一次请求内的共享 KV + 消息列表。

Worker 可 post(artifact) / read(tag)，Orchestrator 管理生命周期。
全方法 threading.Lock 保护，parallel 模式下多 Worker 线程并发读写安全。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Artifact:
    """黑板上的产出物。"""

    key: str
    task_id: str
    producer: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BoardMessage:
    """黑板操作日志。"""

    actor: str
    action: str
    content: str
    timestamp: float = field(default_factory=time.time)


class Blackboard:
    """一次请求内的共享黑板，Worker 间传递 artifact。"""

    def __init__(self):
        self._artifacts: Dict[str, Artifact] = {}
        self._messages: List[BoardMessage] = []
        self._lock = threading.Lock()

    # ── Artifact 操作 ────────────────────────────────────────────────────────
    def post_artifact(
        self,
        key: str,
        task_id: str,
        producer: str,
        summary: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """发布 artifact（同 key 覆盖写）。"""
        with self._lock:
            self._artifacts[key] = Artifact(
                key=key,
                task_id=task_id,
                producer=producer,
                summary=summary,
                data=data or {},
                tags=tags or [],
            )
            self._log(producer, "post", f"{key} ({len(summary)} chars)")

    def read_artifact(self, key: str, reader: str = "") -> Optional[Artifact]:
        """按 key 读取 artifact。"""
        with self._lock:
            art = self._artifacts.get(key)
            if art and reader:
                self._log(reader, "read", key)
            return art

    def find_by_tag(self, tag: str) -> List[Artifact]:
        """按 tag 查找 artifact。"""
        with self._lock:
            return [a for a in self._artifacts.values() if tag in a.tags]

    def find_by_task(self, task_id: str) -> List[Artifact]:
        """按 task_id 查找 artifact。"""
        with self._lock:
            return [a for a in self._artifacts.values() if a.task_id == task_id]

    def all_artifacts(self) -> List[Dict[str, Any]]:
        """返回所有 artifact 的摘要列表（供响应透出）。"""
        with self._lock:
            return [
                {
                    "key": a.key,
                    "task_id": a.task_id,
                    "producer": a.producer,
                    "summary": a.summary[:200],
                    "tags": a.tags,
                }
                for a in self._artifacts.values()
            ]

    # ── 消息/日志 ────────────────────────────────────────────────────────────
    def note(self, actor: str, content: str) -> None:
        """发布自由格式备注。"""
        with self._lock:
            self._log(actor, "note", content)

    def render_log(self) -> List[str]:
        """返回操作日志列表。"""
        with self._lock:
            return [f"[board] {m.actor} {m.action}: {m.content}" for m in self._messages]

    def _log(self, actor: str, action: str, content: str) -> None:
        self._messages.append(BoardMessage(actor=actor, action=action, content=content))

    # ── Prompt 渲染 ──────────────────────────────────────────────────────────
    def render_for_prompt(self, exclude_task: str = "", max_chars: int = 300) -> str:
        """渲染 artifact 列表供 prompt 注入，排除自身任务防自引用。"""
        with self._lock:
            arts = [
                a for a in self._artifacts.values() if a.task_id != exclude_task
            ]
            if not arts:
                return ""
            lines = []
            for a in arts:
                lines.append(
                    f"- [{a.producer}/{a.task_id}] {a.summary[:max_chars]}"
                )
            return "\n".join(lines)
