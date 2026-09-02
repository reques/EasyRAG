"""DeepAgents 结构化黑板（阶段 3）— spawn_tasks DAG 的任务产出物共享层。

与旧版 ``app/agents/blackboard.py``（orchestrator 时代，仅 500 字摘要）的区别：
  - 结构化 Artifact：``{key, producer, summary, data, tags, version}``，
    摘要 + 全量两级——调度注入默认用摘要，按需可取全量 ``data``；
  - 订阅由 ``spawn_tasks`` 的 ``depends_on`` 派生：任务执行前注入依赖
    artifact 摘要；
  - 写通知：post 时经统一事件流发出 ``blackboard/post`` 事件（供前端实时
    展示委派树/黑板状态）。

生命周期 = 一次 spawn_tasks 调用（planner 每次新建实例），无跨请求共享状态。
全方法 threading.Lock 保护：同一层级并发执行的子任务线程安全读写。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

# 注入任务描述时的单条摘要上限（保持 prompt 紧凑，全量仍可从 data 取）
_SUMMARY_LIMIT = 500


@dataclass(frozen=True)
class Artifact:
    """黑板上的结构化产出物（摘要 + 全量两级）。"""

    key: str
    producer: str
    summary: str
    data: Any = None
    tags: Tuple[str, ...] = ()
    version: int = 1


class Blackboard:
    """一次 spawn_tasks 调用内的共享黑板。"""

    def __init__(self) -> None:
        self._artifacts: Dict[str, Artifact] = {}
        self._lock = threading.Lock()

    def post(
        self,
        key: str,
        producer: str,
        summary: str,
        data: Any = None,
        tags: Tuple[str, ...] = (),
    ) -> Artifact:
        """发布产出物（同 key 覆盖写，自动递增 version），并发写通知事件。"""
        with self._lock:
            prev = self._artifacts.get(key)
            version = (prev.version + 1) if prev else 1
            art = Artifact(
                key=key,
                producer=producer,
                summary=str(summary)[:_SUMMARY_LIMIT],
                data=data,
                tags=tuple(tags),
                version=version,
            )
            self._artifacts[key] = art
        # 函数内导入：events → 无 trace 上下文时 no-op
        from app.agents.events import emit

        emit(
            "blackboard", "post", f"产出物 {key}",
            f"producer={producer} version={version} {art.summary[:120]}",
            key=key, producer=producer, version=version, tags=list(art.tags),
        )
        logger.debug("[blackboard] posted %s v%d by %s", key, version, producer)
        return art

    def get(self, key: str) -> Optional[Artifact]:
        with self._lock:
            return self._artifacts.get(key)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._artifacts.keys())

    def render_for_injection(self, keys: List[str]) -> str:
        """把指定依赖的摘要渲染为注入文本（依赖任务产出 → 后续任务描述）。

        缺失的 key 静默跳过（依赖失败时调度层已处理跳过，此处防御）。"""
        with self._lock:
            arts = [self._artifacts[k] for k in keys if k in self._artifacts]
        if not arts:
            return ""
        lines = [f"- [{a.producer}/{a.key}] {a.summary}" for a in arts]
        return "\n".join(lines)
