"""DeepAgents 步骤透传 — 请求级观察者（S3，2026-08-21）。

问题：主 Agent 通过 ``task`` 工具委派 SubAgent 时，子 Agent 的执行过程
（推理/工具调用/工具返回）此前是黑盒——``_run_deep`` 的 on_step/on_artifact
只覆盖主 Agent 的 stream，前端 SSE 只能看到 "调用 task(...)" 与一条
"工具返回"，看不到子 Agent 内部。

方案：task 工具与 SubAgent 同步运行在主 Agent 的 executor 线程内，用两层
ContextVar 把 ``_run_deep`` 的 on_step/on_artifact 回调透传下去：

- ``use_task_observers``：``_run_deep`` 设置（包住主 Agent 执行），
  ``task`` 工具读取——拿到当前请求的 SSE 回调；
- ``use_subagent_observers``：``task`` 工具在调用 ``run_subagent`` 前设置，
  ``run_subagent`` 的子 Agent stream 循环读取——发出带子 Agent 名前缀的步骤。

两层隔离避免子 Agent 步骤与主 Agent 步骤混淆（子 Agent 步骤以
``{subagent_name}/step`` 形式出现）。线程模型：同线程同步调用链，ContextVar
天然可见；跨线程需重新 ``with`` 设置（``_run_deep`` 在 executor 线程内设置，
无需调用方配合）。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional, Tuple

# on_step(step: str, detail: str) / on_artifact(kind, stage, title, content)
Observers = Tuple[
    Optional[Callable[[str, str], None]],
    Optional[Callable[[str, str, str, str], None]],
]

_task_observers: ContextVar[Optional[Observers]] = ContextVar(
    "deep_task_observers", default=None
)
_subagent_observers: ContextVar[Optional[Observers]] = ContextVar(
    "deep_subagent_observers", default=None
)


def get_task_observers() -> Optional[Observers]:
    """当前请求的主 Agent SSE 回调（None = 非请求上下文，如测试/脚本）。"""
    return _task_observers.get()


def get_subagent_observers() -> Optional[Observers]:
    """当前子 Agent 执行的 SSE 回调（None = 不透传，保持原有行为）。"""
    return _subagent_observers.get()


@contextmanager
def use_task_observers(
    on_step: Optional[Callable[[str, str], None]] = None,
    on_artifact: Optional[Callable[[str, str, str, str], None]] = None,
) -> Iterator[None]:
    """在作用域内设置 task 委派观察者（主 Agent 执行期间）；退出恢复。"""
    token = _task_observers.set((on_step, on_artifact))
    try:
        yield
    finally:
        _task_observers.reset(token)


@contextmanager
def use_subagent_observers(
    on_step: Optional[Callable[[str, str], None]] = None,
    on_artifact: Optional[Callable[[str, str, str, str], None]] = None,
) -> Iterator[None]:
    """在作用域内设置子 Agent 执行观察者（run_subagent 期间）；退出恢复。"""
    token = _subagent_observers.set((on_step, on_artifact))
    try:
        yield
    finally:
        _subagent_observers.reset(token)
