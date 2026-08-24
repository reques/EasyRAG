"""请求级知识库授权范围（Request-local authorised KB scope）。

与 ``app/skills/context.py`` 同一模式：用 ContextVar 在请求内携带"当前用户
已授权的知识库 UUID 列表"，供需要访问知识库的注册表工具（如 ``kb_search``）
读取 —— 工具函数签名统一为 ``fn(**kwargs) -> str``，没有渠道接收请求上下文，
必须从 contextvar 取。

线程模型：contextvars 对同一线程的同步调用链（含 task 工具内联执行的
SubAgent）全部可见；跨线程（executor）需在进入时 ``with`` 重新设置 ——
``_run_deep`` 在 executor 线程内自行设置，无需调用方配合。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, List, Optional, Sequence

_authorised_kb_ids: ContextVar[Optional[List[str]]] = ContextVar(
    "authorised_kb_ids", default=None
)


def get_authorised_kb_ids() -> Optional[List[str]]:
    """当前请求已授权的知识库 UUID 列表；None 表示未设置（工具应拒绝检索）。"""
    return _authorised_kb_ids.get()


@contextmanager
def use_authorised_kb_ids(
    ids: Optional[Sequence[str]],
) -> Iterator[Optional[List[str]]]:
    """在作用域内设置请求级授权知识库；退出时恢复原值。

    ids 为空/None 时同样设置（显式"无授权"），避免继承外层遗留值。
    """
    token = _authorised_kb_ids.set(list(ids) if ids else None)
    try:
        yield _authorised_kb_ids.get()
    finally:
        _authorised_kb_ids.reset(token)
