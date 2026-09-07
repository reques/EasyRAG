"""LangGraph checkpoint 生命周期与运行配置。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_checkpointer: Optional[Any] = None
_backend = ""
_connection: Optional[sqlite3.Connection] = None


class CheckpointResumeError(RuntimeError):
    """请求恢复，但对应会话没有可继续的 pending checkpoint。"""


def get_agent_checkpointer() -> Any:
    """返回进程级 saver；SQLite 不可用时降级为内存 saver。"""
    global _checkpointer, _backend, _connection
    if _checkpointer is not None:
        return _checkpointer

    with _lock:
        if _checkpointer is not None:
            return _checkpointer
        cfg = get_settings()
        configured_backend = getattr(cfg, "AGENT_CHECKPOINT_BACKEND", "sqlite")
        if configured_backend == "sqlite":
            try:
                from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
                from langgraph.checkpoint.sqlite import SqliteSaver

                path = Path(cfg.AGENT_CHECKPOINT_PATH).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                _connection = sqlite3.connect(str(path), check_same_thread=False)
                _checkpointer = SqliteSaver(
                    _connection,
                    serde=JsonPlusSerializer(pickle_fallback=False),
                )
                _checkpointer.setup()
                _backend = "sqlite"
                logger.info("[checkpoint] SQLite saver ready: %s", path)
                return _checkpointer
            except Exception as exc:
                logger.warning("[checkpoint] SQLite unavailable, using memory: %s", exc)

        from langgraph.checkpoint.memory import InMemorySaver

        _checkpointer = InMemorySaver()
        _backend = "memory"
        return _checkpointer


def checkpoint_run_config(
    thread_id: str,
    namespace: str,
    recursion_limit: int,
) -> Dict[str, Any]:
    """构造所有 Agent 共用的 checkpoint 配置。"""
    return {
        "configurable": {
            # 顶层 LangGraph 会把 checkpoint_ns 归一化为空；用带模式前缀的
            # thread_id 隔离 dynamic/deep，才能可靠地跨进程读取同一状态。
            "thread_id": f"{namespace}:{thread_id}",
        },
        "recursion_limit": recursion_limit,
    }


def _delete_checkpoint_key(checkpoint_thread_id: str) -> None:
    get_agent_checkpointer().delete_thread(checkpoint_thread_id)


def abandon_checkpoint_run(thread_id: str, namespace: str) -> None:
    """放弃已终止且不会恢复的单个 Agent 模式状态。"""
    try:
        _delete_checkpoint_key(f"{namespace}:{thread_id}")
    except Exception as exc:
        logger.warning("[checkpoint] abandon thread %s failed: %s", thread_id, exc)


def delete_checkpoint_thread(thread_id: str) -> None:
    """删除一个会话的历史 checkpoint；失败不阻塞聊天主链路。"""
    try:
        for namespace in ("dynamic", "deep"):
            _delete_checkpoint_key(f"{namespace}:{thread_id}")
    except Exception as exc:
        logger.warning("[checkpoint] delete thread %s failed: %s", thread_id, exc)


def begin_checkpoint_run(thread_id: str, namespace: str) -> bool:
    """开始一轮可恢复执行，返回该会话是否已有持久图状态。

    已有状态不会在这里清除：LangGraph 以增量消息继续同一 thread，进程重启后
    也能从 SQLite 恢复。调用方应在返回 True 时省略已经 checkpoint 的 DB 历史。
    """
    try:
        config = checkpoint_run_config(thread_id, namespace, recursion_limit=1)
        return get_agent_checkpointer().get_tuple(config) is not None
    except Exception as exc:
        logger.warning("[checkpoint] inspect thread %s failed: %s", thread_id, exc)
        return False


def checkpoint_turn_message_id(
    thread_id: str,
    query: str,
    history: Sequence[Mapping[str, Any]],
    input_message_id: Optional[str] = None,
) -> str:
    """为本轮用户输入生成可重试的稳定 ID，防止恢复时重复追加消息。"""
    if input_message_id:
        return f"turn:{input_message_id}"
    marker = [
        (str(item.get("role") or ""), str(item.get("content") or ""))
        for item in history[-4:]
    ]
    payload = json.dumps(
        [thread_id, query, len(history), marker],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "turn:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def prepare_checkpoint_input(
    agent: Any,
    config: Dict[str, Any],
    messages: Sequence[Any],
    *,
    resume: bool = False,
) -> tuple[Optional[dict[str, Any]], int, bool]:
    """准备增量图输入；恢复请求从 checkpoint 的待执行节点继续。

    返回 ``(graph_input, processed_message_count, resumed)``。正常新轮次只追加
    本轮消息；恢复时传 ``None``，这是 LangGraph 继续 pending task 的正式语义。
    """
    snapshot = None
    try:
        snapshot = agent.get_state(config)
    except Exception:
        if resume:
            raise CheckpointResumeError("无法读取工作记忆 checkpoint")
    existing_messages = []
    if snapshot is not None and isinstance(snapshot.values, dict):
        existing_messages = list(snapshot.values.get("messages") or [])
    if resume:
        if snapshot is not None and snapshot.next:
            return None, len(existing_messages), True
        # 进程可能在首个 checkpoint 前退出；复用已落库用户消息重新执行，仍然
        # 比创建重复消息更安全。已有但已完成的快照依靠稳定消息 ID 保持幂等。
    graph_input = {"messages": list(messages)}

    # add_messages 会按稳定 ID 原位替换事实/经历/知识库上下文；这些替换不会
    # 增加 state 消息数，processed 基线只能统计真正新增的消息。
    existing_ids = {
        getattr(message, "id", None)
        or (message.get("id") if isinstance(message, Mapping) else None)
        for message in existing_messages
    }
    added_count = 0
    for message in messages:
        message_id = getattr(message, "id", None)
        if message_id is None and isinstance(message, Mapping):
            message_id = message.get("id")
        if message_id is None or message_id not in existing_ids:
            added_count += 1
    return graph_input, len(existing_messages) + added_count, False


def checkpoint_backend() -> str:
    """返回实际启用的 backend，供诊断与响应元数据使用。"""
    get_agent_checkpointer()
    return _backend


def reset_checkpointer_for_tests() -> None:
    """关闭并清空 saver 单例，仅供测试隔离。"""
    global _checkpointer, _backend, _connection
    with _lock:
        if _connection is not None:
            _connection.close()
        _checkpointer = None
        _connection = None
        _backend = ""


__all__ = [
    "abandon_checkpoint_run",
    "CheckpointResumeError",
    "begin_checkpoint_run",
    "checkpoint_backend",
    "checkpoint_run_config",
    "checkpoint_turn_message_id",
    "delete_checkpoint_thread",
    "get_agent_checkpointer",
    "prepare_checkpoint_input",
]
