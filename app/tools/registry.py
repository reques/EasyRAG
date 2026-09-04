"""Tool registry – central place to register and invoke tools.

Each tool is registered as a ToolDefinition containing:
  - name        : unique string key
  - description : short human-readable description
  - fn          : callable that accepts **kwargs and returns str
  - arg_schema  : dict mapping arg name -> (type, description, required)
  - timeout_s   : execution budget in seconds (0 = no outer timeout; 2026-08-26 阶段 1)
  - max_retries : extra attempts on transient failures (default 0)
  - metadata    : capability metadata for tool discovery (阶段 2)

invoke() wraps execution with timeout + retry and emits structured events
into the request trace (``app/agents/events.py``; no-op without a trace).
"""
from __future__ import annotations

import contextvars
import inspect
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


def _args_digest(kwargs: Dict[str, Any]) -> str:
    """工具参数的事件摘要（JSON 截断，避免事件流膨胀/大载荷泄露）。"""
    try:
        text = json.dumps(kwargs, ensure_ascii=False, default=str)
    except Exception:
        text = str(kwargs)
    return text[:400]


def _accepts_progress_callback(fn: Callable[..., Any]) -> bool:
    """工具函数是否接受 ``progress_callback`` 参数（显式声明或 **kwargs）。"""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        return True
    return "progress_callback" in sig.parameters


def _build_progress_emitter(name: str, callback: Callable[..., Any]):
    """包装调用方进度回调：经统一事件流发出结构化进度事件后再转原回调。

    约定签名 ``callback(message, percent=None, **extra)``；回调/事件消费侧异常不传播。"""
    from app.agents.events import emit

    def _emit_progress(message: str = "", percent=None, **extra: Any) -> None:
        try:
            emit(
                "tool", "progress", f"{name}: {str(message)[:60]}", str(message)[:300],
                tool=name, percent=percent, **extra,
            )
        except Exception:
            pass
        try:
            callback(message, percent=percent, **extra)
        except Exception:
            pass

    return _emit_progress


def _skill_allows(tool: "ToolDefinition") -> bool:
    """Skill 门控裁决（第二层，ContextVar 侧）。

    2026-09-04 Skill 重构：判据从"静态白名单"改为"已激活 Skill 的工具并集
    + 公共工具"（渐进式披露）。第一层门控在 ``app/skills/middleware.py`` 的
    ``wrap_tool_call``；这一层覆盖 middleware 触达不到的路径 —— 子 Agent 的
    ThreadPoolExecutor 线程、graph 节点、MCP 桥接。两层判据同为
    ``SkillRuntimeContext.allows_tool``，不存在两套真相。

    ``public`` 直接从已持有的 ToolDefinition 取，避免注册表回查（否则
    list_all 遍历 N 个工具会产生 N 次带锁 get）。
    """
    from app.skills.runtime import get_active_skill_context

    return get_active_skill_context().allows_tool(
        tool.name, public=bool((tool.metadata or {}).get("public"))
    )


@dataclass
class ToolDefinition:
    name: str
    description: str
    fn: Callable[..., str]
    # {arg_name: (python_type_str, description, is_required)}
    arg_schema: Dict[str, Tuple[str, str, bool]] = field(default_factory=dict)
    # 工具可用性自检：返回 False 的工具不出现在 schema/react prompt，invoke 时拒绝
    check_fn: Optional[Callable[[], bool]] = None
    # 执行预算（秒）。0 = 不包外层超时（如 MCP 桥接工具自带 120s 超时）。
    timeout_s: float = 60.0
    # 瞬时失败重试次数（额外尝试次数，指数退避）；超时不重试（只会翻倍等待）
    max_retries: int = 0
    # 能力元数据（阶段 2 工具发现用）：{"scenarios": [...], "tags": [...], ...}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """工具是否可用（check_fn 通过）。None 表示总是可用。"""
        if self.check_fn is None:
            return True
        try:
            return bool(self.check_fn())
        except Exception:
            return False

    def to_llm_schema(self) -> Dict[str, Any]:
        """本工具的 OpenAI function-call 格式 schema。"""
        properties: Dict[str, Any] = {}
        required_args: List[str] = []
        for arg_name, (type_str, desc, is_req) in self.arg_schema.items():
            properties[arg_name] = {"type": type_str, "description": desc}
            if is_req:
                required_args.append(arg_name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_args,
                },
            },
        }


class ToolRegistry:
    """Registry that maps tool names to their definitions.

    Thread-safe: all read/write operations are guarded by an RLock so that
    MCP server stop (unregister, main thread) and LangGraph node traversal
    (list_all/to_react_prompt, worker thread) can run concurrently without
    raising ``RuntimeError: dictionary changed size during iteration``.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = threading.RLock()

    def register(self, tool: ToolDefinition) -> None:
        with self._lock:
            self._tools[tool.name] = tool
        logger.debug("Tool registered: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not present).

        Used by MCPManager.stop() to cleanly remove a server's tools without
        touching the private ``_tools`` dict directly.
        """
        with self._lock:
            self._tools.pop(name, None)
        logger.debug("Tool unregistered: %s", name)

    def get(self, name: str) -> ToolDefinition:
        with self._lock:
            if name not in self._tools:
                # 附可用清单（阶段 2）：让主 Agent/子 Agent 能自我修正工具名
                available = ", ".join(self.list_names()) or "(none)"
                raise ToolNotFoundError(
                    name, f"tool not registered. Available tools: {available}"
                )
            return self._tools[name]

    def list_names(self, available_only: bool = True) -> List[str]:
        """Return tool names. available_only=True 时只含 check_fn 通过的工具。"""
        with self._lock:
            snapshot = list(self._tools.values())
        if not available_only:
            candidates = snapshot
        else:
            candidates = [t for t in snapshot if t.is_available()]
        return [tool.name for tool in candidates if _skill_allows(tool)]

    def list_all(self, available_only: bool = True) -> List[ToolDefinition]:
        """Return all ToolDefinition objects (for schema/prompt building).
        available_only=True 时只含 check_fn 通过的工具。"""
        with self._lock:
            snapshot = list(self._tools.values())
        candidates = snapshot if not available_only else [
            tool for tool in snapshot if tool.is_available()
        ]
        return [tool for tool in candidates if _skill_allows(tool)]

    def invoke(self, name: str, **kwargs: Any) -> str:
        """Execute a registered tool by name.

        The tool lookup is performed under the registry lock, but the actual
        ``fn`` call runs *outside* the lock so a slow tool (e.g. web_search
        HTTP request) never blocks other registry readers.

        Execution is wrapped with the tool's ``timeout_s`` budget and
        ``max_retries`` retry policy (2026-08-26, 阶段 1):
          - timeout abandons the call (the worker thread keeps draining until
            the tool's own I/O timeout fires — threads cannot be killed);
          - timeouts are NOT retried (retrying only doubles the wait);
          - transient failures are retried with exponential backoff.

        Structured tool events (tool_start / tool_end / tool_error, with args
        digest and elapsed_ms) are emitted into the request trace; without a
        trace context this is a no-op.

        Progress callback (2026-08-26, 阶段 5): kwargs 可携带可选的
        ``progress_callback(message, percent=None, **extra)``；会包装后经统一事件流
        发出 ``tool/progress`` 事件并转发给原回调。仅当工具函数签名接受该参数时传入。
        参数摘要/日志不含回调本身。

        Args:
            name:   Tool name.
            **kwargs: Arguments forwarded to the tool function.

        Returns:
            String output of the tool.

        Raises:
            ToolNotFoundError:   Tool not registered.
            ToolExecutionError:  Tool unavailable (check_fn failed) or raised during execution.
        """
        tool = self.get(name)  # acquires+releases lock internally
        if not _skill_allows(tool):
            raise ToolExecutionError(
                f"Tool '{name}' is not allowed by the selected Skills. "
                f"Available tools: {', '.join(self.list_names()) or '(none)'}"
            )
        if not tool.is_available():
            raise ToolExecutionError(
                f"Tool '{name}' is not available (check_fn failed — missing config or dependency)"
            )
        # 阶段 5：进度回调提取（不进入参数摘要/日志/工具入参转发判断）
        progress_cb = kwargs.pop("progress_callback", None)
        if callable(progress_cb):
            progress_cb = _build_progress_emitter(name, progress_cb)
            if not _accepts_progress_callback(tool.fn):
                progress_cb = None
        else:
            progress_cb = None
        logger.info("Invoking tool '%s' with args: %s", name, kwargs)
        # 函数内导入：避免模块级循环依赖（事件流/遥测为可选旁路）
        from app.agents.events import emit
        from app.observability.tracing import trace_span

        started = time.perf_counter()
        emit("tool", "tool_start", f"调用 {name}", _args_digest(kwargs), tool=name)

        attempts = max(1, int(tool.max_retries) + 1)
        last_exc: Optional[BaseException] = None
        with trace_span(f"tool.invoke.{name}", tool=name, attempts=attempts):
            for attempt in range(attempts):
                try:
                    result = self._execute(tool, kwargs, progress_cb)
                    emit(
                        "tool", "tool_end", f"{name} 完成", "",
                        tool=name,
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                    )
                    logger.info("Tool '%s' succeeded.", name)
                    return result
                except (ToolNotFoundError, ToolTimeoutError) as exc:
                    # 未注册 / 超时不可重试：重试只会翻倍等待
                    last_exc = exc
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < attempts - 1:
                        delay = 0.5 * (2 ** attempt)
                        logger.warning(
                            "Tool '%s' failed (attempt %d/%d): %s — retrying in %.1fs",
                            name, attempt + 1, attempts, exc, delay,
                        )
                        time.sleep(delay)
                        continue
                    break

        emit(
            "tool", "tool_error", f"{name} 失败", str(last_exc)[:200],
            tool=name,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        if isinstance(last_exc, (ToolExecutionError, ToolNotFoundError)):
            raise last_exc
        raise ToolExecutionError(
            f"Tool '{name}' raised an unexpected error: {last_exc}"
        ) from last_exc

    def _execute(
        self,
        tool: ToolDefinition,
        kwargs: Dict[str, Any],
        progress_cb: Optional[Callable[..., Any]] = None,
    ) -> str:
        """Run the tool fn, bounded by ``timeout_s`` (阶段 1).

        contextvars are explicitly propagated into the worker thread via
        ``contextvars.copy_context()`` — ThreadPoolExecutor does NOT copy
        context automatically, and tools like kb_search rely on request-local
        ContextVars (KB authorization, skill whitelist, event trace).

        timeout_s <= 0 executes directly in the current thread (MCP bridge
        tools carry their own 120s timeout; direct call preserves contextvars
        for free).
        """
        call_kwargs = dict(kwargs)
        if progress_cb is not None:
            call_kwargs["progress_callback"] = progress_cb
        timeout = float(getattr(tool, "timeout_s", 0) or 0)
        if timeout <= 0:
            return tool.fn(**call_kwargs)
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"tool-{tool.name}"
        )
        try:
            context = contextvars.copy_context()
            future = executor.submit(context.run, tool.fn, **call_kwargs)
            try:
                return future.result(timeout=timeout)
            except _FutureTimeoutError:
                raise ToolTimeoutError(tool.name, timeout)
        finally:
            # 不等待（wait=False）：超时场景下线程无法强杀，靠工具自身
            # I/O 超时收尾；正常完成时线程已结束，shutdown 即时返回。
            executor.shutdown(wait=False)

    def discover(self, task_description: str, limit: int = 8) -> List[ToolDefinition]:
        """按任务描述筛出相关工具（阶段 2，v1 关键词/标签匹配，不做向量）。

        计分规则（同一工具取各项之和）：
          - metadata["scenarios"] 短语出现在任务描述中：+2/条；
          - metadata["tags"] 以 ``@tag`` 显式提及：+3；作为单词出现：+1；
          - 工具名（或其中较长的词）出现在描述中：+1。
        候选池为 ``list_all()``（已过 check_fn + skills 白名单——发现结果只能收窄权限，
        不能放大）。只返回得分 > 0 的工具，按得分降序、名称稳定排序，截断至 limit。
        无任何匹配时返回空列表（由调用方决定回退全量）。"""
        text = (task_description or "").lower()
        if not text.strip():
            return []
        scored: List[Tuple[int, str, ToolDefinition]] = []
        for tool in self.list_all():
            meta = tool.metadata or {}
            score = 0
            for phrase in meta.get("scenarios", []):
                if str(phrase).lower() in text:
                    score += 2
            for tag in meta.get("tags", []):
                tag_l = str(tag).lower()
                if f"@{tag_l}" in text:
                    score += 3
                elif re.search(rf"(?<![a-z0-9]){re.escape(tag_l)}(?![a-z0-9])", text):
                    score += 1
            name_words = [w for w in tool.name.lower().replace("-", "_").split("_") if len(w) > 2]
            if tool.name.lower() in text or any(w in text for w in name_words):
                score += 1
            if score > 0:
                scored.append((score, tool.name, tool))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [tool for _, _, tool in scored[:limit]]

    def to_llm_schema(self) -> List[Dict[str, Any]]:
        """Return a list of tool descriptions in OpenAI function-call format (仅可用工具)."""
        return [t.to_llm_schema() for t in self.list_all()]

    def to_react_prompt(self) -> str:
        """生成 ReAct reasoning prompt 用的工具描述文本（仅含可用工具）。"""
        lines = []
        for t in self.list_all():
            args = ", ".join(
                f"{k}: {v[0]}" for k, v in t.arg_schema.items()
            ) or "无参数"
            lines.append(f"- {t.name}: {t.description}（参数: {args}）")
        return "\n".join(lines) or "（无可用工具）"


# ── Default registry singleton ─────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Return the global ToolRegistry, populated with built-in tools."""
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry


def _build_default_registry() -> ToolRegistry:
    """通过自动发现构建注册表：扫描 app/tools/ 下所有模块，注册导出
    TOOL（ToolDefinition 实例）的模块。新增工具 = 放一个模块进去并导出 TOOL，
    无需修改任何现有代码。"""
    reg = discover_tools()
    logger.info("Tool registry initialised with tools: %s", reg.list_names())
    return reg


def discover_tools() -> ToolRegistry:
    """扫描 app/tools/ 下所有模块，注册带 TOOL 全局变量的 ToolDefinition。

    跳过 registry / __init__ 等非工具模块。模块需导出:
        TOOL = ToolDefinition(name=..., description=..., fn=..., arg_schema=..., check_fn=...)
    """
    import importlib
    import pkgutil
    import app.tools as tools_pkg

    reg = ToolRegistry()
    for info in pkgutil.iter_modules(tools_pkg.__path__):
        if info.name in ("registry", "__init__"):
            continue
        try:
            mod = importlib.import_module(f"app.tools.{info.name}")
        except Exception as exc:
            logger.warning("[discover_tools] failed to import app.tools.%s: %s", info.name, exc)
            continue
        tool = getattr(mod, "TOOL", None)
        if isinstance(tool, ToolDefinition):
            reg.register(tool)
            logger.debug("[discover_tools] registered %s from %s", tool.name, info.name)
    return reg
