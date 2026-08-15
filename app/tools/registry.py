"""Tool registry – central place to register and invoke tools.

Each tool is registered as a ToolDefinition containing:
  - name        : unique string key
  - description : short human-readable description
  - fn          : callable that accepts **kwargs and returns str
  - arg_schema  : dict mapping arg name -> (type, description, required)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.exceptions import ToolExecutionError, ToolNotFoundError
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    fn: Callable[..., str]
    # {arg_name: (python_type_str, description, is_required)}
    arg_schema: Dict[str, Tuple[str, str, bool]] = field(default_factory=dict)
    # 工具可用性自检：返回 False 的工具不出现在 schema/react prompt，invoke 时拒绝
    check_fn: Optional[Callable[[], bool]] = None

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
                raise ToolNotFoundError(name, "tool not registered")
            return self._tools[name]

    def list_names(self, available_only: bool = True) -> List[str]:
        """Return tool names. available_only=True 时只含 check_fn 通过的工具。"""
        with self._lock:
            snapshot = list(self._tools.values())
        if not available_only:
            candidates = snapshot
        else:
            candidates = [t for t in snapshot if t.is_available()]
        from app.skills.context import get_active_skill_context

        skill_context = get_active_skill_context()
        return [
            tool.name
            for tool in candidates
            if skill_context.allows_tool(tool.name)
        ]

    def list_all(self, available_only: bool = True) -> List[ToolDefinition]:
        """Return all ToolDefinition objects (for schema/prompt building).
        available_only=True 时只含 check_fn 通过的工具。"""
        with self._lock:
            snapshot = list(self._tools.values())
        candidates = snapshot if not available_only else [
            tool for tool in snapshot if tool.is_available()
        ]
        from app.skills.context import get_active_skill_context

        skill_context = get_active_skill_context()
        return [
            tool for tool in candidates if skill_context.allows_tool(tool.name)
        ]

    def invoke(self, name: str, **kwargs: Any) -> str:
        """Execute a registered tool by name.

        The tool lookup is performed under the registry lock, but the actual
        ``fn`` call runs *outside* the lock so a slow tool (e.g. web_search
        HTTP request) never blocks other registry readers.

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
        from app.skills.context import get_active_skill_context

        if not get_active_skill_context().allows_tool(name):
            raise ToolExecutionError(
                f"Tool '{name}' is not allowed by the selected Skills"
            )
        if not tool.is_available():
            raise ToolExecutionError(
                f"Tool '{name}' is not available (check_fn failed — missing config or dependency)"
            )
        logger.info("Invoking tool '%s' with args: %s", name, kwargs)
        try:
            result = tool.fn(**kwargs)
            logger.info("Tool '%s' succeeded.", name)
            return result
        except (ToolExecutionError, ToolNotFoundError):
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{name}' raised an unexpected error: {exc}"
            ) from exc

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
