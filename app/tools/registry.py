"""Tool registry – central place to register and invoke tools.

Each tool is registered as a ToolDefinition containing:
  - name        : unique string key
  - description : short human-readable description
  - fn          : callable that accepts **kwargs and returns str
  - arg_schema  : dict mapping arg name -> (type, description, required)
"""
from __future__ import annotations

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


class ToolRegistry:
    """Registry that maps tool names to their definitions."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.debug("Tool registered: %s", tool.name)

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise ToolNotFoundError(name, "tool not registered")
        return self._tools[name]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def invoke(self, name: str, **kwargs: Any) -> str:
        """Execute a registered tool by name.

        Args:
            name:   Tool name.
            **kwargs: Arguments forwarded to the tool function.

        Returns:
            String output of the tool.

        Raises:
            ToolNotFoundError:   Tool not registered.
            ToolExecutionError:  Tool raised an error during execution.
        """
        tool = self.get(name)
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
        """Return a list of tool descriptions in OpenAI function-call format."""
        schema = []
        for tool in self._tools.values():
            properties: Dict[str, Any] = {}
            required_args: List[str] = []
            for arg_name, (type_str, desc, is_req) in tool.arg_schema.items():
                properties[arg_name] = {"type": type_str, "description": desc}
                if is_req:
                    required_args.append(arg_name)
            schema.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_args,
                    },
                },
            })
        return schema


# ── Default registry singleton ─────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Return the global ToolRegistry, populated with built-in tools."""
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry


def _build_default_registry() -> ToolRegistry:
    from app.tools.calculator import calculator
    from app.tools.datetime_tool import datetime_tool
    from app.tools.text_tool import text_tool

    reg = ToolRegistry()

    reg.register(ToolDefinition(
        name="calculator",
        description="Evaluate a safe mathematical expression and return the numeric result.",
        fn=lambda expression, **_: calculator(expression),
        arg_schema={
            "expression": ("string", "Math expression to evaluate, e.g. '(12+34)*2'", True),
        },
    ))

    reg.register(ToolDefinition(
        name="datetime_tool",
        description="Return the current date and time, optionally formatted.",
        fn=lambda fmt=None, tz="local", timestamp=None, **_: datetime_tool(
            fmt=fmt, tz=tz, timestamp=timestamp
        ),
        arg_schema={
            "fmt": ("string", "strftime format, e.g. '%Y-%m-%d'", False),
            "tz": ("string", "'local' or 'utc'", False),
            "timestamp": ("number", "Unix timestamp in seconds (optional)", False),
        },
    ))

    reg.register(ToolDefinition(
        name="text_tool",
        description="Perform text processing: word_count, char_count, sentence_count, clean, uppercase, lowercase, reverse, extract_numbers, stats.",
        fn=lambda operation, text, **_: text_tool(operation=operation, text=text),
        arg_schema={
            "operation": ("string", "One of: word_count | char_count | sentence_count | clean | uppercase | lowercase | reverse | extract_numbers | stats", True),
            "text": ("string", "Input text to process", True),
        },
    ))

    logger.info("Tool registry initialised with tools: %s", reg.list_names())
    return reg
