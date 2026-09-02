"""DeepAgents 集成 — 项目 ToolRegistry → langchain 工具转换。

EasyRAG 的工具中心是 ``app/tools/registry.py`` 的 ``ToolRegistry``（自动发现
``app/tools/*.py`` 导出的 ``TOOL`` + MCP 桥接），工具函数签名统一为
``fn(**kwargs) -> str``。langchain ``create_agent`` 需要 langchain
BaseTool。这里把 ``ToolDefinition`` 包装成 ``StructuredTool``（执行时仍走
``registry.invoke``，因此技能白名单、check_fn 可用性检查、MCP 动态注册
全部自动生效，不复制任何工具实现）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, create_model

from app.core.logger import get_logger
from app.tools.registry import ToolDefinition, get_tool_registry

logger = get_logger(__name__)

# python 类型字符串 → pydantic 类型映射（ToolDefinition.arg_schema 的约定）
_TYPE_MAP: Dict[str, Any] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "dict": dict,
    "any": Any,
}


def _args_model(tool: ToolDefinition) -> type[BaseModel]:
    """根据 ToolDefinition.arg_schema 动态构造 pydantic 参数模型。"""
    fields: Dict[str, Any] = {}
    for arg_name, (type_str, desc, is_required) in (tool.arg_schema or {}).items():
        # 跳过下划线开头的参数（Pydantic 不允许）
        if arg_name.startswith("_"):
            logger.debug("[deepagents] skip arg %r in tool %r (leading underscore)", arg_name, tool.name)
            continue
        py_type = _TYPE_MAP.get((type_str or "").lower(), str)
        # 必填字段没有默认值；可选字段给 None 默认
        if is_required:
            fields[arg_name] = (py_type, Field(description=desc or arg_name))
        else:
            fields[arg_name] = (
                Optional[py_type],
                Field(default=None, description=desc or arg_name),
            )
    if not fields:
        # Pydantic 不允许下划线开头的字段名，改用 noop
        fields["noop"] = (Optional[str], Field(default=None, description="无参数"))
    return create_model(f"{tool.name}Args", **fields)


def _to_structured(tool: ToolDefinition) -> Any:
    """单个 ToolDefinition → langchain StructuredTool（执行走 registry.invoke）。"""
    from langchain_core.tools import StructuredTool

    args_model = _args_model(tool)

    def _run(**kwargs: Any) -> str:
        return get_tool_registry().invoke(tool.name, **kwargs)

    st = StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,
        args_schema=args_model,
    )
    st.__easyrag_tool_name__ = tool.name  # 便于测试/日志回查
    return st


def registry_to_langchain_tools(
    tool_names: Optional[List[str]] = None,
) -> List[Any]:
    """把注册表（可用 + 技能白名单放行）的工具转换为 langchain 工具列表。

    Args:
        tool_names: 白名单过滤（None = 全部可用工具）。SubAgent 用它做
            工具子集配置；主 Agent 传 None 拿全量。
    """
    registry = get_tool_registry()
    tools = []
    for t in registry.list_all():  # 已含 check_fn + 技能白名单过滤
        if tool_names is not None and t.name not in tool_names:
            continue
        try:
            tools.append(_to_structured(t))
        except Exception as exc:
            logger.warning("[deepagents] skip tool %s: %s", t.name, exc)
    return tools


def tools_prompt(tools: List[Any]) -> str:
    """生成工具清单文本（注入 system prompt，帮助模型理解可用工具）。"""
    lines = []
    for t in tools:
        schema = getattr(t, "args", None) or {}
        args_desc = ", ".join(schema.keys()) or "无参数"
        lines.append(f"- {t.name}: {t.description}（参数: {args_desc}）")
    return "\n".join(lines) or "（无可用工具）"


def _json_dumps_default(obj: Any) -> str:
    """JSON 序列化兜底（dataclass / pydantic 对象等）。"""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), ensure_ascii=False, default=str)
    return str(obj)
