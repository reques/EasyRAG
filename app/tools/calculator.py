"""Calculator tool – evaluates safe mathematical expressions."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from app.core.exceptions import ToolExecutionError
from app.core.logger import get_logger

logger = get_logger(__name__)

# Allowed operators
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Allowed math functions
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using whitelisted ops only."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolExecutionError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ToolExecutionError(f"Unsupported operator: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ToolExecutionError(f"Unsupported unary op: {op_type}")
        return _OPS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolExecutionError("Function calls must be simple names.")
        fname = node.func.id
        if fname not in _FUNCS:
            raise ToolExecutionError(f"Function '{fname}' is not allowed.")
        args = [_safe_eval(a) for a in node.args]
        return _FUNCS[fname](*args)
    if isinstance(node, ast.Name):
        if node.id in _FUNCS:
            return _FUNCS[node.id]  # e.g. pi, e
        raise ToolExecutionError(f"Unknown name: {node.id}")
    raise ToolExecutionError(f"Unsupported AST node: {type(node)}")


def calculator(expression: str) -> str:
    """Evaluate a mathematical *expression* and return the result as a string.

    Args:
        expression: e.g. "(12 + 34) * 2", "sqrt(144)", "2 ** 10"

    Returns:
        String representation of the numeric result.

    Raises:
        ToolExecutionError: if the expression is invalid or unsafe.
    """
    logger.debug("Calculator: %s", expression)
    expr = expression.strip()
    if not expr:
        raise ToolExecutionError("Empty expression.")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        # Format nicely: drop .0 for whole numbers
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        else:
            formatted = str(round(result, 10)).rstrip("0").rstrip(".")
        logger.debug("Calculator result: %s", formatted)
        return f"{expression} = {formatted}"
    except ToolExecutionError:
        raise
    except ZeroDivisionError:
        raise ToolExecutionError("Division by zero.")
    except Exception as exc:
        raise ToolExecutionError(f"Failed to evaluate '{expression}': {exc}") from exc
