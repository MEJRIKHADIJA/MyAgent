from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from errors import ToolError


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs":  abs,
    "round": round,
    "floor": math.floor,
    "ceil":  math.ceil,
}

_FILLER_PATTERNS = [
    r"\bwhat\s+is\b",
    r"\bwhat's\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\beval\b",
    r"\bevaluate\b",
    r"\bplease\b",
    r"\bsolve\b",
    r"\bthe\s+result\s+of\b",
]


def run(query: str) -> str:
    expression = normalize_expression(query)
    if not expression:
        raise ToolError("expression cannot be parsed", tool_name="calculator")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("division by zero", tool_name="calculator") from exc
    except Exception as exc:
        raise ToolError("expression cannot be parsed", tool_name="calculator") from exc

    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ToolError("result is not a number", tool_name="calculator")

    if isinstance(result, float) and math.isnan(result):
        raise ToolError("low confidence", tool_name="calculator")

    result_string = (
        str(int(result))
        if isinstance(result, float) and result.is_integer()
        else str(result)
    )

    if not result_string or result_string.lower() == "nan":
        raise ToolError("low confidence", tool_name="calculator")

    return result_string


def calculate(query: str) -> str:
    return run(query)


def normalize_expression(query: str) -> str:
    expression = query.strip().lower()

    for pattern in _FILLER_PATTERNS:
        expression = re.sub(pattern, "", expression)

    expression = expression.replace("?", " ")
    expression = expression.replace(",", " ")
    expression = expression.replace("^", "**")
    expression = re.sub(r"\bsquare\s+root\s+of\b", "sqrt", expression)
    expression = re.sub(r"\bsqrt\s+of\b",          "sqrt", expression)
    expression = re.sub(r"\s+", " ", expression).strip()

    if re.search(r"[^0-9+\-*/().\sA-Za-z_]", expression):
        return ""

    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression))
    if names - set(_FUNCTIONS):
        return ""

    if not re.search(r"\d|\bsqrt\b", expression):
        return ""

    return expression


def looks_like_math(query: str) -> bool:
    expression = normalize_expression(query)
    if not expression:
        return False
    if "sqrt" in expression:
        return True
    return bool(re.search(r"\d\s*(\*\*|[+\-*/])\s*\d", expression))


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError("result is not a number", tool_name="calculator")

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ToolError("unsupported operator", tool_name="calculator")
        return op_func(_evaluate(node.left), _evaluate(node.right))

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ToolError("unsupported operator", tool_name="calculator")
        return op_func(_evaluate(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolError("unsupported function", tool_name="calculator")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ToolError("unsupported function", tool_name="calculator")
        if node.keywords:
            raise ToolError("expression cannot be parsed", tool_name="calculator")
        return func(*[_evaluate(arg) for arg in node.args])

    raise ToolError("expression cannot be parsed", tool_name="calculator")