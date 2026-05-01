"""Calculator plugin for OpenCAS — safe expression evaluator + unit conversion."""

from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}
_FUNCS = {
    name: getattr(math, name)
    for name in (
        "sqrt", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "floor", "ceil", "fabs", "factorial", "gcd",
    )
}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"unsupported operator: {op_type.__name__}")
        return _BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id not in _NAMES:
            raise ValueError(f"unknown name: {node.id}")
        return _NAMES[node.id]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError(f"unsupported function call: {ast.dump(node.func)}")
        if node.keywords:
            raise ValueError("keyword arguments not supported")
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt) for elt in node.elts)
    raise ValueError(f"unsupported node: {type(node).__name__}")


def _calculate(args: Dict[str, Any]) -> ToolResult:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return ToolResult(success=False, output="expression is required", metadata={})
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
    except (ValueError, SyntaxError, ZeroDivisionError, ArithmeticError) as exc:
        return ToolResult(success=False, output=f"eval error: {exc}", metadata={"expression": expr})
    return ToolResult(
        success=True,
        output=str(value),
        metadata={"expression": expr, "type": type(value).__name__},
    )


# Conversion factors keyed to a base unit per family.
_LENGTH = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}
_MASS = {"g": 1.0, "kg": 1000.0, "mg": 0.001, "lb": 453.59237, "oz": 28.349523125, "ton": 1_000_000.0}
_TIME = {"s": 1.0, "ms": 0.001, "us": 1e-6, "ns": 1e-9, "min": 60.0, "h": 3600.0, "d": 86400.0, "wk": 604800.0}
_BYTES = {"B": 1.0, "KB": 1024.0, "MB": 1024.0**2, "GB": 1024.0**3, "TB": 1024.0**4, "PB": 1024.0**5}
_FAMILIES = {"length": _LENGTH, "mass": _MASS, "time": _TIME, "bytes": _BYTES}


def _find_family(unit: str) -> tuple[str | None, dict | None]:
    for family, table in _FAMILIES.items():
        if unit in table:
            return family, table
    return None, None


def _unit_convert(args: Dict[str, Any]) -> ToolResult:
    try:
        value = float(args.get("value"))
    except (TypeError, ValueError):
        return ToolResult(success=False, output="value must be numeric", metadata={})
    src = str(args.get("from", "")).strip()
    dst = str(args.get("to", "")).strip()
    if not src or not dst:
        return ToolResult(success=False, output="'from' and 'to' units are required", metadata={})

    if src.lower() in {"c", "f", "k"} and dst.lower() in {"c", "f", "k"}:
        s, d = src.lower(), dst.lower()
        if s == "c":
            kelvin = value + 273.15
        elif s == "f":
            kelvin = (value - 32) * 5.0 / 9.0 + 273.15
        else:
            kelvin = value
        if d == "c":
            result = kelvin - 273.15
        elif d == "f":
            result = (kelvin - 273.15) * 9.0 / 5.0 + 32
        else:
            result = kelvin
        return ToolResult(
            success=True,
            output=json.dumps({"value": result, "from": src, "to": dst, "family": "temperature"}, indent=2),
            metadata={"family": "temperature"},
        )

    fam_a, table_a = _find_family(src)
    fam_b, table_b = _find_family(dst)
    if fam_a is None or fam_b is None:
        return ToolResult(success=False, output=f"unknown unit(s): {src}, {dst}", metadata={})
    if fam_a != fam_b:
        return ToolResult(
            success=False,
            output=f"unit families differ: {fam_a} vs {fam_b}",
            metadata={"from_family": fam_a, "to_family": fam_b},
        )
    base = value * table_a[src]
    result = base / table_b[dst]
    return ToolResult(
        success=True,
        output=json.dumps({"value": result, "from": src, "to": dst, "family": fam_a}, indent=2),
        metadata={"family": fam_a},
    )


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "calculate",
        "Evaluate an arithmetic expression. Supports +-*/%**, parens, and math functions (sqrt, log, sin, abs, floor, ceil, factorial, gcd, ...) plus pi/e/tau.",
        lambda name, args: _calculate(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )
    tools.register(
        "unit_convert",
        "Convert a value between units. Supported families: length, mass, time, bytes, temperature (C/F/K).",
        lambda name, args: _unit_convert(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from": {"type": "string"},
                "to": {"type": "string"},
            },
            "required": ["value", "from", "to"],
        },
    )
