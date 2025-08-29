from __future__ import annotations

import ast
import operator
from typing import Any


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_expr(node: ast.AST) -> Any:
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant")
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        op = OPS.get(type(node.op))
        if not op:
            raise ValueError("Unsupported operator")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = OPS.get(type(node.op))
        if not op:
            raise ValueError("Unsupported unary op")
        return op(_eval_expr(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    tree = ast.parse(expression, mode="eval")
    result = _eval_expr(tree.body)
    return str(result)

