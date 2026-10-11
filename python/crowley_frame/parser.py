"""
Compile expression strings such as "x / sum(x)" to deferred expressions.

The string is parsed with `ast` and walked against a whitelist; it is never
evaluated. Bare names are columns, quoted strings and numbers are literals,
and only functions registered in `functions.REGISTRY` can be called.
"""
from __future__ import annotations

import ast
import difflib
import operator
from functools import lru_cache
from typing import Any

from .expr import ColumnExpr, Expr, TranslationError, as_expr
from .functions import REGISTRY

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _Compiler:
    def __init__(self, text: str):
        self.text = text

    def fail(self, node: ast.AST, msg: str) -> TranslationError:
        offset = getattr(node, "col_offset", 0)
        return TranslationError(f"{msg}\n  {self.text}\n  {' ' * offset}^")

    def visit(self, node: ast.AST) -> Any:
        """Returns an Expr, or a plain Python value for constants."""
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (bool, int, float, str)):
                return node.value
            raise self.fail(node, f"Unsupported literal {node.value!r}")

        if isinstance(node, ast.Name):
            return ColumnExpr(node.id)

        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~as_expr(operand)

        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)) and not (
                isinstance(left, Expr) and isinstance(right, Expr)
            ):
                # `x > 1 & y > 1` parses as `x > (1 & y) > 1`
                raise self.fail(
                    node,
                    "& and | bind tighter than comparisons; write and/or, "
                    "or put each comparison in parentheses",
                )
            return _BINARY[type(node.op)](left, right)

        if isinstance(node, ast.BoolOp):
            values = [as_expr(self.visit(v)) for v in node.values]
            op = operator.and_ if isinstance(node.op, ast.And) else operator.or_
            out = values[0]
            for v in values[1:]:
                out = op(out, v)
            return out

        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            out = None
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    term = as_expr(left).is_in(self.literal_list(comparator))
                    if isinstance(op, ast.NotIn):
                        term = ~term
                    right = None
                elif type(op) in _COMPARE:
                    right = self.visit(comparator)
                    term = _COMPARE[type(op)](as_expr(left), right)
                else:
                    raise self.fail(comparator, "Unsupported comparison")
                out = term if out is None else out & term
                left = right
            return out

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise self.fail(node, "Only plain function calls are supported in string expressions")
            fn = REGISTRY.get(node.func.id)
            if fn is None:
                hint = difflib.get_close_matches(node.func.id, REGISTRY, n=1)
                suffix = f"; did you mean {hint[0]}()?" if hint else ""
                raise self.fail(node, f"Unknown function {node.func.id}(){suffix}")
            args = [self.arg(a) for a in node.args]
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    raise self.fail(node, "**kwargs is not supported in string expressions")
                kwargs[kw.arg] = self.arg(kw.value)
            try:
                return fn(*args, **kwargs)
            except TypeError as e:
                raise self.fail(node, f"Bad call to {node.func.id}(): {e}") from None

        raise self.fail(
            node,
            f"{type(node).__name__} is not supported in string expressions "
            "(allowed: column names, literals, arithmetic, comparisons, and/or/not, "
            "and calls to crowley_frame functions)",
        )

    def arg(self, node: ast.AST) -> Any:
        if isinstance(node, (ast.List, ast.Tuple)):
            return self.literal_list(node)
        return self.visit(node)

    def literal_list(self, node: ast.AST) -> list:
        if not isinstance(node, (ast.List, ast.Tuple)):
            raise self.fail(node, "`in` needs a literal list, e.g. x in [1, 2, 3]")
        values = [self.visit(e) for e in node.elts]
        if builtins_any(isinstance(v, Expr) for v in values):
            raise self.fail(node, "Lists in string expressions may only contain literals")
        return values


builtins_any = any


@lru_cache(maxsize=1024)
def parse_expr(text: str) -> Expr:
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise TranslationError(f"Could not parse expression {text!r}: {e.msg}") from None
    compiler = _Compiler(text.strip())
    return as_expr(compiler.visit(tree.body))
