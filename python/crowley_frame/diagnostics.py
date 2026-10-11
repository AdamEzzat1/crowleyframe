"""
Stage 1 inspection helpers that read the schema or the plan, never the data:
compare_schema, explain_plan, explain_expr, resolve_columns.
(audit and lineage live in tracking.py.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import polars as pl

from .col import resolve_selectors
from .expr import Ctx
from .tracking import format_step


class SchemaComparison:
    """Result of compare_schema(a, b). Truthy when the schemas are identical."""

    def __init__(self, a: Any, b: Any):
        names_a, names_b = list(a), list(b)
        self.missing: List[str] = [c for c in names_b if c not in a]  # in b, not in a
        self.extra: List[str] = [c for c in names_a if c not in b]  # in a, not in b
        self.dtype_mismatches: Dict[str, Tuple[str, str]] = {
            c: (str(a[c]), str(b[c])) for c in names_a if c in b and a[c] != b[c]
        }
        shared_a = [c for c in names_a if c in b]
        shared_b = [c for c in names_b if c in a]
        self.order_differs: bool = shared_a != shared_b
        self.equal: bool = not (self.missing or self.extra or self.dtype_mismatches or self.order_differs)

    def __bool__(self) -> bool:
        return self.equal

    def to_polars(self) -> pl.DataFrame:
        rows = (
            [{"column": c, "difference": "missing", "left": None, "right": "present"} for c in self.missing]
            + [{"column": c, "difference": "extra", "left": "present", "right": None} for c in self.extra]
            + [
                {"column": c, "difference": "dtype", "left": left, "right": right}
                for c, (left, right) in self.dtype_mismatches.items()
            ]
        )
        if self.order_differs:
            rows.append({"column": None, "difference": "column order", "left": None, "right": None})
        schema = {"column": pl.String, "difference": pl.String, "left": pl.String, "right": pl.String}
        return pl.DataFrame(rows, schema=schema)

    def __repr__(self) -> str:
        if self.equal:
            return "SchemaComparison: identical"
        parts = []
        if self.missing:
            parts.append(f"missing: {self.missing}")
        if self.extra:
            parts.append(f"extra: {self.extra}")
        if self.dtype_mismatches:
            parts.append(f"dtype mismatches: {self.dtype_mismatches}")
        if self.order_differs:
            parts.append("column order differs")
        return "SchemaComparison: " + "; ".join(parts)


def compare_schema(frame: Any, other: Any) -> SchemaComparison:
    """
    Compare this frame's schema with another's: columns missing here, extra
    here, dtype mismatches and column order. Reads no data.
    """
    from .api import df

    return SchemaComparison(frame.schema, df(other).schema)


def resolve_columns(frame: Any, *selectors: Any) -> List[str]:
    """The column names the given tidyselect selectors resolve to. Reads no data."""
    return resolve_selectors(frame.schema, selectors)


def explain_expr(frame: Any, expr: Any, context: str = "mutate") -> str:
    """
    The Polars expression a crowley expression (object or string) compiles to
    on this frame. `context` is "mutate" (aggregates become per-group windows
    on a grouped frame) or "summarise". Reads no data.
    """
    from .api import _to_expr

    if context not in ("mutate", "summarise", "summarize"):
        raise ValueError("context must be 'mutate' or 'summarise'")
    ctx = Ctx(set(frame.columns), frame._groups, window=context == "mutate", schema=frame.schema)
    return str(_to_expr(expr)._compile(ctx))


# (substring of the optimised plan, warning)
_PLAN_WARNINGS = [
    ("SORT BY", "sorts rows: cost grows faster than the data"),
    ("CROSS JOIN", "cross join: output is left rows x right rows"),
    ("UNIQUE", "de-duplicates rows: holds every distinct row in memory"),
    ("PYTHON", "runs Python code in the engine: not optimised and single-threaded"),
    ("MAP", "runs an opaque function in the engine: blocks plan optimisation"),
    ("UNPIVOT", "unpivots: output has one row per input row per pivoted column"),
]
_EAGER_VERBS = {
    "collect": "the plan was executed here",
    "pivot_wider": "the plan up to here was executed (pivot_wider is eager)",
    "snapshot": "the plan was executed and written to disk here",
}


def explain_plan(frame: Any, optimized: bool = True) -> str:
    """
    The crowley verbs applied so far, the Polars plan they produce, and
    warnings for expensive operations. Reads no data.
    """
    plan = frame._lf.explain(optimized=optimized)
    lines = ["Crowley pipeline:"]
    warnings: List[str] = []
    if not frame._audit:
        lines.append("  (no verbs applied)")
    for i, step in enumerate(frame._audit, start=1):
        verb, arguments = format_step(step)
        lines.append(f"  {i}. {verb}({arguments})")
        if verb in _EAGER_VERBS:
            warnings.append(f"step {i}: {_EAGER_VERBS[verb]}")
    upper = plan.upper()
    for needle, message in _PLAN_WARNINGS:
        if needle in upper:
            warnings.append(message)
    lines += ["", f"Polars plan ({'optimised' if optimized else 'as written'}):"]
    lines += ["  " + line for line in plan.splitlines()]
    lines += ["", "Warnings:"]
    lines += ["  - " + w for w in warnings] if warnings else ["  (none)"]
    return "\n".join(lines)
