"""
Per-frame bookkeeping: the audit trail and (opt-in) column lineage.

`instrument(Frame)` wraps every public verb once, at import. The wrapper
appends (verb, args, kwargs) to the result's audit trail; nothing is formatted
until audit() is called. Lineage is only computed for frames that asked for it
with track_lineage(), because it re-reads the verb's arguments on every call.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import polars as pl

from .col import ColSelector, resolve_selectors
from .expr import Ctx, Expr

Lineage = Dict[str, FrozenSet[str]]
UNKNOWN = frozenset({"(unknown)"})


def instrument(cls: type) -> None:
    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or not inspect.isfunction(fn):
            continue
        setattr(cls, name, _wrap(cls, name, fn))


def _wrap(cls: type, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def verb(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = fn(self, *args, **kwargs)
        if out is self:
            return out
        if isinstance(out, cls):
            _record(self, out, name, args, kwargs)
        elif isinstance(out, (tuple, dict)):
            # Several frames (partition_valid, branch). Frames that already
            # recorded their own steps, as branches do, keep that trail.
            for item in out.values() if isinstance(out, dict) else out:
                if isinstance(item, cls) and item._audit is self._audit:
                    _record(self, item, name, args, kwargs)
        return out

    return verb


def _record(before: Any, after: Any, name: str, args: tuple, kwargs: dict) -> None:
    after._audit = before._audit + ((name, args, kwargs),)
    if before._lineage is not None:
        after._lineage = _update_lineage(name, before, after, args, kwargs)


# ---------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, Expr):
        return value._text
    if isinstance(value, ColSelector):
        return f"{value.kind}({value.value!r})" if value.value is not None else f"{value.kind}()"
    if hasattr(value, "_lf") and hasattr(value, "_audit"):
        return f"<Frame: {len(value.columns)} columns>"
    if isinstance(value, (pl.DataFrame, pl.LazyFrame, pl.Series)):
        return f"<polars {type(value).__name__}>"
    if callable(value) and not isinstance(value, type):
        return getattr(value, "__name__", type(value).__name__)
    if isinstance(value, (list, tuple)) and len(value) <= 6:
        inner = ", ".join(_fmt(v) for v in value)
        return f"[{inner}]" if isinstance(value, list) else f"({inner})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def format_step(step: Tuple[str, tuple, dict]) -> Tuple[str, str]:
    name, args, kwargs = step
    parts = [_fmt(a) for a in args] + [f"{k}={_fmt(v)}" for k, v in kwargs.items()]
    return name, ", ".join(parts)


def audit(frame: Any) -> pl.DataFrame:
    """
    The verbs applied to this frame since it was created, in order, as a small
    Polars DataFrame (step, verb, arguments). Reads no data.
    """
    steps = [format_step(s) for s in frame._audit]
    return pl.DataFrame(
        {
            "step": list(range(1, len(steps) + 1)),
            "verb": [s[0] for s in steps],
            "arguments": [s[1] for s in steps],
        },
        schema={"step": pl.Int64, "verb": pl.String, "arguments": pl.String},
    )


# ---------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------


def track_lineage(frame: Any) -> Any:
    """
    Start tracking which of the current columns each later column derives from.
    Tracking costs a little on every following verb, so it is opt-in.
    """
    out = frame._new(frame._lf)
    out._lineage = {c: frozenset({c}) for c in frame.columns}
    return out


def lineage(frame: Any) -> Dict[str, List[str]]:
    """
    {column: [source columns]} relative to where track_lineage() was called.
    Columns from the other side of a join appear as "right.<name>" unless that
    frame was tracking its own lineage. "(unknown)" marks columns produced by a
    verb lineage does not model. Reads no data.
    """
    if frame._lineage is None:
        from .expr import CrowleyValidationError

        raise CrowleyValidationError("lineage() needs track_lineage() to be called earlier in the pipeline")
    return {c: sorted(frame._lineage.get(c, UNKNOWN)) for c in frame.columns}


def _sources(lin: Lineage, names: Any) -> FrozenSet[str]:
    out: FrozenSet[str] = frozenset()
    for name in names:
        out |= lin.get(name, UNKNOWN)
    return out


def _lineage_of(frame: Any, prefix: str) -> Lineage:
    if frame._lineage is not None:
        return frame._lineage
    return {c: frozenset({prefix + c}) for c in frame.columns}


def _expr_lineage(before: Any, items: List[Tuple[str, Any]], window: bool, lin: Lineage) -> Lineage:
    from .api import _to_expr

    known = set(before.columns)
    ctx = Ctx(known, before._groups, window=window, schema=before.schema)
    out = dict(lin)
    for name, value in items:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            roots = [value[0]]  # summarise(name=("column", "function"))
        else:
            roots = _to_expr(value)._compile(ctx).meta.root_names()
        out[name] = _sources(out, roots)
        known.add(name)
    return out


def _update_lineage(name: str, before: Any, after: Any, args: tuple, kwargs: dict) -> Lineage:
    lin: Lineage = before._lineage
    new_cols = after.columns
    try:
        if name in ("rename", "rename_with", "normalize_names"):
            return {new: lin.get(old, UNKNOWN) for old, new in zip(before.columns, new_cols)}

        if name in ("mutate", "transmute"):
            out = _expr_lineage(before, before._named(args, kwargs), True, lin)
            return {c: out.get(c, UNKNOWN) for c in new_cols}

        if name in ("summarise", "summarize"):
            out = _expr_lineage(before, before._named(args, kwargs), False, lin)
            return {c: out.get(c, UNKNOWN) for c in new_cols}

        if name in ("count", "add_count"):
            keys = list(before._groups) + [a for a in args if isinstance(a, str)]
            counted = kwargs.get("name", "n")
            out = {c: lin.get(c, UNKNOWN) for c in new_cols}
            out[counted] = _sources(lin, keys + ([kwargs["wt"]] if kwargs.get("wt") else []))
            if kwargs.get("prop"):
                out["prop"] = out[counted]
            return out

        if name.endswith("_join") and args:
            from .api import df

            right = _lineage_of(df(args[0]), "right.")
            suffixes = kwargs.get("suffixes", ("_x", "_y"))
            out = {}
            for c in new_cols:
                if c in lin:
                    out[c] = lin[c]
                elif c in right:
                    out[c] = right[c]
                elif suffixes[0] and c.endswith(suffixes[0]) and c[: -len(suffixes[0])] in lin:
                    out[c] = lin[c[: -len(suffixes[0])]]
                elif suffixes[1] and c.endswith(suffixes[1]) and c[: -len(suffixes[1])] in right:
                    out[c] = right[c[: -len(suffixes[1])]]
                else:
                    out[c] = UNKNOWN
            return out

        if name == "pivot_longer":
            cols = kwargs.get("cols")
            selectors = (list(cols) if isinstance(cols, (list, tuple)) else [cols]) if cols is not None else (
                list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args)
            )
            pivoted = _sources(lin, resolve_selectors(before.schema, selectors))
            out = {c: lin.get(c, UNKNOWN) for c in new_cols}
            out[kwargs.get("names_to", "name")] = pivoted
            out[kwargs.get("values_to", "value")] = pivoted
            return out

        if name == "unite":
            new_column, columns = args[0], (args[1] if len(args) > 1 else kwargs["columns"])
            out = {c: lin.get(c, UNKNOWN) for c in new_cols}
            out[new_column] = _sources(lin, columns)
            return out

        if name == "separate":
            column, into = args[0], (args[1] if len(args) > 1 else kwargs["into"])
            out = {c: lin.get(c, UNKNOWN) for c in new_cols}
            for piece in into:
                out[piece] = lin.get(column, UNKNOWN)
            return out

        if name == "fingerprint_rows":
            hashed = resolve_selectors(before.schema, args) if args else before.columns
            out = {c: lin.get(c, UNKNOWN) for c in new_cols}
            out[kwargs.get("name", "row_hash")] = _sources(lin, hashed)
            return out
    except Exception:
        # Lineage is advisory; never let it break the pipeline.
        pass
    return {c: lin.get(c, UNKNOWN) for c in new_cols}
