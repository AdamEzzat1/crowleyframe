"""
R-named functions for use inside mutate / filter / summarise / arrange.

Every function returns a deferred `Expr` that compiles to native Polars.
On a grouped frame, aggregates and window functions are evaluated per group
in mutate/filter (via `.over()`) and as plain aggregates in summarise.

A string in a column position names a column (`mean("x")`); a string in a
value position is a literal (`if_else(col.x > 0, "pos", "neg")`).
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import polars as pl

from .expr import Ctx, Expr, TranslationError, as_expr, col_arg, lit, over  # noqa: F401

_ROLLING_MIN_KW = (
    "min_samples" if "min_samples" in inspect.signature(pl.Expr.rolling_mean).parameters else "min_periods"
)


def _is_missing_default(value: Any) -> bool:
    return value is None or type(value).__name__ == "NAType"


def _aggregate(name: str, fn: Callable[[pl.Expr], pl.Expr]):
    def agg(x: Any, na_rm: bool = True) -> Expr:
        e = col_arg(x)

        def build(ctx: Ctx) -> pl.Expr:
            c = e._compile(ctx)
            out = fn(c)
            if not na_rm:
                # R semantics: any missing value makes the result missing.
                out = pl.when(c.null_count() > 0).then(None).otherwise(out)
            return over(out, ctx)

        return Expr(build, f"{name}({e._text})")

    agg.__name__ = name
    agg.__doc__ = f"{name}(x, na_rm=True): missing values are skipped unless na_rm=False."
    return agg


mean = _aggregate("mean", lambda c: c.mean())
sum = _aggregate("sum", lambda c: c.sum())  # noqa: A001
min = _aggregate("min", lambda c: c.min())  # noqa: A001
max = _aggregate("max", lambda c: c.max())  # noqa: A001
median = _aggregate("median", lambda c: c.median())
sd = _aggregate("sd", lambda c: c.std())
var = _aggregate("var", lambda c: c.var())
first = _aggregate("first", lambda c: c.first())
last = _aggregate("last", lambda c: c.last())
any = _aggregate("any", lambda c: c.any())  # noqa: A001
all = _aggregate("all", lambda c: c.all())  # noqa: A001


class IntAgg(Expr):
    """
    A counting aggregate. Polars counts in UInt32; R users expect a signed
    integer (so `n() - 5` cannot wrap). `_raw` builds the uncast aggregate so
    summarise can cast once on the aggregated result: a cast inside
    group_by().agg() takes Polars off its fast aggregation path.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Callable[[Ctx], pl.Expr], text: str):
        self._raw = raw
        super().__init__(lambda ctx: over(raw(ctx).cast(pl.Int64), ctx), text)


def n() -> Expr:
    """Number of rows in the current group."""
    return IntAgg(lambda ctx: pl.len(), "n()")


def n_distinct(x: Any, na_rm: bool = False) -> Expr:
    e = col_arg(x)

    def raw(ctx: Ctx) -> pl.Expr:
        c = e._compile(ctx)
        if na_rm:
            c = c.drop_nulls()
        return c.n_unique()

    return IntAgg(raw, f"n_distinct({e._text})")


def _count_non_null(x: Any) -> Expr:
    e = col_arg(x)
    return IntAgg(lambda ctx: e._compile(ctx).count(), f"count({e._text})")


# ---------------------------------------------------------------------
# window functions
# ---------------------------------------------------------------------


def lag(x: Any, n: int = 1, default: Any = None) -> Expr:
    e = col_arg(x)
    fill = None if _is_missing_default(default) else default
    return Expr(
        lambda ctx: over(e._compile(ctx).shift(n, fill_value=fill), ctx),
        f"lag({e._text}, {n})",
    )


def lead(x: Any, n: int = 1, default: Any = None) -> Expr:
    e = col_arg(x)
    fill = None if _is_missing_default(default) else default
    return Expr(
        lambda ctx: over(e._compile(ctx).shift(-n, fill_value=fill), ctx),
        f"lead({e._text}, {n})",
    )


def rolling_mean(x: Any, window: int, min_periods: Optional[int] = None) -> Expr:
    e = col_arg(x)
    kwargs = {_ROLLING_MIN_KW: window if min_periods is None else min_periods}
    return Expr(
        lambda ctx: over(e._compile(ctx).rolling_mean(window_size=window, **kwargs), ctx),
        f"rolling_mean({e._text}, {window})",
    )


def cumsum(x: Any) -> Expr:
    e = col_arg(x)
    return Expr(lambda ctx: over(e._compile(ctx).cum_sum(), ctx), f"cumsum({e._text})")


def row_number() -> Expr:
    return Expr(lambda ctx: over(pl.int_range(1, pl.len() + 1, dtype=pl.Int64), ctx), "row_number()")


def _rank(name: str, method: str):
    def rank(x: Any) -> Expr:
        e = col_arg(x)
        return Expr(
            lambda ctx: over(e._compile(ctx).rank(method, descending=e._is_desc).cast(pl.Int64), ctx),
            f"{name}({e._text})",
        )

    rank.__name__ = name
    return rank


min_rank = _rank("min_rank", "min")
dense_rank = _rank("dense_rank", "dense")


def percent_rank(x: Any) -> Expr:
    """Rank rescaled to [0, 1]: (min_rank - 1) / (number of non-missing values - 1)."""
    e = col_arg(x)

    def build(ctx: Ctx) -> pl.Expr:
        c = e._compile(ctx)
        return over((c.rank("min", descending=e._is_desc) - 1) / (c.count() - 1), ctx)

    return Expr(build, f"percent_rank({e._text})")


def desc(x: Any) -> Expr:
    """Mark a sort key (in arrange, or a ranking function) as descending."""
    e = col_arg(x)
    return Expr(e._build, f"desc({e._text})", is_desc=True)


# ---------------------------------------------------------------------
# elementwise
# ---------------------------------------------------------------------


def _elementwise(name: str, fn: Callable[..., pl.Expr]):
    def f(x: Any, *args: Any) -> Expr:
        e = col_arg(x)
        return Expr(lambda ctx: fn(e._compile(ctx), *args), f"{name}({e._text})")

    f.__name__ = name
    return f


abs = _elementwise("abs", lambda c: c.abs())  # noqa: A001
sqrt = _elementwise("sqrt", lambda c: c.sqrt())
exp = _elementwise("exp", lambda c: c.exp())
log = _elementwise("log", lambda c, *base: c.log(*base))
floor = _elementwise("floor", lambda c: c.floor())
ceiling = _elementwise("ceiling", lambda c: c.ceil())
round = _elementwise("round", lambda c, digits=0: c.round(digits))  # noqa: A001


def is_na(x: Any) -> Expr:
    return col_arg(x).is_na()


def not_na(x: Any) -> Expr:
    e = col_arg(x)
    return Expr(lambda ctx: e._compile(ctx).is_not_null(), f"not_na({e._text})")


def na_if(x: Any, y: Any) -> Expr:
    """Replace values equal to `y` with a missing value."""
    e, other = col_arg(x), as_expr(y)

    def build(ctx: Ctx) -> pl.Expr:
        c = e._compile(ctx)
        return pl.when(c == other._compile(ctx)).then(None).otherwise(c)

    return Expr(build, f"na_if({e._text}, {other._text})")


def near(x: Any, y: Any, tol: float = 1.5e-8) -> Expr:
    """Whether two numbers are equal within a tolerance."""
    a, b = col_arg(x), as_expr(y)
    return Expr(
        lambda ctx: (a._compile(ctx) - b._compile(ctx)).abs() < tol,
        f"near({a._text}, {b._text})",
    )


def pick(*selectors: Any) -> Expr:
    """
    The selected columns as one struct value per row, for use where a single
    expression is expected: n_distinct(pick("a", "b")), arrange(pick(...)).
    """
    from .col import resolve_selectors

    def build(ctx: Ctx) -> pl.Expr:
        if ctx.schema is None:
            raise TranslationError("pick() needs a frame schema; use it inside a Frame verb")
        names = [c for c in resolve_selectors(ctx.schema, selectors) if c not in ctx.groups]
        if not names:
            raise TranslationError("pick() selected no columns")
        return pl.struct(names)

    return Expr(build, "pick(...)")


def is_in(x: Any, values: Iterable[Any]) -> Expr:
    return col_arg(x).is_in(values)


def between(x: Any, left: Any, right: Any) -> Expr:
    e, lo, hi = col_arg(x), as_expr(left), as_expr(right)
    return Expr(
        lambda ctx: e._compile(ctx).is_between(lo._compile(ctx), hi._compile(ctx)),
        f"between({e._text}, {lo._text}, {hi._text})",
    )


def coalesce(*xs: Any) -> Expr:
    es = [col_arg(x) for x in xs]
    return Expr(lambda ctx: pl.coalesce([e._compile(ctx) for e in es]), "coalesce(...)")


def if_else(condition: Any, true: Any, false: Any) -> Expr:
    """Vectorised if; a missing condition gives a missing result, as in dplyr."""
    c, t, f = as_expr(condition), as_expr(true), as_expr(false)

    def build(ctx: Ctx) -> pl.Expr:
        cond = c._compile(ctx)
        return pl.when(cond.is_null()).then(None).when(cond).then(t._compile(ctx)).otherwise(f._compile(ctx))

    return Expr(build, f"if_else({c._text}, {t._text}, {f._text})")


def case_when(*cases: Tuple[Any, Any], _default: Any = None) -> Expr:
    """case_when((cond, value), (cond, value), _default=value): first matching condition wins."""
    if not cases:
        raise TranslationError("case_when() needs at least one (condition, value) pair")
    pairs = []
    for case in cases:
        if not (isinstance(case, tuple) and len(case) == 2):
            raise TranslationError("case_when() takes (condition, value) tuples")
        pairs.append((as_expr(case[0]), as_expr(case[1])))
    default = as_expr(_default)

    def build(ctx: Ctx) -> pl.Expr:
        chain: Any = pl
        for cond, value in pairs:
            chain = chain.when(cond._compile(ctx)).then(value._compile(ctx))
        return chain.otherwise(default._compile(ctx))

    return Expr(build, "case_when(...)")


# Names callable from string expressions, e.g. mutate(r="x / sum(x)").
REGISTRY: Dict[str, Callable[..., Expr]] = {
    "mean": mean, "sum": sum, "min": min, "max": max, "median": median,
    "sd": sd, "var": var, "first": first, "last": last, "any": any, "all": all,
    "n": n, "n_distinct": n_distinct,
    "lag": lag, "lead": lead, "rolling_mean": rolling_mean, "cumsum": cumsum,
    "row_number": row_number, "min_rank": min_rank, "dense_rank": dense_rank,
    "percent_rank": percent_rank, "desc": desc,
    "abs": abs, "sqrt": sqrt, "exp": exp, "log": log, "floor": floor,
    "ceiling": ceiling, "round": round,
    "is_na": is_na, "not_na": not_na, "na_if": na_if, "near": near,
    "is_in": is_in, "between": between, "coalesce": coalesce,
    "if_else": if_else, "lit": lit,
}

# Function names accepted in the legacy summarise(name=("col", "fn")) form.
SUMMARY_NAMES: Dict[str, Callable[..., Expr]] = {
    "mean": mean, "sum": sum, "min": min, "max": max, "median": median,
    "std": sd, "sd": sd, "var": var, "first": first, "last": last,
    "count": _count_non_null, "nunique": n_distinct, "n_distinct": n_distinct,
    "size": lambda _x: n(),
}


# ---------------------------------------------------------------------
# across()
# ---------------------------------------------------------------------


class Across:
    """Result of across(): expands to one named expression per column and function."""

    __slots__ = ("cols", "fns", "names")

    def __init__(self, cols: Any, fns: Any, names: Optional[str]):
        self.cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        several = isinstance(fns, (dict, list, tuple))
        if fns is None:
            pairs = [(None, lambda x: x)]
        elif isinstance(fns, dict):
            pairs = list(fns.items())
        elif several:
            pairs = [(self._fn_name(f, i), f) for i, f in enumerate(fns, start=1)]
        else:
            pairs = [(self._fn_name(fns, 1), fns)]
        self.fns = [(name, self._resolve_fn(f)) for name, f in pairs]
        self.names = names if names is not None else ("{col}_{fn}" if several else "{col}")

    @staticmethod
    def _fn_name(fn: Any, position: int) -> str:
        if isinstance(fn, str):
            return fn
        name = getattr(fn, "__name__", "")
        return name if name.isidentifier() else str(position)

    @staticmethod
    def _resolve_fn(fn: Any) -> Callable[[Expr], Any]:
        if isinstance(fn, str):
            if fn not in REGISTRY:
                raise TranslationError(f"across(): unknown function {fn!r}")
            return REGISTRY[fn]
        if not callable(fn):
            raise TranslationError(f"across(): expected a function or function name, got {type(fn)!r}")
        return fn

    def expand(self, schema: Any, groups: Tuple[str, ...]) -> list:
        from .col import resolve_selectors
        from .expr import ColumnExpr

        columns = [c for c in resolve_selectors(schema, self.cols) if c not in groups]
        out = []
        for column in columns:
            for fn_name, fn in self.fns:
                result = fn(ColumnExpr(column))
                if not isinstance(result, Expr):
                    raise TranslationError(
                        "across(): the function must return a crowley_frame expression "
                        f"(got {type(result)!r}); Python functions are not run on column data"
                    )
                out.append((self.names.format(col=column, fn=fn_name), result))
        names = [name for name, _ in out]
        if len(set(names)) != len(names):
            raise TranslationError(
                f"across(): _names={self.names!r} gives duplicate column names; use both {{col}} and {{fn}}"
            )
        return out


def across(cols: Any, fns: Any = None, _names: Optional[str] = None) -> Across:
    """
    Apply functions to several columns, inside mutate(), transmute() or summarise().

        summarise(across(where(is_numeric), mean))
        summarise(across(starts_with("score"), {"mean": mean, "sd": sd}))   # score_a_mean, ...
        mutate(across(["a", "b"], lambda x: x - lag(x), _names="{col}_diff"))

    `cols` is a selector, a column name, or a list of them. `fns` is a
    function, a function name, a list of either, or a {name: function} dict.
    `_names` is a template using {col} and {fn}; it defaults to "{col}" for a
    single function and "{col}_{fn}" otherwise. Grouping columns are skipped.
    """
    return Across(cols, fns, _names)
