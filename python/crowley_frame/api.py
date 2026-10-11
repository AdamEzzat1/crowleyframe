from __future__ import annotations

import inspect
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import polars as pl

from .col import resolve_selectors
from .expr import (
    ColumnNotFoundError,
    Ctx,
    CrowleyTranslationError,
    Expr,
    TranslationError,
    column_not_found,
    lit,
)
from .functions import SUMMARY_NAMES, Across, IntAgg, lag, lead, rolling_mean  # noqa: F401  (re-exported)
from .parser import parse_expr

# Polars renamed / added a few keyword arguments across 1.x; look them up once.
_JOIN_PARAMS = set(inspect.signature(pl.LazyFrame.join).parameters)
_NULLS_EQUAL_KW = "nulls_equal" if "nulls_equal" in _JOIN_PARAMS else "join_nulls"
_HAS_JOIN_ORDER = "maintain_order" in _JOIN_PARAMS

_ORDERS = ("dplyr", "fast")
_ROW = "__crowley_row__"
_SPLIT = "\x1f"
_REGEX_CHARS = set(r".^$*+?{}[]\|()")
_VALIDATE = {"one_to_one": "1:1", "one_to_many": "1:m", "many_to_one": "m:1", "many_to_many": "m:m"}


class _Options:
    """
    order : "dplyr" (default) or "fast".

    "dplyr" keeps the row order dplyr users can see: summarise/count sorted by
    group keys, stable arrange, distinct keeps first occurrences in place,
    joins keep left row order, pivot_longer is row-major.
    "fast" emits the plain Polars plan and makes no ordering promises beyond
    what the verb itself asks for (arrange still sorts).
    """

    order = "dplyr"


options = _Options()


def set_order(order: str) -> None:
    """Set the default ordering mode for frames that do not carry their own."""
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
    options.order = order


def _to_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return parse_expr(value)
    if isinstance(value, pl.Expr):
        return Expr(lambda ctx: value, str(value))
    if isinstance(value, (bool, int, float)) or value is None:
        return lit(value)
    try:
        series = pl.Series(value)
    except Exception:
        raise TranslationError(f"Cannot translate a value of type {type(value)!r} to a column") from None
    return lit(series)


def _as_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def df(obj: Any, order: Optional[str] = None) -> "Frame":
    """Create a lazy Frame from polars, pandas, pyarrow or dict-like data."""
    if isinstance(obj, Frame):
        return obj if order is None else obj.with_order(order)
    if isinstance(obj, pl.LazyFrame):
        return Frame(obj, order=order)
    if isinstance(obj, pl.DataFrame):
        return Frame(obj.lazy(), order=order)
    if isinstance(obj, Mapping):
        return Frame(pl.DataFrame({k: list(v) for k, v in obj.items()}).lazy(), order=order)
    module = type(obj).__module__
    if module.startswith("pandas"):
        return Frame(pl.from_pandas(obj).lazy(), order=order)
    if module.startswith("pyarrow"):
        return Frame(pl.from_arrow(obj).lazy(), order=order)
    raise TypeError(f"Unsupported input type for df(): {type(obj)!r}")


class Frame:
    """
    A tidyverse-style frame over a `polars.LazyFrame`.

    Every verb returns a new Frame holding a longer lazy plan; nothing is
    computed until collect(), to_polars(), to_pandas() or to_arrow().
    """

    __slots__ = ("_lf", "_groups", "_order", "_schema_cache", "_audit", "_expect", "_lineage")

    def __init__(self, inner: pl.LazyFrame, groups: Tuple[str, ...] = (), order: Optional[str] = None):
        if isinstance(inner, pl.DataFrame):
            inner = inner.lazy()
        if order is not None and order not in _ORDERS:
            raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
        self._lf = inner
        self._groups = tuple(groups)
        self._order = order
        self._schema_cache = None
        self._audit: Tuple[Any, ...] = ()  # verbs applied so far, see tracking.py
        self._expect: Tuple[Any, ...] = ()  # expectations attached with expect()
        self._lineage: Any = None  # column lineage, only when track_lineage() was called

    # -----------------------------
    # internals
    # -----------------------------
    def _new(self, lf: pl.LazyFrame, groups: Optional[Tuple[str, ...]] = None) -> "Frame":
        out = Frame(lf, self._groups if groups is None else groups, self._order)
        out._audit, out._expect, out._lineage = self._audit, self._expect, self._lineage
        return out

    def _dplyr(self) -> bool:
        return (self._order or options.order) == "dplyr"

    @property
    def schema(self) -> Any:
        if self._schema_cache is None:
            self._schema_cache = self._lf.collect_schema()
        return self._schema_cache

    @property
    def columns(self) -> List[str]:
        return self.schema.names()

    @property
    def groups(self) -> Tuple[str, ...]:
        return self._groups

    def _require(self, *names: str) -> None:
        schema = self.schema
        for name in names:
            if name not in schema:
                raise column_not_found(name, schema.names())

    def _over(self, e: pl.Expr) -> pl.Expr:
        return e.over(list(self._groups)) if self._groups else e

    def with_order(self, order: str) -> "Frame":
        """Same plan, different ordering mode ("dplyr" or "fast") for later verbs."""
        out = Frame(self._lf, self._groups, order)
        out._audit, out._expect, out._lineage = self._audit, self._expect, self._lineage
        return out

    def __rshift__(self, other: Any) -> Any:
        """Pipe operator support: Frame >> callable"""
        if not callable(other):
            raise TypeError(f"unsupported operand type(s) for >>: 'Frame' and {type(other)!r}")
        return other(self)

    def __repr__(self) -> str:
        head = self._lf.head(10).collect()
        grouped = f", groups: {list(self._groups)}" if self._groups else ""
        return f"# crowley Frame (lazy; preview of first 10 rows{grouped})\n{head}"

    # -----------------------------
    # Constructors / converters
    # -----------------------------
    @staticmethod
    def from_pandas(pdf: Any) -> "Frame":
        return Frame(pl.from_pandas(pdf).lazy())

    @staticmethod
    def from_polars(pldf: Union[pl.DataFrame, pl.LazyFrame]) -> "Frame":
        return Frame(pldf.lazy())

    @staticmethod
    def from_arrow(tbl: Any) -> "Frame":
        return Frame(pl.from_arrow(tbl).lazy())

    def collect(self, **kwargs: Any) -> "Frame":
        """Run the plan now and return a Frame over the result."""
        return self._new(self._lf.collect(**kwargs).lazy())

    def to_polars(self, **kwargs: Any) -> pl.DataFrame:
        return self._lf.collect(**kwargs)

    def to_lazy(self) -> pl.LazyFrame:
        """The underlying lazy plan."""
        return self._lf

    def to_pandas(self) -> Any:
        return self._lf.collect().to_pandas()

    def to_arrow(self) -> Any:
        return self._lf.collect().to_arrow()

    def explain(self, optimized: bool = True) -> str:
        """The Polars query plan this frame will run."""
        return self._lf.explain(optimized=optimized)

    def glimpse(self, n: int = 5) -> "Frame":
        """
        Print each column's name, dtype and first few values, then return the frame.

        Reads only the first `n` rows of the result (the plan still has to run
        far enough to produce them). The row count is not computed.
        """
        head = self._lf.head(n).collect()
        lines = ["Rows: ?? (lazy)", f"Columns: {head.width}"]
        if self._groups:
            lines.append(f"Groups: {', '.join(self._groups)}")
        width = max((len(c) for c in head.columns), default=0)
        for name in head.columns:
            values = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in head[name].to_list())
            lines.append(f"$ {name:<{width}} <{head.schema[name]}> {values}")
        print("\n".join(lines))
        return self

    def pull(self, column: Optional[str] = None) -> pl.Series:
        """
        Run the plan and return one column (the last one by default) as a
        Polars Series. This is an eager step: extracting values is its purpose.
        """
        name = self.columns[-1] if column is None else column
        self._require(name)
        return self._lf.select(name).collect().to_series()

    # -----------------------------
    # Core verbs
    # -----------------------------
    def select(self, *selectors: Any) -> "Frame":
        if len(selectors) == 1 and isinstance(selectors[0], (list, tuple)):
            selectors = tuple(selectors[0])
        cols = resolve_selectors(self.schema, selectors)
        cols = [g for g in self._groups if g not in cols] + cols
        return self._new(self._lf.select(cols))

    def _named(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """(name, expression) pairs from across(...) arguments followed by name=expression ones."""
        items: List[Tuple[str, Any]] = []
        for arg in args:
            if not isinstance(arg, Across):
                raise TypeError(
                    "Positional arguments must be across(...); give other expressions a name, e.g. z=col.x * 2"
                )
            items.extend(arg.expand(self.schema, self._groups))
        items.extend(kwargs.items())
        return items

    def _mutate_batches(self, items: List[Tuple[str, Any]]) -> List[List[pl.Expr]]:
        """
        Compile to as few with_columns steps as possible, splitting only where an
        expression uses a column created earlier in the same call (dplyr evaluates in order).
        """
        known = set(self.columns)
        ctx = Ctx(known, self._groups, window=True, schema=self.schema)
        batches: List[List[pl.Expr]] = []
        batch: List[pl.Expr] = []
        created: set = set()
        for name, value in items:
            e = _to_expr(value)._compile(ctx)
            if created and not created.isdisjoint(e.meta.root_names()):
                batches.append(batch)
                batch, created = [], set()
            batch.append(e.alias(name))
            created.add(name)
            known.add(name)
        if batch:
            batches.append(batch)
        return batches

    def mutate(self, *args: Any, **kwargs: Any) -> "Frame":
        lf = self._lf
        for batch in self._mutate_batches(self._named(args, kwargs)):
            lf = lf.with_columns(batch)
        return self._new(lf)

    def transmute(self, *args: Any, **kwargs: Any) -> "Frame":
        """Like mutate(), but keeps only the new columns (and any grouping columns)."""
        items = self._named(args, kwargs)
        batches = self._mutate_batches(items)
        names = list(dict.fromkeys(name for name, _ in items))
        keep: List[Any] = [g for g in self._groups if g not in names]
        if len(batches) == 1 and len(batches[0]) == len(names):
            return self._new(self._lf.select(keep + batches[0]))
        lf = self._lf
        for batch in batches:
            lf = lf.with_columns(batch)
        return self._new(lf.select(keep + names))

    def filter(self, *predicates: Any) -> "Frame":
        ctx = Ctx(set(self.columns), self._groups, window=True, schema=self.schema)
        compiled = [_to_expr(p)._compile(ctx) for p in predicates]
        return self._new(self._lf.filter(*compiled))

    def arrange(self, *keys: Any) -> "Frame":
        cols = set(self.columns)
        ctx = Ctx(cols, (), window=True, schema=self.schema)
        by: List[Any] = []
        descending: List[bool] = []

        for key in keys:
            if isinstance(key, str):
                text = key.strip()
                is_desc = text.startswith("-")
                name = text[1:].strip() if is_desc else text
                if name in cols:
                    by.append(name)
                    descending.append(is_desc)
                    continue
                if name.isidentifier():
                    raise column_not_found(name, cols)
            e = _to_expr(key)
            by.append(e._compile(ctx))
            descending.append(e._is_desc)

        if not by:
            return self
        lf = self._lf.sort(by, descending=descending, nulls_last=True, maintain_order=self._dplyr())
        return self._new(lf)

    def rename(self, **mapping: str) -> "Frame":
        # mapping is new_name=old_name; polars wants old->new
        self._require(*mapping.values())
        return self._rename({old: new for new, old in mapping.items()})

    def _rename(self, old_to_new: Dict[str, str]) -> "Frame":
        groups = tuple(old_to_new.get(g, g) for g in self._groups)
        return self._new(self._lf.rename(old_to_new), groups=groups)

    def rename_with(self, fn: Callable[[str], str], *selectors: Any) -> "Frame":
        """Rename the selected columns (all by default) with a function of the name, e.g. str.upper."""
        cols = resolve_selectors(self.schema, selectors) if selectors else self.columns
        mapping = {c: fn(c) for c in cols}
        for old, new in mapping.items():
            if not isinstance(new, str):
                raise TypeError(f"rename_with(): the function returned {new!r} for {old!r}, expected a string")
        return self._rename(mapping)

    def relocate(self, *cols: Any, before: Optional[str] = None, after: Optional[str] = None) -> "Frame":
        all_cols = self.columns
        move = resolve_selectors(self.schema, cols)

        remaining = [c for c in all_cols if c not in move]

        if before is not None and after is not None:
            raise ValueError("Provide only one of before= or after=")

        if before is not None:
            if before not in remaining:
                raise column_not_found(before, remaining)
            idx = remaining.index(before)
            new_order = remaining[:idx] + move + remaining[idx:]
        elif after is not None:
            if after not in remaining:
                raise column_not_found(after, remaining)
            idx = remaining.index(after) + 1
            new_order = remaining[:idx] + move + remaining[idx:]
        else:
            new_order = move + remaining

        return self._new(self._lf.select(new_order))

    def distinct(self, *cols: str, keep_all: bool = False) -> "Frame":
        """
        Unique rows. With columns, returns only those columns (as dplyr does)
        unless keep_all=True, which keeps the first row of each combination.
        """
        self._require(*cols)
        dplyr = self._dplyr()
        if cols and keep_all:
            lf = self._lf.unique(subset=list(cols), keep="first" if dplyr else "any", maintain_order=dplyr)
        else:
            lf = self._lf.select(list(self._groups) + [c for c in cols if c not in self._groups]) if cols else self._lf
            lf = lf.unique(keep="first" if dplyr else "any", maintain_order=dplyr)
        return self._new(lf)

    # -----------------------------
    # group_by / summarise
    # -----------------------------
    def group_by(self, *cols: str) -> "Frame":
        self._require(*cols)
        return self._new(self._lf, groups=tuple(cols))

    def ungroup(self) -> "Frame":
        return self._new(self._lf, groups=())

    def summarise(self, *args: Any, **agg_specs: Any) -> "Frame":
        """
        One row per group (or one row in total when ungrouped).

        Each named argument is an expression (`mean_x=mean(col.x)`), an expression
        string (`mean_x="mean(x)"`), or the older `("column", "function")` pair.
        Positional arguments are across(...) calls.
        """
        ctx = Ctx(set(self.columns), self._groups, window=False, schema=self.schema)
        exprs: List[pl.Expr] = []
        counts: List[str] = []

        for out_name, spec in self._named(args, agg_specs):
            if isinstance(spec, tuple) and len(spec) == 2:
                col_name, fn = spec
                if not isinstance(fn, str):
                    raise TranslationError(
                        f"summarise({out_name}=...): Python callables are not translated to Polars. "
                        "Use a crowley_frame function (e.g. mean, sd, n) or an expression string."
                    )
                if fn not in SUMMARY_NAMES:
                    raise TranslationError(
                        f"summarise({out_name}=...): unknown summary function {fn!r}; "
                        f"available: {sorted(SUMMARY_NAMES)}"
                    )
                e = SUMMARY_NAMES[fn](col_name)
            else:
                e = _to_expr(spec)
            if isinstance(e, IntAgg):
                exprs.append(e._raw(ctx).alias(out_name))
                counts.append(out_name)
            else:
                exprs.append(e._compile(ctx).alias(out_name))

        keys = list(self._groups)
        lf = self._lf.group_by(keys).agg(exprs) if keys else self._lf.select(exprs)
        if counts:
            lf = lf.with_columns(pl.col(counts).cast(pl.Int64))
        if not keys:
            return self._new(lf)
        if self._dplyr():
            lf = lf.sort(keys, nulls_last=True)
        return self._new(lf, groups=())

    summarize = summarise

    # -----------------------------
    # count()
    # -----------------------------
    def count(
        self,
        *cols: str,
        sort: bool = False,
        prop: bool = False,
        name: str = "n",
        wt: Optional[str] = None,
    ) -> "Frame":
        """
        Tidyverse-style count().

        Examples
        --------
        cf.count("grp", sort=True, prop=True)
        """
        self._require(*cols)
        keys = list(self._groups) + [c for c in cols if c not in self._groups]
        dplyr = self._dplyr()

        if wt is None:
            n_expr = pl.len()
        else:
            self._require(wt)
            n_expr = pl.col(wt).sum()

        if keys:
            lf = self._lf.group_by(keys).agg(n_expr.alias(name))
            if wt is None:
                lf = lf.with_columns(pl.col(name).cast(pl.Int64))
            if dplyr:
                lf = lf.sort(keys, nulls_last=True)
        else:
            lf = self._lf.select(n_expr.cast(pl.Int64).alias(name) if wt is None else n_expr.alias(name))

        if sort:
            lf = lf.sort(name, descending=True, maintain_order=dplyr)
        if prop:
            lf = lf.with_columns((pl.col(name) / pl.col(name).sum()).alias("prop"))

        return self._new(lf, groups=())

    def add_count(self, *cols: str, name: str = "n", wt: Optional[str] = None, sort: bool = False) -> "Frame":
        """Add a column with the number of rows in each group, keeping every row."""
        self._require(*cols)
        keys = list(self._groups) + [c for c in cols if c not in self._groups]
        if wt is None:
            # Cast the per-group count before it is broadcast to every row;
            # casting the broadcast column instead costs about 15% more.
            n_expr = pl.len().cast(pl.Int64)
        else:
            self._require(wt)
            n_expr = pl.col(wt).sum()
        if keys:
            n_expr = n_expr.over(keys)
        lf = self._lf.with_columns(n_expr.alias(name))
        if sort:
            lf = lf.sort(name, descending=True, maintain_order=self._dplyr())
        return self._new(lf)

    # -----------------------------
    # missing values
    # -----------------------------
    def drop_na(self, *selectors: Any) -> "Frame":
        """Drop rows with a missing value in the selected columns (any column by default)."""
        subset = resolve_selectors(self.schema, selectors) if selectors else None
        return self._new(self._lf.drop_nulls(subset=subset))

    def replace_na(self, values: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> "Frame":
        """Replace missing values per column: replace_na({"x": 0}) or replace_na(x=0, y="unknown")."""
        mapping = {**(dict(values) if values else {}), **kwargs}
        self._require(*mapping)
        if not mapping:
            return self
        return self._new(self._lf.with_columns([pl.col(c).fill_null(v) for c, v in mapping.items()]))

    def fill(self, *selectors: Any, direction: str = "down") -> "Frame":
        """
        Fill missing values in the selected columns from the previous ("down") or
        next ("up") value, or both ("downup", "updown"), within groups if grouped.
        """
        strategies = {
            "down": ["forward"],
            "up": ["backward"],
            "downup": ["forward", "backward"],
            "updown": ["backward", "forward"],
        }
        if direction not in strategies:
            raise ValueError(f"direction must be one of {sorted(strategies)}, got {direction!r}")
        cols = resolve_selectors(self.schema, selectors)
        if not cols:
            return self
        exprs = []
        for c in cols:
            e = pl.col(c)
            for strategy in strategies[direction]:
                e = e.fill_null(strategy=strategy)
            exprs.append(self._over(e))
        return self._new(self._lf.with_columns(exprs))

    # -----------------------------
    # tidyr-ish: separate/unite/pivot
    # -----------------------------
    def separate(
        self,
        column: str,
        into: Sequence[str],
        sep: str = r"[^A-Za-z0-9]+",
        remove: bool = True,
        convert: bool = False,
        extra: str = "warn",
        fill: str = "warn",
    ) -> "Frame":
        self._require(column)
        if extra not in ("warn", "drop", "merge"):
            raise ValueError("extra must be 'warn', 'drop', or 'merge'")
        if fill not in ("warn", "left", "right"):
            raise ValueError("fill must be 'warn', 'left', or 'right'")
        if convert:
            raise TranslationError("separate(convert=True) is not translated yet; cast the new columns in mutate()")
        if fill == "left":
            raise TranslationError("separate(fill='left') is not translated yet; short rows are padded on the right")

        into = list(into)
        k = len(into)
        src = pl.col(column).cast(pl.String)
        is_regex = bool(_REGEX_CHARS.intersection(sep))

        if extra == "merge":
            if is_regex:
                raise TranslationError(
                    "separate(extra='merge') needs a literal separator; "
                    f"{sep!r} is a regular expression"
                )
            parts = src.str.splitn(sep, k)
            pieces = [parts.struct.field(f"field_{i}").alias(name) for i, name in enumerate(into)]
        else:
            if is_regex:
                parts = src.str.replace_all(sep, _SPLIT).str.split(_SPLIT)
            else:
                parts = src.str.split(sep)
            pieces = [parts.list.get(i, null_on_oob=True).alias(name) for i, name in enumerate(into)]

        out_cols: List[Any] = []
        for c in self.columns:
            if c == column:
                if not remove and column not in into:
                    out_cols.append(c)
                out_cols.extend(pieces)
            elif c not in into:
                out_cols.append(c)
        return self._new(self._lf.select(out_cols))

    def unite(
        self,
        new_column: str,
        columns: Sequence[str],
        sep: str = "_",
        remove: bool = True,
        na_rm: bool = False,
    ) -> "Frame":
        """
        Tidyverse-style unite:

        - new_column: name of the combined column
        - columns: list of columns to concatenate
        - sep: separator string
        - remove: if True, drop the original columns after uniting
        - na_rm:
            * False (default): if ANY source column is missing in a row, result is missing
            * True: drop missing values in that row before joining; if all missing -> missing
        """
        cols = list(columns)
        self._require(*cols)
        if not cols:
            raise ValueError("unite() needs at least one column")

        parts = [pl.col(c).cast(pl.String) for c in cols]
        if na_rm:
            united = (
                pl.when(pl.all_horizontal([p.is_null() for p in parts]))
                .then(None)
                .otherwise(pl.concat_str(parts, separator=sep, ignore_nulls=True))
            )
        else:
            united = pl.concat_str(parts, separator=sep)

        out_cols: List[Any] = []
        for c in self.columns:
            if c == cols[0]:
                out_cols.append(united.alias(new_column))
            if c in cols and remove:
                continue
            if c != new_column:
                out_cols.append(c)
        return self._new(self._lf.select(out_cols))

    def pivot_longer(
        self,
        *cols_args: Any,
        names_to: str = "name",
        values_to: str = "value",
        cols: Any = None,
        values_drop_na: bool = False,
        names_prefix: Optional[str] = None,
    ) -> "Frame":
        """Tidyverse-style pivot_longer.

        Supports either:
          - varargs selectors: cf.pivot_longer(col.matches("^year_"), names_to="year", values_to="value")
          - a single list/tuple: cf.pivot_longer([col.starts_with("year_")], names_to=..., values_to=...)
          - keyword form: cf.pivot_longer(cols=[...], names_to=..., values_to=...)
        """
        if cols is not None:
            sel_seq = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        elif len(cols_args) == 1 and isinstance(cols_args[0], (list, tuple)):
            sel_seq = list(cols_args[0])
        else:
            sel_seq = list(cols_args)

        select_cols = resolve_selectors(self.schema, sel_seq)
        if not select_cols:
            raise ValueError("pivot_longer(): no columns selected to pivot")
        id_cols = [c for c in self.columns if c not in select_cols]
        for name in (names_to, values_to):
            if name in id_cols:
                raise ValueError(f"pivot_longer(): {name!r} already exists as a column")

        if self._dplyr():
            # Polars unpivots column by column; tidyr returns rows in input order.
            lf = (
                self._lf.with_row_index(_ROW)
                .unpivot(on=select_cols, index=[_ROW] + id_cols, variable_name=names_to, value_name=values_to)
                .sort(_ROW, maintain_order=True)
                .drop(_ROW)
            )
        else:
            lf = self._lf.unpivot(on=select_cols, index=id_cols, variable_name=names_to, value_name=values_to)

        if names_prefix:
            lf = lf.with_columns(pl.col(names_to).str.strip_prefix(names_prefix))
        if values_drop_na:
            lf = lf.drop_nulls(values_to)
        return self._new(lf)

    def pivot_wider(
        self,
        names_from: str,
        values_from: str,
        values_fill: Any = None,
        id_cols: Optional[Sequence[str]] = None,
        values_fn: str = "first",
    ) -> "Frame":
        """
        Tidyverse-style pivot_wider.

        The output columns depend on the data, so this verb runs the plan so
        far (it is the one eager step) and continues lazily from the result.
        """
        self._require(names_from, values_from)
        ids = list(id_cols) if id_cols is not None else [
            c for c in self.columns if c not in (names_from, values_from)
        ]
        self._require(*ids)

        data = self._lf.collect()
        dummy = not ids
        if dummy:
            data = data.with_columns(pl.lit(0).alias(_ROW))
            ids = [_ROW]

        wide = data.pivot(on=names_from, index=ids, values=values_from, aggregate_function=values_fn)
        if values_fill is not None:
            new_cols = [c for c in wide.columns if c not in ids]
            wide = wide.with_columns(pl.col(new_cols).fill_null(values_fill))
        if dummy:
            wide = wide.drop(_ROW)
        return self._new(wide.lazy())

    # -----------------------------
    # slice helpers
    # -----------------------------
    def slice_head(self, n: int = 5) -> "Frame":
        if not self._groups:
            return self._new(self._lf.head(n))
        return self._new(self._lf.filter(self._over(pl.int_range(pl.len())) < n))

    def slice_tail(self, n: int = 5) -> "Frame":
        if not self._groups:
            return self._new(self._lf.tail(n))
        position = self._over(pl.int_range(pl.len()))
        return self._new(self._lf.filter(position >= self._over(pl.len()) - n))

    def slice_sample(
        self,
        n: Optional[int] = None,
        prop: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "Frame":
        if (n is None) == (prop is None):
            raise ValueError("Provide exactly one of n= or prop=")
        if self._groups:
            raise TranslationError("slice_sample() on a grouped frame is not translated yet")
        # A fixed seed makes every column draw the same rows.
        seed = random.randrange(2**31) if random_state is None else random_state
        sampled = pl.all().sample(n=n, fraction=prop, with_replacement=replace, seed=seed)
        return self._new(self._lf.select(sampled))

    def _slice_extreme(self, order_by: str, n: int, with_ties: bool, descending: bool) -> "Frame":
        self._require(order_by)
        c = pl.col(order_by)
        if with_ties or not self._groups:
            # Keep every row at or beyond the n-th value: one top-k pass, no full sort.
            cutoff = self._over(c.top_k(n).min() if descending else c.bottom_k(n).max())
            lf = self._lf.filter(c >= cutoff if descending else c <= cutoff)
            if not with_ties:
                # Only the tied rows are left, so trimming to exactly n is cheap.
                # (Much faster than LazyFrame.top_k on the full frame, which sorts it.)
                lf = lf.top_k(n, by=order_by) if descending else lf.bottom_k(n, by=order_by)
        else:
            lf = self._lf.filter(self._over(c.rank("ordinal", descending=descending)) <= n)
        if self._dplyr() and not self._groups:
            # dplyr returns the selected rows ordered by the slicing column.
            lf = lf.sort(order_by, descending=descending, nulls_last=True, maintain_order=True)
        return self._new(lf)

    def slice_max(self, order_by: str, n: int = 1, with_ties: bool = True) -> "Frame":
        return self._slice_extreme(order_by, n, with_ties, descending=True)

    def slice_min(self, order_by: str, n: int = 1, with_ties: bool = True) -> "Frame":
        return self._slice_extreme(order_by, n, with_ties, descending=False)

    # -----------------------------
    # joins
    # -----------------------------
    def _join(
        self,
        other: Any,
        how: str,
        on: Union[str, Sequence[str], None],
        left_on: Union[str, Sequence[str], None],
        right_on: Union[str, Sequence[str], None],
        suffixes: Tuple[str, str],
        by: Union[str, Sequence[str], None],
        na_matches: str,
        validate: Optional[str],
    ) -> "Frame":
        other = df(other)
        if na_matches not in ("na", "never"):
            raise ValueError("na_matches must be 'na' or 'never'")

        left_cols, right_cols = self.columns, other.columns
        keys = _as_list(by if by is not None else on)
        if keys:
            left_keys = right_keys = keys
        elif left_on is not None or right_on is not None:
            left_keys, right_keys = _as_list(left_on), _as_list(right_on)
            if len(left_keys) != len(right_keys) or not left_keys:
                raise ValueError("left_on and right_on must name the same number of columns")
        else:
            left_keys = right_keys = [c for c in left_cols if c in right_cols]
            if not left_keys:
                raise ValueError("No common columns to join on; pass on= or left_on=/right_on=")
        self._require(*left_keys)
        other._require(*right_keys)

        left_lf, right_lf = self._lf, other._lf
        filtering = how in ("semi", "anti")

        if not filtering:
            # Polars only suffixes the right side; rename both, as dplyr/pandas do.
            overlap = [c for c in left_cols if c in right_cols and c not in left_keys and c not in right_keys]
            if overlap:
                left_lf = left_lf.rename({c: c + suffixes[0] for c in overlap})
                right_lf = right_lf.rename({c: c + suffixes[1] for c in overlap})

        kwargs: Dict[str, Any] = {"how": how}
        if left_keys == right_keys:
            kwargs["on"] = left_keys
        else:
            kwargs["left_on"], kwargs["right_on"] = left_keys, right_keys
        if na_matches == "na":
            kwargs[_NULLS_EQUAL_KW] = True
        if not filtering and (how == "full" or left_keys != right_keys):
            kwargs["coalesce"] = True
        if validate is not None:
            kwargs["validate"] = _VALIDATE.get(validate, validate)

        restore_order = False
        if self._dplyr() and not filtering:
            if _HAS_JOIN_ORDER:
                kwargs["maintain_order"] = "right" if how == "right" else "left"
            elif how == "inner":
                # Older Polars cannot ask the join to keep left order.
                left_lf = left_lf.with_row_index(_ROW)
                restore_order = True

        lf = left_lf.join(right_lf, **kwargs)
        if restore_order:
            lf = lf.sort(_ROW, maintain_order=True).drop(_ROW)
        return self._new(lf)

    def left_join(self, other, on=None, left_on=None, right_on=None, suffixes=("_x", "_y"),
                  by=None, na_matches="na", validate=None) -> "Frame":
        return self._join(other, "left", on, left_on, right_on, suffixes, by, na_matches, validate)

    def inner_join(self, other, on=None, left_on=None, right_on=None, suffixes=("_x", "_y"),
                   by=None, na_matches="na", validate=None) -> "Frame":
        return self._join(other, "inner", on, left_on, right_on, suffixes, by, na_matches, validate)

    def right_join(self, other, on=None, left_on=None, right_on=None, suffixes=("_x", "_y"),
                   by=None, na_matches="na", validate=None) -> "Frame":
        return self._join(other, "right", on, left_on, right_on, suffixes, by, na_matches, validate)

    def full_join(self, other, on=None, left_on=None, right_on=None, suffixes=("_x", "_y"),
                  by=None, na_matches="na", validate=None) -> "Frame":
        return self._join(other, "full", on, left_on, right_on, suffixes, by, na_matches, validate)

    def cross_join(self, other: Any, suffixes: Tuple[str, str] = ("_x", "_y")) -> "Frame":
        """Every row of this frame paired with every row of `other`."""
        other = df(other)
        left_lf, right_lf = self._lf, other._lf
        right_cols = set(other.columns)
        overlap = [c for c in self.columns if c in right_cols]
        if overlap:
            left_lf = left_lf.rename({c: c + suffixes[0] for c in overlap})
            right_lf = right_lf.rename({c: c + suffixes[1] for c in overlap})
        return self._new(left_lf.join(right_lf, how="cross"))

    def semi_join(self, other, on=None, left_on=None, right_on=None, by=None, na_matches="na") -> "Frame":
        return self._join(other, "semi", on, left_on, right_on, ("", ""), by, na_matches, None)

    def anti_join(self, other, on=None, left_on=None, right_on=None, by=None, na_matches="na") -> "Frame":
        return self._join(other, "anti", on, left_on, right_on, ("", ""), by, na_matches, None)


# A grouped frame is a Frame carrying group columns; kept as a name for old code.
GroupedFrame = Frame

__all__ = [
    "Frame",
    "GroupedFrame",
    "df",
    "options",
    "set_order",
    "lag",
    "lead",
    "rolling_mean",
    "CrowleyTranslationError",
    "TranslationError",
    "ColumnNotFoundError",
]
