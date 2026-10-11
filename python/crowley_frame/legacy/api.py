from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa


# Try to import the Rust extension module, but fall back to pure Python if unavailable
try:
    from .. import _crowley  # type: ignore
    # Check if _crowley is actually the Rust module by verifying it has the Frame class
    # with from_pandas method, not Python's built-in Frame type
    if hasattr(_crowley, 'Frame') and hasattr(_crowley.Frame, 'from_pandas'):
        _CROWLEY_AVAILABLE = True
    else:
        _crowley = None  # type: ignore
        _CROWLEY_AVAILABLE = False
except (ImportError, ModuleNotFoundError, AttributeError):
    _crowley = None  # type: ignore
    _CROWLEY_AVAILABLE = False


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------


def _to_pandas_dict(obj: Mapping[str, Sequence[Any]]) -> Dict[str, List[Any]]:
    return {k: list(v) for k, v in obj.items()}


def _clean_column_name(name: str) -> str:
    return str(name)


def _select_columns_from_selectors(
    pdf: pd.DataFrame,
    selectors: Sequence[Any],
) -> List[str]:
    """
    Interpret crowley_frame.col selectors (and plain strings) against a pandas DataFrame.
    We assume selector objects have at least `kind` and `value` attributes,
    matching what `selectors.rs` expects.
    """
    cols = list(pdf.columns)

    # If no selectors, return all columns
    if not selectors:
        return cols

    selected: List[str] = []

    def add_unique(col_name: str) -> None:
        if col_name in cols and col_name not in selected:
            selected.append(col_name)

    for obj in selectors:
        # Plain string column name
        if isinstance(obj, str):
            add_unique(obj)
            continue

        # Accept selector-like objects with .kind / .value
        kind = getattr(obj, "kind", None)
        value = getattr(obj, "value", None)

        if kind is None:
            raise TypeError(f"Unsupported selector type: {type(obj)!r}")

        if kind == "name":
            add_unique(str(value))
        elif kind == "starts_with":
            prefix = str(value)
            for c in cols:
                if str(c).startswith(prefix):
                    add_unique(c)
        elif kind == "ends_with":
            suffix = str(value)
            for c in cols:
                if str(c).endswith(suffix):
                    add_unique(c)
        elif kind == "contains":
            needle = str(value)
            for c in cols:
                if needle in str(c):
                    add_unique(c)
        elif kind == "matches":
            import re

            pattern = re.compile(str(value))
            for c in cols:
                if pattern.search(str(c)):
                    add_unique(c)
        elif kind == "where_numeric":
            from pandas.api.types import is_numeric_dtype

            for c in cols:
                if is_numeric_dtype(pdf[c]):
                    add_unique(c)
        elif kind == "where_string":
            from pandas.api.types import is_string_dtype

            for c in cols:
                if is_string_dtype(pdf[c]):
                    add_unique(c)
        else:
            raise ValueError(f"Unknown selector kind: {kind!r}")

    return selected


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def df(obj: Any) -> "Frame":
    """Create a Frame from pandas, polars, pyarrow, dict-like, etc."""
    if isinstance(obj, Frame):
        return obj
    if isinstance(obj, pd.DataFrame):
        return Frame.from_pandas(obj)
    if isinstance(obj, pl.DataFrame):
        return Frame.from_polars(obj)
    if isinstance(obj, pa.Table):
        return Frame.from_arrow(obj)
    if isinstance(obj, Mapping):
        return Frame.from_pandas(pd.DataFrame(_to_pandas_dict(obj)))
    raise TypeError(f"Unsupported input type for df(): {type(obj)!r}")


@dataclass(frozen=True)
class _LagSpec:
    col: str
    n: int = 1
    default: Any = pd.NA


@dataclass(frozen=True)
class _LeadSpec:
    col: str
    n: int = 1
    default: Any = pd.NA


@dataclass(frozen=True)
class _RollingMeanSpec:
    col: str
    window: int
    min_periods: Optional[int] = None


def lag(col: str, n: int = 1, default: Any = pd.NA) -> _LagSpec:
    return _LagSpec(col=col, n=n, default=default)


def lead(col: str, n: int = 1, default: Any = pd.NA) -> _LeadSpec:
    return _LeadSpec(col=col, n=n, default=default)


def rolling_mean(col: str, window: int, min_periods: Optional[int] = None) -> _RollingMeanSpec:
    return _RollingMeanSpec(col=col, window=window, min_periods=min_periods)


class Frame:
    """
    Python-facing wrapper around the Rust core.

    Internally stores a Rust-backed frame (if available) but most v0.1 verbs
    are implemented against pandas for now.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def __rshift__(self, other: Any) -> Any:
        """Pipe operator support: Frame >> callable"""
        if not callable(other):
            raise TypeError(f"unsupported operand type(s) for >>: 'Frame' and {type(other)!r}")
        return other(self)

    # -----------------------------
    # Constructors / converters
    # -----------------------------
    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> "Frame":
        if _CROWLEY_AVAILABLE:
            inner = _crowley.Frame.from_pandas(pdf)
        else:
            # Pure Python fallback: store the pandas DataFrame directly
            inner = pdf.copy()
        return Frame(inner)

    @staticmethod
    def from_polars(pldf: pl.DataFrame) -> "Frame":
        return Frame.from_pandas(pldf.to_pandas())

    @staticmethod
    def from_arrow(tbl: pa.Table) -> "Frame":
        return Frame.from_pandas(tbl.to_pandas())

    def to_pandas(self) -> pd.DataFrame:
        if _CROWLEY_AVAILABLE and hasattr(self._inner, 'to_pandas'):
            return self._inner.to_pandas()
        else:
            # Pure Python fallback: inner is already a pandas DataFrame
            return self._inner.copy()

    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self.to_pandas())

    def to_arrow(self) -> pa.Table:
        return pa.Table.from_pandas(self.to_pandas())

    # -----------------------------
    # Core verbs (v0.1 + v0.2 chunks)
    # -----------------------------
    def select(self, *selectors: Any) -> "Frame":
        pdf = self.to_pandas().copy()

        # Flatten: allow either varargs or a single list/tuple
        if len(selectors) == 1 and isinstance(selectors[0], (list, tuple)):
            sel_seq = list(selectors[0])
        else:
            sel_seq = list(selectors)

        cols = _select_columns_from_selectors(pdf, sel_seq)
        out = pdf[cols].copy()
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    def mutate(self, **kwargs: Any) -> "Frame":
        pdf = self.to_pandas().copy()

        for new_col, expr in kwargs.items():
            if isinstance(expr, str):
                # Use pandas.eval for simple expressions; falls back to python eval if needed.
                try:
                    pdf[new_col] = pdf.eval(expr)
                except Exception:
                    env: Dict[str, Any] = {c: pdf[c] for c in pdf.columns}
                    env["np"] = np
                    pdf[new_col] = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
            elif isinstance(expr, _LagSpec):
                s = pdf[expr.col].shift(expr.n)
                if expr.default is not pd.NA:
                    s = s.fillna(expr.default)
                pdf[new_col] = s
            elif isinstance(expr, _LeadSpec):
                s = pdf[expr.col].shift(-expr.n)
                if expr.default is not pd.NA:
                    s = s.fillna(expr.default)
                pdf[new_col] = s
            elif isinstance(expr, _RollingMeanSpec):
                mp = expr.min_periods if expr.min_periods is not None else expr.window
                pdf[new_col] = pdf[expr.col].rolling(window=expr.window, min_periods=mp).mean()
            else:
                pdf[new_col] = expr

        pdf = pdf.reset_index(drop=True)
        return Frame.from_pandas(pdf)

    def filter(self, expr: str) -> "Frame":
        pdf = self.to_pandas().copy()
        try:
            mask = pdf.eval(expr)
        except Exception:
            env: Dict[str, Any] = {c: pdf[c] for c in pdf.columns}
            env["np"] = np
            mask = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
        out = pdf.loc[mask].reset_index(drop=True)
        return Frame.from_pandas(out)

    def arrange(self, *cols: str) -> "Frame":
        pdf = self.to_pandas().copy()
        by: List[str] = []
        ascending: List[bool] = []

        for c in cols:
            if c.startswith("-"):
                by.append(c[1:])
                ascending.append(False)
            else:
                by.append(c)
                ascending.append(True)

        out = pdf.sort_values(by=by, ascending=ascending, kind="mergesort").reset_index(drop=True)
        return Frame.from_pandas(out)

    def rename(self, **mapping: str) -> "Frame":
        pdf = self.to_pandas().copy()
        # mapping is new_name=old_name; pandas wants old->new
        inv = {old: new for new, old in mapping.items()}
        out = pdf.rename(columns=inv).reset_index(drop=True)
        return Frame.from_pandas(out)

    def relocate(self, *cols: str, before: str | None = None, after: str | None = None) -> "Frame":
        pdf = self.to_pandas().copy()
        all_cols = list(pdf.columns)

        move = [c for c in cols]
        for c in move:
            if c not in all_cols:
                raise KeyError(f"Column {c!r} not found in DataFrame")

        remaining = [c for c in all_cols if c not in move]

        if before is not None and after is not None:
            raise ValueError("Provide only one of before= or after=")

        if before is not None:
            if before not in remaining:
                raise KeyError(f"Column {before!r} not found in DataFrame")
            idx = remaining.index(before)
            new_order = remaining[:idx] + move + remaining[idx:]
        elif after is not None:
            if after not in remaining:
                raise KeyError(f"Column {after!r} not found in DataFrame")
            idx = remaining.index(after) + 1
            new_order = remaining[:idx] + move + remaining[idx:]
        else:
            new_order = move + remaining

        out = pdf[new_order].reset_index(drop=True)
        return Frame.from_pandas(out)

    def distinct(self, *cols: str, keep: str = "first") -> "Frame":
        pdf = self.to_pandas().copy()

        if keep not in ("first", "last", False):
            raise ValueError(f"keep must be 'first', 'last', or False, got {keep!r}")

        if cols:
            out = pdf.drop_duplicates(subset=list(cols), keep=keep)
        else:
            out = pdf.drop_duplicates(keep=keep)

        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    # -----------------------------
    # group_by / summarise
    # -----------------------------
    def group_by(self, *cols: str) -> "GroupedFrame":
        return GroupedFrame(self, list(cols))

    # -----------------------------
    # count()
    # -----------------------------
    def count(
        self,
        *cols: str,
        sort: bool = False,
        prop: bool = False,
        name: str = "n",
    ) -> "Frame":
        """
        Tidyverse-style count().

        Examples
        --------
        cf.count("grp", sort=True, prop=True)
        """
        pdf = self.to_pandas().copy()
        temp_col = "_crowley_n"
        used_temp = False

        # If no grouping columns provided, use a temporary column of 1s
        # and group by that. We guard against accidental collisions.
        if not cols:
            if temp_col in pdf.columns:
                raise ValueError(
                    f"Temporary column {temp_col!r} already exists; "
                    "pass explicit columns to count() instead."
                )
            cols = (temp_col,)
            pdf[temp_col] = 1
            used_temp = True

        group_cols = list(cols)
        grouped = pdf.groupby(group_cols, dropna=False)

        out = grouped.size().reset_index(name=name)

        if used_temp:
            # We only used temp_col as a dummy; drop it from the final result
            out = out.drop(columns=[temp_col])

        if sort:
            out = out.sort_values(by=name, ascending=False)

        if prop:
            total = float(out[name].sum())
            out["prop"] = out[name] / total

        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

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
        pdf = self.to_pandas().copy()
        if column not in pdf.columns:
            raise KeyError(f"Column {column!r} not found in DataFrame")

        split = pdf[column].astype("string").str.split(sep, expand=True, regex=True)

        k = len(into)
        if split.shape[1] < k:
            if fill == "right":
                for _ in range(k - split.shape[1]):
                    split[split.shape[1]] = pd.NA
            elif fill == "left":
                for _ in range(k - split.shape[1]):
                    split.insert(0, -1, pd.NA)  # temp col name
                split.columns = range(split.shape[1])
            elif fill == "warn":
                # pad on right, but warn-like behavior
                for _ in range(k - split.shape[1]):
                    split[split.shape[1]] = pd.NA
            else:
                raise ValueError("fill must be 'warn', 'left', or 'right'")

        if split.shape[1] > k:
            if extra == "drop":
                split = split.iloc[:, :k]
            elif extra == "merge":
                # merge extras into last column
                head = split.iloc[:, : k - 1]
                tail = split.iloc[:, k - 1 :].astype("string").agg(lambda r: "".join([x if x is not pd.NA else "" for x in r]), axis=1)
                split = head.copy()
                split[k - 1] = tail
            elif extra == "warn":
                split = split.iloc[:, :k]
            else:
                raise ValueError("extra must be 'warn', 'drop', or 'merge'")

        split = split.iloc[:, :k]
        split.columns = list(into)

        if convert:
            for c in split.columns:
                split[c] = pd.to_numeric(split[c], errors="ignore")

        for c in split.columns:
            pdf[c] = split[c]

        if remove:
            pdf = pdf.drop(columns=[column])

        pdf = pdf.reset_index(drop=True)
        return Frame.from_pandas(pdf)

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
            * False (default): if ANY source column is NA in a row, result is NA
            * True: drop NA values in that row before joining; if all NA -> NA
        """
        import math

        pdf = self.to_pandas().copy()
        cols = list(columns)

        for c in cols:
            if c not in pdf.columns:
                raise KeyError(f"Column {c!r} not found in DataFrame")

        sub = pdf[cols]

        def _is_missing(val: Any) -> bool:
            # Try to robustly detect "missingness" even if values became strings
            if val is None or val is pd.NA:
                return True
            if isinstance(val, float) and math.isnan(val):
                return True
            if isinstance(val, str) and val.strip().lower() in {"none", "nan", ""}:
                return True
            return False

        if na_rm:
            # Drop NAs row-wise; if all missing -> NA
            def combine_row_rm(row: pd.Series) -> Any:
                values = [v for v in row if not _is_missing(v)]
                if not values:
                    return pd.NA
                return sep.join(str(v) for v in values)

            pdf[new_column] = sub.apply(combine_row_rm, axis=1)
        else:
            # If ANY missing -> entire result is NA
            def combine_row_strict(row: pd.Series) -> Any:
                if any(_is_missing(v) for v in row):
                    return pd.NA
                return sep.join(str(v) for v in row)

            pdf[new_column] = sub.apply(combine_row_strict, axis=1)

        if remove:
            pdf = pdf.drop(columns=cols)

        return Frame.from_pandas(pdf)

    def pivot_longer(
        self,
        *cols_args: Any,
        names_to: str,
        values_to: str,
        cols: Any = None,
    ) -> "Frame":
        """Tidyverse-style pivot_longer.

        Supports either:
          - varargs selectors: cf.pivot_longer(col.matches("^year_"), names_to="year", values_to="value")
          - a single list/tuple: cf.pivot_longer([col.starts_with("year_")], names_to=..., values_to=...)
          - keyword form for older tests: cf.pivot_longer(cols=[...], names_to=..., values_to=...)

        Parameters
        ----------
        cols_args / cols:
            Column selectors or column names to pivot into long form.
        names_to:
            Name of the new column that will contain former column names.
        values_to:
            Name of the new column that will contain values.
        """
        pdf = self.to_pandas().copy()

        # Determine which form was used
        if cols is not None:
            # Keyword form: pivot_longer(cols=[...], ...)
            sel_seq = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        elif len(cols_args) == 1 and isinstance(cols_args[0], (list, tuple)):
            # Single list/tuple: pivot_longer([...], ...)
            sel_seq = list(cols_args[0])
        else:
            # Varargs: pivot_longer(..., ..., ...)
            sel_seq = list(cols_args)

        select_cols = _select_columns_from_selectors(pdf, sel_seq)
        id_cols = [c for c in pdf.columns if c not in select_cols]

        melted = pdf.melt(
            id_vars=id_cols,
            value_vars=select_cols,
            var_name=names_to,
            value_name=values_to,
        )
        melted = melted.reset_index(drop=True)
        return Frame.from_pandas(melted)

    def pivot_wider(
        self,
        names_from: str,
        values_from: str,
        values_fill: Any = None,
    ) -> "Frame":
        pdf = self.to_pandas().copy()

        if names_from not in pdf.columns:
            raise KeyError(f"Column {names_from!r} not found in DataFrame")
        if values_from not in pdf.columns:
            raise KeyError(f"Column {values_from!r} not found in DataFrame")

        id_cols = [c for c in pdf.columns if c not in {names_from, values_from}]

        wide = pdf.pivot_table(
            index=id_cols,
            columns=names_from,
            values=values_from,
            aggfunc="first",
        )

        if values_fill is not None:
            wide = wide.fillna(values_fill)

        # Flatten columns if needed
        if isinstance(wide.columns, pd.MultiIndex):
            wide.columns = ["_".join(map(str, tup)).strip() for tup in wide.columns.values]
        else:
            wide.columns = [str(c) for c in wide.columns]

        wide = wide.reset_index()
        wide.columns.name = None
        wide = wide.reset_index(drop=True)
        return Frame.from_pandas(wide)

    # -----------------------------
    # slice helpers
    # -----------------------------
    def slice_head(self, n: int = 5) -> "Frame":
        pdf = self.to_pandas().copy()
        out = pdf.head(n).reset_index(drop=True)
        return Frame.from_pandas(out)

    def slice_tail(self, n: int = 5) -> "Frame":
        pdf = self.to_pandas().copy()
        out = pdf.tail(n).reset_index(drop=True)
        return Frame.from_pandas(out)

    def slice_sample(
        self,
        n: Optional[int] = None,
        prop: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "Frame":
        pdf = self.to_pandas().copy()
        if (n is None) == (prop is None):
            raise ValueError("Provide exactly one of n= or prop=")
        if prop is not None:
            n = int(round(len(pdf) * float(prop)))
        assert n is not None
        out = pdf.sample(n=n, replace=replace, random_state=random_state).reset_index(drop=True)
        return Frame.from_pandas(out)

    def slice_max(self, order_by: str, n: int = 1, with_ties: bool = True) -> "Frame":
        pdf = self.to_pandas().copy()
        if order_by not in pdf.columns:
            raise KeyError(f"Column {order_by!r} not found in DataFrame")

        # nlargest gives descending order
        out = pdf.nlargest(n=n, columns=order_by, keep="all" if with_ties else "first")
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    def slice_min(self, order_by: str, n: int = 1, with_ties: bool = True) -> "Frame":
        pdf = self.to_pandas().copy()
        if order_by not in pdf.columns:
            raise KeyError(f"Column {order_by!r} not found in DataFrame")

        # nsmallest gives ascending order
        out = pdf.nsmallest(n=n, columns=order_by, keep="all" if with_ties else "first")
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    # -----------------------------
    # joins
    # -----------------------------
    def left_join(
        self,
        other: "Frame",
        on: Union[str, Sequence[str], None] = None,
        left_on: Union[str, Sequence[str], None] = None,
        right_on: Union[str, Sequence[str], None] = None,
        suffixes: Tuple[str, str] = ("_x", "_y"),
    ) -> "Frame":
        left_pdf = self.to_pandas().copy()
        right_pdf = other.to_pandas().copy()
        out = left_pdf.merge(
            right_pdf,
            how="left",
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
        )
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    def inner_join(
        self,
        other: "Frame",
        on: Union[str, Sequence[str], None] = None,
        left_on: Union[str, Sequence[str], None] = None,
        right_on: Union[str, Sequence[str], None] = None,
        suffixes: Tuple[str, str] = ("_x", "_y"),
    ) -> "Frame":
        left_pdf = self.to_pandas().copy()
        right_pdf = other.to_pandas().copy()
        out = left_pdf.merge(
            right_pdf,
            how="inner",
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
        )
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)


class GroupedFrame:
    def __init__(self, frame: Frame, group_cols: List[str]):
        self._frame = frame
        self._group_cols = group_cols

    def __rshift__(self, other: Any) -> Any:
        """Pipe operator support: GroupedFrame >> callable"""
        if not callable(other):
            raise TypeError(f"unsupported operand type(s) for >>: 'GroupedFrame' and {type(other)!r}")
        return other(self)

    def summarise(
        self,
        **agg_specs: Tuple[str, Union[str, Callable[..., Any]]],
    ) -> Frame:
        pdf = self._frame.to_pandas().copy()
        gcols = list(self._group_cols)

        grouped = pdf.groupby(gcols, dropna=False)

        out_parts: Dict[str, Any] = {}
        for out_name, (col_name, fn) in agg_specs.items():
            if isinstance(fn, str):
                out_parts[out_name] = getattr(grouped[col_name], fn)()
            else:
                out_parts[out_name] = grouped[col_name].apply(fn)

        out = pd.concat(out_parts, axis=1).reset_index()
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)