from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from . import _crowley  # type: ignore


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------


def _to_pandas_dict(obj: Mapping[str, Sequence[Any]]) -> Dict[str, List[Any]]:
    return {k: list(v) for k, v in obj.items()}


def _clean_column_name(name: str) -> str:
    """Simple janitor/clean_names style normalization."""
    import re

    s = str(name).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s.lower()


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

    # Normalize selectors so we can handle:
    # - a single ColSelector (col.matches(...))
    # - a list/tuple of selectors or strings
    # - a single string column name
    if selectors is None:
        selectors_list: List[Any] = []
    elif isinstance(selectors, (list, tuple)):
        selectors_list = list(selectors)
    else:
        selectors_list = [selectors]

    # If no selectors, return all columns
    if not selectors_list:
        return cols

    selected: List[str] = []

    def add_unique(col_name: str) -> None:
        if col_name in cols and col_name not in selected:
            selected.append(col_name)

    for obj in selectors_list:
        # Plain string => direct column name
        if isinstance(obj, str):
            add_unique(obj)
            continue

        # Anything with a 'kind' attribute we treat as a selector
        kind = getattr(obj, "kind", None)
        value = getattr(obj, "value", None)

        if kind is None:
            # unknown object: ignore
            continue

        if kind == "name" and isinstance(value, str):
            # Exact name selection from col("name")
            add_unique(value)
        elif kind == "starts_with" and isinstance(value, str):
            for c in cols:
                if c.startswith(value):
                    add_unique(c)
        elif kind == "ends_with" and isinstance(value, str):
            for c in cols:
                if c.endswith(value):
                    add_unique(c)
        elif kind == "contains" and isinstance(value, str):
            for c in cols:
                if value in c:
                    add_unique(c)
        elif kind == "matches" and isinstance(value, str):
            import re

            pattern = re.compile(value)
            for c in cols:
                if pattern.search(c):
                    add_unique(c)
        elif kind == "where_numeric":
            for c in cols:
                if pd.api.types.is_numeric_dtype(pdf[c].dtype):
                    add_unique(c)
        elif kind == "where_string":
            for c in cols:
                if pd.api.types.is_string_dtype(pdf[c].dtype):
                    add_unique(c)

    # If no columns matched, just return empty list (consistent with tidyselect)
    return selected


# --------------------------------------------------------------------
# Expression helpers (lag, lead, rolling_mean) for mutate
# --------------------------------------------------------------------


@dataclass
class LagExpr:
    column: str
    n: int = 1


@dataclass
class LeadExpr:
    column: str
    n: int = 1


@dataclass
class RollingMeanExpr:
    column: str
    window: int
    min_periods: int = 1


def lag(column: str, n: int = 1) -> LagExpr:
    return LagExpr(column=column, n=n)


def lead(column: str, n: int = 1) -> LeadExpr:
    return LeadExpr(column=column, n=n)


def rolling_mean(column: str, window: int, min_periods: int = 1) -> RollingMeanExpr:
    return RollingMeanExpr(column=column, window=window, min_periods=min_periods)


# --------------------------------------------------------------------
# Core Frame types
# --------------------------------------------------------------------


class Frame:
    """
    High-level API wrapper around the Rust core `_crowley.Frame`.

    We use the Rust side primarily as a storage engine via from_dict / to_dict,
    and implement higher-level verbs in Python (pandas).

    For v0.1, we also keep a pandas cache (`_pdf_cache`) so NA semantics
    and complex operations behave exactly as the Python tests expect.
    """

    def __init__(self, inner: _crowley.Frame, pdf_cache: Optional[pd.DataFrame] = None):
        self._inner = inner
        self._pdf_cache: Optional[pd.DataFrame] = pdf_cache

    # --------------------------------------------------------------
    # Constructors / converters
    # --------------------------------------------------------------
    @classmethod
    def from_pandas(cls, pdf: pd.DataFrame) -> "Frame":
        data = pdf.to_dict(orient="list")
        inner = _crowley.Frame.from_dict(data)  # type: ignore[attr-defined]
        # cache the pandas view so NA behavior is preserved
        return cls(inner, pdf.copy())

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> "Frame":
        pdf = df.to_pandas()
        return cls.from_pandas(pdf)

    def to_dict(self) -> Dict[str, List[Any]]:
        if self._pdf_cache is not None:
            return {c: self._pdf_cache[c].tolist() for c in self._pdf_cache.columns}
        return self._inner.to_dict()

    def to_pandas(self) -> pd.DataFrame:
        if self._pdf_cache is not None:
            return self._pdf_cache.copy()
        return pd.DataFrame(self._inner.to_dict())

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self.to_dict())

    def to_numpy(self):
        return self.to_pandas().to_numpy()

    def to_arrow(self):
        return pa.Table.from_pandas(self.to_pandas())

    # Nicely reuse the Rust / Polars __repr__ if available
    def __repr__(self) -> str:
        try:
            return repr(self._inner)
        except Exception:
            return repr(self.to_pandas())

    # --------------------------------------------------------------
    # Pipe operator (Frame >> f)
    # --------------------------------------------------------------
    def __rshift__(self, other: Any) -> "Frame":
        """
        Enable pipe syntax:

            cf >> pipe.group_by("user_id") >> pipe.summarise(...)

        where `pipe.group_by("user_id")` returns a callable that
        takes a Frame and returns a GroupedFrame, and
        `pipe.summarise(...)` takes a GroupedFrame and returns a Frame.
        """
        if callable(other):
            result = other(self)
            if isinstance(result, (Frame, GroupedFrame)):
                return result  # type: ignore[return-value]
            raise TypeError(
                f"Pipe function returned unsupported type: {type(result)!r}"
            )
        raise TypeError(f"unsupported operand type(s) for >>: 'Frame' and {type(other)!r}")

    # --------------------------------------------------------------
    # Basic verbs (Python implementations)
    # --------------------------------------------------------------
    def clean_names(self) -> "Frame":
        pdf = self.to_pandas().copy()
        pdf.columns = [_clean_column_name(c) for c in pdf.columns]
        return Frame.from_pandas(pdf)

    def select(self, *selectors: Any) -> "Frame":
        pdf = self.to_pandas().copy()
        # Flatten selectors in case we get a single list/tuple
        if len(selectors) == 1 and isinstance(selectors[0], (list, tuple)):
            sel_seq = list(selectors[0])
        else:
            sel_seq = list(selectors)

        cols = _select_columns_from_selectors(pdf, sel_seq)
        out = pdf.loc[:, cols]
        return Frame.from_pandas(out)

    def filter(self, expr: str) -> "Frame":
        """
        Simple row filter using pandas.query.

        Example: cf.filter("user_score > 10 and user_id != 3")
        """
        pdf = self.to_pandas().copy()
        out = pdf.query(expr, engine="python")
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    def arrange(self, expr: Union[str, Sequence[str]]) -> "Frame":
        """
        Sort rows by one or more columns.

        expr can be:
        - "col" for ascending
        - "-col" for descending
        - list/tuple of such strings
        """
        pdf = self.to_pandas().copy()

        if isinstance(expr, str):
            exprs = [expr]
        else:
            exprs = list(expr)

        by: List[str] = []
        ascending: List[bool] = []

        for e in exprs:
            if e.startswith("-"):
                by.append(e[1:])
                ascending.append(False)
            else:
                by.append(e)
                ascending.append(True)

        out = pdf.sort_values(by=by, ascending=ascending)
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    def rename(self, **kwargs: str) -> "Frame":
        """
        Rename columns: cf.rename(old_name="new_name")
        """
        pdf = self.to_pandas().copy()
        out = pdf.rename(columns=kwargs)
        return Frame.from_pandas(out)

    def drop(self, *cols: str) -> "Frame":
        pdf = self.to_pandas().copy()
        out = pdf.drop(columns=list(cols))
        return Frame.from_pandas(out)

    def mutate(self, **kwargs: Any) -> "Frame":
        """
        Add/modify columns.

        Supports:
        - string expressions evaluated via pandas.eval (column-wise)
        - lag/lead/rolling_mean helpers
        - callables: lambda pdf: ...
        """
        pdf = self.to_pandas().copy()

        for name, spec in kwargs.items():
            # 1) Lag / lead / rolling expressions
            if isinstance(spec, LagExpr):
                pdf[name] = pdf[spec.column].shift(spec.n)
                continue
            if isinstance(spec, LeadExpr):
                pdf[name] = pdf[spec.column].shift(-spec.n)
                continue
            if isinstance(spec, RollingMeanExpr):
                pdf[name] = (
                    pdf[spec.column]
                    .rolling(window=spec.window, min_periods=spec.min_periods)
                    .mean()
                )
                continue

            # 2) Callable taking whole DataFrame
            if callable(spec):
                pdf[name] = spec(pdf)
                continue

            # 3) String expression evaluated with pandas.eval
            if isinstance(spec, str):
                # Use existing columns as local namespace
                env = {col: pdf[col] for col in pdf.columns}
                pdf[name] = pd.eval(spec, local_dict=env, engine="python")
                continue

            # 4) Fallback: treat as literal vector
            pdf[name] = spec

        return Frame.from_pandas(pdf)

    # --------------------------------------------------------------
    # Grouping
    # --------------------------------------------------------------
    def group_by(self, *cols: str) -> "GroupedFrame":
        pdf = self.to_pandas().copy()
        group_cols = list(cols)
        for c in group_cols:
            if c not in pdf.columns:
                raise KeyError(f"Column {c!r} not found for group_by")
        return GroupedFrame(pdf, group_cols)

    def summarise(self, **kwargs: Any) -> "Frame":
        """
        Ungrouped summarise: reduce entire frame to a single row.

        Example:
        cf.summarise(
            mean_x=("x", "mean"),
            max_y=("y", "max"),
        )
        """
        pdf = self.to_pandas().copy()

        agg_spec: Dict[str, Tuple[str, Union[str, Callable]]] = {}
        for new_col, spec in kwargs.items():
            if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                raise ValueError(
                    f"summarise value for {new_col!r} must be (column, agg), "
                    f"got {spec!r}"
                )
            src, fn = spec
            agg_spec[new_col] = (src, fn)

        data: Dict[str, Any] = {}
        for new_col, (src, fn) in agg_spec.items():
            if src not in pdf.columns:
                raise KeyError(f"Column {src!r} not found for summarise")
            series = pdf[src]
            if isinstance(fn, str):
                if fn == "mean":
                    data[new_col] = float(series.mean())
                elif fn == "sum":
                    data[new_col] = series.sum()
                elif fn == "max":
                    data[new_col] = series.max()
                elif fn == "min":
                    data[new_col] = series.min()
                elif fn == "count":
                    data[new_col] = int(series.count())
                else:
                    # let pandas handle unknown agg names
                    data[new_col] = series.aggregate(fn)
            else:
                data[new_col] = fn(series)

        out = pd.DataFrame([data])
        return Frame.from_pandas(out)

    # --------------------------------------------------------------
    # Joins
    # --------------------------------------------------------------
    def left_join(self, other: "Frame", on: Union[str, Sequence[str]]) -> "Frame":
        left = self.to_pandas().copy()
        right = other.to_pandas().copy()
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        out = left.merge(right, how="left", on=on_cols)
        return Frame.from_pandas(out)

    def inner_join(self, other: "Frame", on: Union[str, Sequence[str]]) -> "Frame":
        left = self.to_pandas().copy()
        right = other.to_pandas().copy()
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)
        out = left.merge(right, how="inner", on=on_cols)
        return Frame.from_pandas(out)

    # --------------------------------------------------------------
    # distinct / count
    # --------------------------------------------------------------
    def distinct(self, *cols: str) -> "Frame":
        pdf = self.to_pandas().copy()
        if cols:
            out = pdf.drop_duplicates(subset=list(cols))
        else:
            out = pdf.drop_duplicates()
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

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

    # --------------------------------------------------------------
    # skim (minimal profile)
    # --------------------------------------------------------------
    def skim(self) -> "Frame":
        pdf = self.to_pandas().copy()
        rows = []

        for col in pdf.columns:
            s = pdf[col]
            dtype = str(s.dtype)
            n = len(s)
            n_missing = int(s.isna().sum())
            n_unique = int(s.nunique(dropna=True))

            if pd.api.types.is_numeric_dtype(s.dtype):
                mean = float(s.mean())
                sd = float(s.std())
                q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
                min_v = float(q.loc[0.0])
                q25 = float(q.loc[0.25])
                median = float(q.loc[0.5])
                q75 = float(q.loc[0.75])
                max_v = float(q.loc[1.0])
            else:
                mean = np.nan
                sd = np.nan
                min_v = np.nan
                q25 = np.nan
                median = np.nan
                q75 = np.nan
                max_v = np.nan

            rows.append(
                dict(
                    variable=col,
                    type=dtype,
                    n=n,
                    n_missing=n_missing,
                    n_unique=n_unique,
                    mean=mean,
                    sd=sd,
                    min=min_v,
                    q25=q25,
                    median=median,
                    q75=q75,
                    max=q75 if not np.isnan(q75) else max_v,
                )
            )

        out = pd.DataFrame(rows)
        return Frame.from_pandas(out)

    # --------------------------------------------------------------
    # tidyr-like verbs: pivot_longer / pivot_wider
    # --------------------------------------------------------------
    def pivot_longer(
        self,
        cols: Sequence[Any],
        names_to: str,
        values_to: str,
    ) -> "Frame":
        """
        Tidyverse-style pivot_longer.

        cols: selectors (e.g. col.matches("^year_")) or column names
        names_to: new column name for former column names
        values_to: new column name for values
        """
        pdf = self.to_pandas().copy()
        select_cols = _select_columns_from_selectors(pdf, cols)
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
        """
        Tidyverse-style pivot_wider.

        names_from: column whose distinct values become new columns
        values_from: column whose values are spread across the new columns
        values_fill: fill value for missing entries
        """
        pdf = self.to_pandas().copy()
        id_cols = [c for c in pdf.columns if c not in (names_from, values_from)]

        wide = (
            pdf.pivot_table(
                index=id_cols,
                columns=names_from,
                values=values_from,
                aggfunc="first",
            )
            .reset_index()
        )

        # Flatten MultiIndex columns if needed
        wide.columns = [
            c if isinstance(c, str) else c[1] if c[0] == "" else "_".join(map(str, c))
            for c in wide.columns
        ]

        if values_fill is not None:
            # Fill only the pivoted columns (non-id columns)
            for c in wide.columns:
                if c not in id_cols:
                    wide[c] = wide[c].fillna(values_fill)

        wide = wide.reset_index(drop=True)
        return Frame.from_pandas(wide)

    # --------------------------------------------------------------
    # separate / unite
    # --------------------------------------------------------------
    def separate(
        self,
        column: str,
        into: Sequence[str],
        sep: Union[str, int] = r"\s+",
        remove: bool = True,
        convert: bool = False,
    ) -> "Frame":
        """
        Tidyverse-style separate:

        - column: column to split
        - into: names for new columns
        - sep: regex string or integer position
        - remove: if True, drop original column
        - convert: if True, try to infer better dtypes
        """
        pdf = self.to_pandas().copy()

        if column not in pdf.columns:
            raise KeyError(f"Column {column!r} not found for separate")

        if isinstance(sep, int):
            # Fixed position split
            s = pdf[column].astype(str)
            left = s.str.slice(0, sep)
            right = s.str.slice(sep)
            parts = pd.concat([left, right], axis=1)
        else:
            # Regex / string split
            parts = pdf[column].astype(str).str.split(sep, expand=True)

        if parts.shape[1] != len(into):
            raise ValueError(
                f"separate produced {parts.shape[1]} columns, "
                f"but into has length {len(into)}"
            )

        parts.columns = list(into)

        if convert:
            parts = parts.apply(pd.to_numeric, errors="ignore")

        for c in into:
            pdf[c] = parts[c]

        if remove:
            pdf = pdf.drop(columns=[column])

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
            # robust missing detection across weird round-trips
            if val is None or val is pd.NA:
                return True
            if isinstance(val, float):
                try:
                    if math.isnan(val):
                        return True
                except TypeError:
                    pass
            if isinstance(val, str) and val.strip().lower() in {"none", "nan", ""}:
                return True
            return False

        if na_rm:
            # Drop NAs row-wise; if all missing -> NA
            def combine_row_rm(row: pd.Series) -> Any:
                values = [v for v in row if not _is_missing(v)]
                if not values:
                    return np.nan
                return sep.join(str(v) for v in values)

            pdf[new_column] = sub.apply(combine_row_rm, axis=1)
        else:
            # If ANY missing -> entire result is NA
            def combine_row_strict(row: pd.Series) -> Any:
                if any(_is_missing(v) for v in row):
                    return np.nan
                return sep.join(str(v) for v in row)

            pdf[new_column] = sub.apply(combine_row_strict, axis=1)

        if remove:
            pdf = pdf.drop(columns=cols)

        return Frame.from_pandas(pdf)

    # --------------------------------------------------------------
    # slice_* helpers
    # --------------------------------------------------------------
    def slice_head(self, n: int = 5) -> "Frame":
        """
        Return the first n rows (like dplyr::slice_head).
        """
        pdf = self.to_pandas().copy()
        if n is None:
            n = 5
        if n < 0:
            raise ValueError("n must be non-negative")
        out = pdf.head(n)
        return Frame.from_pandas(out)

    def slice_tail(self, n: int = 5) -> "Frame":
        """
        Return the last n rows (like dplyr::slice_tail).
        """
        pdf = self.to_pandas().copy()
        if n is None:
            n = 5
        if n < 0:
            raise ValueError("n must be non-negative")
        out = pdf.tail(n)
        return Frame.from_pandas(out)

    def slice_sample(
        self,
        n: Optional[int] = None,
        prop: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False,
    ) -> "Frame":
        """
        Randomly sample rows.

        - Exactly one of n or prop must be provided.
        - prop is a fraction in [0, 1].
        - random_state for reproducibility.
        """
        pdf = self.to_pandas().copy()
        if (n is None and prop is None) or (n is not None and prop is not None):
            raise ValueError("Exactly one of n or prop must be provided")

        if prop is not None:
            if prop < 0.0 or prop > 1.0:
                raise ValueError("prop must be between 0 and 1")
            sampled = pdf.sample(
                frac=prop,
                replace=replace,
                random_state=random_state,
            )
        else:
            if n < 0:
                raise ValueError("n must be non-negative")
            sampled = pdf.sample(
                n=n,
                replace=replace,
                random_state=random_state,
            )

        sampled = sampled.reset_index(drop=True)
        return Frame.from_pandas(sampled)

    def slice_max(self, column: str, n: int = 1) -> "Frame":
        """
        Take the rows with the largest values in `column`.
        """
        pdf = self.to_pandas().copy()
        if column not in pdf.columns:
            raise KeyError(f"Column {column!r} not found")

        if n <= 0:
            raise ValueError("n must be positive")

        try:
            out = pdf.nlargest(n, column)
        except TypeError:
            raise TypeError(
                f"slice_max currently supports numeric-like columns only "
                f"(got dtype {pdf[column].dtype!r})"
            )

        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)

    def slice_min(self, column: str, n: int = 1) -> "Frame":
        """
        Take the rows with the smallest values in `column`.
        """
        pdf = self.to_pandas().copy()
        if column not in pdf.columns:
            raise KeyError(f"Column {column!r} not found")

        if n <= 0:
            raise ValueError("n must be positive")

        try:
            out = pdf.nsmallest(n, column)
        except TypeError:
            raise TypeError(
                f"slice_min currently supports numeric-like columns only "
                f"(got dtype {pdf[column].dtype!r})"
            )

        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)


class GroupedFrame:
    """
    Simple grouped wrapper around a pandas DataFrame.

    Created via Frame.group_by(...). We don't store back into Rust until
    summarise() finishes.
    """

    def __init__(self, pdf: pd.DataFrame, group_cols: Sequence[str]):
        self._pdf = pdf
        self._group_cols = list(group_cols)

    def __rshift__(self, other: Any) -> Union["GroupedFrame", Frame]:
        """
        Enable pipe syntax for grouped frames:

            cf >> pipe.group_by(...) >> pipe.summarise(...)
        """
        if callable(other):
            result = other(self)
            if isinstance(result, (Frame, GroupedFrame)):
                return result
            raise TypeError(
                f"Pipe function returned unsupported type from GroupedFrame: {type(result)!r}"
            )
        raise TypeError(
            f"unsupported operand type(s) for >>: 'GroupedFrame' and {type(other)!r}"
        )

    def summarise(self, **kwargs: Any) -> Frame:
        """
        Example:
        cf.group_by("user_id").summarise(
            mean_score=("score", "mean"),
            n=("score", "count"),
        )
        """
        grouped = self._pdf.groupby(self._group_cols, dropna=False)

        agg_map: Dict[str, Tuple[str, Union[str, Callable]]] = {}
        for new_col, spec in kwargs.items():
            if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                raise ValueError(
                    f"summarise value for {new_col!r} must be (column, agg), "
                    f"got {spec!r}"
                )
            src, fn = spec
            agg_map[new_col] = (src, fn)

        # Build dictionary suitable for groupby.agg
        agg_for_groupby: Dict[str, List[Tuple[str, Union[str, Callable]]]] = {}
        for new_col, (src, fn) in agg_map.items():
            if src not in agg_for_groupby:
                agg_for_groupby[src] = []
            agg_for_groupby[src].append((new_col, fn))

        # Apply aggregations
        result_pieces: Dict[str, Any] = {}
        for src, ops in agg_for_groupby.items():
            for new_col, fn in ops:
                series = grouped[src].agg(fn)
                result_pieces[new_col] = series

        out = pd.DataFrame(result_pieces)
        # The group keys are part of the index right now
        out.index = out.index.set_names(self._group_cols)
        out = out.reset_index()
        out = out.reset_index(drop=True)
        return Frame.from_pandas(out)


# --------------------------------------------------------------------
# Top-level helpers
# --------------------------------------------------------------------


def df(obj: Union[Mapping[str, Sequence[Any]], pd.DataFrame, pl.DataFrame]) -> Frame:
    """
    Construct a Frame from:
    - a dict-of-sequences
    - a pandas DataFrame
    - a polars DataFrame
    """
    if isinstance(obj, Frame):
        return obj
    if isinstance(obj, pd.DataFrame):
        return Frame.from_pandas(obj)
    if isinstance(obj, pl.DataFrame):
        return Frame.from_polars(obj)
    if isinstance(obj, Mapping):
        data = _to_pandas_dict(obj)
        pdf = pd.DataFrame(data)
        return Frame.from_pandas(pdf)
    raise TypeError(f"Unsupported type for df(): {type(obj)!r}")
