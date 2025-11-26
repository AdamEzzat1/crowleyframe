from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from . import _crowley  # Rust extension module


# ---------------------------------------------------------------------------
# Window specs: lag / lead / rolling_mean
# ---------------------------------------------------------------------------


@dataclass
class LagSpec:
    column: str
    n: int = 1


@dataclass
class LeadSpec:
    column: str
    n: int = 1


@dataclass
class RollingMeanSpec:
    column: str
    window: int
    min_periods: int = 1


def lag(column: str, n: int = 1) -> LagSpec:
    """Specification for lag(column, n) inside mutate."""
    return LagSpec(column=column, n=n)


def lead(column: str, n: int = 1) -> LeadSpec:
    """Specification for lead(column, n) inside mutate."""
    return LeadSpec(column=column, n=n)


def rolling_mean(column: str, window: int, min_periods: int = 1) -> RollingMeanSpec:
    """Specification for rolling_mean(column, window) inside mutate."""
    return RollingMeanSpec(column=column, window=window, min_periods=min_periods)


# ---------------------------------------------------------------------------
# Core Frame wrapper
# ---------------------------------------------------------------------------


class Frame:
    """
    High-level crowleyframe dataframe wrapper.

    Under the hood, holds a Rust/Polars-backed `_crowley.Frame`, but most
    high-level verbs are currently implemented via pandas for v0.1/v0.2.
    """

    def __init__(self, inner: "_crowley.Frame"):
        self._inner = inner

    # -----------------------
    # Constructors & interop
    # -----------------------

    @classmethod
    def from_pandas(cls, pdf: pd.DataFrame) -> "Frame":
        """
        Build a Frame from a pandas DataFrame by converting to dict-of-lists
        and calling the Rust from_dict constructor.
        """
        data = {col: pdf[col].tolist() for col in pdf.columns}
        inner = _crowley.Frame.from_dict(data)  # type: ignore[attr-defined]
        return cls(inner)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Return a dict-of-lists representation."""
        return self._inner.to_dict()  # type: ignore[attr-defined]

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        return pd.DataFrame(self._inner.to_dict())  # type: ignore[attr-defined]

    def to_polars(self) -> pl.DataFrame:
        """Convert to a polars DataFrame."""
        return pl.DataFrame(self._inner.to_dict())  # type: ignore[attr-defined]

    def to_numpy(self) -> np.ndarray:
        """Convert to a numpy array."""
        return self.to_pandas().to_numpy()

    def to_arrow(self) -> pa.Table:
        """Convert to a pyarrow Table."""
        return pa.table(self._inner.to_dict())  # type: ignore[attr-defined]

    # -----------------------
    # Display
    # -----------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return repr(self.to_pandas())

    # -----------------------
    # Core tidy-like verbs
    # -----------------------

    def clean_names(self) -> "Frame":
        """
        Standardize column names (implemented in Rust/Polars).
        """
        inner = self._inner.clean_names()  # type: ignore[attr-defined]
        return Frame(inner)

    def select(self, *selectors: Any) -> "Frame":
        """
        Column selection using the `col` DSL or raw names.

        Example:
            cf.select(col.starts_with("user_"), "id")
        """
        if len(selectors) == 1 and isinstance(selectors[0], (list, tuple)):
            arg = selectors[0]
        else:
            arg = list(selectors)
        inner = self._inner.select(arg)  # type: ignore[attr-defined]
        return Frame(inner)

    def filter(self, expr: str) -> "Frame":
        """
        Row filtering using a pandas-style expression string.

        Example:
            cf.filter("user_score > 10 & user_id != 3")
        """
        pdf = self.to_pandas()
        mask = pdf.eval(expr, engine="python")
        out = pdf[mask]
        return Frame.from_pandas(out)

    def arrange(self, *keys: str) -> "Frame":
        """
        Sort rows by one or more columns.

        Use a leading '-' for descending order:
            cf.arrange("-user_score", "user_id")
        """
        if not keys:
            return self

        pdf = self.to_pandas()
        by: List[str] = []
        ascending: List[bool] = []

        for k in keys:
            if k.startswith("-"):
                by.append(k[1:])
                ascending.append(False)
            elif k.startswith("+"):
                by.append(k[1:])
                ascending.append(True)
            else:
                by.append(k)
                ascending.append(True)

        out = pdf.sort_values(by=by, ascending=ascending)
        return Frame.from_pandas(out)

    # -----------------------
    # Grouped semantics
    # -----------------------

    def group_by(self, *cols: str) -> "GroupedFrame":
        """
        Returns a GroupedFrame; subsequent summarise() acts per-group.
        """
        pdf = self.to_pandas()
        return GroupedFrame(pdf, list(cols))

    def summarise(self, **kwargs: Any) -> "Frame":
        """
        Ungrouped summarise over the whole frame.
        """
        pdf = self.to_pandas()
        row: Dict[str, Any] = {}

        for out_name, spec in kwargs.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                colname, func = spec
                s = pdf[colname]
                if isinstance(func, str):
                    val = getattr(s, func)()
                else:
                    val = func(s)
            else:
                if callable(spec):
                    val = spec(pdf)
                else:
                    val = spec
            row[out_name] = val

        out = pd.DataFrame([row])
        return Frame.from_pandas(out)

    # -----------------------
    # mutate / rename / drop
    # -----------------------

    def mutate(self, **kwargs: Any) -> "Frame":
        """
        Add or modify columns.

        Each kwarg value can be:
        - a string expression (evaluated with pandas.eval using existing columns)
        - a LagSpec / LeadSpec / RollingMeanSpec
        - a callable(pdf) -> Series or array-like
        - a scalar / list-like
        """
        pdf = self.to_pandas()
        env: Dict[str, Any] = {col: pdf[col] for col in pdf.columns}

        for name, spec in kwargs.items():
            if isinstance(spec, LagSpec):
                pdf[name] = pdf[spec.column].shift(spec.n)
            elif isinstance(spec, LeadSpec):
                pdf[name] = pdf[spec.column].shift(-spec.n)
            elif isinstance(spec, RollingMeanSpec):
                pdf[name] = (
                    pdf[spec.column]
                    .rolling(spec.window, min_periods=spec.min_periods)
                    .mean()
                )
            elif isinstance(spec, str):
                pdf[name] = pd.eval(spec, engine="python", local_dict=env)
            elif callable(spec):
                val = spec(pdf)
                pdf[name] = val
            else:
                pdf[name] = spec

            env[name] = pdf[name]

        return Frame.from_pandas(pdf)

    def rename(self, **kwargs: str) -> "Frame":
        """
        Rename columns by name:

            cf.rename(user_id="id", user_score="score")
        """
        pdf = self.to_pandas()
        out = pdf.rename(columns=kwargs)
        return Frame.from_pandas(out)

    def drop(self, *cols: str) -> "Frame":
        """
        Drop one or more columns:

            cf.drop("temp", "debug_flag")
        """
        pdf = self.to_pandas()
        out = pdf.drop(columns=list(cols))
        return Frame.from_pandas(out)

    # -----------------------
    # Distinct / count
    # -----------------------

    def distinct(self, *cols: str) -> "Frame":
        """
        Distinct rows, optionally on a subset of columns.
        """
        pdf = self.to_pandas()
        if cols:
            out = pdf.drop_duplicates(subset=list(cols))
        else:
            out = pdf.drop_duplicates()
        return Frame.from_pandas(out)

    def count(self, *cols: str, sort: bool = False, prop: bool = False) -> "Frame":
        """
        Grouped counts, similar to dplyr::count().
        """
        pdf = self.to_pandas()

        if cols:
            grouped = pdf.groupby(list(cols), dropna=False)
            counts = grouped.size().reset_index(name="n")
        else:
            counts = pd.DataFrame({"n": [len(pdf)]})

        if sort:
            counts = counts.sort_values("n", ascending=False)

        if prop:
            total = counts["n"].sum()
            if total:
                counts["prop"] = counts["n"] / float(total)
            else:
                counts["prop"] = 0.0

        return Frame.from_pandas(counts)

    # -----------------------
    # Joins
    # -----------------------

    def left_join(
        self,
        other: "Frame",
        on: Union[str, Sequence[str]],
    ) -> "Frame":
        """Tidyverse-style left_join."""
        pdf_left = self.to_pandas()
        pdf_right = other.to_pandas()
        out = pdf_left.merge(pdf_right, how="left", on=on)
        return Frame.from_pandas(out)

    def inner_join(
        self,
        other: "Frame",
        on: Union[str, Sequence[str]],
    ) -> "Frame":
        """Tidyverse-style inner_join."""
        pdf_left = self.to_pandas()
        pdf_right = other.to_pandas()
        out = pdf_left.merge(pdf_right, how="inner", on=on)
        return Frame.from_pandas(out)

    # -----------------------
    # skim: quick profiling
    # -----------------------

    def skim(self) -> "Frame":
        """
        Simple skim() summary for each column:
        variable, type, n, n_missing, n_unique, mean, sd, min, q25, median, q75, max
        """
        pdf = self.to_pandas()
        rows: List[Dict[str, Any]] = []

        for col in pdf.columns:
            s = pdf[col]
            n = len(s)
            n_missing = int(s.isna().sum())
            n_unique = int(s.nunique(dropna=True))
            dtype = str(s.dtype)

            if pd.api.types.is_numeric_dtype(s):
                mean = s.mean()
                sd = s.std()
                minv = s.min()
                maxv = s.max()
                q = s.quantile([0.25, 0.5, 0.75])
                q25 = q.get(0.25, np.nan)
                q50 = q.get(0.5, np.nan)
                q75 = q.get(0.75, np.nan)
            else:
                mean = sd = minv = maxv = np.nan
                q25 = q50 = q75 = np.nan

            rows.append(
                {
                    "variable": col,
                    "type": dtype,
                    "n": n,
                    "n_missing": n_missing,
                    "n_unique": n_unique,
                    "mean": mean,
                    "sd": sd,
                    "min": minv,
                    "q25": q25,
                    "median": q50,
                    "q75": q75,
                    "max": maxv,
                }
            )

        out = pd.DataFrame(
            rows,
            columns=[
                "variable",
                "type",
                "n",
                "n_missing",
                "n_unique",
                "mean",
                "sd",
                "min",
                "q25",
                "median",
                "q75",
                "max",
            ],
        )
        return Frame.from_pandas(out)

    # -----------------------
    # Tidyr-style reshaping: pivot_longer / pivot_wider
    # -----------------------

    def pivot_longer(
        self,
        *cols: Any,
        names_to: str = "name",
        values_to: str = "value",
    ) -> "Frame":
        """
        Tidyverse-style pivot_longer.
        """
        pdf = self.to_pandas()

        if cols:
            selected = self.select(*cols)
            selected_cols = list(selected.to_pandas().columns)
        else:
            selected_cols = list(pdf.columns)

        id_cols = [c for c in pdf.columns if c not in selected_cols]

        long_pdf = pdf.melt(
            id_vars=id_cols,
            value_vars=selected_cols,
            var_name=names_to,
            value_name=values_to,
        )

        return Frame.from_pandas(long_pdf)

    def pivot_wider(
        self,
        names_from: str,
        values_from: str,
        values_fill: Optional[Any] = None,
    ) -> "Frame":
        """
        Tidyverse-style pivot_wider.
        """
        pdf = self.to_pandas()

        id_cols = [c for c in pdf.columns if c not in (names_from, values_from)]

        wide = pdf.pivot_table(
            index=id_cols if id_cols else None,
            columns=names_from,
            values=values_from,
            aggfunc="first",
            fill_value=values_fill,
        )

        wide = wide.reset_index()
        wide.columns = [str(c) for c in wide.columns]

        return Frame.from_pandas(wide)

    # -----------------------
    # separate / unite (tidyr-style)
    # -----------------------

    def separate(
        self,
        column: str,
        into: Sequence[str],
        sep: str = r"\s+",
        remove: bool = True,
    ) -> "Frame":
        """
        Tidyverse-style separate: split one column into multiple columns.
        """
        pdf = self.to_pandas().copy()

        if column not in pdf.columns:
            raise KeyError(f"Column {column!r} not found in frame")

        split = (
            pdf[column]
            .astype("string")
            .str.split(sep, expand=True, regex=True)
        )

        if split.shape[1] < len(into):
            for _ in range(len(into) - split.shape[1]):
                split[split.shape[1]] = np.nan

        for i, name in enumerate(into):
            pdf[name] = split.iloc[:, i]

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
        Tidyverse-style unite: combine multiple columns into one.

        Semantics:
        - na_rm=False (default): if any value in the row is NA -> result is NA
        - na_rm=True: drop NA values and join the remaining parts
        """
        pdf = self.to_pandas().copy()
        cols = list(columns)

        # 1. Check columns exist
        for c in cols:
            if c not in pdf.columns:
                raise KeyError(f"Column {c!r} not found in frame")

        if not na_rm:
            # Row is NA if ANY of the component columns is NA
            row_has_na = pdf[cols].isna().any(axis=1)

            def combine_row_no_rm(row: pd.Series) -> Any:
                if row_has_na.loc[row.name]:
                    return np.nan
                return sep.join(str(v) for v in row[cols])

            pdf[new_column] = pdf.apply(combine_row_no_rm, axis=1)
        else:
            # na_rm=True: drop NA values and join remaining
            def combine_row_rm(row: pd.Series) -> str:
                s = row[cols]
                non_missing = s[~s.isna()]
                return sep.join(str(v) for v in non_missing)

            pdf[new_column] = pdf.apply(combine_row_rm, axis=1)

        if remove:
            pdf = pdf.drop(columns=cols)

        return Frame.from_pandas(pdf)


# ---------------------------------------------------------------------------
# GroupedFrame wrapper
# ---------------------------------------------------------------------------


class GroupedFrame:
    """
    Wrapper around a pandas GroupBy used by Frame.group_by(...).
    """

    def __init__(self, pdf: pd.DataFrame, by: List[str]):
        self._pdf = pdf
        self._by = by

    def summarise(self, **kwargs: Any) -> Frame:
        """
        Group-wise aggregation.
        """
        grouped = self._pdf.groupby(self._by, dropna=False)
        rows: List[Dict[str, Any]] = []

        for key, sub in grouped:
            row: Dict[str, Any] = {}

            if len(self._by) == 1:
                row[self._by[0]] = key
            else:
                for col, val in zip(self._by, key):
                    row[col] = val

            for out_name, spec in kwargs.items():
                if isinstance(spec, tuple) and len(spec) == 2:
                    colname, func = spec
                    s = sub[colname]
                    if isinstance(func, str):
                        val = getattr(s, func)()
                    else:
                        val = func(s)
                else:
                    if callable(spec):
                        val = spec(sub)
                    else:
                        val = spec

                row[out_name] = val

            rows.append(row)

        out = pd.DataFrame(rows)
        return Frame.from_pandas(out)


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------


def df(data: Mapping[str, Sequence[Any]]) -> Frame:
    """
    Construct a crowleyframe Frame from a dict-of-lists-style mapping.
    """
    inner = _crowley.Frame.from_dict(dict(data))  # type: ignore[attr-defined]
    return Frame(inner)
