from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import _crowley


# -----------------------------
# Window helper marker classes
# -----------------------------
class _Lag:
    def __init__(self, col: str, n: int = 1):
        self.col = col
        self.n = n


class _Lead:
    def __init__(self, col: str, n: int = 1):
        self.col = col
        self.n = n


class _RollingMean:
    def __init__(self, col: str, window: int, min_periods: int | None = None):
        self.col = col
        self.window = window
        self.min_periods = min_periods if min_periods is not None else 1


def lag(col: str, n: int = 1) -> _Lag:
    """Declare a lag window op for use inside mutate()."""
    return _Lag(col, n)


def lead(col: str, n: int = 1) -> _Lead:
    """Declare a lead window op for use inside mutate()."""
    return _Lead(col, n)


def rolling_mean(col: str, window: int, min_periods: int | None = None) -> _RollingMean:
    """Declare a rolling mean op for use inside mutate()."""
    return _RollingMean(col, window, min_periods=min_periods)


class Frame:
    """
    Python wrapper over the Rust Frame.

    For v0.1:
    - Heavy lifting for `select` + `clean_names` happens in Rust.
    - Higher-level verbs (filter, arrange, group_by/summarise, distinct, count, skim,
      mutate, rename, drop, joins, windows) are implemented in Python on top of pandas,
      then passed back into Rust via from_dict().
    """

    def __init__(self, inner: _crowley.Frame):
        self._inner = inner

    # -----------------------------
    # Core Rust-backed operations
    # -----------------------------
    def select(self, *selectors: Any) -> "Frame":
        """
        Column selection via crowleyframe.col selectors.
        """
        return Frame(self._inner.select(list(selectors)))

    def clean_names(self) -> "Frame":
        """
        Clean column names to snake_case-ish style (Rust side).
        """
        return Frame(self._inner.clean_names())

    def to_dict(self) -> dict:
        """
        Round-trip the internal Rust/Polars DataFrame back to a
        Python dict-of-lists.
        """
        return self._inner.to_dict()  # type: ignore[no-any-return]

    # -----------------------------
    # Interop: pandas / polars / numpy / arrow
    # -----------------------------
    def to_pandas(self):
        """
        Convert to a pandas DataFrame.

        For v0.1 this is implemented via dict-of-lists. Good enough and simple.
        """
        import pandas as pd

        return pd.DataFrame(self.to_dict())

    def to_polars(self):
        """
        Convert to a Python polars.DataFrame.

        Again, via dict-of-lists for v0.1.
        """
        import polars as pl  # type: ignore[import-not-found]

        return pl.DataFrame(self.to_dict())

    def to_numpy(self):
        """
        Convert to a NumPy array (values only).
        """
        return self.to_pandas().to_numpy()

    def to_arrow(self):
        """
        Convert to a pyarrow.Table.
        """
        import pyarrow as pa  # type: ignore[import-not-found]

        return pa.Table.from_pydict(self.to_dict())

    # -----------------------------
    # A. filter()
    # -----------------------------
    def filter(self, expr: str | None = None, mask: Any | None = None) -> "Frame":
        """
        Filter rows.

        For v0.1:
        - If `expr` is given: uses pandas.DataFrame.query(expr).
          Example: cf.filter("user_score > 10 and user_id != 2")
        - If `mask` is given: boolean mask applied like df[mask].
        """
        import pandas as pd  # noqa: F401

        pdf = self.to_pandas()

        if expr is not None:
            pdf2 = pdf.query(expr, engine="python")
        elif mask is not None:
            pdf2 = pdf[mask]
        else:
            # nothing to filter, just return self
            return self

        inner = _crowley.Frame.from_dict(pdf2.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # -----------------------------
    # B. arrange()
    # -----------------------------
    def arrange(self, *columns: str) -> "Frame":
        """
        Sort rows by one or more columns.

        Usage:
            cf.arrange("user_id")             # ascending
            cf.arrange("-user_score")         # descending on user_score
            cf.arrange("user_id", "-user_score")

        A leading '-' means descending.
        """
        import pandas as pd  # noqa: F401

        pdf = self.to_pandas()

        if not columns:
            return self

        by: list[str] = []
        ascending: list[bool] = []

        for col in columns:
            if isinstance(col, str) and col.startswith("-"):
                by.append(col[1:])
                ascending.append(False)
            else:
                by.append(col)
                ascending.append(True)

        pdf2 = pdf.sort_values(by=by, ascending=ascending, kind="mergesort")

        inner = _crowley.Frame.from_dict(pdf2.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # -----------------------------
    # C. group_by() / summarise()
    # -----------------------------
    def group_by(self, *cols: str) -> "GroupedFrame":
        """
        Start a grouped pipeline.

        Example:
            (cf.clean_names()
               .group_by("user_id")
               .summarise(
                   mean_score=("user_score", "mean"),
                   n=("user_score", "count"),
               ))
        """
        if not cols:
            raise ValueError("group_by(...) requires at least one column name")
        return GroupedFrame(self, list(cols))

    # -----------------------------
    # D. distinct() and count()
    # -----------------------------
    def distinct(self, *cols: str) -> "Frame":
        """
        Drop duplicate rows.

        If cols are provided, dedupe on those columns only.
        Otherwise, dedupe on all columns.
        """
        import pandas as pd  # noqa: F401

        pdf = self.to_pandas()

        if cols:
            pdf2 = pdf.drop_duplicates(subset=list(cols))
        else:
            pdf2 = pdf.drop_duplicates()

        inner = _crowley.Frame.from_dict(pdf2.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    def count(self, *cols: str, sort: bool = False, prop: bool = False) -> "Frame":
        """
        Tidyverse-style count(): group + summarise(n = n()).

        Examples:
            cf.count("user_id")
            cf.count("user_id", "group", sort=True, prop=True)
        """
        import pandas as pd  # noqa: F401

        pdf = self.to_pandas()

        if cols:
            g = pdf.groupby(list(cols), dropna=False)
            out = g.size().reset_index(name="n")
        else:
            # Treat as a single group: just count rows
            n = len(pdf)
            out = pd.DataFrame({"n": [n]})  # type: ignore[name-defined]

        if prop:
            total = out["n"].sum()
            if total > 0:
                out["prop"] = out["n"] / total
            else:
                out["prop"] = 0.0

        if sort:
            out = out.sort_values(by="n", ascending=False)

        inner = _crowley.Frame.from_dict(out.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # -----------------------------
    # E. skim(): minimal profiling
    # -----------------------------
    def skim(self) -> "Frame":
        """
        Minimal skim/profile of each column, inspired by skimr.

        Output columns (for v0.1):
            variable, type, n, n_missing, n_unique,
            mean, sd, min, q25, median, q75, max (for numerics)
        """
        import pandas as pd
        from pandas.api import types as ptypes

        pdf = self.to_pandas()

        rows: list[dict[str, Any]] = []

        for col in pdf.columns:
            s = pdf[col]
            dtype = str(s.dtype)
            n = int(len(s))
            n_missing = int(s.isna().sum())
            n_unique = int(s.nunique(dropna=True))

            row: dict[str, Any] = {
                "variable": col,
                "type": dtype,
                "n": n,
                "n_missing": n_missing,
                "n_unique": n_unique,
            }

            if ptypes.is_numeric_dtype(s):
                row.update(
                    {
                        "mean": float(s.mean()) if n > 0 else None,
                        "sd": float(s.std()) if n > 1 else None,
                        "min": float(s.min()) if n > 0 else None,
                        "q25": float(s.quantile(0.25)) if n > 0 else None,
                        "median": float(s.median()) if n > 0 else None,
                        "q75": float(s.quantile(0.75)) if n > 0 else None,
                        "max": float(s.max()) if n > 0 else None,
                    }
                )
            else:
                # Non-numeric: basic placeholders
                row.setdefault("mean", None)
                row.setdefault("sd", None)
                row.setdefault("min", None)
                row.setdefault("q25", None)
                row.setdefault("median", None)
                row.setdefault("q75", None)
                row.setdefault("max", None)

            rows.append(row)

        skim_df = pd.DataFrame(rows)
        inner = _crowley.Frame.from_dict(skim_df.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # -----------------------------
    # NEW: mutate / rename / drop
    # -----------------------------
    def mutate(self, **new_cols: Any) -> "Frame":
        """
        Add or transform columns.

        Supported value types:
        - callable(pdf) -> Series or array-like
        - window ops: lag("col"), lead("col"), rolling_mean("col", window)
        - string: expression evaluated with pandas.eval
        - scalar / list-like: assigned directly

        Example:
            cf.mutate(
                z = "(score - score.mean()) / score.std()",
                lag_score = lag("score"),
            )
        """
        import pandas as pd  # noqa: F401

        pdf = self.to_pandas()

        for name, expr in new_cols.items():
            if isinstance(expr, _Lag):
                pdf[name] = pdf[expr.col].shift(expr.n)
            elif isinstance(expr, _Lead):
                pdf[name] = pdf[expr.col].shift(-expr.n)
            elif isinstance(expr, _RollingMean):
                pdf[name] = (
                    pdf[expr.col]
                    .rolling(expr.window, min_periods=expr.min_periods)
                    .mean()
                )
            elif callable(expr):
                pdf[name] = expr(pdf)
            elif isinstance(expr, str):
                # Use pandas.eval in the DataFrame context
                pdf[name] = pdf.eval(expr)
            else:
                # Fallback: assign directly (scalar or list-like)
                pdf[name] = expr

        inner = _crowley.Frame.from_dict(pdf.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    def rename(self, **mapping: str) -> "Frame":
        """
        Rename columns: cf.rename(old_name="new_name").
        """
        import pandas as pd  # noqa: F401

        pdf = self.to_pandas()
        pdf2 = pdf.rename(columns=mapping)
        inner = _crowley.Frame.from_dict(pdf2.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    def drop(self, *cols: str) -> "Frame":
        """
        Drop one or more columns.

        Example:
            cf.drop("temp", "debug_flag")
        """
        import pandas as pd  # noqa: F401

        if not cols:
            return self

        pdf = self.to_pandas()
        pdf2 = pdf.drop(columns=list(cols), errors="ignore")
        inner = _crowley.Frame.from_dict(pdf2.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # -----------------------------
    # NEW: Joins
    # -----------------------------
    def left_join(self, other: "Frame", on: str | Sequence[str]) -> "Frame":
        """
        Left join with another Frame.

        Example:
            cf1.left_join(cf2, on="id")
            cf1.left_join(cf2, on=["id", "date"])
        """
        import pandas as pd  # noqa: F401

        pdf1 = self.to_pandas()
        pdf2 = other.to_pandas()

        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)

        out = pdf1.merge(pdf2, how="left", on=on_cols)
        inner = _crowley.Frame.from_dict(out.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    def inner_join(self, other: "Frame", on: str | Sequence[str]) -> "Frame":
        """
        Inner join with another Frame.
        """
        import pandas as pd  # noqa: F401

        pdf1 = self.to_pandas()
        pdf2 = other.to_pandas()

        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = list(on)

        out = pdf1.merge(pdf2, how="inner", on=on_cols)
        inner = _crowley.Frame.from_dict(out.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # -----------------------------
    # Pipe operator support (>> and |)
    # -----------------------------
    def __rshift__(self, other: Any):
        """
        Allow: cf >> pipe.group_by("id") >> pipe.summarise(...)
        """
        if callable(other):
            return other(self)
        return NotImplemented

    def __or__(self, other: Any):
        """
        Allow: cf | pipe.group_by("id") | pipe.summarise(...)
        """
        if callable(other):
            return other(self)
        return NotImplemented

    # -----------------------------
    # Misc
    # -----------------------------
    def __repr__(self) -> str:  # pragma: no cover - passthrough
        return repr(self._inner)


class GroupedFrame:
    """
    Lightweight grouped wrapper for v0.1, implemented in Python on top of pandas.

    You obtain this via Frame.group_by(...)
    and then call summarise(...) on it.
    """

    def __init__(self, frame: Frame, group_cols: Sequence[str]):
        self._frame = frame
        self.group_cols = list(group_cols)

    def summarise(self, **agg_specs: tuple[str, str]) -> Frame:
        """
        Summarise grouped data.

        agg_specs: new_column = (source_column, agg_func)

        Example:
            cf.group_by("user_id").summarise(
                mean_score=("user_score", "mean"),
                n=("user_score", "count"),
            )
        """
        import pandas as pd  # noqa: F401

        pdf = self._frame.to_pandas()
        g = pdf.groupby(self.group_cols, dropna=False)

        # Use pandas named aggregation
        named_aggs: dict[str, tuple[str, str]] = {}
        for new_name, (src_col, func) in agg_specs.items():
            named_aggs[new_name] = (src_col, func)

        if not named_aggs:
            # If no aggs are provided, just return unique groups
            out = g.size().reset_index(name="n")
        else:
            out = g.agg(**named_aggs).reset_index()

        inner = _crowley.Frame.from_dict(out.to_dict(orient="list"))  # type: ignore[attr-defined]
        return Frame(inner)

    # Pipe support for grouped objects too
    def __rshift__(self, other: Any):
        if callable(other):
            return other(self)
        return NotImplemented

    def __or__(self, other: Any):
        if callable(other):
            return other(self)
        return NotImplemented


def df(obj: Any) -> Frame:
    """
    Construct a Frame from:
    - dict-of-lists
    - pandas.DataFrame

    For dicts, we normalize through pandas.DataFrame first so that
    None in numeric columns becomes NaN and types are inferred properly.
    """
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        pdf = obj
    elif isinstance(obj, dict):
        # Let pandas handle None / NaN / dtype inference
        pdf = pd.DataFrame(obj)
    else:
        raise TypeError("df() currently accepts dict or pandas.DataFrame")

    data: Mapping[str, list] = pdf.to_dict(orient="list")
    inner = _crowley.Frame.from_dict(data)  # type: ignore[attr-defined]
    return Frame(inner)
