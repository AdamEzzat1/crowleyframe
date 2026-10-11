"""
Reproducibility and pipeline ergonomics: fingerprint_rows, snapshot /
restore_snapshot, expectations, branch, define_pipeline, collect_all,
compare_outputs.

Lazy: fingerprint_rows, expect, clear_expectations, branch, define_pipeline.
Eager: snapshot (runs the plan and writes files), assert_expectations (one
aggregation query), collect_all, compare_outputs.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import polars as pl

from .col import resolve_selectors
from .diagnostics import SchemaComparison
from .expr import CrowleyValidationError
from .quality import RULE_NAMES, _finish, run_checks
from .tracking import format_step

# ---------------------------------------------------------------------
# row fingerprints
# ---------------------------------------------------------------------


def fingerprint_rows(frame: Any, *selectors: Any, name: str = "row_hash", seed: int = 0) -> Any:
    """
    Add a UInt64 hash of the selected columns (all by default) for each row. Lazy.

    Equal rows get equal hashes for the same column selection, seed and Polars
    version. Polars does not promise the same hash values across versions, so
    do not compare fingerprints stored by a different Polars version.
    """
    if name in frame.schema:
        raise ValueError(f"fingerprint_rows(): column {name!r} already exists; pass name=")
    cols = resolve_selectors(frame.schema, selectors) if selectors else frame.columns
    if not cols:
        raise ValueError("fingerprint_rows(): no columns selected")
    return frame._new(frame._lf.with_columns(pl.struct(cols).hash(seed).alias(name)))


# ---------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------

_META_SUFFIX = ".meta.json"
_SNAPSHOT_VERSION = 1


def _meta_path(path: str) -> str:
    return path + _META_SUFFIX


def snapshot(frame: Any, path: Union[str, os.PathLike], overwrite: bool = False) -> Any:
    """
    Run the plan, write the result to `path` as Parquet with a metadata file
    next to it (`<path>.meta.json`: schema, row count, groups, audit trail,
    Polars version, time), and return a frame that reads from the snapshot, so
    later verbs do not recompute the plan.

    Eager, and writes to disk. The path must be explicit; its directory must
    already exist. Existing files are never replaced unless overwrite=True.
    Files are written under a temporary name and renamed into place.
    """
    path = os.fspath(path)
    if not path.endswith(".parquet"):
        raise ValueError("snapshot(): path must end in .parquet")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"snapshot(): directory {directory!r} does not exist")
    meta_path = _meta_path(path)
    for target in (path, meta_path):
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(f"snapshot(): {target!r} already exists; pass overwrite=True to replace it")

    data = frame._lf.collect()
    meta = {
        "snapshot_version": _SNAPSHOT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "file": os.path.basename(path),
        "rows": data.height,
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in data.schema.items()],
        "groups": list(frame._groups),
        "polars_version": pl.__version__,
        "audit": [{"verb": verb, "arguments": arguments} for verb, arguments in map(format_step, frame._audit)],
    }
    tmp_data, tmp_meta = path + ".tmp", meta_path + ".tmp"
    try:
        data.write_parquet(tmp_data)
        with open(tmp_meta, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        os.replace(tmp_data, path)
        os.replace(tmp_meta, meta_path)
    finally:
        for tmp in (tmp_data, tmp_meta):
            if os.path.exists(tmp):
                os.remove(tmp)
    return frame._new(pl.scan_parquet(path))


def snapshot_metadata(path: Union[str, os.PathLike]) -> dict:
    """The metadata written next to a snapshot."""
    meta_path = _meta_path(os.fspath(path))
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no snapshot metadata at {meta_path!r}")
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def restore_snapshot(path: Union[str, os.PathLike], check: bool = True) -> Any:
    """
    A lazy frame over a snapshot written by snapshot(), with its grouping
    restored. With check=True (default) the Parquet schema must match the
    metadata, otherwise CrowleyValidationError is raised. Reads the Parquet
    footer and the metadata file, not the data.
    """
    from .api import Frame

    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no snapshot at {path!r}")
    lf = pl.scan_parquet(path)
    if not check:
        return Frame(lf)
    meta = snapshot_metadata(path)
    schema = lf.collect_schema()
    expected = [(c["name"], c["dtype"]) for c in meta["columns"]]
    actual = [(name, str(dtype)) for name, dtype in schema.items()]
    if expected != actual:
        raise CrowleyValidationError(
            f"restore_snapshot(): {path!r} does not match its metadata "
            f"(metadata: {expected}, file: {actual})"
        )
    return Frame(lf, groups=tuple(meta.get("groups", ())))


# ---------------------------------------------------------------------
# expectations
# ---------------------------------------------------------------------


def expect(frame: Any, **rules: Any) -> Any:
    """
    Attach expectations to the pipeline without checking them yet. Takes the
    same rules as validate(): columns, dtypes, not_null, unique, ranges,
    allowed, strict_columns. They travel with the frame through later verbs
    and are checked, against whatever the frame looks like then, by
    assert_expectations(). Lazy; reads no data.
    """
    unknown = sorted(set(rules) - set(RULE_NAMES))
    if unknown:
        raise TypeError(f"expect(): unknown rule(s) {unknown}; available: {list(RULE_NAMES)}")
    if not rules:
        raise TypeError("expect() needs at least one rule")
    out = frame._new(frame._lf)
    out._expect = frame._expect + (dict(rules),)
    return out


def clear_expectations(frame: Any) -> Any:
    """Remove every expectation attached with expect(). Lazy."""
    out = frame._new(frame._lf)
    out._expect = ()
    return out


def expectations(frame: Any) -> List[dict]:
    """The rule sets attached with expect(), in order."""
    return [dict(rules) for rules in frame._expect]


def assert_expectations(frame: Any, on_fail: str = "raise") -> Any:
    """
    Check every attached expectation against the frame as it is now, and
    return the frame unchanged. Eager: all data rules of all expectations run
    in one aggregation query (none if only schema rules are attached).
    """
    return _finish(frame, run_checks(frame, frame._expect), on_fail)


# ---------------------------------------------------------------------
# branching and reusable pipelines
# ---------------------------------------------------------------------


class Pipeline:
    """
    A named, reusable sequence of steps. Each step is a callable that takes a
    frame and returns a frame, e.g. `pipe.filter("x > 0")` or a function.

        clean = define_pipeline("clean", pipe.drop_na("x"), pipe.mutate(y="x * 2"))
        out = clean(cf)          # or: cf >> clean
    """

    def __init__(self, name: str, steps: Sequence[Callable[[Any], Any]]):
        for step in steps:
            if not callable(step):
                raise TypeError(f"define_pipeline({name!r}): step {step!r} is not callable")
        self.name = name
        self.steps = tuple(steps)

    def __call__(self, frame: Any) -> Any:
        for step in self.steps:
            frame = step(frame)
        return frame

    def __repr__(self) -> str:
        names = ", ".join(getattr(s, "__name__", type(s).__name__) for s in self.steps)
        return f"Pipeline({self.name!r}: {names})"


def define_pipeline(name: str, *steps: Callable[[Any], Any]) -> Pipeline:
    """Define a named, reusable transformation. Nothing runs until it is applied to a frame."""
    return Pipeline(name, steps)


def branch(frame: Any, **branches: Callable[[Any], Any]) -> Dict[str, Any]:
    """
    Apply several transformations to the same lazy frame and return
    {name: frame}. Lazy: the shared upstream plan is not run here. Collect the
    branches together with crowley_frame.collect_all so the shared part runs once.
    """
    if not branches:
        raise TypeError("branch() needs at least one name=transformation")
    out = {}
    for name, fn in branches.items():
        if not callable(fn):
            raise TypeError(f"branch({name}=...): expected a callable taking a frame")
        result = fn(frame)
        if not hasattr(result, "_lf"):
            raise TypeError(f"branch({name}=...): the transformation must return a frame")
        out[name] = result
    return out


def collect_all(frames: Any) -> Any:
    """
    Run several lazy frames in one go and return them collected, in the same
    structure (dict, list or tuple). Polars runs sub-plans the frames share
    once. Eager.
    """
    if isinstance(frames, Mapping):
        names, items = list(frames), list(frames.values())
    else:
        names, items = None, list(frames)
    collected = pl.collect_all([f._lf for f in items])
    out = [f._new(d.lazy()) for f, d in zip(items, collected)]
    if names is not None:
        return dict(zip(names, out))
    return tuple(out) if isinstance(frames, tuple) else out


# ---------------------------------------------------------------------
# comparing outputs
# ---------------------------------------------------------------------


class OutputComparison:
    """Result of compare_outputs(). Truthy when the two outputs are equal."""

    def __init__(
        self,
        schema: SchemaComparison,
        rows_left: int,
        rows_right: int,
        only_left: Optional[int],
        only_right: Optional[int],
        mismatches: Optional[Dict[str, int]],
        check_dtypes: bool,
    ):
        self.schema = schema
        self.rows_left, self.rows_right = rows_left, rows_right
        self.only_left, self.only_right = only_left, only_right
        self.mismatches = mismatches  # {column: differing values}; None if rows could not be aligned
        schema_ok = not (schema.missing or schema.extra or (check_dtypes and schema.dtype_mismatches))
        rows_ok = rows_left == rows_right and not only_left and not only_right
        values_ok = mismatches is not None and not any(mismatches.values())
        self.equal = bool(schema_ok and rows_ok and values_ok)

    def __bool__(self) -> bool:
        return self.equal

    def to_polars(self) -> pl.DataFrame:
        rows = [
            {"item": "rows_left", "value": self.rows_left},
            {"item": "rows_right", "value": self.rows_right},
            {"item": "rows_only_left", "value": self.only_left},
            {"item": "rows_only_right", "value": self.only_right},
        ]
        for name, count in (self.mismatches or {}).items():
            rows.append({"item": f"mismatches:{name}", "value": count})
        return pl.DataFrame(rows, schema={"item": pl.String, "value": pl.Int64})

    def __repr__(self) -> str:
        if self.equal:
            return f"OutputComparison: equal ({self.rows_left} rows)"
        parts = []
        if not self.schema.equal:
            parts.append(repr(self.schema))
        if self.rows_left != self.rows_right:
            parts.append(f"row counts differ: {self.rows_left} vs {self.rows_right}")
        if self.only_left or self.only_right:
            parts.append(f"{self.only_left} rows only on the left, {self.only_right} only on the right")
        if self.mismatches is None:
            parts.append("rows could not be aligned, values not compared")
        else:
            differing = {k: v for k, v in self.mismatches.items() if v}
            if differing:
                parts.append(f"differing values per column: {differing}")
        return "OutputComparison: not equal\n  " + "\n  ".join(parts)


def _values_equal(left: pl.Expr, right: pl.Expr, dtype: Any, rtol: float, atol: float) -> pl.Expr:
    if dtype.is_float() and (rtol or atol):
        close = (left - right).abs() <= atol + rtol * right.abs()
        both_missing = left.is_null() & right.is_null()
        both_nan = left.is_nan() & right.is_nan()
        return (close | both_nan).fill_null(False) | both_missing
    return left.eq_missing(right)


def compare_outputs(
    frame: Any,
    other: Any,
    by: Union[str, Sequence[str], None] = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
    check_order: bool = False,
    check_dtypes: bool = True,
) -> OutputComparison:
    """
    Compare two frames (or the results of two pipelines): schema, row counts,
    rows present on one side only, and differing values per column, with a
    tolerance for floats.

    - by: column(s) that identify a row. Rows are matched on them, so row
      order does not matter. They must be unique on both sides.
    - without `by`: rows are compared by position; with check_order=False
      (default) both sides are first sorted by every shared column.

    Eager: runs both plans. Returns an OutputComparison (truthy when equal).
    """
    from .api import _NULLS_EQUAL_KW, df

    right_frame = df(other)
    schema = SchemaComparison(frame.schema, right_frame.schema)
    left_schema = frame.schema
    shared = [c for c in frame.columns if c in right_frame.schema]
    keys = [by] if isinstance(by, str) else list(by or [])
    for k in keys:
        if k not in shared:
            raise ValueError(f"compare_outputs(): key column {k!r} must exist on both sides")
    values = [c for c in shared if c not in keys]
    left_lf, right_lf = frame._lf.select(shared), right_frame._lf.select(shared)

    if keys:
        left_lf = left_lf.rename({c: c + "__l" for c in values}).with_columns(pl.lit(True).alias("__in_l"))
        right_lf = right_lf.rename({c: c + "__r" for c in values}).with_columns(pl.lit(True).alias("__in_r"))
        joined = left_lf.join(right_lf, on=keys, how="full", coalesce=True, **{_NULLS_EQUAL_KW: True})
        both = pl.col("__in_l").is_not_null() & pl.col("__in_r").is_not_null()
        aggs = [
            pl.col("__in_l").is_not_null().sum().alias("__rows_l"),
            pl.col("__in_r").is_not_null().sum().alias("__rows_r"),
            (pl.col("__in_l").is_not_null() & pl.col("__in_r").is_null()).sum().alias("__only_l"),
            (pl.col("__in_l").is_null() & pl.col("__in_r").is_not_null()).sum().alias("__only_r"),
        ]
        for i, c in enumerate(values):
            if left_schema[c] != right_frame.schema[c]:
                aggs.append(pl.lit(0).alias(f"__m{i}"))  # reported as a dtype mismatch instead
                continue
            same = _values_equal(pl.col(c + "__l"), pl.col(c + "__r"), left_schema[c], rtol, atol)
            aggs.append((both & ~same).sum().alias(f"__m{i}"))
        stats = joined.select(aggs).collect().row(0, named=True)
        mismatches = {c: stats[f"__m{i}"] for i, c in enumerate(values)}
        return OutputComparison(
            schema, stats["__rows_l"], stats["__rows_r"], stats["__only_l"], stats["__only_r"], mismatches,
            check_dtypes,
        )

    if not check_order and shared:
        left_lf = left_lf.sort(shared, nulls_last=True)
        right_lf = right_lf.sort(shared, nulls_last=True)
    left_data, right_data = pl.collect_all([left_lf, right_lf])
    if left_data.height != right_data.height:
        return OutputComparison(schema, left_data.height, right_data.height, None, None, None, check_dtypes)
    paired = pl.concat([left_data.rename({c: c + "__l" for c in shared}),
                        right_data.rename({c: c + "__r" for c in shared})], how="horizontal")
    aggs = []
    for i, c in enumerate(shared):
        left_col, right_col = pl.col(c + "__l"), pl.col(c + "__r")
        if left_schema[c] != right_frame.schema[c]:
            aggs.append(pl.lit(0).alias(f"__m{i}"))  # reported as a dtype mismatch instead
            continue
        aggs.append((~_values_equal(left_col, right_col, left_schema[c], rtol, atol)).sum().alias(f"__m{i}"))
    stats = paired.select(aggs).row(0, named=True) if aggs else {}
    mismatches = {c: stats[f"__m{i}"] for i, c in enumerate(shared)}
    return OutputComparison(schema, left_data.height, right_data.height, 0, 0, mismatches, check_dtypes)
