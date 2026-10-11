"""
Profiling, validation and data-quality helpers.

Laziness, per function:

- lazy (returns a Frame over a longer plan): profile_by, partition_valid,
  normalize_names, normalize_values, replace_missing, drop_missing
- eager (runs ONE aggregation query over the frame): profile, profile_diff,
  validate (only when a data check is requested), validate_bounds,
  validate_categories, detect_anomalies, detect_drift

The eager ones never pull rows into Python: they collect one row of
aggregates and report on it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import polars as pl

from .col import resolve_selectors
from .expr import Ctx, CrowleyTranslationError, CrowleyValidationError, column_not_found

Key = Union[str, Sequence[str]]
Bounds = Tuple[Any, Any]


def _frame(data: pl.DataFrame) -> Any:
    from .api import Frame

    return Frame(data.lazy())


def _key_cols(key: Key) -> List[str]:
    return [key] if isinstance(key, str) else list(key)


def _key_label(key: Key) -> str:
    return key if isinstance(key, str) else ", ".join(key)


def _key_expr(key: Key) -> pl.Expr:
    cols = _key_cols(key)
    return pl.col(cols[0]) if len(cols) == 1 else pl.struct(cols)


def _orderable(dtype: Any) -> bool:
    return dtype.is_numeric() or dtype.is_temporal() or dtype in (pl.String, pl.Boolean)


def _distinct_expr(name: str, dtype: Any, exact: bool) -> pl.Expr:
    e = pl.col(name)
    if exact:
        return e.n_unique()
    if dtype.is_numeric() or dtype in (pl.String, pl.Boolean):
        return e.approx_n_unique()
    if dtype.is_temporal() or dtype in (pl.Categorical, pl.Enum):
        return e.to_physical().approx_n_unique()
    return e.n_unique()


def _in_bounds(value: Any, bounds: Bounds) -> bool:
    lo, hi = bounds
    if value is None:
        return False
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _fmt_bounds(bounds: Bounds) -> str:
    lo, hi = bounds
    return f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"


# ---------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------

PROFILE_COLUMNS = ["column", "dtype", "rows", "nulls", "null_rate", "distinct", "min", "max", "mean"]
_PROFILE_SCHEMA = {
    "column": pl.String, "dtype": pl.String, "rows": pl.Int64, "nulls": pl.Int64, "null_rate": pl.Float64,
    "distinct": pl.Int64, "min": pl.String, "max": pl.String, "mean": pl.Float64,
}


def _profile_aggs(schema: Any, cols: List[str], exact: bool) -> List[pl.Expr]:
    aggs: List[pl.Expr] = [pl.len().alias("__rows")]
    for i, name in enumerate(cols):
        dtype = schema[name]
        aggs.append(pl.col(name).null_count().alias(f"__nulls{i}"))
        aggs.append(_distinct_expr(name, dtype, exact).alias(f"__distinct{i}"))
        if _orderable(dtype):
            aggs.append(pl.col(name).min().alias(f"__min{i}"))
            aggs.append(pl.col(name).max().alias(f"__max{i}"))
        if dtype.is_numeric():
            aggs.append(pl.col(name).mean().alias(f"__mean{i}"))
    return aggs


def profile(frame: Any, *selectors: Any, exact_distinct: bool = False) -> Any:
    """
    One row per column: dtype, row count, missing count and rate, distinct
    count (approximate unless exact_distinct=True), min, max and mean.

    Eager: runs one aggregation query over the frame. min and max are shown as
    text so that columns of different types fit in one table.
    """
    schema = frame.schema
    cols = resolve_selectors(schema, selectors) if selectors else frame.columns
    stats = frame._lf.select(_profile_aggs(schema, cols, exact_distinct)).collect().row(0, named=True)
    return _frame(_profile_table(schema, cols, stats))


def _profile_table(schema: Any, cols: List[str], stats: dict) -> pl.DataFrame:
    rows = stats["__rows"]
    out = []
    for i, name in enumerate(cols):
        nulls = stats[f"__nulls{i}"]
        low, high = stats.get(f"__min{i}"), stats.get(f"__max{i}")
        out.append(
            {
                "column": name,
                "dtype": str(schema[name]),
                "rows": rows,
                "nulls": nulls,
                "null_rate": nulls / rows if rows else None,
                "distinct": stats[f"__distinct{i}"],
                "min": None if low is None else str(low),
                "max": None if high is None else str(high),
                "mean": stats.get(f"__mean{i}"),
            }
        )
    return pl.DataFrame(out, schema=_PROFILE_SCHEMA)


def profile_by(frame: Any, *by: str, cols: Any = None, exact_distinct: bool = False) -> Any:
    """
    One row per group: `rows`, then for each profiled column `<col>_nulls`,
    `<col>_distinct`, `<col>_min`, `<col>_max` and (numeric) `<col>_mean`.

    Lazy. `by` defaults to the frame's grouping columns. In order="dplyr" the
    result is sorted by the group columns.
    """
    keys = list(by) or list(frame._groups)
    if not keys:
        raise ValueError("profile_by() needs at least one column to group by")
    frame._require(*keys)
    schema = frame.schema
    selected = cols if isinstance(cols, (list, tuple)) else ([cols] if cols is not None else [])
    targets = resolve_selectors(schema, selected) if selected else frame.columns
    targets = [c for c in targets if c not in keys]

    aggs: List[pl.Expr] = [pl.len().alias("rows")]
    counts = ["rows"]
    for name in targets:
        dtype = schema[name]
        aggs.append(pl.col(name).null_count().alias(f"{name}_nulls"))
        aggs.append(_distinct_expr(name, dtype, exact_distinct).alias(f"{name}_distinct"))
        counts += [f"{name}_nulls", f"{name}_distinct"]
        if _orderable(dtype):
            aggs.append(pl.col(name).min().alias(f"{name}_min"))
            aggs.append(pl.col(name).max().alias(f"{name}_max"))
        if dtype.is_numeric():
            aggs.append(pl.col(name).mean().alias(f"{name}_mean"))

    lf = frame._lf.group_by(keys).agg(aggs).with_columns(pl.col(counts).cast(pl.Int64))
    if frame._dplyr():
        lf = lf.sort(keys, nulls_last=True)
    return frame._new(lf, groups=())


def _profiles(objects: Sequence[Any]) -> List[List[dict]]:
    """Profile rows for each input (a frame, or a profile() result), all computed in one collect_all."""
    from .api import df

    frames = [df(obj) for obj in objects]
    queries = [
        f._lf if f.columns == PROFILE_COLUMNS else f._lf.select(_profile_aggs(f.schema, f.columns, False))
        for f in frames
    ]
    out = []
    for f, data in zip(frames, pl.collect_all(queries)):
        if f.columns != PROFILE_COLUMNS:
            data = _profile_table(f.schema, f.columns, data.row(0, named=True))
        out.append(data.to_dicts())
    return out


def profile_diff(frame: Any, other: Any, rtol: float = 1e-9) -> Any:
    """
    Compare two frames, or two profile() results, column by column.

    Returns one row per column with a `status` of "same", "changed",
    "dtype_changed", "added" (only in `other`) or "removed" (only here), and
    the statistics of both sides. Eager when given frames: both are profiled
    in one collect_all.
    """
    left_rows, right_rows = _profiles([frame, other])
    left = {r["column"]: r for r in left_rows}
    right = {r["column"]: r for r in right_rows}
    stats = ["rows", "nulls", "null_rate", "distinct", "min", "max", "mean"]

    def differs(a: Any, b: Any) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) > rtol * max(abs(a), abs(b), 1.0)
        return a != b

    rows = []
    for name in list(left) + [c for c in right if c not in left]:
        a, b = left.get(name), right.get(name)
        if a is None:
            status = "added"
        elif b is None:
            status = "removed"
        elif a["dtype"] != b["dtype"]:
            status = "dtype_changed"
        elif any(differs(a[s], b[s]) for s in stats):
            status = "changed"
        else:
            status = "same"
        row: Dict[str, Any] = {"column": name, "status": status}
        for side, values in (("left", a), ("right", b)):
            row[f"dtype_{side}"] = values["dtype"] if values else None
            for s in stats:
                row[f"{s}_{side}"] = values[s] if values else None
        rows.append(row)

    schema: Dict[str, Any] = {"column": pl.String, "status": pl.String}
    for side in ("left", "right"):
        schema[f"dtype_{side}"] = pl.String
        for s in stats:
            schema[f"{s}_{side}"] = _PROFILE_SCHEMA[s]
    return _frame(pl.DataFrame(rows, schema=schema))


# ---------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------

_REPORT_SCHEMA = {
    "check": pl.String, "column": pl.String, "passed": pl.Boolean,
    "observed": pl.String, "expected": pl.String,
}


class ValidationReport:
    """Outcome of a validation: one row per check. Truthy when every check passed."""

    def __init__(self, rows: List[dict]):
        self.rows = rows

    @property
    def ok(self) -> bool:
        return all(r["passed"] for r in self.rows)

    @property
    def failures(self) -> List[dict]:
        return [r for r in self.rows if not r["passed"]]

    def __bool__(self) -> bool:
        return self.ok

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, schema=_REPORT_SCHEMA)

    def __repr__(self) -> str:
        if self.ok:
            return f"ValidationReport: all {len(self.rows)} checks passed"
        lines = [f"ValidationReport: {len(self.failures)} of {len(self.rows)} checks failed"]
        for r in self.failures:
            lines.append(f"  - {r['check']}({r['column']}): observed {r['observed']}, expected {r['expected']}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise CrowleyValidationError(repr(self), report=self)


def _check(rows: List[dict], check: str, column: Any, passed: bool, observed: Any, expected: Any) -> None:
    rows.append(
        {"check": check, "column": None if column is None else str(column), "passed": bool(passed),
         "observed": str(observed), "expected": str(expected)}
    )


def _finish(frame: Any, rows: List[dict], on_fail: str) -> Any:
    if on_fail not in ("raise", "report"):
        raise ValueError("on_fail must be 'raise' or 'report'")
    report = ValidationReport(rows)
    if on_fail == "report":
        return report
    report.raise_if_failed()
    return frame


def _dtype_matches(dtype: Any, spec: Any) -> bool:
    if isinstance(spec, tuple):
        return dtype in spec
    if callable(spec) and not (isinstance(spec, type) and issubclass(spec, pl.DataType)):
        return bool(spec(dtype))
    return dtype == spec


def _present(frame: Any, rows: List[dict], check: str, key: Key) -> bool:
    """Record a failure and return False if any column of `key` is absent."""
    missing = [c for c in _key_cols(key) if c not in frame.schema]
    if missing:
        _check(rows, check, _key_label(key), False, f"column {missing[0]!r} not found", "column present")
    return not missing


RULE_NAMES = ("columns", "dtypes", "not_null", "unique", "ranges", "allowed", "strict_columns")


def run_checks(frame: Any, rule_sets: Sequence[Mapping[str, Any]]) -> List[dict]:
    """
    Evaluate one or more rule sets (dicts keyed by RULE_NAMES). Schema rules
    read the schema; every data rule of every set goes into ONE aggregation query.
    """
    schema = frame.schema
    rows: List[dict] = []
    aggs: List[pl.Expr] = []
    pending: List[Tuple[str, str, Any]] = []  # (kind, label, extra)

    def add(kind: str, label: str, extra: Any, *exprs: pl.Expr) -> None:
        for j, e in enumerate(exprs):
            aggs.append(e.alias(f"__c{len(pending)}_{j}"))
        pending.append((kind, label, extra))

    for rules in rule_sets:
        unknown = sorted(set(rules) - set(RULE_NAMES))
        if unknown:
            raise TypeError(f"unknown validation rule(s) {unknown}; available: {list(RULE_NAMES)}")

        # --- schema checks: no data read ---
        required = list(rules.get("columns") or [])
        for name in required:
            present = name in schema
            _check(rows, "column_present", name, present, "present" if present else "missing", "present")
        if rules.get("strict_columns"):
            extra = [c for c in schema.names() if c not in required]
            _check(rows, "no_extra_columns", None, not extra, extra or "none", "none")
        for name, spec in (rules.get("dtypes") or {}).items():
            if _present(frame, rows, "dtype", name):
                expected = getattr(spec, "__name__", None) or str(spec)
                _check(rows, "dtype", name, _dtype_matches(schema[name], spec), schema[name], expected)

        # --- data checks: queued for the single aggregation query ---
        for name in rules.get("not_null") or []:
            if _present(frame, rows, "not_null", name):
                add("not_null", name, None, pl.col(name).null_count())
        for key in rules.get("unique") or []:
            if _present(frame, rows, "unique", key):
                add("unique", _key_label(key), None, pl.len() - _key_expr(key).n_unique())
        for name, bounds in (rules.get("ranges") or {}).items():
            if _present(frame, rows, "range", name):
                lo, hi = bounds
                c = pl.col(name)
                outside = pl.lit(False)
                if lo is not None:
                    outside = outside | (c < lo)
                if hi is not None:
                    outside = outside | (c > hi)
                add("range", name, bounds, outside.sum())
        for name, values in (rules.get("allowed") or {}).items():
            if _present(frame, rows, "allowed_values", name):
                values = list(values)
                add("allowed_values", name, values, (~pl.col(name).is_in(values)).sum())

    if aggs:
        stats = frame._lf.select(aggs).collect().row(0)
        named = dict(zip([e.meta.output_name() for e in aggs], stats))
        # Only when an allowed-values check failed: one more query for a few
        # offending values to show. The passing path stays a single query.
        failed = [
            (i, label, extra) for i, (kind, label, extra) in enumerate(pending)
            if kind == "allowed_values" and named[f"__c{i}_0"]
        ]
        examples_by_check = {}
        if failed:
            shown = frame._lf.select(
                [pl.col(label).filter(~pl.col(label).is_in(extra)).unique().head(5).implode().alias(f"__e{i}")
                 for i, label, extra in failed]
            ).collect().row(0, named=True)
            examples_by_check = {i: list(shown[f"__e{i}"]) for i, _, _ in failed}
        for i, (kind, label, extra) in enumerate(pending):
            violations = named[f"__c{i}_0"] or 0
            if kind == "not_null":
                _check(rows, kind, label, violations == 0, f"{violations} missing", "0 missing")
            elif kind == "unique":
                _check(rows, kind, label, violations == 0, f"{violations} duplicate rows", "0 duplicate rows")
            elif kind == "range":
                _check(rows, kind, label, violations == 0, f"{violations} out of range", f"all in {_fmt_bounds(extra)}")
            else:
                examples = examples_by_check.get(i, [])
                observed = f"{violations} unexpected (e.g. {examples})" if violations else "0 unexpected"
                _check(rows, kind, label, violations == 0, observed, f"values in {extra}")
    return rows


def validate(
    frame: Any,
    columns: Optional[Iterable[str]] = None,
    dtypes: Optional[Mapping[str, Any]] = None,
    not_null: Optional[Iterable[str]] = None,
    unique: Optional[Iterable[Key]] = None,
    ranges: Optional[Mapping[str, Bounds]] = None,
    allowed: Optional[Mapping[str, Iterable[Any]]] = None,
    strict_columns: bool = False,
    on_fail: str = "raise",
) -> Any:
    """
    Assert facts about the frame and return it unchanged, so it can sit in a pipeline.

    - columns: names that must exist (strict_columns=True: and no others)
    - dtypes: {column: Polars dtype, tuple of dtypes, or a predicate such as is_numeric}
    - not_null: columns with no missing values
    - unique: columns, or tuples of columns, with no duplicate rows
    - ranges: {column: (low, high)}, either bound may be None
    - allowed: {column: permitted values}
    Missing values never fail a range or allowed-values check; use not_null.

    `columns`, `dtypes` and `strict_columns` read the schema only. Any other
    check is eager: all of them run in one aggregation query (a failed
    allowed-values check adds one more, to fetch example values).
    on_fail="raise" raises CrowleyValidationError; on_fail="report" returns a
    ValidationReport instead of the frame.
    """
    rules = dict(columns=columns, dtypes=dtypes, not_null=not_null, unique=unique, ranges=ranges,
                 allowed=allowed, strict_columns=strict_columns)
    return _finish(frame, run_checks(frame, [rules]), on_fail)


def validate_categories(frame: Any, allowed: Mapping[str, Iterable[Any]], on_fail: str = "raise") -> Any:
    """
    Check that each listed column only holds permitted values (missing values
    are ignored). Eager: one aggregation query. The failure lists a few of the
    unexpected values.
    """
    return _finish(frame, run_checks(frame, [{"allowed": allowed}]), on_fail)


def validate_bounds(
    frame: Any,
    rows: Optional[Bounds] = None,
    null_rate: Optional[Mapping[str, float]] = None,
    duplicate_rate: Optional[Mapping[Any, float]] = None,
    ranges: Optional[Mapping[str, Bounds]] = None,
    cardinality: Optional[Mapping[str, Bounds]] = None,
    on_fail: str = "raise",
) -> Any:
    """
    Check whole-frame statistics against bounds and return the frame unchanged.

    - rows: (min, max) row count
    - null_rate: {column: highest acceptable share of missing values}
    - duplicate_rate: {column or tuple of columns: highest acceptable share of duplicate rows}
    - ranges: {column: (low, high)} for the column's minimum and maximum
    - cardinality: {column: (min, max)} number of distinct values
    Bounds may use None for an open end.

    Eager: one aggregation query over the frame.
    """
    out: List[dict] = []
    aggs: List[pl.Expr] = [pl.len().alias("__rows")]
    pending: List[Tuple[str, Any, Any]] = []

    def add(kind: str, key: Any, bound: Any, *exprs: pl.Expr) -> None:
        for j, e in enumerate(exprs):
            aggs.append(e.alias(f"__b{len(pending)}_{j}"))
        pending.append((kind, key, bound))

    for name, limit in (null_rate or {}).items():
        if _present(frame, out, "null_rate", name):
            add("null_rate", name, limit, pl.col(name).null_count())
    for key, limit in (duplicate_rate or {}).items():
        if _present(frame, out, "duplicate_rate", key):
            add("duplicate_rate", key, limit, _key_expr(key).n_unique())
    for name, bounds in (ranges or {}).items():
        if _present(frame, out, "range", name):
            add("range", name, bounds, pl.col(name).min(), pl.col(name).max())
    for name, bounds in (cardinality or {}).items():
        if _present(frame, out, "cardinality", name):
            add("cardinality", name, bounds, pl.col(name).n_unique())

    stats = frame._lf.select(aggs).collect().row(0, named=True)
    total = stats["__rows"]
    if rows is not None:
        _check(out, "rows", None, _in_bounds(total, rows), total, _fmt_bounds(rows))
    for i, (kind, key, bound) in enumerate(pending):
        first = stats[f"__b{i}_0"]
        if kind == "null_rate":
            rate = first / total if total else 0.0
            _check(out, kind, key, rate <= bound, f"{rate:.6g}", f"<= {bound}")
        elif kind == "duplicate_rate":
            rate = 1 - first / total if total else 0.0
            _check(out, kind, _key_label(key), rate <= bound, f"{rate:.6g}", f"<= {bound}")
        elif kind == "range":
            low, high = first, stats[f"__b{i}_1"]
            passed = low is None or (_in_bounds(low, bound) and _in_bounds(high, bound))
            _check(out, kind, key, passed, f"[{low}, {high}]", f"within {_fmt_bounds(bound)}")
        else:
            _check(out, kind, key, _in_bounds(first, bound), first, _fmt_bounds(bound))
    return _finish(frame, out, on_fail)


# ---------------------------------------------------------------------
# quarantine: split rows into valid / invalid
# ---------------------------------------------------------------------


def partition_valid(
    frame: Any,
    not_null: Optional[Iterable[str]] = None,
    unique: Optional[Iterable[Key]] = None,
    ranges: Optional[Mapping[str, Bounds]] = None,
    allowed: Optional[Mapping[str, Iterable[Any]]] = None,
    where: Optional[Mapping[str, Any]] = None,
    reason: str = "failed_checks",
) -> Tuple[Any, Any]:
    """
    Split into (valid, invalid) frames by row-level rules. Both are lazy.

    Rules are as in validate(), plus `where`: {label: expression that must be
    true}. The invalid frame gains a `reason` column naming the failed rules.
    Missing values pass range and allowed-values rules (use not_null), but a
    missing `where` result counts as a failure. For `unique`, every copy of a
    duplicated key is invalid.

    To run both halves in one pass over the data, use crowley_frame.collect_all.
    """
    from .api import _to_expr

    rules: List[Tuple[str, pl.Expr]] = []
    for name in not_null or []:
        frame._require(name)
        rules.append((f"not_null({name})", pl.col(name).is_not_null()))
    for key in unique or []:
        frame._require(*_key_cols(key))
        rules.append((f"unique({_key_label(key)})", ~_key_expr(key).is_duplicated()))
    for name, (lo, hi) in (ranges or {}).items():
        frame._require(name)
        ok = pl.lit(True)
        if lo is not None:
            ok = ok & (pl.col(name) >= lo)
        if hi is not None:
            ok = ok & (pl.col(name) <= hi)
        rules.append((f"range({name})", ok.fill_null(True)))
    for name, values in (allowed or {}).items():
        frame._require(name)
        rules.append((f"allowed_values({name})", pl.col(name).is_in(list(values)).fill_null(True)))
    if where:
        ctx = Ctx(set(frame.columns), (), window=True, schema=frame.schema)
        for label, predicate in where.items():
            rules.append((str(label), _to_expr(predicate)._compile(ctx).fill_null(False)))
    if not rules:
        raise ValueError("partition_valid() needs at least one rule")
    if reason in frame.schema:
        raise ValueError(f"partition_valid(): column {reason!r} already exists; pass reason= to rename it")

    all_ok = pl.all_horizontal([ok for _, ok in rules])
    failed = pl.concat_str(
        [pl.when(ok).then(pl.lit("")).otherwise(pl.lit(label + "; ")) for label, ok in rules]
    ).str.strip_chars_end("; ")
    valid = frame._new(frame._lf.filter(all_ok))
    invalid = frame._new(frame._lf.filter(~all_ok).with_columns(failed.alias(reason)))
    return valid, invalid


# ---------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------


def clean_name(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))  # camelCase -> camel_Case
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not text:
        text = "column"
    return "x" + text if text[0].isdigit() else text


def normalize_names(frame: Any) -> Any:
    """
    Rename every column to snake_case: lower case, runs of other characters
    become one underscore, camelCase is split, a leading digit gets an "x"
    prefix. Raises CrowleyValidationError if two columns would get the same
    name. Lazy; reads the schema only.
    """
    mapping = {c: clean_name(c) for c in frame.columns}
    seen: Dict[str, str] = {}
    for old, new in mapping.items():
        if new in seen:
            raise CrowleyValidationError(
                f"normalize_names(): columns {seen[new]!r} and {old!r} would both become {new!r}"
            )
        seen[new] = old
    changed = {old: new for old, new in mapping.items() if old != new}
    return frame._rename(changed) if changed else frame


DEFAULT_NULL_VALUES = ("", "NA", "N/A", "NULL", "null", "None", "NaN", "nan")


def normalize_values(
    frame: Any,
    *selectors: Any,
    trim: bool = True,
    null_values: Optional[Iterable[str]] = DEFAULT_NULL_VALUES,
    case: Optional[str] = None,
    collapse_whitespace: bool = False,
) -> Any:
    """
    Clean text columns (all string columns by default), in this order: trim
    surrounding whitespace, collapse internal whitespace runs to one space,
    change case ("lower", "upper" or "title"), then turn the listed null-like
    strings into missing values. `null_values` is matched exactly, after
    trimming; pass None to keep them. Lazy.
    """
    if case not in (None, "lower", "upper", "title"):
        raise ValueError("case must be None, 'lower', 'upper' or 'title'")
    schema = frame.schema
    cols = resolve_selectors(schema, selectors) if selectors else [c for c in schema.names() if schema[c] == pl.String]
    for name in cols:
        if schema[name] != pl.String:
            raise CrowleyTranslationError(
                f"normalize_values(): column {name!r} is {schema[name]}, not a string column"
            )
    nulls = list(null_values) if null_values else []
    exprs = []
    for name in cols:
        e = pl.col(name)
        if trim:
            e = e.str.strip_chars()
        if collapse_whitespace:
            e = e.str.replace_all(r"\s+", " ")
        if case == "lower":
            e = e.str.to_lowercase()
        elif case == "upper":
            e = e.str.to_uppercase()
        elif case == "title":
            e = e.str.to_titlecase()
        if nulls:
            e = pl.when(e.is_in(nulls)).then(None).otherwise(e)
        exprs.append(e.alias(name))
    return frame._new(frame._lf.with_columns(exprs)) if exprs else frame


_FILL_STRATEGIES = {
    "mean": lambda c: c.fill_null(c.mean()),
    "median": lambda c: c.fill_null(c.median()),
    "min": lambda c: c.fill_null(c.min()),
    "max": lambda c: c.fill_null(c.max()),
    "zero": lambda c: c.fill_null(0),
    "forward": lambda c: c.fill_null(strategy="forward"),
    "backward": lambda c: c.fill_null(strategy="backward"),
}


def replace_missing(
    frame: Any,
    values: Optional[Mapping[str, Any]] = None,
    strategy: Optional[Mapping[str, str]] = None,
) -> Any:
    """
    Fill missing values with explicit per-column rules:

    - values: {column: replacement value}
    - strategy: {column: "mean" | "median" | "min" | "max" | "zero" | "forward" | "backward"}
    Strategies are computed within groups on a grouped frame. A column may
    appear in only one of the two. Lazy.
    """
    values, strategy = dict(values or {}), dict(strategy or {})
    both = sorted(set(values) & set(strategy))
    if both:
        raise ValueError(f"replace_missing(): {both} given in both values= and strategy=")
    frame._require(*values, *strategy)
    exprs = [pl.col(name).fill_null(value) for name, value in values.items()]
    for name, how in strategy.items():
        if how not in _FILL_STRATEGIES:
            raise ValueError(f"replace_missing(): unknown strategy {how!r}; choose from {sorted(_FILL_STRATEGIES)}")
        exprs.append(frame._over(_FILL_STRATEGIES[how](pl.col(name))))
    return frame._new(frame._lf.with_columns(exprs)) if exprs else frame


def drop_missing(frame: Any, *selectors: Any, how: str = "any", max_missing: Union[int, float, None] = None) -> Any:
    """
    Drop rows by missing values in the selected columns (all by default).

    - how="any": drop rows with any missing value
    - how="all": drop rows where every selected column is missing
    - max_missing: keep rows with at most this many missing values; a float
      below 1 is a share of the selected columns
    Lazy.
    """
    if how not in ("any", "all"):
        raise ValueError("how must be 'any' or 'all'")
    cols = resolve_selectors(frame.schema, selectors) if selectors else frame.columns
    if max_missing is not None:
        if how != "any":
            raise ValueError("drop_missing(): pass either how='all' or max_missing=, not both")
        limit = max_missing * len(cols) if isinstance(max_missing, float) and max_missing < 1 else max_missing
        missing = pl.sum_horizontal([pl.col(c).is_null() for c in cols])
        return frame._new(frame._lf.filter(missing <= limit))
    if how == "all":
        return frame._new(frame._lf.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols])))
    return frame._new(frame._lf.drop_nulls(subset=cols if selectors else None))


# ---------------------------------------------------------------------
# anomalies and drift
# ---------------------------------------------------------------------

_FINDING_SCHEMA = {
    "column": pl.String, "issue": pl.String, "value": pl.Float64, "threshold": pl.Float64, "detail": pl.String,
}


def detect_anomalies(
    frame: Any,
    max_null_rate: float = 0.5,
    iqr_factor: Optional[float] = 3.0,
    max_outlier_rate: float = 0.01,
    allowed: Optional[Mapping[str, Iterable[Any]]] = None,
    keys: Optional[Iterable[Key]] = None,
) -> Any:
    """
    Flag suspicious columns. Returns one row per finding (empty if none):

    - high_null_rate: share of missing values above max_null_rate
    - constant: a single distinct value across two or more non-missing rows
    - outliers: share of numeric values beyond iqr_factor x IQR from the
      quartiles above max_outlier_rate (iqr_factor=None skips this)
    - unexpected_values: values outside `allowed` ({column: permitted values})
    - duplicate_keys: duplicate rows for a key in `keys`

    Eager: one aggregation query (it computes quartiles for numeric columns).
    """
    schema = frame.schema
    aggs: List[pl.Expr] = [pl.len().alias("__rows")]
    for i, name in enumerate(schema.names()):
        dtype = schema[name]
        c = pl.col(name)
        aggs.append(c.null_count().alias(f"__nulls{i}"))
        if _orderable(dtype):
            aggs.append((c.min() == c.max()).alias(f"__const{i}"))
        if iqr_factor is not None and dtype.is_numeric():
            q1, q3 = c.quantile(0.25, "linear"), c.quantile(0.75, "linear")
            spread = (q3 - q1) * iqr_factor
            aggs.append(((c < q1 - spread) | (c > q3 + spread)).sum().alias(f"__out{i}"))
    allowed = {k: list(v) for k, v in (allowed or {}).items()}
    frame._require(*allowed)
    for name, values in allowed.items():
        aggs.append((~pl.col(name).is_in(values)).sum().alias(f"__bad_{name}"))
    keys = list(keys or [])
    for j, key in enumerate(keys):
        frame._require(*_key_cols(key))
        aggs.append((pl.len() - _key_expr(key).n_unique()).alias(f"__dup{j}"))

    stats = frame._lf.select(aggs).collect().row(0, named=True)
    rows = stats["__rows"]
    findings: List[dict] = []

    def found(column: str, issue: str, value: float, threshold: Optional[float], detail: str) -> None:
        findings.append({"column": column, "issue": issue, "value": value, "threshold": threshold, "detail": detail})

    for i, name in enumerate(schema.names()):
        nulls = stats[f"__nulls{i}"]
        rate = nulls / rows if rows else 0.0
        if rate > max_null_rate:
            found(name, "high_null_rate", rate, max_null_rate, f"{nulls} of {rows} values missing")
        if stats.get(f"__const{i}") and rows - nulls > 1:
            found(name, "constant", 1.0, None, "one distinct non-missing value")
        outliers = stats.get(f"__out{i}")
        if outliers and rows and outliers / rows > max_outlier_rate:
            found(name, "outliers", outliers / rows, max_outlier_rate,
                  f"{outliers} values beyond {iqr_factor} x IQR from the quartiles")
    for name in allowed:
        bad = stats[f"__bad_{name}"] or 0
        if bad:
            found(name, "unexpected_values", bad / rows, 0.0, f"{bad} values outside the allowed set")
    for j, key in enumerate(keys):
        dup = stats[f"__dup{j}"]
        if dup:
            found(_key_label(key), "duplicate_keys", dup / rows, 0.0, f"{dup} duplicate rows")
    return _frame(pl.DataFrame(findings, schema=_FINDING_SCHEMA))


_DRIFT_SCHEMA = {
    "column": pl.String, "issue": pl.String, "reference": pl.String, "current": pl.String, "detail": pl.String,
}


def _drift_query(frame: Any, cols: List[str], categories: List[str]) -> pl.LazyFrame:
    schema = frame.schema
    aggs: List[pl.Expr] = [pl.len().alias("__rows")]
    for i, name in enumerate(cols):
        c = pl.col(name)
        aggs.append(c.null_count().alias(f"__nulls{i}"))
        if schema[name].is_numeric():
            aggs += [c.mean().alias(f"__mean{i}"), c.std().alias(f"__sd{i}")]
        if name in categories:
            aggs.append(c.drop_nulls().unique().implode().alias(f"__cats{i}"))
    return frame._lf.select(aggs)


def detect_drift(
    frame: Any,
    reference: Any,
    null_rate_tol: float = 0.05,
    mean_shift_tol: float = 0.25,
    row_count_tol: Optional[float] = None,
    categories: Optional[Iterable[str]] = None,
) -> Any:
    """
    Compare this (current) frame with a reference frame. One row per finding:

    - column_added / column_removed / dtype_changed: schema drift
    - null_rate_change: missing-value share moved by more than null_rate_tol
    - mean_shift: a numeric mean moved by more than mean_shift_tol reference
      standard deviations
    - new_categories / missing_categories: for the columns named in `categories`
    - row_count_change: relative change above row_count_tol, if given

    Eager: one aggregation query per frame, run together. Distribution change is judged by
    the mean only; `categories` collects each named column's distinct values.
    """
    from .api import df

    ref = df(reference)
    cur_schema, ref_schema = frame.schema, ref.schema
    findings: List[dict] = []

    def found(column: Optional[str], issue: str, old: Any, new: Any, detail: str) -> None:
        findings.append(
            {"column": column, "issue": issue, "reference": None if old is None else str(old),
             "current": None if new is None else str(new), "detail": detail}
        )

    for name in cur_schema.names():
        if name not in ref_schema:
            found(name, "column_added", None, cur_schema[name], "not in the reference")
    for name in ref_schema.names():
        if name not in cur_schema:
            found(name, "column_removed", ref_schema[name], None, "missing from the current frame")

    shared = [c for c in cur_schema.names() if c in ref_schema]
    comparable = []
    for name in shared:
        if cur_schema[name] != ref_schema[name]:
            found(name, "dtype_changed", ref_schema[name], cur_schema[name], "")
        else:
            comparable.append(name)
    cats = [c for c in (categories or []) if c in comparable]

    cur, old = (
        d.row(0, named=True)
        for d in pl.collect_all([_drift_query(frame, comparable, cats), _drift_query(ref, comparable, cats)])
    )
    rows_cur, rows_old = cur["__rows"], old["__rows"]
    if row_count_tol is not None and rows_old:
        change = (rows_cur - rows_old) / rows_old
        if abs(change) > row_count_tol:
            found(None, "row_count_change", rows_old, rows_cur, f"{change:+.1%}")
    for i, name in enumerate(comparable):
        rate_old = old[f"__nulls{i}"] / rows_old if rows_old else 0.0
        rate_cur = cur[f"__nulls{i}"] / rows_cur if rows_cur else 0.0
        if abs(rate_cur - rate_old) > null_rate_tol:
            found(name, "null_rate_change", f"{rate_old:.4g}", f"{rate_cur:.4g}", f"{rate_cur - rate_old:+.4g}")
        mean_old, mean_cur, sd_old = old.get(f"__mean{i}"), cur.get(f"__mean{i}"), old.get(f"__sd{i}")
        if mean_old is not None and mean_cur is not None and sd_old:
            shift = (mean_cur - mean_old) / sd_old
            if abs(shift) > mean_shift_tol:
                found(name, "mean_shift", f"{mean_old:.6g}", f"{mean_cur:.6g}", f"{shift:+.3g} reference SDs")
        if name in cats:
            set_old, set_cur = set(old[f"__cats{i}"]), set(cur[f"__cats{i}"])
            added, gone = sorted(set_cur - set_old, key=str), sorted(set_old - set_cur, key=str)
            if added:
                found(name, "new_categories", None, added[:10], f"{len(added)} new values")
            if gone:
                found(name, "missing_categories", gone[:10], None, f"{len(gone)} values no longer present")
    return _frame(pl.DataFrame(findings, schema=_DRIFT_SCHEMA))
