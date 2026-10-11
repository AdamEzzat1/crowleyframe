"""
Key integrity and join safety: check_keys, validate_relationship,
join_diagnostics, safe_join.

All of these except safe_join(check="lazy") are eager: they run one
aggregation query (join_diagnostics: one query over both frames' key counts).
None of them collects rows.
"""
from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import polars as pl

from .expr import CrowleyTranslationError, CrowleyValidationError
from .quality import _check, _finish, _frame

Keys = Union[str, Sequence[str], None]

_RELATIONSHIPS = {
    "one-to-one": "1:1", "one_to_one": "1:1", "1:1": "1:1",
    "one-to-many": "1:m", "one_to_many": "1:m", "1:m": "1:m",
    "many-to-one": "m:1", "many_to_one": "m:1", "m:1": "m:1",
    "many-to-many": "m:m", "many_to_many": "m:m", "m:m": "m:m",
}


def _relationship(value: str) -> str:
    if value not in _RELATIONSHIPS:
        raise ValueError(
            "relationship must be 'one-to-one', 'one-to-many', 'many-to-one' or 'many-to-many', "
            f"got {value!r}"
        )
    return _RELATIONSHIPS[value]


def _as_list(value: Keys) -> List[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _join_keys(frame: Any, other: Any, by: Keys, left_on: Keys, right_on: Keys) -> Tuple[List[str], List[str]]:
    keys = _as_list(by)
    if keys:
        left, right = keys, keys
    elif left_on is not None or right_on is not None:
        left, right = _as_list(left_on), _as_list(right_on)
        if not left or len(left) != len(right):
            raise ValueError("left_on and right_on must name the same number of columns")
    else:
        right_cols = set(other.columns)
        left = right = [c for c in frame.columns if c in right_cols]
        if not left:
            raise ValueError("No common columns to join on; pass by= or left_on=/right_on=")
    frame._require(*left)
    other._require(*right)
    return left, right


def _check_key_dtypes(frame: Any, other: Any, left: List[str], right: List[str]) -> List[Tuple[str, str, Any, Any]]:
    return [
        (a, b, frame.schema[a], other.schema[b])
        for a, b in zip(left, right)
        if frame.schema[a] != other.schema[b]
    ]


def _key_stats(frame: Any, keys: List[str]) -> Tuple[int, int, int]:
    """(rows, rows with a missing key part, duplicate rows) in one aggregation query."""
    key = pl.col(keys[0]) if len(keys) == 1 else pl.struct(keys)
    rows, nulls, distinct = (
        frame._lf.select(
            pl.len().alias("__rows"),
            pl.any_horizontal([pl.col(k).is_null() for k in keys]).sum().alias("__nulls"),
            key.n_unique().alias("__distinct"),
        )
        .collect()
        .row(0)
    )
    return rows, nulls, (rows - distinct) if rows else 0


def check_keys(
    frame: Any,
    by: Keys,
    other: Any = None,
    other_by: Keys = None,
    allow_nulls: bool = False,
    allow_duplicates: bool = False,
    on_fail: str = "raise",
) -> Any:
    """
    Check the key column(s) `by`: present, no missing values, no duplicates,
    and (when `other` is given) the same dtypes as its key columns `other_by`
    (default: the same names). Returns the frame unchanged.

    Presence and dtype checks read the schema only. The missing and duplicate
    checks are eager: one aggregation query, skipped if both are allowed.
    """
    from .api import df

    keys = _as_list(by)
    if not keys:
        raise ValueError("check_keys() needs at least one key column")
    rows: List[dict] = []
    missing = [k for k in keys if k not in frame.schema]
    for k in keys:
        _check(rows, "key_present", k, k not in missing, "missing" if k in missing else "present", "present")

    if other is not None:
        right = df(other)
        right_keys = _as_list(other_by) or keys
        if len(right_keys) != len(keys):
            raise ValueError("by and other_by must name the same number of columns")
        for a, b in zip(keys, right_keys):
            if b not in right.schema:
                _check(rows, "key_present_in_other", b, False, "missing", "present")
            elif a not in missing:
                same = frame.schema[a] == right.schema[b]
                _check(rows, "key_dtype", a, same, frame.schema[a], right.schema[b])

    if not missing and not (allow_nulls and allow_duplicates):
        _, nulls, duplicates = _key_stats(frame, keys)
        label = ", ".join(keys)
        if not allow_nulls:
            _check(rows, "key_not_null", label, nulls == 0, f"{nulls} rows with a missing key", "0")
        if not allow_duplicates:
            _check(rows, "key_unique", label, duplicates == 0, f"{duplicates} duplicate rows", "0")
    return _finish(frame, rows, on_fail)


def validate_relationship(
    frame: Any,
    other: Any,
    by: Keys = None,
    relationship: str = "many-to-one",
    left_on: Keys = None,
    right_on: Keys = None,
    on_fail: str = "raise",
) -> Any:
    """
    Check key cardinality between this frame (left) and `other` (right):
    "one-to-one", "one-to-many", "many-to-one" or "many-to-many". A "one" side
    must have no duplicate keys. Returns the frame unchanged.

    Eager: one aggregation query per side that must be unique (none for
    many-to-many).
    """
    from .api import df

    right = df(other)
    code = _relationship(relationship)
    left_keys, right_keys = _join_keys(frame, right, by, left_on, right_on)
    rows: List[dict] = []
    if code[0] == "1":
        duplicates = _key_stats(frame, left_keys)[2]
        _check(rows, "left_keys_unique", ", ".join(left_keys), duplicates == 0, f"{duplicates} duplicate rows", "0")
    if code[2] == "1":
        duplicates = _key_stats(right, right_keys)[2]
        _check(rows, "right_keys_unique", ", ".join(right_keys), duplicates == 0, f"{duplicates} duplicate rows", "0")
    return _finish(frame, rows, on_fail)


DIAGNOSTIC_COLUMNS = [
    "left_rows", "right_rows", "left_null_keys", "right_null_keys",
    "left_duplicate_rows", "right_duplicate_rows",
    "left_matched_rows", "left_unmatched_rows", "right_matched_rows", "right_unmatched_rows",
    "inner_join_rows", "left_join_rows", "full_join_rows", "left_match_rate", "row_multiplication",
]


def diagnostics_plan(
    left_lf: pl.LazyFrame, right_lf: pl.LazyFrame, left_keys: List[str], right_keys: List[str], nulls_match: bool
) -> pl.LazyFrame:
    """The lazy query behind join_diagnostics: count rows per key on each side, join the counts, sum."""
    from .api import _NULLS_EQUAL_KW

    left = left_lf.group_by(left_keys).agg(pl.len().alias("__nl"))
    right = right_lf.group_by(right_keys).agg(pl.len().alias("__nr")).rename(dict(zip(right_keys, left_keys)))
    joined = left.join(right, on=left_keys, how="full", coalesce=True, **{_NULLS_EQUAL_KW: nulls_match})
    nl, nr = pl.col("__nl").cast(pl.Int64), pl.col("__nr").cast(pl.Int64)
    null_key = pl.any_horizontal([pl.col(k).is_null() for k in left_keys])
    return joined.select(
        nl.sum().alias("left_rows"),
        nr.sum().alias("right_rows"),
        nl.filter(null_key).sum().alias("left_null_keys"),
        nr.filter(null_key).sum().alias("right_null_keys"),
        (nl - 1).sum().alias("left_duplicate_rows"),
        (nr - 1).sum().alias("right_duplicate_rows"),
        nl.filter(nr.is_not_null()).sum().alias("left_matched_rows"),
        nl.filter(nr.is_null()).sum().alias("left_unmatched_rows"),
        nr.filter(nl.is_not_null()).sum().alias("right_matched_rows"),
        nr.filter(nl.is_null()).sum().alias("right_unmatched_rows"),
        (nl * nr).sum().alias("inner_join_rows"),
    ).with_columns(
        pl.all().cast(pl.Int64).fill_null(0)
    ).with_columns(
        (pl.col("inner_join_rows") + pl.col("left_unmatched_rows")).alias("left_join_rows"),
        (pl.col("inner_join_rows") + pl.col("left_unmatched_rows") + pl.col("right_unmatched_rows")).alias(
            "full_join_rows"
        ),
        (pl.col("left_matched_rows") / pl.col("left_rows")).alias("left_match_rate"),
        ((pl.col("inner_join_rows") + pl.col("left_unmatched_rows")) / pl.col("left_rows")).alias(
            "row_multiplication"
        ),
    )


def _diagnose(frame: Any, right: Any, by: Keys, left_on: Keys, right_on: Keys, na_matches: str) -> pl.DataFrame:
    if na_matches not in ("na", "never"):
        raise ValueError("na_matches must be 'na' or 'never'")
    left_keys, right_keys = _join_keys(frame, right, by, left_on, right_on)
    mismatched = _check_key_dtypes(frame, right, left_keys, right_keys)
    if mismatched:
        a, b, left_type, right_type = mismatched[0]
        raise CrowleyValidationError(
            f"join key dtypes differ: left {a!r} is {left_type}, right {b!r} is {right_type}; cast one side first"
        )
    return diagnostics_plan(frame._lf, right._lf, left_keys, right_keys, na_matches == "na").collect()


def join_diagnostics(
    frame: Any,
    other: Any,
    by: Keys = None,
    left_on: Keys = None,
    right_on: Keys = None,
    na_matches: str = "na",
) -> Any:
    """
    What a join between this frame and `other` would do, without doing it.
    Returns a one-row frame: rows, missing keys, duplicate rows, matched and
    unmatched rows on each side, the row counts an inner, left and full join
    would produce, the left match rate, and `row_multiplication` (left-join
    rows / left rows; above 1 means the join duplicates left rows).

    Eager: one query that counts rows per key on each side and joins the counts.
    """
    from .api import df

    return _frame(_diagnose(frame, df(other), by, left_on, right_on, na_matches))


def safe_join(
    frame: Any,
    other: Any,
    by: Keys = None,
    how: str = "left",
    left_on: Keys = None,
    right_on: Keys = None,
    relationship: Optional[str] = None,
    unmatched: str = "ignore",
    na_matches: str = "na",
    suffixes: Tuple[str, str] = ("_x", "_y"),
    check: str = "eager",
) -> Any:
    """
    A join that checks its assumptions first.

    - relationship: required key cardinality ("one-to-one", "one-to-many",
      "many-to-one", "many-to-many"); a violation raises CrowleyValidationError
    - unmatched: what to do when left rows have no match in `other`:
      "ignore" (default), "warn" or "error"
    - check="eager" (default): run join_diagnostics now, so problems surface
      here with counts, before anything downstream runs
    - check="lazy": hand `relationship` to Polars, which checks it when the
      plan runs (no extra query; the error then comes from Polars). `unmatched`
      cannot be checked lazily.

    With no relationship and unmatched="ignore" this is an ordinary lazy join.
    """
    from .api import df

    if how not in ("left", "inner", "right", "full"):
        raise ValueError("how must be 'left', 'inner', 'right' or 'full'")
    if unmatched not in ("ignore", "warn", "error"):
        raise ValueError("unmatched must be 'ignore', 'warn' or 'error'")
    if check not in ("eager", "lazy"):
        raise ValueError("check must be 'eager' or 'lazy'")
    right = df(other)
    code = _relationship(relationship) if relationship is not None else None
    validate = None

    if check == "lazy":
        if unmatched != "ignore":
            raise CrowleyTranslationError(
                "safe_join(check='lazy') cannot check unmatched rows; use check='eager' or unmatched='ignore'"
            )
        validate = code
    elif code is not None or unmatched != "ignore":
        stats = _diagnose(frame, right, by, left_on, right_on, na_matches).row(0, named=True)
        problems = []
        if code is not None and code[0] == "1" and stats["left_duplicate_rows"]:
            problems.append(f"left keys are not unique ({stats['left_duplicate_rows']} duplicate rows)")
        if code is not None and code[2] == "1" and stats["right_duplicate_rows"]:
            problems.append(f"right keys are not unique ({stats['right_duplicate_rows']} duplicate rows)")
        if problems:
            raise CrowleyValidationError(
                f"safe_join(): relationship {relationship!r} violated: " + "; ".join(problems), report=stats
            )
        if unmatched != "ignore" and stats["left_unmatched_rows"]:
            message = (
                f"safe_join(): {stats['left_unmatched_rows']} of {stats['left_rows']} left rows "
                "have no match in the right frame"
            )
            if unmatched == "error":
                raise CrowleyValidationError(message, report=stats)
            warnings.warn(message, stacklevel=3)

    return frame._join(right, how, None, left_on, right_on, suffixes, by, na_matches, validate)
