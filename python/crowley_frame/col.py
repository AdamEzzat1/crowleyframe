from __future__ import annotations

import re
from typing import Any, Callable, Iterable, List, Sequence, Union

import polars as pl

from .expr import ColumnExpr, CrowleyTranslationError, column_not_found


class ColSelector:
    """
    Lightweight description of a column selection, resolved against a schema
    (names and dtypes, never data), so selection stays lazy.

    Kinds:

    - "name"          : exact column name, from col("name")
    - "starts_with"   : prefix match
    - "ends_with"     : suffix match
    - "contains"      : substring match
    - "matches"       : regex match
    - "where"         : dtype predicate, from where(...)
    - "where_numeric" : all numeric columns
    - "where_string"  : all string / categorical columns
    - "everything"    : all columns
    - "last_col"      : the last column
    - "all_of"        : listed names, error if any is missing
    - "any_of"        : listed names, missing ones ignored

    Selectors combine with ~ (complement), | (union), & (intersection)
    and - (difference).
    """

    __slots__ = ("kind", "value", "ignore_case")

    def __init__(self, kind: str, value: Any = None, ignore_case: bool = False):
        self.kind = kind
        self.value = value
        self.ignore_case = ignore_case

    def __repr__(self) -> str:
        return f"ColSelector({self.kind!r}, {self.value!r})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ColSelector) and (self.kind, self.value) == (other.kind, other.value)

    def __invert__(self) -> "ColSelector":
        return ColSelector("not", self)

    def __or__(self, other: Any) -> "ColSelector":
        return ColSelector("or", (self, other))

    def __and__(self, other: Any) -> "ColSelector":
        return ColSelector("and", (self, other))

    def __sub__(self, other: Any) -> "ColSelector":
        return ColSelector("diff", (self, other))

    def _match_names(self, cols: List[str], test: Callable[[str, str], bool]) -> List[str]:
        patterns = [self.value] if isinstance(self.value, str) else list(self.value)
        if self.ignore_case:
            patterns = [p.lower() for p in patterns]
            return [c for c in cols if any(test(c.lower(), p) for p in patterns)]
        return [c for c in cols if any(test(c, p) for p in patterns)]

    def _resolve(self, schema: Any) -> List[str]:
        cols = list(schema)
        kind, value = self.kind, self.value

        if kind == "name":
            if value not in schema:
                raise column_not_found(value, cols)
            return [value]
        if kind == "starts_with":
            return self._match_names(cols, str.startswith)
        if kind == "ends_with":
            return self._match_names(cols, str.endswith)
        if kind == "contains":
            return self._match_names(cols, lambda c, p: p in c)
        if kind == "matches":
            patterns = [value] if isinstance(value, str) else list(value)
            flags = re.IGNORECASE if self.ignore_case else 0
            compiled = [re.compile(p, flags) for p in patterns]
            return [c for c in cols if any(p.search(c) for p in compiled)]
        if kind == "where":
            return [c for c in cols if value(schema[c])]
        if kind == "where_numeric":
            return [c for c in cols if schema[c].is_numeric()]
        if kind == "where_string":
            return [c for c in cols if is_string(schema[c])]
        if kind == "everything":
            return cols
        if kind == "last_col":
            return cols[-1:]
        if kind == "all_of":
            for name in value:
                if name not in schema:
                    raise column_not_found(name, cols)
            return list(value)
        if kind == "any_of":
            return [name for name in value if name in schema]
        if kind == "not":
            drop = set(_resolve_one(value, schema))
            return [c for c in cols if c not in drop]
        if kind in ("or", "and", "diff"):
            left = _resolve_one(value[0], schema)
            right = _resolve_one(value[1], schema)
            if kind == "or":
                return left + [c for c in right if c not in left]
            if kind == "and":
                return [c for c in left if c in right]
            return [c for c in left if c not in right]
        raise ValueError(f"Unknown selector kind: {kind!r}")


def _resolve_one(obj: Any, schema: Any) -> List[str]:
    if isinstance(obj, str):
        if obj not in schema:
            raise column_not_found(obj, list(schema))
        return [obj]
    if isinstance(obj, (ColSelector, ColumnExpr)):
        return obj._resolve(schema)
    raise TypeError(f"Unsupported selector type: {type(obj)!r}")


def resolve_selectors(schema: Any, selectors: Sequence[Any]) -> List[str]:
    """Column names matched by `selectors`, in the order given, without duplicates."""
    out: List[str] = []
    for obj in selectors:
        for name in _resolve_one(obj, schema):
            if name not in out:
                out.append(name)
    return out


# ---------------------------------------------------------------------
# dtype predicates for where()
# ---------------------------------------------------------------------


def is_numeric(dtype: Any) -> bool:
    return dtype.is_numeric()


def is_integer(dtype: Any) -> bool:
    return dtype.is_integer()


def is_float(dtype: Any) -> bool:
    return dtype.is_float()


def is_string(dtype: Any) -> bool:
    return dtype in (pl.String, pl.Categorical, pl.Enum)


def is_boolean(dtype: Any) -> bool:
    return dtype == pl.Boolean


def is_temporal(dtype: Any) -> bool:
    return dtype.is_temporal()


# ---------------------------------------------------------------------
# tidyselect helpers
# ---------------------------------------------------------------------


def _patterns(patterns: Sequence[str]) -> Union[str, tuple]:
    if not patterns:
        raise TypeError("expected at least one pattern")
    return patterns[0] if len(patterns) == 1 else tuple(patterns)


def starts_with(*prefixes: str, ignore_case: bool = False) -> ColSelector:
    """Columns whose names start with any of the prefixes."""
    return ColSelector("starts_with", _patterns(prefixes), ignore_case)


def ends_with(*suffixes: str, ignore_case: bool = False) -> ColSelector:
    """Columns whose names end with any of the suffixes."""
    return ColSelector("ends_with", _patterns(suffixes), ignore_case)


def contains(*substrings: str, ignore_case: bool = False) -> ColSelector:
    """Columns whose names contain any of the substrings."""
    return ColSelector("contains", _patterns(substrings), ignore_case)


def matches(*patterns: str, ignore_case: bool = False) -> ColSelector:
    """Columns whose names match any of the regular expressions."""
    return ColSelector("matches", _patterns(patterns), ignore_case)


def everything() -> ColSelector:
    """All columns."""
    return ColSelector("everything")


def last_col() -> ColSelector:
    """The last column."""
    return ColSelector("last_col")


def all_of(names: Iterable[str]) -> ColSelector:
    """The listed columns; error if any is missing."""
    return ColSelector("all_of", tuple(names))


def any_of(names: Iterable[str]) -> ColSelector:
    """The listed columns that exist."""
    return ColSelector("any_of", tuple(names))


def num_range(prefix: str, numbers: Iterable[int], width: int = 0) -> ColSelector:
    """Columns named prefix + number, e.g. num_range("wk", range(1, 4)) -> wk1, wk2, wk3."""
    return ColSelector("any_of", tuple(f"{prefix}{i:0{width}d}" for i in numbers))


def where(predicate: Any) -> ColSelector:
    """
    Columns whose dtype satisfies `predicate`: a function of the Polars dtype
    (is_numeric, is_string, ...), a dtype, or a tuple of dtypes.

    The predicate sees the schema only. Predicates on column *values* would
    need a data scan and are not translated.
    """
    if isinstance(predicate, tuple) or isinstance(predicate, pl.DataType) or (
        isinstance(predicate, type) and issubclass(predicate, pl.DataType)
    ):
        dtypes = predicate if isinstance(predicate, tuple) else (predicate,)
        return ColSelector("where", lambda dtype: dtype in dtypes)
    if not callable(predicate):
        raise TypeError("where() takes a dtype predicate such as is_numeric, a dtype, or a tuple of dtypes")

    def check(dtype: Any) -> bool:
        try:
            result = predicate(dtype)
        except Exception as e:
            raise CrowleyTranslationError(
                "where() predicates receive the column's dtype, not its values, "
                f"so selection never scans data ({type(e).__name__}: {e})"
            ) from None
        if not isinstance(result, bool):
            raise CrowleyTranslationError(
                f"where() predicate must return True or False for a dtype, got {type(result)!r}"
            )
        return result

    return ColSelector("where", check)


class _ColNamespace:
    """
    Column DSL for crowley-frame.

    Usage examples
    --------------
    col("user_id")                    # a column: selector and expression
    col.user_id, col["user id"]       # same, attribute / item form
    col.user_id * 2 > col.limit       # expressions for mutate / filter
    col.starts_with("user_")          # the tidyselect helpers, also available
    col.where_numeric()               #   as plain functions (starts_with, where, ...)
    ~col.starts_with("tmp_")          # everything except
    """

    def __call__(self, name: str) -> ColumnExpr:
        """A single column by exact name."""
        return ColumnExpr(name)

    def __getitem__(self, name: str) -> ColumnExpr:
        return ColumnExpr(name)

    def __getattr__(self, name: str) -> ColumnExpr:
        if name.startswith("__"):
            raise AttributeError(name)
        return ColumnExpr(name)

    starts_with = staticmethod(starts_with)
    ends_with = staticmethod(ends_with)
    contains = staticmethod(contains)
    matches = staticmethod(matches)
    everything = staticmethod(everything)
    last_col = staticmethod(last_col)
    all_of = staticmethod(all_of)
    any_of = staticmethod(any_of)
    num_range = staticmethod(num_range)
    where = staticmethod(where)

    @staticmethod
    def where_numeric() -> ColSelector:
        """Select all numeric columns."""
        return ColSelector("where_numeric")

    @staticmethod
    def where_string() -> ColSelector:
        """Select all string/categorical columns."""
        return ColSelector("where_string")


col = _ColNamespace()
