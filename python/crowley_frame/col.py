from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ColSelector:
    """
    Lightweight description of a column selection.

    The Rust side (selectors.rs) and the Python helper
    `_select_columns_from_selectors` both understand these `kind` strings:

    - "name"          : exact column name, from col("name")
    - "starts_with"   : prefix match
    - "ends_with"     : suffix match
    - "contains"      : substring match
    - "matches"       : regex match
    - "where_numeric" : all numeric columns
    - "where_string"  : all string / categorical columns
    """
    kind: str
    value: Any | None = None


class _ColNamespace:
    """
    Column selection DSL for crowley-frame.

    Usage examples
    --------------
    col("user_id")                    # exact name
    col.starts_with("user_")          # prefix
    col.ends_with("_date")            # suffix
    col.contains("time")              # substring
    col.matches(r"^x_\\d+$")          # regex
    col.where_numeric()               # all numeric columns
    col.where_string()                # all string-like columns
    """

    def __call__(self, name: str) -> ColSelector:
        """Select a single column by exact name."""
        return ColSelector("name", name)

    def starts_with(self, prefix: str) -> ColSelector:
        """Select all columns whose names start with the given prefix."""
        return ColSelector("starts_with", prefix)

    def ends_with(self, suffix: str) -> ColSelector:
        """Select all columns whose names end with the given suffix."""
        return ColSelector("ends_with", suffix)

    def contains(self, substring: str) -> ColSelector:
        """Select all columns whose names contain the given substring."""
        return ColSelector("contains", substring)

    def matches(self, pattern: str) -> ColSelector:
        """Regex-based column name match."""
        return ColSelector("matches", pattern)

    def where_numeric(self) -> ColSelector:
        """Select all numeric columns."""
        return ColSelector("where_numeric")

    def where_string(self) -> ColSelector:
        """Select all string/categorical columns."""
        return ColSelector("where_string")


col = _ColNamespace()
