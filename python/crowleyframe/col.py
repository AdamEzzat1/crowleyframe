from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ColSelector:
    kind: str
    value: Any | None = None


class _ColNamespace:
    """
    Column selection DSL for crowley-frame.

    Examples:
        col("user_id")
        col.starts_with("user_")
        col.ends_with("_id")
        col.contains("date")
        col.matches("^user_.*_score$")
        col.where_numeric()
        col.where_string()
    """

    def __call__(self, name: str) -> ColSelector:
        return ColSelector("name", name)

    def starts_with(self, prefix: str) -> ColSelector:
        return ColSelector("starts_with", prefix)

    def ends_with(self, suffix: str) -> ColSelector:
        return ColSelector("ends_with", suffix)

    def contains(self, pattern: str) -> ColSelector:
        return ColSelector("contains", pattern)

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
