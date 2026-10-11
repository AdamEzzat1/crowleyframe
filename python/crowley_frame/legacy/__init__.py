"""
Frozen copy of the pandas-backed implementation that preceded the lazy
Polars rewrite. Kept only so the A/B benchmark can run old and new side by
side; not part of the public API.
"""
from .api import Frame, GroupedFrame, df, lag, lead, rolling_mean
from .col import col
from .pipe import pipe

__all__ = ["Frame", "GroupedFrame", "df", "col", "pipe", "lag", "lead", "rolling_mean"]
