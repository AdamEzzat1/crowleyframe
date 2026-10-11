from . import diagnostics, functions, keys, quality, tracking, workflow
from .api import Frame, GroupedFrame, df, options, set_order
from .col import (
    all_of,
    any_of,
    col,
    contains,
    ends_with,
    everything,
    is_boolean,
    is_float,
    is_integer,
    is_numeric,
    is_string,
    is_temporal,
    last_col,
    matches,
    num_range,
    starts_with,
    where,
)
from .diagnostics import SchemaComparison
from .expr import ColumnNotFoundError, CrowleyTranslationError, CrowleyValidationError, TranslationError, lit
from .functions import (
    across,
    between,
    case_when,
    ceiling,
    coalesce,
    cumsum,
    dense_rank,
    desc,
    exp,
    first,
    floor,
    if_else,
    is_in,
    is_na,
    lag,
    last,
    lead,
    log,
    mean,
    median,
    min_rank,
    n,
    n_distinct,
    na_if,
    near,
    not_na,
    percent_rank,
    pick,
    rolling_mean,
    row_number,
    sd,
    sqrt,
    var,
)
from .pipe import pipe
from .quality import ValidationReport
from .workflow import OutputComparison, Pipeline, collect_all, define_pipeline, restore_snapshot, snapshot_metadata

# Workflow helpers live in their own modules as functions of a frame; attach
# them as Frame methods so they chain and pipe like any other verb.
_METHODS = {
    diagnostics: ["compare_schema", "resolve_columns", "explain_expr", "explain_plan"],
    tracking: ["audit", "track_lineage", "lineage"],
    quality: [
        "profile", "profile_by", "profile_diff", "validate", "validate_categories", "validate_bounds",
        "partition_valid", "normalize_names", "normalize_values", "replace_missing", "drop_missing",
        "detect_anomalies", "detect_drift",
    ],
    keys: ["check_keys", "validate_relationship", "join_diagnostics", "safe_join"],
    workflow: [
        "fingerprint_rows", "snapshot", "expect", "clear_expectations", "expectations",
        "assert_expectations", "branch", "compare_outputs",
    ],
}
for _module, _names in _METHODS.items():
    for _name in _names:
        setattr(Frame, _name, getattr(_module, _name))
tracking.instrument(Frame)

# Available as crowley_frame.sum etc., but left out of __all__ so that
# `from crowley_frame import *` does not shadow Python builtins.
sum = functions.sum  # noqa: A001
min = functions.min  # noqa: A001
max = functions.max  # noqa: A001
abs = functions.abs  # noqa: A001
round = functions.round  # noqa: A001
any = functions.any  # noqa: A001
all = functions.all  # noqa: A001

__all__ = [
    "Frame",
    "GroupedFrame",
    "df",
    "col",
    "pipe",
    "options",
    "set_order",
    "CrowleyTranslationError",
    "CrowleyValidationError",
    "TranslationError",
    "ColumnNotFoundError",
    "ValidationReport",
    "SchemaComparison",
    "OutputComparison",
    "Pipeline",
    "define_pipeline",
    "collect_all",
    "restore_snapshot",
    "snapshot_metadata",
    "lit",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "last_col",
    "where",
    "all_of",
    "any_of",
    "num_range",
    "is_numeric",
    "is_integer",
    "is_float",
    "is_string",
    "is_boolean",
    "is_temporal",
    "across",
    "pick",
    "na_if",
    "near",
    "not_na",
    "percent_rank",
    "between",
    "case_when",
    "ceiling",
    "coalesce",
    "cumsum",
    "dense_rank",
    "desc",
    "exp",
    "first",
    "floor",
    "if_else",
    "is_in",
    "is_na",
    "lag",
    "last",
    "lead",
    "log",
    "mean",
    "median",
    "min_rank",
    "n",
    "n_distinct",
    "rolling_mean",
    "row_number",
    "sd",
    "sqrt",
    "var",
]
