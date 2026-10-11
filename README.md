# crowley-frame

Tidyverse-style data manipulation for Python that compiles to Polars lazy plans.

You write dplyr/tidyr-shaped pipelines; crowley-frame turns each verb into
part of a `polars.LazyFrame` plan and runs nothing until you ask for the
result. Polars does the work, so performance tracks hand-written Polars.

```python
import crowley_frame as cr
from crowley_frame import df, col, pipe, mean, n, desc

sales = df({
    "team":  ["a", "a", "b", "b", "b"],
    "score": [10.0, 30.0, 5.0, None, 25.0],
})

out = (
    sales
    >> pipe.drop_na("score")
    >> pipe.group_by("team")
    >> pipe.mutate(share="score / sum(score)")      # per-team, because the frame is grouped
    >> pipe.summarise(avg=mean(col.score), rows=n())
    >> pipe.arrange(desc(col.avg))
)
out.to_polars()
# team  avg   rows
# a     20.0  2
# b     15.0  2
```

## Status

Alpha (0.2.x). The API is still moving.

- **Engine:** Polars, through its Python API. Tested on Polars 1.17 and 1.23;
  Polars 2.x is not yet supported (`polars>=1.17,<2`).
- **Tests:** 118, run on both Polars versions.
- **Rust:** the `src/` extension is not used. Earlier versions of this README
  described a Rust core; the speed comes from Polars' own Rust engine. The
  build backend is still maturin, so installing from source needs a Rust
  toolchain for now.
- **Old implementation:** the previous pandas-backed code is kept as
  `crowley_frame.legacy` for benchmarking only.

## Install

From a checkout:

```bash
pip install -e .
```

Requires Python 3.8+, `polars`, and (for `to_pandas()` / `to_arrow()`) `pandas` and `pyarrow`.

## Core ideas

### Frames are lazy

`df()` accepts a dict, a Polars `DataFrame`/`LazyFrame`, a pandas `DataFrame`
or a PyArrow table. Every verb returns a new `Frame` holding a longer plan.

These run the plan:

| Call | Returns |
|---|---|
| `collect()` | a `Frame` over the computed result |
| `to_polars()`, `to_pandas()`, `to_arrow()` | the data |
| `pull("col")` | one column as a Polars Series |
| `glimpse()`, printing a frame | a preview of the first rows |
| `pivot_wider()` | runs the plan so far (its output columns depend on the data), then continues lazily |

`explain()` shows the Polars plan without running it.

### Two ways to write an expression

```python
cf.mutate(r=col.x / cr.sum(col.x), big=(col.x > 2) & ~cr.is_na(col.y))   # column objects
cf.mutate(r="x / sum(x)", big="x > 2 and not is_na(y)")                   # strings
```

Both compile to the same Polars expression. Strings are parsed against a
whitelist and never evaluated: bare names are columns, quoted text and numbers
are literals, and only crowley-frame functions can be called. Anything else
raises `CrowleyTranslationError`.

On a grouped frame, aggregates and window functions in `mutate` and `filter`
are computed per group. In `summarise` they are plain aggregates.

`sum`, `min`, `max`, `abs`, `round`, `any` and `all` are available as
`cr.sum` etc.; they are left out of `from crowley_frame import *` so they do
not shadow Python's builtins.

### Method chaining or pipes

Every verb is a `Frame` method and is also available as `pipe.<verb>(...)` for `>>`:

```python
cf.filter("x > 0").arrange("-x")
cf >> pipe.filter("x > 0") >> pipe.arrange("-x")
```

### Ordering modes

| Mode | Behaviour |
|---|---|
| `order="dplyr"` (default) | Keeps the row order dplyr users see: `summarise`/`count` sorted by group keys, stable `arrange`, `distinct` keeps first occurrences in place, joins keep left row order, `pivot_longer` returns rows in input order. |
| `order="fast"` | Emits the plain Polars plan with no ordering guarantees beyond what the verb asks for. |

Set it per frame with `df(data, order="fast")` or `cf.with_order("fast")`, or
globally with `cr.set_order("fast")`. The only large measured cost of dplyr
mode is `pivot_longer` (about 3x to 4x).

### Errors instead of slow fallbacks

There is no pandas fallback and no hidden Python-per-row path. Unknown columns
raise `ColumnNotFoundError` (a `KeyError`) with a suggestion; anything that
cannot be expressed as a lazy Polars plan raises `CrowleyTranslationError`;
failed data checks raise `CrowleyValidationError`.

## Verbs

| Area | Verbs |
|---|---|
| Rows | `filter`, `arrange`, `distinct`, `slice_head`, `slice_tail`, `slice_min`, `slice_max`, `slice_sample` |
| Columns | `select`, `mutate`, `transmute`, `rename`, `rename_with`, `relocate`, `pull`, `glimpse` |
| Groups | `group_by`, `ungroup`, `summarise` / `summarize`, `count`, `add_count` |
| Joins | `left_join`, `inner_join`, `right_join`, `full_join`, `semi_join`, `anti_join`, `cross_join` |
| Reshaping | `pivot_longer`, `pivot_wider`, `separate`, `unite` |
| Missing values | `drop_na`, `replace_na`, `fill`, `drop_missing`, `replace_missing` |

### Selecting columns

```python
from crowley_frame import starts_with, ends_with, contains, matches, everything, last_col
from crowley_frame import all_of, any_of, num_range, where, is_numeric, is_string

cf.select("id", starts_with("score"))
cf.select(where(is_numeric) & ~starts_with("tmp_"))
cf.select(num_range("wk", range(1, 5)))
```

Selectors combine with `~`, `|`, `&` and `-`, and are resolved from the schema
only. `where()` takes a predicate on the column's dtype, not its values.

### Expression functions

- **Summaries:** `mean`, `median`, `sd`, `var`, `first`, `last`, `n`, `n_distinct`, plus `cr.sum`, `cr.min`, `cr.max`, `cr.any`, `cr.all`
- **Windows:** `lag`, `lead`, `cumsum`, `rolling_mean`, `row_number`, `min_rank`, `dense_rank`, `percent_rank`
- **Values:** `if_else`, `case_when`, `coalesce`, `na_if`, `between`, `near`, `is_na`, `not_na`, `is_in`, `lit`, plus `cr.abs`, `cr.round`, `sqrt`, `exp`, `log`, `floor`, `ceiling`
- **Several columns at once:** `across`, `pick`

```python
from crowley_frame import across, case_when, sd

cf.group_by("team").summarise(across(where(is_numeric), {"mean": mean, "sd": sd}))   # score_mean, score_sd
cf.mutate(band=case_when((col.score < 10, "low"), (col.score < 20, "mid"), _default="high"))
```

## Workflow helpers

Beyond the verbs: checks and tooling for pipelines that have to be trusted.

| Area | Helpers |
|---|---|
| Inspection | `profile`, `profile_by`, `profile_diff`, `compare_schema`, `explain_plan`, `explain_expr`, `resolve_columns`, `audit`, `track_lineage` / `lineage` |
| Validation | `validate`, `validate_bounds`, `validate_categories`, `expect` / `assert_expectations` / `clear_expectations` |
| Joins | `check_keys`, `validate_relationship`, `join_diagnostics`, `safe_join` |
| Data quality | `partition_valid`, `normalize_names`, `normalize_values`, `detect_anomalies`, `detect_drift` |
| Reproducibility | `fingerprint_rows`, `snapshot` / `restore_snapshot`, `branch`, `define_pipeline`, `collect_all`, `compare_outputs` |

```python
orders = df({"cust": [1, 1, 2, 9], "amount": [10.0, 20.0, -5.0, 40.0]})
customers = df({"cust": [1, 2, 3], "name": ["Ada", "Bo", "Cy"]})

# Fail early, with counts, if the join would not behave as assumed
orders.safe_join(customers, by="cust", relationship="many-to-one", unmatched="error")
# CrowleyValidationError: safe_join(): 1 of 4 left rows have no match in the right frame

# Assert facts and keep going; all data rules run in one pass
checked = orders.validate(not_null=["cust"], ranges={"amount": (0, None)}, on_fail="report")
checked.ok          # False: one amount is out of range

# Quarantine bad rows instead of failing
valid, invalid = orders.partition_valid(ranges={"amount": (0, None)})
invalid.to_polars() # the -5.0 row, with failed_checks = "range(amount)"
```

Helpers that need the data run one aggregation query and never pull rows into
Python; helpers that only need the schema read no data. Which is which, with
limitations per function, is in [docs/workflow-helpers.md](docs/workflow-helpers.md).

## Performance

Benchmarks compare hand-written Polars lazy queries, the old pandas
implementation, and this implementation in both ordering modes, on generated
data from 100k to 15M rows. Results are checked for equality before timing.

- **Against the old implementation:** faster on all 12 benchmark queries it
  can run, at every size. On a five-verb pipeline at 15M rows it uses about
  0.5 GB of extra memory against 1.6 GB.
- **Against raw Polars:** `order="fast"` is within 10% on 16 to 18 of 22
  queries by median, and 16 of 22 produce a plan identical to the hand-written
  one. Where the plan differs it is for a stated reason (for example, counts
  are returned as signed 64-bit integers).
- **Caveat:** timings on the development machine vary by 20% or more between
  runs, including for identical plans, so they cannot confirm single-digit
  overhead per query. Plan identity is the firmer evidence.
- **Building a plan** costs up to about 5x raw Polars, under half a
  millisecond for a ten-verb pipeline. It only matters on very small data.

Full tables: [benchmarks/REPORT.md](benchmarks/REPORT.md) (verbs) and
[benchmarks/WORKFLOW_REPORT.md](benchmarks/WORKFLOW_REPORT.md) (workflow helpers).

```bash
python benchmarks/ab_bench.py --with-15m
```

```bash
python benchmarks/workflow_bench.py --with-15m
```

## Differences from dplyr and tidyr

- Aggregates skip missing values by default (`mean(x)` is R's `mean(x, na.rm = TRUE)`); pass `na_rm=False` for R's behaviour.
- `NaN` is not treated as missing. pandas `NaN` becomes a missing value on the way in.
- R's dot-arguments use an underscore or a plain name: `_names`, `_default`, `keep_all`, `before`, `after`.
- `case_when` takes `(condition, value)` tuples.
- `summarise` always returns an ungrouped frame.
- `slice_head` and `slice_tail` default to 5 rows.
- `unite` gives a missing result if any part is missing, unless `na_rm=True`.

Not available yet: `rowwise`, `if_any` / `if_all`, inequality and rolling
joins, per-verb `.by`, `nest` / `unnest`, `complete` / `expand`, the
`separate_wider_*` family, and stringr / lubridate / forcats helpers. The full
list with reasons is in [docs/lazy-backend-notes.md](docs/lazy-backend-notes.md).

## Development

```bash
python -m pytest
```

Tests cover each verb against literal expected output or the equivalent
Polars query, check that lazy verbs do not execute the plan, and check that
untranslatable input raises rather than falling back.

## License

MIT
