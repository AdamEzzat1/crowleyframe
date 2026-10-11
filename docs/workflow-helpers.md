# Workflow helpers: validation, profiling, safe joins, lineage

Notes for the four-stage workflow layer. Timings are from
`benchmarks/WORKFLOW_REPORT.md` (Polars 1.23.0, 8 threads); rerun with
`python benchmarks/workflow_bench.py --with-15m`.

## Where things live

| Module | Contents |
|---|---|
| `diagnostics.py` | `compare_schema`, `resolve_columns`, `explain_expr`, `explain_plan` |
| `tracking.py` | audit trail, `track_lineage`, `lineage`; wraps every verb once at import |
| `quality.py` | profiling, validation, quarantine, cleaning, anomaly and drift detection |
| `keys.py` | `check_keys`, `validate_relationship`, `join_diagnostics`, `safe_join` |
| `workflow.py` | fingerprints, snapshots, expectations, `branch`, `define_pipeline`, `collect_all`, `compare_outputs` |

Each helper is a plain function of a frame, attached to `Frame` in
`__init__.py`, so it chains (`cf.validate(...)`) and pipes (`cf >> pipe.validate(...)`).

Two error types: `CrowleyTranslationError` (cannot be expressed lazily) and
`CrowleyValidationError` (the data failed a check; `.report` holds the details).

## Lazy or eager

**Schema or plan only (no data read):** `compare_schema`, `resolve_columns`,
`explain_expr`, `explain_plan`, `audit`, `track_lineage`, `lineage`,
`normalize_names`, `expect`, `clear_expectations`, `expectations`,
`define_pipeline`, and `validate` / `assert_expectations` / `check_keys` when
only schema rules are asked for.

**Lazy (extend the plan):** `profile_by`, `partition_valid`,
`normalize_values`, `replace_missing`, `drop_missing`, `fingerprint_rows`,
`branch`, `safe_join` with no checks or with `check="lazy"`.

**Eager (run a query; this is their purpose):**

| Helper | What it runs |
|---|---|
| `profile`, `validate`, `validate_bounds`, `validate_categories`, `assert_expectations`, `check_keys`, `detect_anomalies` | one aggregation query |
| `validate` / `validate_categories` when an allowed-values check fails | one more query, to fetch example values |
| `profile_diff`, `detect_drift` | one aggregation query per frame, collected together |
| `validate_relationship` | one aggregation query per side that must be unique |
| `join_diagnostics`, `safe_join(check="eager")` with a relationship or `unmatched` | one query: rows per key on each side, joined |
| `compare_outputs` | runs both plans |
| `collect_all` | runs every frame given, sharing common sub-plans |
| `snapshot` | runs the plan and writes two files |

No eager helper pulls rows into Python: each collects one row of aggregates.

## Every function

Ratio is crowley / hand-written Polars, best run of each, at 5M rows (15M in
brackets where measured). Medians on the development machine scatter by ±20%
or more between runs, so read these as "level with raw Polars" unless noted.

| Function | Kind | Equivalent Polars | Ratio | Tests | Known limitations |
|---|---|---|---|---|---|
| `profile` | eager | one `select` of `len`, `null_count`, `approx_n_unique`, `min`, `max`, `mean` | 1.12x (0.87x) | stage 1 | distinct counts are approximate unless `exact_distinct=True`; min/max shown as text |
| `profile_by` | lazy | `group_by().agg()` of the same statistics | 1.16x | stage 1 | wide output, one set of columns per profiled column |
| `profile_diff` | eager | two profile queries in `collect_all` | 0.97x | stage 1 | compares summary statistics, not rows |
| `validate` | eager / schema | one `select` of violation counts | 0.93x (1.04x) | stage 1 | missing values never fail range or allowed-values rules |
| `validate_bounds` | eager | one `select` of `len`, `null_count`, `n_unique`, `min`, `max` | 0.92x | stage 1 | exact `n_unique`, so cost grows with cardinality |
| `compare_schema` | schema | `collect_schema()` on both | 81 µs vs 117 µs | stage 1 | names, dtypes and order only |
| `explain_plan` | schema | `LazyFrame.explain()` | 1.0 ms vs 0.5 ms | stage 1 | warnings are keyword matches on the plan text |
| `explain_expr` | schema | `str(pl.Expr)` | 118 µs vs 64 µs | stage 1 | |
| `resolve_columns` | schema | `cs.expand_selector` | 57 µs vs 218 µs | stage 1 | |
| `audit` | schema | none | 0.9 ms incl. building a six-verb pipeline | stage 1 | records verbs and arguments, not timings; `validate` and other checks that return the same frame are not recorded |
| `track_lineage` / `lineage` | schema | none | 0.9 ms vs 0.2 ms to build a tracked pipeline | stage 1 | opt-in; `pivot_wider` outputs and unmodelled verbs show `(unknown)`; right-side join columns show as `right.<name>` |
| `check_keys` | eager / schema | one `select` of `len`, null count, `n_unique` | 1.28x | stage 2 | |
| `validate_relationship` | eager | `n_unique` on the side(s) that must be unique | 1.09x | stage 2 | sub-millisecond on a 100k-row lookup table |
| `join_diagnostics` | eager | group both sides by key, full join the counts, sum | 1.09x (1.21x) | stage 2 | key dtypes must match; predicts equi-joins only |
| `safe_join`, eager checks | eager + lazy | `join(validate="m:1")` | 2.24x (1.49x) | stage 2 | runs `join_diagnostics` before the join, by design |
| `safe_join`, `check="lazy"` | lazy | `join(validate="m:1")` | 1.06x (1.02x) | stage 2 | error comes from Polars at collect time; cannot check unmatched rows |
| `semi_join`, `anti_join`, `cross_join` | lazy | `join(how=...)` | see `REPORT.md` | verb layer | added in the previous layer |
| `partition_valid` | lazy | two `filter`s, one with a reason column | 1.05x (1.11x) | stage 3 | row-level rules only; every copy of a duplicated key is invalid |
| `normalize_names` | schema | `rename` | 274 µs vs 62 µs | stage 3 | raises if two names collide |
| `normalize_values` | lazy | `str.strip_chars`, `when/then/otherwise` | 1.01x (0.88x) | stage 3 | string columns only; null-like list is matched exactly |
| `replace_missing` | lazy | `fill_null` | 1.08x | stage 3 | plan identical to hand-written |
| `drop_missing` | lazy | `drop_nulls` / `filter` on `sum_horizontal` | 1.04x | stage 3 | |
| `validate_categories` | eager | one `select` of `(~is_in).sum()` | 0.94x | stage 3 | |
| `detect_anomalies` | eager | one `select` with quartiles per numeric column | 1.09x | stage 3 | quartile-based outliers only; quartiles make it the slowest check |
| `detect_drift` | eager | two statistics queries in `collect_all` | 0.99x | stage 3 | distribution change judged by the mean only; categories only for columns you name |
| `fingerprint_rows` | lazy | `pl.struct(cols).hash(seed)` | 1.22x (1.21x) | stage 4 | plan identical to hand-written; hashes are not stable across Polars versions |
| `snapshot` | eager, writes files | `collect().write_parquet()` | 0.84x | stage 4 | about 20 ms fixed cost for metadata and rename; Parquet only |
| `restore_snapshot` | lazy | `scan_parquet` | 0.92x | stage 4 | needs the metadata file unless `check=False` |
| `expect`, `clear_expectations` | schema | none | part of the next row | stage 4 | expectations refer to columns by name; a later rename is not followed |
| `assert_expectations` | eager | one `select` of violation counts | 1.20x | stage 4 | 17 ms at 5M rows, so the ratio is mostly fixed cost |
| `branch` + `collect_all` | lazy + eager | `pl.collect_all([...])` | 0.82x | stage 4 | sharing the upstream work relies on Polars' common-sub-plan elimination |
| `define_pipeline` | lazy | the same verbs chained | 0.99x (0.98x) | stage 4 | steps are opaque callables; not serialisable |
| `compare_outputs` | eager | full join on the key, sum of differences | 0.82x (0.98x) | stage 4 | `by` must be unique on both sides; without `by`, both sides are sorted on every column |

## Findings from the benchmark

- **Two overheads were found and fixed.** `detect_drift` ran its two
  statistics queries one after the other (1.4x to 2x); they now run in one
  `collect_all`. `validate` collected example offending values on every run
  (up to 1.46x); it now does so only when an allowed-values check fails.
- **One overhead is deliberate.** `safe_join` with eager checks is about 2x a
  validated Polars join, because it counts keys on both sides first so that it
  can fail before the join with exact numbers. `check="lazy"` avoids it.
- **Schema-only helpers** cost 1.9x to 4.4x their raw equivalents, all at or
  below about a millisecond, independent of data size.
- **Ordering modes** only matter for `profile_by` (sorted by group in dplyr
  mode), `safe_join` (keeps left order) and pipelines containing `arrange`;
  none showed a measurable difference with 100 groups.
- **Memory** was not measured for these helpers. The eager ones return one row
  of aggregates; `compare_outputs` holds a full join of both inputs.

## Deferred

| Item | Why |
|---|---|
| Tracking lineage through expectations and renames | needs expectations to hold column identities rather than names |
| Row-level difference output from `compare_outputs` | returns counts only; a diff frame needs a design for wide outputs |
| Distribution tests in `detect_drift` (PSI, KS) | no native Polars form for KS; histograms need a binning policy |
| Snapshot formats other than Parquet, and a content hash in the metadata | kept to one format; hashing a large file doubles the write time |
| A fingerprint that is stable across Polars versions | needs a hashing plugin (Rust), which is out of the core path |
| Serialisable pipelines | steps are Python callables |
| Unmatched-row handling for right and full joins in `safe_join` | `unmatched` covers left rows only |

## Risks

- **Polars 2.0 is untested** (`polars>=1.17,<2`). `join_diagnostics` uses
  `join_nulls`/`nulls_equal` through the same compatibility lookup as the joins.
- **Benchmark noise** hides anything under roughly 20%. The lazy helpers whose
  plans are asserted identical to hand-written Polars in the tests
  (`replace_missing`, `fingerprint_rows`, `drop_missing`, `define_pipeline`,
  `safe_join` without checks) are the ones that can be trusted to be level.
- **`explain_plan` warnings** match keywords in Polars' plan text, which can
  change between versions.
- **Lineage never raises**: if it cannot model a verb it marks new columns
  `(unknown)` rather than failing the pipeline.
