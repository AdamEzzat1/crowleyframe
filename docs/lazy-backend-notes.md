# Lazy Polars backend: engineering notes

These notes cover the lazy rewrite and the second verb layer. Numbers come from
`benchmarks/REPORT.md` (Polars 1.23.0, 8 threads); rerun
`python benchmarks/ab_bench.py --with-15m` to refresh them.

## How it is put together

| Module | Responsibility |
|---|---|
| `api.py` | `Frame`: one method per verb; each returns a new `Frame` over a longer `pl.LazyFrame` plan |
| `expr.py` | Deferred expressions (`Expr`), the compile context (`Ctx`), error types |
| `functions.py` | R-named expression helpers, `across()`, `pick()` |
| `parser.py` | Expression strings to `Expr` through an `ast` whitelist; nothing is evaluated |
| `col.py` | Column selectors, resolved against the schema only |
| `pipe.py` | `>>` support: `pipe.verb(...)` for every `Frame` method |
| `legacy/` | The previous pandas implementation, kept only for the A/B benchmark |

An expression is compiled to `pl.Expr` when a verb supplies its context. That
is how the same `sum(x)` becomes a per-group window (`.over(groups)`) in
`mutate`/`filter` and a plain aggregate in `summarise`.

## Eager boundaries

Nothing runs until one of these is called:

- `collect()`, `to_polars()`, `to_pandas()`, `to_arrow()`: run the plan.
- `pull()`: runs the plan and returns one column.
- `glimpse(n)`, `repr()`: run the plan far enough to produce the first few rows.
- `pivot_wider()`: runs the plan up to that point, because the output columns
  depend on the data. It then continues lazily. This is the only verb that
  executes; it predates the rewrite and was kept for compatibility.

Selectors, including `where()`, read the schema only.

## What maps cleanly to Polars

In `order="fast"` these produce the plan a Polars user would write by hand
(checked by comparing `explain()` output in the tests and the benchmark):

`select`, `filter`, `mutate`, `transmute`, `arrange`, `rename`, `rename_with`,
`relocate`, `distinct`, `group_by` + `summarise`, `drop_na`, `replace_na`,
`fill`, `pivot_longer`, all joins, `across()` in `mutate` and `summarise`.

## Where the plan differs from hand-written Polars, and why

| Verb | Difference | Reason | Measured effect |
|---|---|---|---|
| `count`, `n()` in `summarise` | one cast of the count on the aggregated result | Polars counts in `UInt32`; a signed `Int64` means `n() - 5` cannot wrap | none measurable; the cast runs on one row per group |
| `add_count`, `n()` in `mutate` | cast of the per-group count before it is broadcast | same | about 4% in a focused test at 5M rows |
| `slice_min`, `slice_max` | filter on a top-k cutoff, then (without ties) `top_k` on the few rows left | keeps ties, as dplyr does by default; avoids a full sort | 6x to 9x faster than `LazyFrame.top_k`/`bottom_k` at 1M rows and above on Polars 1.23, which sorts the whole frame; level on 1.17 |
| joins | `join_nulls=True` | dplyr matches missing keys to each other by default (`na_matches="na"`) | none measurable; pass `na_matches="never"` for the Polars default |
| `if_else` | an extra `is_null` branch | a missing condition gives a missing result, as in dplyr; Polars' `when/then/otherwise` sends it to `otherwise` | not measured separately |

Two casts were moved after measuring. Casting a count inside
`group_by().agg()` took Polars off its fast aggregation path (about 1.3x on the
five-verb pipeline), and casting the broadcast column in `add_count` cost about
19% against 4% for casting before the broadcast.

## Where `order="dplyr"` costs performance

| Verb | What dplyr mode adds | Cost |
|---|---|---|
| `pivot_longer` | a row index and a stable sort, so rows come back in input order (Polars returns column by column) | about 3x to 4x on this verb, the largest ordering cost measured |
| `summarise`, `count` | sort by group keys | a sort over one row per group; not measurable with 100 groups, grows with the number of groups |
| `arrange` | stable sort (`maintain_order=True`) | within noise in the benchmark |
| `distinct` | keeps first occurrences in input order | within noise in the benchmark; expect more with many distinct rows |
| joins | `maintain_order="left"` | within noise in the benchmark |
| `slice_min`, `slice_max` | sort of the selected rows | negligible; only the selected rows are sorted |

`order="fast"` adds none of these.

## Intentional differences from dplyr

- **Missing values in aggregates are skipped by default.** `mean(x)` behaves
  like R's `mean(x, na.rm = TRUE)`. Pass `na_rm=False` for R's default.
- **`NaN` is not treated as missing** by `is_na`, `drop_na`, `fill` or
  `replace_na`. pandas `NaN` becomes a missing value on the way in.
- **Arguments R spells with a leading dot** use a leading underscore or plain
  name: `_names`, `_default`, `keep_all`, `before`, `after`.
- **`case_when`** takes `(condition, value)` tuples instead of `~` formulas.
- **Strings.** A string passed to `mutate`/`filter`/`summarise` is an
  expression; inside a function, a string in a column position is a column
  name and in a value position is a literal. Use `lit("text")` to force a literal.
- **`summarise` always returns an ungrouped frame** (dplyr drops only the last
  grouping level by default).
- **`slice_head`/`slice_tail` default to 5 rows** (dplyr: 1), kept from the
  previous API.
- **`where()` takes a dtype predicate**, not a function of the column's values.
- **`pick()` returns a struct expression**, not a data frame.
- **`slice_min`/`slice_max` never select rows with a missing ordering value.**
- **Grouped `slice_min`/`slice_max` leave rows in input order**; ungrouped
  results are sorted in dplyr mode.
- **`unite` gives a missing result if any part is missing** (tidyr pastes the
  text `"NA"`), unless `na_rm=True`.

## Deferred, and why

| Feature | Why it was left out |
|---|---|
| Lazy `pivot_wider` | Output columns depend on the data. Newer Polars has a lazy pivot that needs the columns declared up front; that needs an API decision first. |
| `rowwise`, `c_across` | No native Polars form; would need a Python function per row. |
| `if_any`, `if_all`, `across()` in `filter` | Straightforward (`any_horizontal`/`all_horizontal`); not in this layer's scope. |
| `where()` on column values | Needs a data scan to pick columns. |
| Inequality and rolling joins (`join_by`) | Polars `join_where` is inner-only on 1.x and does not keep row order. |
| `slice_sample` on grouped frames, `weight_by` | No native weighted sampling. |
| `separate(fill="left")`, `separate(convert=True)`, `extra="merge"` with a regex separator | Each needs more than a single native expression; they raise `CrowleyTranslationError`. |
| `.groups=`, `.by=` per-verb grouping | Design decision pending: argument naming and default regrouping. |
| `nest`/`unnest`, `complete`/`expand`, `separate_wider_*` | Not started. |
| Python functions in `summarise` | Would run once per group in Python. Raises `CrowleyTranslationError`. |
| stringr / lubridate / forcats helpers | Not started; Polars `.str`/`.dt` are not exposed through string expressions yet. |

## Risks and future work

- **Polars 2.0 is untested.** `pyproject.toml` requires `polars>=1.17,<2`.
  2.0 makes the streaming engine the default for lazy queries, which does not
  guarantee row order for `group_by`, joins and unpivot. Every ordering
  guarantee above needs re-checking there, and the costs re-measuring.
- **Benchmark noise.** On the development machine, queries with identical plans
  vary by ±20% or more between runs, so the benchmark cannot confirm
  single-digit overhead per query. Plan identity is the reliable check.
- **Plan construction** costs up to about 5x raw Polars, and stays under 0.4 ms
  for the ten-verb pipeline. It matters only for very small data or very many
  tiny queries. Most of it is expression parsing and schema lookups.
- **Schema lookups.** Each verb resolves the schema of the plan so far, to
  validate names. That is cheap for plans over in-memory data; it has not been
  measured on long plans over `scan_parquet` sources.
- **Tie-breaking** in `slice_min`/`slice_max` with `with_ties=False` is
  arbitrary, as in Polars.
- **The README** still describes the pandas implementation and a Rust core.
- **The Rust extension** is still built by maturin and is unused.
