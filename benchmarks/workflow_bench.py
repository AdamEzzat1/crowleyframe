"""
Benchmark of the workflow helpers (profiling, validation, safe joins, data
quality, reproducibility) against equivalent hand-written Polars.

    python benchmarks/workflow_bench.py                # 100k, 1M, 5M rows
    python benchmarks/workflow_bench.py --with-15m     # plus 15M rows for representative cases
    python benchmarks/workflow_bench.py --sizes 100000 --reps 3 --stages 1 2

Writes benchmarks/results/workflow_results.json and benchmarks/WORKFLOW_REPORT.md.
Exits non-zero if a helper disagrees with its hand-written equivalent.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import polars.selectors as cs  # noqa: E402
from polars.testing import assert_frame_equal  # noqa: E402

import ab_bench  # noqa: E402
import crowley_frame as cr  # noqa: E402
from crowley_frame import col, define_pipeline, df, is_numeric, mean, n, pipe, where  # noqa: E402

GROUPS = [f"g{i:03d}" for i in range(100)]
NUMERIC = ["row", "id", "v1", "v2", "k"]
ALL = ["row", "id", "grp", "v1", "v2", "k", "txt"]
NULLISH = list(cr.quality.DEFAULT_NULL_VALUES)
TMP = Path(tempfile.mkdtemp(prefix="crowley_bench_"))


def make_inputs(n_rows: int) -> dict:
    """Polars frames shared by both sides; conversion and data generation are not timed."""
    pdf, dim_pdf, _ = ab_bench.make_data(n_rows)
    rng = np.random.default_rng(7)
    variants = np.array(["  alpha ", "beta", "NA", "", " gamma  delta ", "NULL"])
    data = pl.from_pandas(pdf).with_columns(pl.Series("txt", variants[rng.integers(0, len(variants), n_rows)]))
    # `other`: the same rows with 1% of v1 changed and the last 1% of rows dropped
    changed = pl.Series(rng.random(n_rows) < 0.01)
    other = data.with_columns(pl.when(changed).then(pl.col("v1") + 1).otherwise(pl.col("v1")).alias("v1")).head(
        n_rows - n_rows // 100
    )
    dim = pl.from_pandas(dim_pdf)
    snap = TMP / f"snap_{n_rows}.parquet"
    data.write_parquet(snap)
    cr.df(data).snapshot(TMP / f"restore_{n_rows}.parquet", overwrite=True)
    return {
        "raw": {"lf": data.lazy(), "dim": dim.lazy(), "other": other.lazy()},
        "fast": {"cf": df(data, order="fast"), "dim": df(dim, order="fast"), "other": df(other, order="fast")},
        "dplyr": {"cf": df(data, order="dplyr"), "dim": df(dim, order="dplyr"), "other": df(other, order="dplyr")},
        "n": n_rows,
        "snap": snap,
        "restore": TMP / f"restore_{n_rows}.parquet",
    }


# ---------------------------------------------------------------------
# hand-written Polars equivalents used by several cases
# ---------------------------------------------------------------------


def raw_profile(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.select(
        pl.len().alias("rows"),
        pl.all().null_count().name.suffix("_nulls"),
        pl.all().approx_n_unique().name.suffix("_distinct"),
        pl.all().min().name.suffix("_min"),
        pl.all().max().name.suffix("_max"),
        pl.col(NUMERIC).mean().name.suffix("_mean"),
    )


def raw_key_counts(lf, dim):
    left = lf.group_by("id").agg(pl.len().alias("nl"))
    right = dim.group_by("id").agg(pl.len().alias("nr"))
    nl, nr = pl.col("nl").cast(pl.Int64), pl.col("nr").cast(pl.Int64)
    return left.join(right, on="id", how="full", coalesce=True, join_nulls=True).select(
        nl.sum().alias("left_rows"),
        nr.sum().alias("right_rows"),
        (nl - 1).sum().alias("left_duplicate_rows"),
        (nr - 1).sum().alias("right_duplicate_rows"),
        nl.filter(nr.is_not_null()).sum().alias("left_matched_rows"),
        nl.filter(nr.is_null()).sum().alias("left_unmatched_rows"),
        (nl * nr).sum().alias("inner_join_rows"),
    )


def raw_partition(lf):
    ok = pl.col("v1").is_not_null() & ((pl.col("v2") >= -2) & (pl.col("v2") <= 2)).fill_null(True)
    reason = pl.concat_str(
        pl.when(pl.col("v1").is_not_null()).then(pl.lit("")).otherwise(pl.lit("not_null(v1); ")),
        pl.when(((pl.col("v2") >= -2) & (pl.col("v2") <= 2)).fill_null(True)).then(pl.lit("")).otherwise(
            pl.lit("range(v2); ")
        ),
    ).str.strip_chars_end("; ")
    return [lf.filter(ok), lf.filter(~ok).with_columns(reason.alias("failed_checks"))]


def raw_anomalies(lf):
    aggs = [pl.len().alias("rows"), pl.all().null_count().name.suffix("_nulls")]
    aggs += [(pl.col(c).min() == pl.col(c).max()).alias(f"{c}_const") for c in ALL]
    for c in NUMERIC:
        q1, q3 = pl.col(c).quantile(0.25, "linear"), pl.col(c).quantile(0.75, "linear")
        spread = (q3 - q1) * 3.0
        aggs.append(((pl.col(c) < q1 - spread) | (pl.col(c) > q3 + spread)).sum().alias(f"{c}_out"))
    aggs.append((pl.len() - pl.col("row").n_unique()).alias("row_dup"))
    return lf.select(aggs)


def raw_drift(lf):
    return lf.select(
        pl.len().alias("rows"),
        pl.all().null_count().name.suffix("_nulls"),
        pl.col(NUMERIC).mean().name.suffix("_mean"),
        pl.col(NUMERIC).std().name.suffix("_sd"),
        pl.col("grp").drop_nulls().unique().implode().alias("grp_cats"),
    )


def raw_compare(lf, other):
    values = [c for c in ALL if c != "row"]
    left = lf.rename({c: c + "__l" for c in values}).with_columns(pl.lit(True).alias("in_l"))
    right = other.rename({c: c + "__r" for c in values}).with_columns(pl.lit(True).alias("in_r"))
    both = pl.col("in_l").is_not_null() & pl.col("in_r").is_not_null()
    aggs = [
        (pl.col("in_l").is_not_null() & pl.col("in_r").is_null()).sum().alias("only_l"),
        (pl.col("in_l").is_null() & pl.col("in_r").is_not_null()).sum().alias("only_r"),
    ]
    for c in values:
        a, b = pl.col(c + "__l"), pl.col(c + "__r")
        if c in ("v1", "v2"):
            same = ((a - b).abs() <= 1e-9 * b.abs()).fill_null(False) | (a.is_null() & b.is_null())
        else:
            same = a.eq_missing(b)
        aggs.append((both & ~same).sum().alias(c))
    return left.join(right, on="row", how="full", coalesce=True, join_nulls=True).select(aggs)


def raw_pipeline(lf):
    return (
        lf.drop_nulls("v1")
        .filter(pl.col("v1") > -1)
        .with_columns((pl.col("v1") * 2 + pl.col("k")).alias("z"))
        .group_by("grp")
        .agg(pl.col("z").mean().alias("z_mean"), pl.len().alias("rows"))
        .sort("z_mean", descending=True)
    )


PIPELINE = define_pipeline(
    "score",
    pipe.drop_na("v1"),
    pipe.filter("v1 > -1"),
    pipe.mutate(z="v1 * 2 + k"),
    pipe.group_by("grp"),
    pipe.summarise(z_mean=mean(col.z), rows=n()),
    pipe.arrange("-z_mean"),
)


def tracked_pipeline(cf):
    return (
        cf.track_lineage()
        .rename(key="id")
        .mutate(z="v1 * 2 + k", w=col.v1 + col.v2)
        .group_by("grp")
        .summarise(z_mean=mean(col.z), rows=n())
    )


# ---------------------------------------------------------------------
# cases
#
# kind  : "lazy"   builds a plan; timed end to end (build + collect)
#         "eager"  the helper itself runs a query
#         "schema" reads the schema/plan only; timed in microseconds at one size
# crowley / raw : callables taking the inputs for one side and returning a result
# check : optional callable(crowley_result, raw_result) that raises on disagreement
# modes : also time order="fast" (default is order="dplyr")
# big   : include at 15M rows
# ---------------------------------------------------------------------

CASES = []


def case(name, stage, kind, crowley, raw, check=None, modes=False, big=False, note=""):
    CASES.append(dict(name=name, stage=stage, kind=kind, crowley=crowley, raw=raw, check=check, modes=modes,
                      big=big, note=note))


def same_frame(keys=None):
    def check(got, want):
        got = got.to_polars() if hasattr(got, "to_polars") else got
        if keys:
            got, want = got.sort(keys), want.sort(keys)
        assert_frame_equal(got, want, check_dtype=False, check_exact=False, rtol=1e-9)

    return check


def _profile_check(got, want):
    rows = {r["column"]: r for r in got.to_polars().to_dicts()}
    w = want.row(0, named=True)
    for c in ALL:
        assert rows[c]["nulls"] == w[f"{c}_nulls"] and rows[c]["rows"] == w["rows"], c
    assert abs(rows["v1"]["mean"] - w["v1_mean"]) < 1e-9


# --- stage 1 ---------------------------------------------------------
case("profile", 1, "eager", lambda c, i: c["cf"].profile(), lambda r, i: raw_profile(r["lf"]).collect(),
     _profile_check, big=True)
case("profile_by", 1, "lazy",
     lambda c, i: c["cf"].profile_by("grp", cols=["v1", "v2"]).to_polars(),
     lambda r, i: r["lf"].group_by("grp").agg(
         pl.len().alias("rows"),
         *[e for c_ in ("v1", "v2") for e in (
             pl.col(c_).null_count().alias(f"{c_}_nulls"), pl.col(c_).approx_n_unique().alias(f"{c_}_distinct"),
             pl.col(c_).min().alias(f"{c_}_min"), pl.col(c_).max().alias(f"{c_}_max"),
             pl.col(c_).mean().alias(f"{c_}_mean"))],
     ).collect(),
     same_frame(["grp"]), modes=True)
case("profile_diff", 1, "eager",
     lambda c, i: c["cf"].profile_diff(c["other"]),
     lambda r, i: pl.collect_all([raw_profile(r["lf"]), raw_profile(r["other"])]),
     lambda got, want: None, note="raw side computes the two profiles only, not the comparison")
case("validate", 1, "eager",
     lambda c, i: c["cf"].validate(not_null=["row"], unique=["row"], ranges={"k": (0, 1000)},
                                   allowed={"grp": GROUPS}, on_fail="report"),
     lambda r, i: r["lf"].select(
         pl.col("row").null_count().alias("nulls"), (pl.len() - pl.col("row").n_unique()).alias("dups"),
         ((pl.col("k") < 0) | (pl.col("k") > 1000)).sum().alias("outside"),
         (~pl.col("grp").is_in(GROUPS)).sum().alias("bad")).collect(),
     lambda got, want: (_ for _ in ()).throw(AssertionError("validate")) if (not got.ok) != any(want.row(0)) else None,
     big=True)
case("validate_bounds", 1, "eager",
     lambda c, i: c["cf"].validate_bounds(rows=(1, None), null_rate={"v1": 0.1}, duplicate_rate={"row": 0.0},
                                          ranges={"k": (0, 1000)}, cardinality={"grp": (1, 200)}, on_fail="report"),
     lambda r, i: r["lf"].select(
         pl.len(), pl.col("v1").null_count().alias("nulls"), pl.col("row").n_unique().alias("distinct"),
         pl.col("k").min().alias("lo"), pl.col("k").max().alias("hi"), pl.col("grp").n_unique().alias("card")).collect(),
     lambda got, want: None if got.ok else (_ for _ in ()).throw(AssertionError(repr(got))))
case("compare_schema", 1, "schema",
     lambda c, i: c["cf"].compare_schema(c["other"]),
     lambda r, i: r["lf"].collect_schema() == r["other"].collect_schema(),
     lambda got, want: None if got.equal == want else (_ for _ in ()).throw(AssertionError("schema")))
case("resolve_columns", 1, "schema",
     lambda c, i: c["cf"].resolve_columns(where(is_numeric)),
     lambda r, i: list(cs.expand_selector(r["lf"], cs.numeric())),
     lambda got, want: None if got == want else (_ for _ in ()).throw(AssertionError("columns")))
case("explain_expr", 1, "schema",
     lambda c, i: c["cf"].explain_expr("v1 / sum(v1)"),
     lambda r, i: str(pl.col("v1") / pl.col("v1").sum()),
     lambda got, want: None if got == want else (_ for _ in ()).throw(AssertionError("expr")))
case("explain_plan", 1, "schema",
     lambda c, i: PIPELINE(c["cf"]).explain_plan(), lambda r, i: raw_pipeline(r["lf"]).explain(),
     note="both sides build the six-verb pipeline and explain it")
case("audit", 1, "schema", lambda c, i: PIPELINE(c["cf"]).audit(), lambda r, i: raw_pipeline(r["lf"]),
     note="raw side only builds the pipeline; Polars has no audit trail")
case("lineage", 1, "schema",
     lambda c, i: tracked_pipeline(c["cf"]).lineage(),
     lambda r, i: r["lf"].rename({"id": "key"}).with_columns(
         (pl.col("v1") * 2 + pl.col("k")).alias("z"), (pl.col("v1") + pl.col("v2")).alias("w")
     ).group_by("grp").agg(pl.col("z").mean().alias("z_mean"), pl.len().alias("rows")),
     note="raw side only builds the pipeline; Polars has no lineage")

# --- stage 2 ---------------------------------------------------------
case("check_keys", 2, "eager",
     lambda c, i: c["cf"].check_keys("row", on_fail="report"),
     lambda r, i: r["lf"].select(pl.len(), pl.col("row").is_null().sum().alias("nulls"),
                                 pl.col("row").n_unique().alias("distinct")).collect(),
     lambda got, want: None if got.ok else (_ for _ in ()).throw(AssertionError(repr(got))))
case("validate_relationship", 2, "eager",
     lambda c, i: c["cf"].validate_relationship(c["dim"], by="id", relationship="many-to-one", on_fail="report"),
     lambda r, i: r["dim"].select(pl.len(), pl.col("id").n_unique().alias("distinct")).collect(),
     lambda got, want: None if got.ok else (_ for _ in ()).throw(AssertionError(repr(got))))
case("join_diagnostics", 2, "eager",
     lambda c, i: c["cf"].join_diagnostics(c["dim"], by="id"),
     lambda r, i: raw_key_counts(r["lf"], r["dim"]).collect(),
     lambda got, want: assert_frame_equal(got.to_polars().select(want.columns), want, check_dtype=False), big=True)
case("safe_join (eager checks)", 2, "eager",
     lambda c, i: c["cf"].safe_join(c["dim"], by="id", relationship="many-to-one", unmatched="error").to_polars(),
     lambda r, i: r["lf"].join(r["dim"], on="id", how="left", validate="m:1").collect(),
     same_frame(["row"]), modes=True, big=True,
     note="crowley runs join_diagnostics first, then the join; raw uses Polars' built-in m:1 validation")
case("safe_join (lazy check)", 2, "lazy",
     lambda c, i: c["cf"].safe_join(c["dim"], by="id", relationship="many-to-one", check="lazy").to_polars(),
     lambda r, i: r["lf"].join(r["dim"], on="id", how="left", validate="m:1").collect(),
     same_frame(["row"]), modes=True, big=True)

# --- stage 3 ---------------------------------------------------------
case("partition_valid", 3, "lazy",
     lambda c, i: cr.collect_all(c["cf"].partition_valid(not_null=["v1"], ranges={"v2": (-2, 2)})),
     lambda r, i: pl.collect_all(raw_partition(r["lf"])),
     lambda got, want: [same_frame(["row"])(g, w) for g, w in zip(got, want)], big=True)
case("normalize_names", 3, "schema",
     lambda c, i: c["cf"].rename(**{"Row ID": "row", "totalV1": "v1"}).normalize_names(),
     lambda r, i: r["lf"].rename({"row": "Row ID", "v1": "totalV1"}).rename({"Row ID": "row_id", "totalV1": "total_v1"}))
case("normalize_values", 3, "lazy",
     lambda c, i: c["cf"].normalize_values("txt").to_polars(),
     lambda r, i: r["lf"].with_columns(
         pl.when(pl.col("txt").str.strip_chars().is_in(NULLISH)).then(None).otherwise(pl.col("txt").str.strip_chars())
         .alias("txt")).collect(),
     same_frame(), big=True)
case("replace_missing", 3, "lazy",
     lambda c, i: c["cf"].replace_missing(values={"v2": 0.0}, strategy={"v1": "mean"}).to_polars(),
     lambda r, i: r["lf"].with_columns(pl.col("v2").fill_null(0.0), pl.col("v1").fill_null(pl.col("v1").mean())).collect(),
     same_frame())
case("drop_missing", 3, "lazy",
     lambda c, i: c["cf"].drop_missing("v1", "v2", how="all").to_polars(),
     lambda r, i: r["lf"].filter(~pl.all_horizontal(pl.col("v1").is_null(), pl.col("v2").is_null())).collect(),
     same_frame())
case("validate_categories", 3, "eager",
     lambda c, i: c["cf"].validate_categories({"grp": GROUPS}, on_fail="report"),
     lambda r, i: r["lf"].select((~pl.col("grp").is_in(GROUPS)).sum().alias("bad")).collect(),
     lambda got, want: None if got.ok == (want.item() == 0) else (_ for _ in ()).throw(AssertionError("categories")))
case("detect_anomalies", 3, "eager",
     lambda c, i: c["cf"].detect_anomalies(keys=["row"]), lambda r, i: raw_anomalies(r["lf"]).collect(),
     lambda got, want: None if (got.to_polars().filter(pl.col("issue") == "duplicate_keys").height > 0)
     == (want["row_dup"].item() > 0) else (_ for _ in ()).throw(AssertionError("anomalies")))
case("detect_drift", 3, "eager",
     lambda c, i: c["cf"].detect_drift(c["other"], categories=["grp"]),
     lambda r, i: pl.collect_all([raw_drift(r["lf"]), raw_drift(r["other"])]),
     note="raw side computes both sets of statistics only, not the comparison")

# --- stage 4 ---------------------------------------------------------
case("fingerprint_rows", 4, "lazy",
     lambda c, i: c["cf"].fingerprint_rows("id", "grp", "k").to_polars(),
     lambda r, i: r["lf"].with_columns(pl.struct(["id", "grp", "k"]).hash(0).alias("row_hash")).collect(),
     same_frame(), big=True)
case("snapshot", 4, "eager",
     lambda c, i: c["cf"].snapshot(TMP / "bench_snapshot.parquet", overwrite=True),
     lambda r, i: r["lf"].collect().write_parquet(TMP / "bench_raw.parquet"),
     note="crowley also writes the metadata file")
case("restore_snapshot", 4, "lazy",
     lambda c, i: cr.restore_snapshot(i["restore"]).to_polars(),
     lambda r, i: pl.scan_parquet(i["snap"]).collect(), same_frame())
case("expect + assert_expectations", 4, "eager",
     lambda c, i: c["cf"].expect(not_null=["row"]).expect(ranges={"k": (0, 1000)}).assert_expectations(on_fail="report"),
     lambda r, i: r["lf"].select(pl.col("row").null_count().alias("nulls"),
                                 ((pl.col("k") < 0) | (pl.col("k") > 1000)).sum().alias("outside")).collect(),
     lambda got, want: None if got.ok == (sum(want.row(0)) == 0) else (_ for _ in ()).throw(AssertionError("expect")))
case("branch + collect_all", 4, "lazy",
     lambda c, i: cr.collect_all(c["cf"].filter("v1 > 0").branch(
         high=pipe.filter("k > 500"), per_group=lambda f: f.group_by("grp").summarise(m=mean(col.v2)))),
     lambda r, i: pl.collect_all([
         r["lf"].filter(pl.col("v1") > 0).filter(pl.col("k") > 500),
         r["lf"].filter(pl.col("v1") > 0).group_by("grp").agg(pl.col("v2").mean().alias("m"))]),
     lambda got, want: (same_frame(["row"])(got["high"], want[0]), same_frame(["grp"])(got["per_group"], want[1])),
     modes=True)
case("define_pipeline", 4, "lazy",
     lambda c, i: PIPELINE(c["cf"]).to_polars(), lambda r, i: raw_pipeline(r["lf"]).collect(),
     same_frame(["grp"]), modes=True, big=True)
case("compare_outputs", 4, "eager",
     lambda c, i: c["cf"].compare_outputs(c["other"], by="row"),
     lambda r, i: raw_compare(r["lf"], r["other"]).collect(),
     lambda got, want: None if (got.only_left, got.mismatches["v1"]) == (want["only_l"].item(), want["v1"].item())
     else (_ for _ in ()).throw(AssertionError("compare_outputs")), big=True)


# ---------------------------------------------------------------------
# running
# ---------------------------------------------------------------------


def stats(samples):
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def time_pair(fns: dict, reps: int) -> dict:
    """Interleaved timing of several callables after one warm-up each."""
    samples = {k: [] for k in fns}
    for f in fns.values():
        f()
    names = list(fns)
    for rep in range(reps):
        for k in names[rep % len(names):] + names[: rep % len(names)]:
            gc.collect()
            t0 = time.perf_counter()
            out = fns[k]()
            samples[k].append(time.perf_counter() - t0)
            del out
    return {k: stats(v) for k, v in samples.items()}


def run_case(c: dict, inputs: dict, reps: int) -> dict:
    if c["check"] is not None:
        c["check"](c["crowley"](inputs["dplyr"], inputs), c["raw"](inputs["raw"], inputs))
    fns = {"raw": lambda: c["raw"](inputs["raw"], inputs), "dplyr": lambda: c["crowley"](inputs["dplyr"], inputs)}
    if c["modes"]:
        fns["fast"] = lambda: c["crowley"](inputs["fast"], inputs)
    return time_pair(fns, reps if c["kind"] != "schema" else 300)


def write_report(results: dict, path: Path) -> None:
    lines = [
        "# crowley-frame workflow helpers: benchmark against hand-written Polars",
        "",
        "Generated by `python benchmarks/workflow_bench.py`. Do not edit by hand.",
        "",
        "## Environment",
        "",
        "| | |",
        "|---|---|",
    ]
    for key, value in results["environment"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        f"| repetitions | {results['reps']} timed runs per cell after one warm-up (300 for schema-only helpers) |",
        "",
        "Each helper is timed end to end against the Polars a user would write by hand for the same job. "
        "Before timing, the helper's result is checked against the hand-written result where the two are "
        "comparable. Times are median (min-max); the min is the best run. "
        "Ratio = crowley median / raw median; best-run ratio compares the fastest run of each.",
        "",
        "Kinds: **lazy** helpers only extend the plan (timed as build + collect); **eager** helpers run a "
        "query themselves; **schema** helpers read the schema or plan only and are timed in microseconds.",
        "",
    ]
    notes = [(c["name"], c["note"]) for c in CASES if c["note"] and c["name"] in results["cases"]]

    schema_rows = {k: v for k, v in results["schema"].items()}
    if schema_rows:
        lines += [
            "## Schema-only helpers (no data read; microseconds)",
            "",
            "| Helper | Stage | Raw Polars | Crowley | Ratio |",
            "|---|---|---|---|---|",
        ]
        for name, row in schema_rows.items():
            raw, crow = row["raw"], row["dplyr"]
            lines.append(
                f"| `{name}` | {results['cases'][name]['stage']} "
                f"| {raw['median'] * 1e6:.0f} ({raw['min'] * 1e6:.0f}-{raw['max'] * 1e6:.0f}) "
                f"| {crow['median'] * 1e6:.0f} ({crow['min'] * 1e6:.0f}-{crow['max'] * 1e6:.0f}) "
                f"| {crow['median'] / raw['median']:.1f}x |"
            )
        lines.append("")

    for size, block in results["sizes"].items():
        lines += [
            f"## {int(size):,} rows (milliseconds)",
            "",
            "| Helper | Stage | Kind | Raw Polars | Crowley (order=dplyr) | Ratio | Best-run ratio | order=fast ratio |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, row in block.items():
            info = results["cases"][name]
            raw, crow = row["raw"], row["dplyr"]
            fast = f"{row['fast']['median'] / raw['median']:.2f}x" if "fast" in row else ""
            lines.append(
                f"| `{name}` | {info['stage']} | {info['kind']} "
                f"| {raw['median'] * 1e3:.1f} ({raw['min'] * 1e3:.1f}-{raw['max'] * 1e3:.1f}) "
                f"| {crow['median'] * 1e3:.1f} ({crow['min'] * 1e3:.1f}-{crow['max'] * 1e3:.1f}) "
                f"| {crow['median'] / raw['median']:.2f}x | {crow['min'] / raw['min']:.2f}x | {fast} |"
            )
        lines.append("")

    if notes:
        lines += ["## Notes on the comparisons", ""]
        lines += [f"- `{name}`: {note}" for name, note in notes]
        lines.append("")
    lines += ["## Correctness", "", results["correctness"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 5_000_000])
    parser.add_argument("--with-15m", action="store_true", help="also run 15,000,000 rows for representative helpers")
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--report", type=Path, default=HERE / "WORKFLOW_REPORT.md")
    args = parser.parse_args()

    cases = [c for c in CASES if c["stage"] in args.stages]
    results = {
        "environment": ab_bench.environment(),
        "reps": args.reps,
        "cases": {c["name"]: {"stage": c["stage"], "kind": c["kind"]} for c in cases},
        "schema": {},
        "sizes": {},
    }
    print(json.dumps(results["environment"], indent=2), flush=True)
    sizes = sorted(set(args.sizes)) + ([15_000_000] if args.with_15m else [])
    failures = []

    for n_rows in sizes:
        print(f"\n== {n_rows:,} rows ==", flush=True)
        inputs = make_inputs(n_rows)
        block = {}
        for c in cases:
            if n_rows == 15_000_000 and args.with_15m and n_rows not in args.sizes and not c["big"]:
                continue
            if c["kind"] == "schema" and results["schema"].get(c["name"]):
                continue
            try:
                row = run_case(c, inputs, args.reps)
            except AssertionError as e:
                failures.append(f"{c['name']} at {n_rows:,} rows: {str(e).splitlines()[0][:160]}")
                print(f"  !! {c['name']}: {failures[-1]}", flush=True)
                continue
            if c["kind"] == "schema":
                results["schema"][c["name"]] = row
                unit, scale = "us", 1e6
            else:
                block[c["name"]] = row
                unit, scale = "ms", 1e3
            print(
                f"  {c['name']:30s} raw={row['raw']['median'] * scale:10.2f}{unit}  "
                f"crowley={row['dplyr']['median'] * scale:10.2f}{unit}  "
                f"ratio={row['dplyr']['median'] / row['raw']['median']:5.2f}",
                flush=True,
            )
        results["sizes"][str(n_rows)] = block
        del inputs
        gc.collect()

    results["correctness"] = (
        "Every helper agreed with its hand-written Polars equivalent at every size."
        if not failures
        else "**Disagreements:**\n\n" + "\n".join(f"- {f}" for f in failures)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "workflow_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.report)
    print(f"\nWrote {args.out / 'workflow_results.json'} and {args.report}")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
