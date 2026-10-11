"""
A/B benchmark: legacy (pandas-backed) crowley-frame vs the lazy Polars-backed
rewrite, against hand-written Polars lazy queries.

    python benchmarks/ab_bench.py                 # 100k, 1M, 5M rows
    python benchmarks/ab_bench.py --with-15m      # also 15M rows
    python benchmarks/ab_bench.py --sizes 100000 --reps 5 --no-memory
    python benchmarks/ab_bench.py --queries count add_count

Writes benchmarks/results/ab_results.json and benchmarks/REPORT.md.
Exits non-zero if any implementation returns a different result.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from polars.testing import assert_frame_equal  # noqa: E402

import crowley_frame as new  # noqa: E402
from crowley_frame import across, col, desc, if_else, legacy, mean, n, sd  # noqa: E402

IMPLS = ["raw_polars", "legacy", "new_fast", "new_dplyr"]
LABELS = {
    "raw_polars": "Raw Polars lazy",
    "legacy": "Current (pandas)",
    "new_fast": "New, order=fast",
    "new_dplyr": "New, order=dplyr",
}
DIM_ROWS = 100_000
SEED = 42


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------


def make_data(n_rows: int):
    rng = np.random.default_rng(SEED)
    groups = np.array([f"g{i:03d}" for i in range(100)])
    v1 = rng.normal(size=n_rows)
    v2 = rng.normal(size=n_rows)
    v1[rng.random(n_rows) < 0.05] = np.nan
    v2[rng.random(n_rows) < 0.05] = np.nan
    pdf = pd.DataFrame(
        {
            "row": np.arange(n_rows, dtype=np.int64),
            "id": rng.integers(0, DIM_ROWS, n_rows),
            "grp": groups[rng.integers(0, 100, n_rows)],
            "v1": v1,
            "v2": v2,
            "k": rng.integers(0, 1000, n_rows),
        }
    )
    dim_pdf = pd.DataFrame(
        {
            "id": np.arange(DIM_ROWS, dtype=np.int64),
            "w": rng.normal(size=DIM_ROWS),
            "label": groups[rng.integers(0, 100, DIM_ROWS)],
        }
    )
    half_pdf = dim_pdf.loc[dim_pdf["w"] > 0, ["id"]].reset_index(drop=True)
    return pdf, dim_pdf, half_pdf


def make_inputs(n_rows: int):
    """
    Each implementation gets (data, dim, half) in its native form; conversion is not timed.
    `dim` has one row per id; `half` holds roughly half of the ids (for semi/anti joins).
    """
    pandas_frames = make_data(n_rows)
    polars_frames = [pl.from_pandas(f) for f in pandas_frames]
    return {
        "raw_polars": tuple(f.lazy() for f in polars_frames),
        "legacy": tuple(legacy.df(f) for f in pandas_frames),
        "new_fast": tuple(new.df(f, order="fast") for f in polars_frames),
        "new_dplyr": tuple(new.df(f, order="dplyr") for f in polars_frames),
    }


# ---------------------------------------------------------------------
# queries
#
# raw    : the query as a Polars user would write it (same semantics)
# cf     : the crowley-frame query, run on the new implementation
# legacy : "same" if the old implementation runs the same code, a separate
#          builder where its API differs, or None where it lacks the verb
# keys   : columns identifying a row, for comparing results whose order is
#          not guaranteed; None means row order itself must match
# ---------------------------------------------------------------------

QUERIES = {}


def query(name, description, keys, legacy_builder="same", legacy_order=True):
    def register(pair):
        raw, cf = pair()
        QUERIES[name] = {
            "raw": raw,
            "cf": cf,
            "legacy": cf if legacy_builder == "same" else legacy_builder,
            "keys": keys,
            # False where order=dplyr deliberately orders rows differently from the old implementation
            "legacy_order": legacy_order,
            "description": description,
        }
        return pair

    return register


@query("select", "select two columns", None)
def _():
    return (
        lambda lf, dim, half: lf.select(["id", "v1"]),
        lambda cf, dim, half: cf.select("id", "v1"),
    )


@query("filter_mutate", "filter, then mutate two columns", ["row"])
def _():
    return (
        lambda lf, dim, half: lf.filter(pl.col("v1") > 0).with_columns(
            (pl.col("v1") * 2).alias("a"), (pl.col("v1") + pl.col("v2")).alias("b")
        ),
        lambda cf, dim, half: cf.filter("v1 > 0").mutate(a="v1 * 2", b="v1 + v2"),
    )


@query("group_mean", "group_by (100 groups), then mean", ["grp"])
def _():
    return (
        lambda lf, dim, half: lf.group_by("grp").agg(pl.col("v1").mean().alias("mean_v1")),
        lambda cf, dim, half: cf.group_by("grp").summarise(mean_v1=("v1", "mean")),
    )


@query("arrange", "arrange by two columns", ["grp", "k", "row"])
def _():
    return (
        lambda lf, dim, half: lf.sort("grp", "k", descending=[False, True]),
        lambda cf, dim, half: cf.arrange("grp", "-k"),
    )


@query("join", f"left join to a {DIM_ROWS:,}-row table", ["row"])
def _():
    return (
        lambda lf, dim, half: lf.join(dim, on="id", how="left"),
        lambda cf, dim, half: cf.left_join(dim, on="id"),
    )


@query("pipeline", "filter, mutate, group_by, summarise, arrange (5 verbs)", ["grp"])
def _():
    return (
        lambda lf, dim, half: (
            lf.filter(pl.col("v1") > 0)
            .with_columns((pl.col("v1") * 2 + pl.col("k")).alias("z"))
            .group_by("grp")
            .agg(pl.col("z").mean().alias("mean_z"), pl.len().alias("n"))
            .sort("mean_z", descending=True)
        ),
        lambda cf, dim, half: (
            cf.filter("v1 > 0")
            .mutate(z="v1 * 2 + k")
            .group_by("grp")
            .summarise(mean_z=("z", "mean"), n=("z", "size"))
            .arrange("-mean_z")
        ),
    )


@query(
    "distinct", "distinct combinations of two columns", ["grp", "k"],
    legacy_builder=lambda cf, dim, half: cf.select("grp", "k").distinct(),
)
def _():
    return (
        lambda lf, dim, half: lf.select(["grp", "k"]).unique(),
        lambda cf, dim, half: cf.distinct("grp", "k"),
    )


@query("count", "count rows per group", ["grp"])
def _():
    return (
        lambda lf, dim, half: lf.group_by("grp").agg(pl.len().alias("n")),
        lambda cf, dim, half: cf.count("grp"),
    )


@query("add_count", "add a per-group row count to every row", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.with_columns(pl.len().over("grp").alias("n")),
        lambda cf, dim, half: cf.add_count("grp"),
    )


@query("transmute", "compute two columns, drop the rest", None, legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.select((pl.col("v1") * 2).alias("a"), (pl.col("v1") + pl.col("v2")).alias("b")),
        lambda cf, dim, half: cf.transmute(a="v1 * 2", b="v1 + v2"),
    )


@query("relocate", "move one column to the front", None)
def _():
    return (
        lambda lf, dim, half: lf.select(["k", "row", "id", "grp", "v1", "v2"]),
        lambda cf, dim, half: cf.relocate("k"),
    )


@query("slice_min", "10 smallest rows, keeping ties (raw: bottom_k, no tie handling)", ["row"])
def _():
    return (
        lambda lf, dim, half: lf.bottom_k(10, by="v1"),
        lambda cf, dim, half: cf.slice_min("v1", n=10),
    )


@query("slice_max", "10 largest rows, no ties (raw: top_k)", ["row"])
def _():
    return (
        lambda lf, dim, half: lf.top_k(10, by="v1"),
        lambda cf, dim, half: cf.slice_max("v1", n=10, with_ties=False),
    )


@query("drop_na", "drop rows missing either of two columns", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.drop_nulls(["v1", "v2"]),
        lambda cf, dim, half: cf.drop_na("v1", "v2"),
    )


@query("replace_na", "replace missing values in two columns", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.with_columns(pl.col("v1").fill_null(0.0), pl.col("v2").fill_null(0.0)),
        lambda cf, dim, half: cf.replace_na(v1=0.0, v2=0.0),
    )


@query("fill", "fill missing values downward in one column", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.with_columns(pl.col("v1").fill_null(strategy="forward")),
        lambda cf, dim, half: cf.fill("v1"),
    )


# The old implementation returns all of v1 then all of v2; tidyr (and order=dplyr) keep input row order.
@query("pivot_longer", "two value columns to long form", ["row", "name"], legacy_order=False)
def _():
    return (
        lambda lf, dim, half: lf.select(["row", "v1", "v2"]).unpivot(
            on=["v1", "v2"], index=["row"], variable_name="name", value_name="value"
        ),
        lambda cf, dim, half: cf.select("row", "v1", "v2").pivot_longer(
            "v1", "v2", names_to="name", values_to="value"
        ),
    )


@query("semi_join", "keep rows whose id is in a ~50k-row table", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.join(half, on="id", how="semi"),
        lambda cf, dim, half: cf.semi_join(half, on="id"),
    )


@query("anti_join", "keep rows whose id is not in a ~50k-row table", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.join(half, on="id", how="anti"),
        lambda cf, dim, half: cf.anti_join(half, on="id"),
    )


@query("across_mutate", "centre two columns with across() in mutate", ["row"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.with_columns(
            (pl.col("v1") - pl.col("v1").mean()).alias("v1_c"),
            (pl.col("v2") - pl.col("v2").mean()).alias("v2_c"),
        ),
        lambda cf, dim, half: cf.mutate(across(["v1", "v2"], lambda x: x - mean(x), _names="{col}_c")),
    )


@query("across_summarise", "mean and sd of two columns per group with across()", ["grp"], legacy_builder=None)
def _():
    return (
        lambda lf, dim, half: lf.group_by("grp").agg(
            pl.col("v1").mean().alias("v1_mean"),
            pl.col("v1").std().alias("v1_sd"),
            pl.col("v2").mean().alias("v2_mean"),
            pl.col("v2").std().alias("v2_sd"),
        ),
        lambda cf, dim, half: cf.group_by("grp").summarise(across(["v1", "v2"], {"mean": mean, "sd": sd})),
    )


@query(
    "long_pipeline",
    "drop_na, filter, mutate (if_else), replace_na, left_join, group_by, summarise (across), mutate, filter, arrange",
    ["grp", "sign"],
    legacy_builder=None,
)
def _():
    return (
        lambda lf, dim, half: (
            lf.drop_nulls("v1")
            .filter(pl.col("v1") > -1)
            .with_columns(
                (pl.col("v1") * 2 + pl.col("k")).alias("z"),
                # same semantics as if_else: a missing condition stays missing
                pl.when(pl.col("v2").is_null())
                .then(None)
                .when(pl.col("v2") > 0)
                .then(pl.lit("pos"))
                .otherwise(pl.lit("neg"))
                .alias("sign"),
            )
            .with_columns(pl.col("sign").fill_null("unknown"))
            .join(dim, on="id", how="left")
            .group_by("grp", "sign")
            .agg(
                pl.col("z").mean().alias("z_mean"),
                pl.col("z").std().alias("z_sd"),
                pl.col("w").mean().alias("w_mean"),
                pl.col("w").std().alias("w_sd"),
                pl.len().alias("rows"),
            )
            .with_columns((pl.col("rows") / pl.col("rows").sum()).alias("share"))
            .filter(pl.col("rows") > 10)
            .sort("grp", "z_mean", descending=[False, True])
        ),
        lambda cf, dim, half: (
            cf.drop_na("v1")
            .filter("v1 > -1")
            .mutate(z="v1 * 2 + k", sign=if_else(col.v2 > 0, "pos", "neg"))
            .replace_na(sign="unknown")
            .left_join(dim, on="id")
            .group_by("grp", "sign")
            .summarise(across(["z", "w"], {"mean": mean, "sd": sd}), rows=n())
            .mutate(share="rows / sum(rows)")
            .filter("rows > 10")
            .arrange("grp", desc("z_mean"))
        ),
    )


def impls_for(name: str):
    return [i for i in IMPLS if i != "legacy" or QUERIES[name]["legacy"] is not None]


def build(impl: str, name: str, inputs):
    q = QUERIES[name]
    builder = q["raw"] if impl == "raw_polars" else q["legacy"] if impl == "legacy" else q["cf"]
    return builder(*inputs[impl])


def run(impl: str, name: str, inputs):
    """End to end: build the query and produce the result in the library's native form."""
    out = build(impl, name, inputs)
    if impl == "raw_polars":
        return out.collect()
    if impl == "legacy":
        return out  # already materialised (pandas inside)
    return out.to_polars()


def as_polars(impl: str, result) -> pl.DataFrame:
    if impl == "legacy":
        return pl.from_pandas(result.to_pandas())
    return result


# ---------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------


def validate(inputs, names) -> dict:
    """
    The reference is the current implementation where it has the verb, and the
    hand-written Polars query otherwise. Against the current implementation,
    order=dplyr must match row for row. Everything else must match after
    sorting on the row-identifying columns (or as is, where keys is None).
    """
    report = {}
    for name in names:
        keys = QUERIES[name]["keys"]
        has_legacy = QUERIES[name]["legacy"] is not None
        ref_impl = "legacy" if has_legacy else "raw_polars"
        reference = as_polars(ref_impl, run(ref_impl, name, inputs))
        status = {"reference": LABELS[ref_impl]}
        for impl in ("raw_polars", "new_fast", "new_dplyr"):
            if impl == ref_impl:
                status[impl] = "reference"
                continue
            got = as_polars(impl, run(impl, name, inputs))
            exact_order = keys is None or (impl == "new_dplyr" and has_legacy and QUERIES[name]["legacy_order"])
            left, right = (got, reference) if exact_order else (got.sort(keys), reference.sort(keys))
            try:
                assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
                status[impl] = "ok (same row order)" if exact_order else "ok (after sorting)"
            except AssertionError as e:
                status[impl] = "MISMATCH: " + str(e).splitlines()[0][:200]
            del got
        report[name] = status
        del reference
        gc.collect()
    return report


def plan_identity(inputs, names) -> dict:
    # Polars may print pushed-down predicates in a different order from one
    # explain() to the next, so compare against a few renderings of each plan.
    out = {}
    for name in names:
        raw = {build("raw_polars", name, inputs).explain() for _ in range(8)}
        out[name] = {
            impl: any(build(impl, name, inputs).explain() in raw for _ in range(8))
            for impl in ("new_fast", "new_dplyr")
        }
    return out


# ---------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------


def summarise_samples(samples):
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples), "reps": len(samples)}


def time_end_to_end(inputs, names, reps: int) -> dict:
    results = {}
    for name in names:
        impls = impls_for(name)
        samples = {impl: [] for impl in impls}
        for impl in impls:  # warm-up
            run(impl, name, inputs)
        for rep in range(reps):
            order = impls[rep % len(impls):] + impls[: rep % len(impls)]  # rotate to spread drift
            for impl in order:
                gc.collect()
                t0 = time.perf_counter()
                out = run(impl, name, inputs)
                samples[impl].append(time.perf_counter() - t0)
                del out
        results[name] = {impl: summarise_samples(s) for impl, s in samples.items()}
        print(
            f"  {name:17s} " + "  ".join(f"{impl}={results[name][impl]['median'] * 1e3:9.2f}ms" for impl in impls),
            flush=True,
        )
    return results


def time_plan_construction(inputs, names, iterations: int = 300) -> dict:
    """Time to build the lazy plan without running it. Legacy is eager, so it has none."""
    results = {}
    for name in names:
        row = {}
        for impl in ("raw_polars", "new_fast", "new_dplyr"):
            build(impl, name, inputs)
            samples = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                build(impl, name, inputs)
                samples.append(time.perf_counter() - t0)
            row[impl] = summarise_samples(samples)
        results[name] = row
    return results


# ---------------------------------------------------------------------
# memory (one subprocess per implementation and query)
# ---------------------------------------------------------------------

MEMORY_QUERIES = {"pipeline": IMPLS, "long_pipeline": ["raw_polars", "new_fast", "new_dplyr"]}


def memory_child(impl: str, n_rows: int, name: str) -> None:
    import psutil

    inputs = {impl: make_inputs(n_rows)[impl]}
    gc.collect()
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    out = run(impl, name, inputs)
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set()
    thread.join()
    del out
    print(json.dumps({"baseline_mb": baseline / 2**20, "extra_peak_mb": (peak[0] - baseline) / 2**20}))


def measure_memory(n_rows: int, names):
    try:
        import psutil  # noqa: F401
    except ImportError:
        return None
    out = {}
    for name, impls in MEMORY_QUERIES.items():
        if name not in names:
            continue
        out[name] = {}
        for impl in impls:
            proc = subprocess.run(
                [sys.executable, __file__, "--mem-child", impl, str(n_rows), name],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                error = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"
                out[name][impl] = {"error": error}
            else:
                out[name][impl] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "polars": pl.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "polars_threads": pl.thread_pool_size(),
    }
    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        env["ram_gb"] = None
    return env


def fmt_ms(stat) -> str:
    if stat is None:
        return "n/a"
    return f"{stat['median'] * 1e3:.2f} ({stat['min'] * 1e3:.2f}–{stat['max'] * 1e3:.2f})"


def write_report(results: dict, path: Path) -> None:
    env = results["environment"]
    names = results["queries"]
    lines = [
        "# crowley-frame A/B benchmark",
        "",
        "Generated by `python benchmarks/ab_bench.py`. Do not edit by hand.",
        "",
        "## Environment",
        "",
        "| | |",
        "|---|---|",
    ]
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        f"| repetitions | {results['reps']} timed runs per cell after one warm-up |",
        "",
        "Implementations:",
        "",
        "- **Raw Polars lazy**: the query as a Polars user would write it, `LazyFrame` then `collect()`.",
        "- **Current (pandas)**: the previous crowley-frame implementation, kept in `crowley_frame.legacy`. "
        "Shown as n/a where it does not have the verb.",
        "- **New, order=fast** / **New, order=dplyr**: the lazy Polars-backed rewrite in each ordering mode.",
        "",
        "Input conversion is outside the timed region for every implementation. "
        "Times are milliseconds, shown as median (min–max); the min is the best run. "
        "Ratio is the median divided by the raw Polars median for the same query; "
        "the last column compares the best run of each instead, which is less sensitive "
        "to background load on the machine.",
        "",
        "## Queries",
        "",
    ]
    for name in names:
        lines.append(f"- `{name}`: {QUERIES[name]['description']}")

    lines += ["", "## Correctness", ""]
    any_mismatch = False
    for size, block in results["sizes"].items():
        lines += [
            f"### {int(size):,} rows",
            "",
            "| Query | Reference | Raw Polars | New, fast | New, dplyr |",
            "|---|---|---|---|---|",
        ]
        for name, status in block["validation"].items():
            lines.append(
                f"| `{name}` | {status['reference']} | {status['raw_polars']} "
                f"| {status['new_fast']} | {status['new_dplyr']} |"
            )
            any_mismatch |= any(s.startswith("MISMATCH") for s in status.values())
        lines.append("")
    lines.append(
        "**Result: outputs differ, see MISMATCH rows above.**"
        if any_mismatch
        else "Every implementation returns the same data as its reference for every query and size."
    )

    lines += ["", "## End-to-end time", ""]
    for size, block in results["sizes"].items():
        lines += [
            f"### {int(size):,} rows",
            "",
            "| Query | " + " | ".join(LABELS[i] for i in IMPLS)
            + " | Current ÷ raw | Fast ÷ raw | dplyr ÷ raw | Fast ÷ raw (best run) |",
            "|---|" + "---|" * (len(IMPLS) + 4),
        ]
        for name, row in block["end_to_end"].items():
            raw = row["raw_polars"]["median"]
            legacy_ratio = f"{row['legacy']['median'] / raw:.1f}x" if "legacy" in row else "n/a"
            lines.append(
                f"| `{name}` | "
                + " | ".join(fmt_ms(row.get(i)) for i in IMPLS)
                + f" | {legacy_ratio}"
                + f" | {row['new_fast']['median'] / raw:.2f}x"
                + f" | {row['new_dplyr']['median'] / raw:.2f}x"
                + f" | {row['new_fast']['min'] / row['raw_polars']['min']:.2f}x |"
            )
        lines.append("")

    lines += [
        "## Plan construction time",
        "",
        "Time to build the lazy plan without running it, in microseconds, median (min–max). "
        "It does not depend on row count. The current implementation is eager and has no separate plan step.",
        "",
        "| Query | Raw Polars | New, fast | New, dplyr | Fast ÷ raw |",
        "|---|---|---|---|---|",
    ]
    for name, row in results["plan_construction"].items():
        cells = [
            f"{row[i]['median'] * 1e6:.0f} ({row[i]['min'] * 1e6:.0f}–{row[i]['max'] * 1e6:.0f})"
            for i in ("raw_polars", "new_fast", "new_dplyr")
        ]
        ratio = row["new_fast"]["median"] / row["raw_polars"]["median"]
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {ratio:.1f}x |")

    lines += [
        "",
        "## Plan identity",
        "",
        "Whether the optimised plan (`explain()`) is text-identical to the hand-written Polars plan.",
        "",
        "| Query | New, fast | New, dplyr |",
        "|---|---|---|",
    ]
    for name, row in results["plan_identity"].items():
        lines.append(
            f"| `{name}` | {'identical' if row['new_fast'] else 'differs'} "
            f"| {'identical' if row['new_dplyr'] else 'differs'} |"
        )

    lines += ["", "## Peak extra memory", ""]
    memory = {size: block.get("memory") for size, block in results["sizes"].items()}
    if not any(memory.values()):
        lines.append("Not measured (psutil unavailable or `--no-memory`).")
    else:
        lines += [
            "Resident memory above the loaded-data baseline, sampled every 2 ms in a fresh process "
            "per implementation. One run each, so treat as indicative.",
            "",
            "| Query | Rows | " + " | ".join(LABELS[i] for i in IMPLS) + " |",
            "|---|---|" + "---|" * len(IMPLS),
        ]
        for size, mem in memory.items():
            for name, by_impl in (mem or {}).items():
                cells = []
                for impl in IMPLS:
                    cell = by_impl.get(impl)
                    cells.append("n/a" if cell is None else f"{cell['extra_peak_mb']:.0f} MB" if "extra_peak_mb" in cell else "error")
                lines.append(f"| `{name}` | {int(size):,} | " + " | ".join(cells) + " |")

    lines += ["", "## Summary against the success criteria", ""]
    for size, block in results["sizes"].items():
        rows = block["end_to_end"]
        fast = {q: r["new_fast"]["median"] / r["raw_polars"]["median"] for q, r in rows.items()}
        best = {q: r["new_fast"]["min"] / r["raw_polars"]["min"] for q, r in rows.items()}
        within = [q for q, v in fast.items() if v < 1.10]
        within_best = sum(1 for v in best.values() if v < 1.10)
        with_legacy = {q: r for q, r in rows.items() if "legacy" in r}
        faster = [q for q, r in with_legacy.items() if r["new_fast"]["median"] < r["legacy"]["median"]]
        outside = ", ".join(f"`{q}` {fast[q]:.2f}x" for q in fast if q not in within)
        lines.append(
            f"- **{int(size):,} rows:** order=fast is within 10% of raw Polars on {len(within)} of {len(fast)} queries"
            + f" by median and {within_best} of {len(best)} by best run"
            + (f" (outside by median: {outside})" if outside else "")
            + f"; faster than the current implementation on {len(faster)} of {len(with_legacy)} queries it can run."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 5_000_000])
    parser.add_argument("--with-15m", action="store_true", help="also run 15,000,000 rows")
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--queries", nargs="+", choices=list(QUERIES), default=list(QUERIES))
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--report", type=Path, default=HERE / "REPORT.md")
    parser.add_argument("--mem-child", nargs=3, metavar=("IMPL", "ROWS", "QUERY"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mem_child:
        memory_child(args.mem_child[0], int(args.mem_child[1]), args.mem_child[2])
        return 0

    names = args.queries
    sizes = sorted(set(args.sizes + ([15_000_000] if args.with_15m else [])))
    results = {"environment": environment(), "reps": args.reps, "queries": names, "sizes": {}}
    print(json.dumps(results["environment"], indent=2), flush=True)

    failed = False
    for n_rows in sizes:
        print(f"\n== {n_rows:,} rows ==", flush=True)
        inputs = make_inputs(n_rows)
        if "plan_construction" not in results:
            results["plan_construction"] = time_plan_construction(inputs, names)
            results["plan_identity"] = plan_identity(inputs, names)
        validation = validate(inputs, names)
        for name, status in validation.items():
            for impl, text in status.items():
                if text.startswith("MISMATCH"):
                    failed = True
                    print(f"  !! {name} / {impl}: {text}", flush=True)
        block = {"validation": validation, "end_to_end": time_end_to_end(inputs, names, args.reps)}
        del inputs
        gc.collect()
        if not args.no_memory:
            block["memory"] = measure_memory(n_rows, names)
        results["sizes"][str(n_rows)] = block

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ab_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, args.report)
    print(f"\nWrote {args.out / 'ab_results.json'} and {args.report}")
    if failed:
        print("FAILED: at least one implementation returned different results.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
