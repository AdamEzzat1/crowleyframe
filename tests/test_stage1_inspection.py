"""Stage 1: inspection, debugging and schema safety."""
import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import crowley_frame as cr
from crowley_frame import CrowleyValidationError, col, df, is_numeric, mean, n, pipe, starts_with, where


def _data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "g": ["a", "a", "b", "b", "b", None],
            "x": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "flag": [True, False, True, None, True, True],
            "day": [dt.date(2024, 1, d) for d in (1, 2, 3, 4, 5, 6)],
        }
    )


def _exploding(data: pl.DataFrame = None) -> pl.LazyFrame:
    def boom(_df):
        raise AssertionError("the plan was executed")

    return (data if data is not None else _data()).lazy().map_batches(boom, schema=(data or _data()).schema)


# ---------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------


def test_profile_matches_raw_polars_aggregates():
    data = _data()
    prof = df(data).profile(exact_distinct=True).to_polars()
    assert prof.columns == ["column", "dtype", "rows", "nulls", "null_rate", "distinct", "min", "max", "mean"]
    assert prof["column"].to_list() == data.columns
    raw = data.lazy().select(
        pl.len().alias("rows"),
        pl.all().null_count().name.suffix("_nulls"),
        pl.all().n_unique().name.suffix("_distinct"),
        pl.col("x").mean().alias("x_mean"),
        pl.col("x").min().alias("x_min"),
        pl.col("day").max().alias("day_max"),
    ).collect().row(0, named=True)
    by_col = {r["column"]: r for r in prof.to_dicts()}
    for name in data.columns:
        assert by_col[name]["rows"] == raw["rows"] == 6
        assert by_col[name]["nulls"] == raw[f"{name}_nulls"]
        assert by_col[name]["distinct"] == raw[f"{name}_distinct"]
    assert by_col["x"]["mean"] == raw["x_mean"] and by_col["x"]["min"] == str(raw["x_min"])
    assert by_col["day"]["max"] == str(raw["day_max"]) and by_col["day"]["mean"] is None
    assert by_col["g"]["null_rate"] == pytest.approx(1 / 6)
    assert by_col["g"]["min"] == "a" and by_col["flag"]["dtype"] == "Boolean"

    approx = df(data).profile("id", where(is_numeric)).to_polars()
    assert approx["column"].to_list() == ["id", "x"]
    assert approx["distinct"].to_list() == [6, 6]  # 5 values + missing


def test_profile_on_empty_frame():
    empty = df(_data().head(0)).profile().to_polars()
    assert empty["rows"].to_list() == [0] * 5
    assert empty["null_rate"].to_list() == [None] * 5


@pytest.mark.parametrize("order", ["dplyr", "fast"])
def test_profile_by_is_lazy_and_matches_raw(order):
    data = _data()
    lazy = df(_exploding(), order=order).profile_by("g")
    assert lazy.columns[:3] == ["g", "rows", "id_nulls"]  # built without running anything

    got = df(data, order=order).profile_by("g", cols=["x", "flag"], exact_distinct=True).to_polars()
    want = data.lazy().group_by("g").agg(
        pl.len().alias("rows"),
        pl.col("x").null_count().alias("x_nulls"),
        pl.col("x").n_unique().alias("x_distinct"),
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("x").mean().alias("x_mean"),
        pl.col("flag").null_count().alias("flag_nulls"),
        pl.col("flag").n_unique().alias("flag_distinct"),
        pl.col("flag").min().alias("flag_min"),
        pl.col("flag").max().alias("flag_max"),
    ).collect()
    assert_frame_equal(got.sort("g"), want.sort("g"), check_dtype=False)
    if order == "dplyr":
        assert got["g"].to_list() == ["a", "b", None]
    assert df(data).group_by("g").profile_by().to_polars().height == 3
    with pytest.raises(ValueError):
        df(data).profile_by()


def test_profile_diff():
    old = _data()
    new = old.with_columns(pl.col("x").fill_null(0.0), pl.col("id").cast(pl.Int32)).drop("flag").with_columns(
        pl.lit(1).alias("extra")
    )
    diff = df(old).profile_diff(new).to_polars()
    status = dict(zip(diff["column"].to_list(), diff["status"].to_list()))
    assert status == {"id": "dtype_changed", "g": "same", "x": "changed", "flag": "removed", "day": "same",
                      "extra": "added"}
    row = diff.filter(pl.col("column") == "x").row(0, named=True)
    assert row["nulls_left"] == 1 and row["nulls_right"] == 0
    # profiles can be passed instead of frames
    again = df(old).profile().profile_diff(df(new).profile()).to_polars()
    assert again["status"].to_list() == diff["status"].to_list()
    assert set(df(old).profile_diff(old).to_polars()["status"]) == {"same"}


# ---------------------------------------------------------------------
# validate / validate_bounds
# ---------------------------------------------------------------------


def test_validate_passes_and_returns_the_frame():
    cf = df(_data())
    out = cf.validate(
        columns=["id", "g"],
        dtypes={"id": pl.Int64, "x": is_numeric, "g": (pl.String, pl.Categorical)},
        not_null=["id"],
        unique=["id", ("id", "g")],
        ranges={"x": (0, 10), "id": (1, None)},
        allowed={"g": ["a", "b", "c"]},
    )
    assert out is cf
    assert (cf >> pipe.validate(not_null=["id"]) >> pipe.filter("id > 3")).to_polars().height == 3


def test_validate_reports_every_failure():
    cf = df(_data())
    report = cf.validate(
        columns=["id", "nope"],
        dtypes={"id": pl.String, "missing_col": pl.Int64},
        not_null=["x", "g"],
        unique=["g"],
        ranges={"x": (2, 5)},
        allowed={"g": ["a"]},
        strict_columns=True,
        on_fail="report",
    )
    assert not report and not report.ok
    failed = {(r["check"], r["column"]) for r in report.failures}
    assert failed == {
        ("column_present", "nope"), ("no_extra_columns", None), ("dtype", "id"), ("dtype", "missing_col"),
        ("not_null", "x"), ("not_null", "g"), ("unique", "g"), ("range", "x"), ("allowed_values", "g"),
    }
    table = report.to_polars()
    observed = dict(zip(zip(table["check"].to_list(), table["column"].to_list()), table["observed"].to_list()))
    assert observed[("not_null", "x")] == "1 missing"
    assert observed[("unique", "g")] == "3 duplicate rows"
    assert observed[("range", "x")] == "2 out of range"  # 1.0 and 6.0; the missing value is ignored
    assert observed[("allowed_values", "g")].startswith("3 unexpected") and "'b'" in observed[("allowed_values", "g")]

    with pytest.raises(CrowleyValidationError, match="not_null\\(x\\)") as err:
        cf.validate(not_null=["x"])
    assert err.value.report.failures[0]["check"] == "not_null"
    with pytest.raises(ValueError):
        cf.validate(not_null=["id"], on_fail="ignore")


def test_schema_only_validation_reads_no_data():
    cf = df(_exploding())
    assert cf.validate(columns=["id", "x"], dtypes={"x": pl.Float64}) is cf
    with pytest.raises(CrowleyValidationError):
        cf.validate(columns=["zzz"])
    with pytest.raises(Exception, match="the plan was executed"):
        cf.validate(not_null=["id"])


def test_validate_bounds_and_categories():
    cf = df(_data())
    assert cf.validate_bounds(
        rows=(1, 10), null_rate={"x": 0.2}, duplicate_rate={"id": 0.0, ("id", "g"): 0.0},
        ranges={"x": (0, 10)}, cardinality={"g": (2, 3)},
    ) is cf
    report = cf.validate_bounds(
        rows=(10, None), null_rate={"x": 0.1}, duplicate_rate={"g": 0.2}, ranges={"x": (2, 5)},
        cardinality={"g": (None, 2)}, on_fail="report",
    )
    assert {r["check"] for r in report.failures} == {"rows", "null_rate", "duplicate_rate", "range", "cardinality"}
    table = dict(zip(report.to_polars()["check"].to_list(), report.to_polars()["observed"].to_list()))
    assert table["rows"] == "6" and table["duplicate_rate"] == "0.5" and table["cardinality"] == "3"

    assert cf.validate_categories({"g": ["a", "b"]}) is cf
    with pytest.raises(CrowleyValidationError, match="allowed_values"):
        cf.validate_categories({"g": ["a"]})
    with pytest.raises(CrowleyValidationError):
        df(_data().head(0)).validate_bounds(rows=(1, None))


# ---------------------------------------------------------------------
# schema, plan and expression inspection
# ---------------------------------------------------------------------


def test_compare_schema():
    a = df(_exploding())
    same = a.compare_schema(_data())
    assert same and same.equal and same.to_polars().height == 0

    other = _data().select("g", "id", pl.col("x").cast(pl.Float32), pl.lit(1).alias("extra"))
    cmp = a.compare_schema(other)
    assert not cmp
    assert cmp.missing == ["extra"] and cmp.extra == ["flag", "day"]
    assert cmp.dtype_mismatches == {"x": ("Float64", "Float32")}
    assert cmp.order_differs
    assert cmp.to_polars()["difference"].to_list() == ["missing", "extra", "extra", "dtype", "column order"]
    assert "dtype mismatches" in repr(cmp)


def test_resolve_columns_and_explain_expr_read_no_data():
    cf = df(_exploding())
    assert cf.resolve_columns(where(is_numeric), starts_with("f")) == ["id", "x", "flag"]
    assert cf.resolve_columns() == []
    with pytest.raises(KeyError):
        cf.resolve_columns("nope")

    assert cf.explain_expr("x / sum(x)") == str(pl.col("x") / pl.col("x").sum())
    grouped = cf.group_by("g")
    assert grouped.explain_expr(col.x / cr.sum(col.x)) == str(pl.col("x") / pl.col("x").sum().over(["g"]))
    assert grouped.explain_expr("mean(x)", context="summarise") == str(pl.col("x").mean())
    with pytest.raises(ValueError):
        cf.explain_expr("x", context="filter")


def test_audit_and_explain_plan():
    cf = df(_exploding())
    assert cf.audit().height == 0
    out = (
        cf.filter("x > 1")
        .mutate(y=col.x * 2)
        .group_by("g")
        .summarise(m=mean(col.y), rows=n())
        .arrange("-m")
    )
    trail = out.audit()
    assert trail["verb"].to_list() == ["filter", "mutate", "group_by", "summarise", "arrange"]
    assert trail["step"].to_list() == [1, 2, 3, 4, 5]
    assert trail["arguments"].to_list()[:2] == ["'x > 1'", "y=(x * 2)"]
    assert cf.audit().height == 0  # frames are immutable; the original trail is untouched

    text = out.explain_plan()
    assert "1. filter('x > 1')" in text and "5. arrange('-m')" in text
    assert "Polars plan (optimised):" in text and "AGGREGATE" in text
    assert "sorts rows" in text and "runs" in text  # SORT, and the map_batches source
    assert "(none)" in df(_data()).select("id").explain_plan()
    assert "the plan was executed here" in df(_data()).filter("id > 1").collect().explain_plan()

    piped = df(_data()) >> pipe.select("id", "x") >> pipe.drop_na()
    assert piped.audit()["verb"].to_list() == ["select", "drop_na"]


# ---------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------


def test_lineage_through_core_verbs():
    cf = df(_exploding())
    with pytest.raises(CrowleyValidationError, match="track_lineage"):
        cf.lineage()

    tracked = cf.track_lineage()
    assert tracked.lineage() == {c: [c] for c in tracked.columns}
    out = (
        tracked.rename(key="id")
        .mutate(double="x * 2", both=col.x + col.key, chained="double + 1")
        .select("key", "g", "double", "both", "chained")
        .rename_with(str.upper, "g")
    )
    assert out.lineage() == {
        "key": ["id"], "G": ["g"], "double": ["x"], "both": ["id", "x"], "chained": ["x"],
    }
    summary = out.group_by("G").summarise(m=mean(col.both), rows=n(), top=("chained", "max")).add_count("G")
    assert summary.lineage() == {"G": ["g"], "m": ["id", "x"], "rows": [], "top": ["x"], "n": ["g"]}


def test_lineage_through_joins_and_reshaping():
    left = df(pl.LazyFrame({"id": [1], "v": [1.0], "w1": [1], "w2": [2]})).track_lineage()
    right = df(pl.LazyFrame({"id": [1], "v": [2.0], "label": ["a"]}))

    joined = left.left_join(right, on="id")
    assert joined.lineage() == {
        "id": ["id"], "v_x": ["v"], "w1": ["w1"], "w2": ["w2"], "v_y": ["right.v"], "label": ["right.label"],
    }
    assert joined.semi_join(right, on="id").lineage() == joined.lineage()

    long = left.pivot_longer(starts_with("w"), names_to="week", values_to="value")
    assert long.lineage() == {"id": ["id"], "v": ["v"], "week": ["w1", "w2"], "value": ["w1", "w2"]}
    united = left.unite("key", ["id", "w1"]).fingerprint_rows("key", "v", name="h")
    assert united.lineage()["key"] == ["id", "w1"] and united.lineage()["h"] == ["id", "v", "w1"]
    wide = df(pl.DataFrame({"id": [1, 1], "k": ["a", "b"], "v": [1, 2]})).track_lineage().pivot_wider("k", "v")
    assert wide.lineage() == {"id": ["id"], "a": ["(unknown)"], "b": ["(unknown)"]}
