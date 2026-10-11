"""Stage 3: data quality and quarantine workflows."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import crowley_frame as cr
from crowley_frame import CrowleyTranslationError, CrowleyValidationError, col, df, starts_with


def _data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 2, 4, 5, 6],
            "amount": [10.0, -5.0, 30.0, None, 5000.0, 60.0],
            "status": ["ok", "ok", "bad", "ok", None, "weird"],
        }
    )


def _exploding(data: pl.DataFrame) -> pl.LazyFrame:
    def boom(_df):
        raise AssertionError("the plan was executed")

    return data.lazy().map_batches(boom, schema=data.schema)


# ---------------------------------------------------------------------
# partition_valid
# ---------------------------------------------------------------------


def test_partition_valid_matches_hand_written_filters():
    data = _data()
    valid, invalid = df(_exploding(data)).partition_valid(not_null=["amount"], ranges={"amount": (0, 1000)})
    assert valid.columns == data.columns and invalid.columns == data.columns + ["failed_checks"]  # still lazy

    valid, invalid = df(data).partition_valid(
        not_null=["amount"],
        unique=["id"],
        ranges={"amount": (0, 1000)},
        allowed={"status": ["ok", "bad"]},
        where={"even_id": "id % 2 == 0"},
    )
    ok = (
        pl.col("amount").is_not_null()
        & ~pl.col("id").is_duplicated()
        & ((pl.col("amount") >= 0) & (pl.col("amount") <= 1000)).fill_null(True)
        & pl.col("status").is_in(["ok", "bad"]).fill_null(True)
        & (pl.col("id") % 2 == 0).fill_null(False)
    )
    assert_frame_equal(valid.to_polars(), data.filter(ok))
    bad = invalid.to_polars()
    assert_frame_equal(bad.drop("failed_checks"), data.filter(~ok))
    assert valid.to_polars().height + bad.height == data.height
    assert bad["failed_checks"].to_list() == [
        "even_id",
        "unique(id); range(amount)",
        "unique(id)",
        "not_null(amount)",
        "range(amount); even_id",
        "allowed_values(status)",
    ]


def test_partition_valid_edge_cases():
    data = _data()
    valid, invalid = df(data).partition_valid(where={"positive": col.amount > 0}, reason="why")
    assert valid.to_polars()["id"].to_list() == [1, 2, 5, 6]
    assert invalid.to_polars()["why"].to_list() == ["positive", "positive"]  # a missing result fails

    both = cr.collect_all((valid, invalid))
    assert both[0].to_polars().height == 4 and both[1].to_polars().height == 2

    empty_valid, empty_invalid = df(data.head(0)).partition_valid(not_null=["id"])
    assert empty_valid.to_polars().height == 0 and empty_invalid.to_polars().height == 0
    with pytest.raises(ValueError, match="at least one rule"):
        df(data).partition_valid()
    with pytest.raises(KeyError):
        df(data).partition_valid(not_null=["nope"])
    with pytest.raises(ValueError, match="already exists"):
        df(data).partition_valid(not_null=["id"], reason="status")
    with pytest.raises(CrowleyTranslationError):
        df(data).partition_valid(where={"bad": "amount.__class__"})


# ---------------------------------------------------------------------
# normalize_names / normalize_values
# ---------------------------------------------------------------------


def test_normalize_names():
    lf = pl.LazyFrame({"User ID": [1], "totalAmount($)": [2], "2024 Sales": [3], "ok_name": [4], "  ": [5]})
    cf = df(_exploding(lf.collect())).group_by("User ID").normalize_names()
    assert cf.columns == ["user_id", "total_amount", "x2024_sales", "ok_name", "column"]
    assert cf.groups == ("user_id",)
    unchanged = df(pl.DataFrame({"a": [1], "b_c": [2]}))
    assert unchanged.normalize_names() is unchanged
    with pytest.raises(CrowleyValidationError, match="would both become 'user_id'"):
        df(pl.DataFrame({"User ID": [1], "user-id": [2]})).normalize_names()


def test_normalize_values():
    data = pl.DataFrame(
        {
            "name": ["  Ada  Lovelace ", "NA", "", None, "bob"],
            "city": [" NULL", "Paris ", "n/a", "nan", "ROME"],
            "n": [1, 2, 3, 4, 5],
        }
    )
    got = df(data).normalize_values().to_polars()
    want = data.with_columns(
        pl.when(pl.col(c).str.strip_chars().is_in(list(cr.quality.DEFAULT_NULL_VALUES)))
        .then(None)
        .otherwise(pl.col(c).str.strip_chars())
        .alias(c)
        for c in ("name", "city")
    )
    assert_frame_equal(got, want)
    assert got["name"].to_list() == ["Ada  Lovelace", None, None, None, "bob"]
    assert got["city"].to_list() == [None, "Paris", "n/a", None, "ROME"]  # "n/a" is not in the default list
    assert got["n"].to_list() == [1, 2, 3, 4, 5]

    tidy = df(data).normalize_values("name", case="title", collapse_whitespace=True, null_values=None).to_polars()
    assert tidy["name"].to_list() == ["Ada Lovelace", "Na", "", None, "Bob"]
    assert tidy["city"].to_list() == data["city"].to_list()  # not selected
    lower = df(data).normalize_values(starts_with("c"), case="lower", null_values=["n/a", "null"]).to_polars()
    assert lower["city"].to_list() == [None, "paris", None, "nan", "rome"]
    assert df(data).normalize_values("name", trim=False, null_values=["NA"]).to_polars()["name"][0] == "  Ada  Lovelace "

    assert df(_exploding(data)).normalize_values().columns == data.columns  # lazy
    with pytest.raises(CrowleyTranslationError, match="not a string column"):
        df(data).normalize_values("n")
    with pytest.raises(ValueError):
        df(data).normalize_values(case="snake")


# ---------------------------------------------------------------------
# replace_missing / drop_missing
# ---------------------------------------------------------------------


def test_replace_missing():
    data = pl.DataFrame(
        {"g": ["a", "a", "b", "b", "b"], "x": [1.0, None, 10.0, None, 30.0], "s": ["p", None, "q", "r", None]}
    )
    got = df(data).replace_missing(values={"s": "unknown"}, strategy={"x": "mean"})
    want = data.lazy().with_columns(pl.col("s").fill_null("unknown"), pl.col("x").fill_null(pl.col("x").mean()))
    assert got.explain() == want.explain()
    assert got.to_polars()["x"].to_list() == [1.0, 41 / 3, 10.0, 41 / 3, 30.0]

    grouped = df(data).group_by("g").replace_missing(strategy={"x": "mean"}).to_polars()
    assert grouped["x"].to_list() == [1.0, 1.0, 10.0, 20.0, 30.0]
    for strategy, expected in {
        "median": 10.0, "min": 1.0, "max": 30.0, "zero": 0.0, "forward": 1.0, "backward": 10.0,
    }.items():
        assert df(data).replace_missing(strategy={"x": strategy}).to_polars()["x"][1] == expected

    cf = df(data)
    assert cf.replace_missing() is cf
    with pytest.raises(ValueError, match="both values= and strategy="):
        cf.replace_missing(values={"x": 0}, strategy={"x": "mean"})
    with pytest.raises(ValueError, match="unknown strategy"):
        cf.replace_missing(strategy={"x": "mode"})
    with pytest.raises(KeyError):
        cf.replace_missing(values={"nope": 0})


def test_drop_missing():
    data = pl.DataFrame({"a": [1, None, None, 4], "b": [1, 2, None, None], "c": ["x", None, None, "y"]})
    cf = df(data)
    assert_frame_equal(cf.drop_missing().to_polars(), data.drop_nulls())
    assert_frame_equal(cf.drop_missing("a", "b").to_polars(), data.drop_nulls(["a", "b"]))
    assert cf.drop_missing("a").explain() == data.lazy().drop_nulls("a").explain()
    assert cf.drop_missing().to_polars()["a"].to_list() == [1]
    assert cf.drop_missing(how="all").to_polars()["a"].to_list() == [1, None, 4]
    assert cf.drop_missing("a", "c", how="all").to_polars().height == 2
    assert cf.drop_missing(max_missing=1).to_polars()["a"].to_list() == [1, 4]
    assert cf.drop_missing(max_missing=2).to_polars().height == 3
    assert cf.drop_missing(max_missing=0.5).to_polars().height == 2  # at most half of 3 columns
    assert cf.drop_missing("a", "b", max_missing=0).to_polars().height == 1
    assert df(_exploding(data)).drop_missing(how="all").columns == ["a", "b", "c"]  # lazy
    with pytest.raises(ValueError):
        cf.drop_missing(how="some")
    with pytest.raises(ValueError):
        cf.drop_missing(how="all", max_missing=1)


# ---------------------------------------------------------------------
# detect_anomalies / detect_drift
# ---------------------------------------------------------------------


def test_detect_anomalies():
    n = 200
    data = pl.DataFrame(
        {
            "id": list(range(n - 1)) + [0],
            "value": [float(i % 10) for i in range(n - 4)] + [1e6, -1e6, 9e5, 8e5],
            "mostly_missing": [1.0] + [None] * (n - 1),
            "constant": ["same"] * n,
            "status": ["ok"] * (n - 3) + ["odd", "odd", None],
        }
    )
    found = df(data).detect_anomalies(allowed={"status": ["ok"]}, keys=["id"]).to_polars()
    assert found.columns == ["column", "issue", "value", "threshold", "detail"]
    issues = set(zip(found["column"].to_list(), found["issue"].to_list()))
    assert issues == {
        ("value", "outliers"), ("mostly_missing", "high_null_rate"), ("constant", "constant"),
        ("status", "unexpected_values"), ("id", "duplicate_keys"),
    }
    by_issue = {r["issue"]: r for r in found.to_dicts()}
    assert by_issue["outliers"]["value"] == pytest.approx(4 / n)
    assert by_issue["high_null_rate"]["value"] == pytest.approx((n - 1) / n)
    assert by_issue["unexpected_values"]["detail"] == "2 values outside the allowed set"
    assert by_issue["duplicate_keys"]["detail"] == "1 duplicate rows"

    relaxed = df(data).detect_anomalies(max_null_rate=1.0, iqr_factor=None).to_polars()
    assert relaxed["issue"].to_list() == ["constant"]
    clean = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["w", "x", "y", "z"]})
    assert df(clean).detect_anomalies().to_polars().height == 0
    assert df(clean.head(0)).detect_anomalies().to_polars().height == 0


def test_detect_drift():
    reference = pl.DataFrame(
        {
            "id": list(range(100)),
            "value": [float(i % 10) for i in range(100)],
            "segment": ["a", "b"] * 50,
            "gone": [1] * 100,
            "kind": [1] * 100,
        }
    )
    current = pl.DataFrame(
        {
            "id": list(range(150)),
            "value": [float(i % 10) + 5 if i % 3 else None for i in range(150)],
            "segment": ["a", "c", "d"] * 50,
            "kind": ["x"] * 150,
            "new": [0] * 150,
        }
    )
    found = df(current).detect_drift(reference, categories=["segment"], row_count_tol=0.2).to_polars()
    issues = {(r["column"], r["issue"]) for r in found.to_dicts()}
    assert issues == {
        ("new", "column_added"), ("gone", "column_removed"), ("kind", "dtype_changed"),
        (None, "row_count_change"), ("value", "null_rate_change"), ("value", "mean_shift"),
        ("id", "mean_shift"),  # ids now run to 149, so their mean moved too
        ("segment", "new_categories"), ("segment", "missing_categories"),
    }
    by_issue = {r["issue"]: r for r in found.to_dicts() if r["column"] != "id"}
    assert by_issue["row_count_change"]["detail"] == "+50.0%"
    assert by_issue["new_categories"]["current"] == "['c', 'd']"
    assert by_issue["missing_categories"]["reference"] == "['b']"
    assert "reference SDs" in by_issue["mean_shift"]["detail"]

    assert df(reference).detect_drift(reference, categories=["segment"], row_count_tol=0.0).to_polars().height == 0
    loose = df(current).detect_drift(reference, null_rate_tol=1.0, mean_shift_tol=100.0).to_polars()
    assert set(loose["issue"]) == {"column_added", "column_removed", "dtype_changed"}
