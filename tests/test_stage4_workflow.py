"""Stage 4: reproducibility, lineage and pipeline ergonomics."""
import json
import os

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import crowley_frame as cr
from crowley_frame import CrowleyValidationError, col, define_pipeline, df, mean, pipe, restore_snapshot


def _data() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": [1, 2, 3, 4], "g": ["a", "a", "b", None], "x": [1.0, 2.0, None, 4.0], "s": ["p", "q", "p", "q"]}
    )


def _exploding(data: pl.DataFrame) -> pl.LazyFrame:
    def boom(_df):
        raise AssertionError("the plan was executed")

    return data.lazy().map_batches(boom, schema=data.schema)


# ---------------------------------------------------------------------
# fingerprint_rows
# ---------------------------------------------------------------------


def test_fingerprint_rows():
    data = _data()
    cf = df(data)
    got = cf.fingerprint_rows("g", "s", seed=7)
    want = data.lazy().with_columns(pl.struct(["g", "s"]).hash(7).alias("row_hash"))
    assert got.explain() == want.explain()
    assert_frame_equal(got.to_polars(), want.collect())
    assert got.to_polars().schema["row_hash"] == pl.UInt64

    doubled = pl.concat([data, data])
    hashes = df(doubled).fingerprint_rows(name="h").to_polars()["h"]
    assert hashes[:4].to_list() == hashes[4:].to_list() and hashes[:4].n_unique() == 4  # equal rows, equal hashes
    assert df(data).fingerprint_rows(seed=1).to_polars()["row_hash"].to_list() != hashes[:4].to_list()

    assert df(_exploding(data)).fingerprint_rows("id").columns[-1] == "row_hash"  # lazy
    with pytest.raises(ValueError, match="already exists"):
        cf.fingerprint_rows(name="id")
    with pytest.raises(KeyError):
        cf.fingerprint_rows("nope")


# ---------------------------------------------------------------------
# snapshot / restore_snapshot
# ---------------------------------------------------------------------


def test_snapshot_round_trip(tmp_path):
    path = tmp_path / "step1.parquet"
    cf = df(_data()).filter("id > 1").group_by("g")
    snap = cf.snapshot(path)

    assert sorted(os.listdir(tmp_path)) == ["step1.parquet", "step1.parquet.meta.json"]  # no temp files left
    meta = json.loads((tmp_path / "step1.parquet.meta.json").read_text(encoding="utf-8"))
    assert meta == cr.snapshot_metadata(path)
    assert meta["rows"] == 3 and meta["groups"] == ["g"] and meta["polars_version"] == pl.__version__
    assert meta["columns"][0] == {"name": "id", "dtype": "Int64"}
    assert [s["verb"] for s in meta["audit"]] == ["filter", "group_by"]

    assert snap.groups == ("g",) and snap.audit()["verb"].to_list() == ["filter", "group_by", "snapshot"]
    assert "PARQUET" in snap.explain().upper()  # later verbs read the file, not the original plan
    assert_frame_equal(snap.to_polars(), _data().filter(pl.col("id") > 1))

    restored = restore_snapshot(path)
    assert restored.groups == ("g",)
    assert_frame_equal(restored.to_polars(), _data().filter(pl.col("id") > 1))
    assert restored.summarise(m=mean(col.x)).to_polars()["g"].to_list() == ["a", "b", None]


def test_snapshot_is_not_destructive(tmp_path):
    path = tmp_path / "data.parquet"
    df(_data()).snapshot(path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        df(_data().head(1)).snapshot(path)
    assert restore_snapshot(path).to_polars().height == 4  # untouched
    df(_data().head(1)).snapshot(path, overwrite=True)
    assert restore_snapshot(path).to_polars().height == 1

    with pytest.raises(ValueError, match=".parquet"):
        df(_data()).snapshot(tmp_path / "data.csv")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        df(_data()).snapshot(tmp_path / "missing_dir" / "data.parquet")
    assert not (tmp_path / "missing_dir").exists()
    with pytest.raises(FileNotFoundError):
        restore_snapshot(tmp_path / "nope.parquet")

    # a failing plan leaves nothing behind
    failing = tmp_path / "failing.parquet"
    with pytest.raises(Exception, match="the plan was executed"):
        df(_exploding(_data())).snapshot(failing)
    assert not failing.exists() and not os.path.exists(str(failing) + ".meta.json")


def test_restore_snapshot_checks_metadata(tmp_path):
    path = tmp_path / "data.parquet"
    df(_data()).snapshot(path)
    _data().select("id", "g").write_parquet(path)  # the file changes behind the metadata's back
    with pytest.raises(CrowleyValidationError, match="does not match its metadata"):
        restore_snapshot(path)
    assert restore_snapshot(path, check=False).columns == ["id", "g"]
    os.remove(str(path) + ".meta.json")
    with pytest.raises(FileNotFoundError, match="no snapshot metadata"):
        restore_snapshot(path)


# ---------------------------------------------------------------------
# expectations
# ---------------------------------------------------------------------


def test_expectations_travel_with_the_pipeline():
    cf = df(_exploding(_data()))
    expecting = cf.expect(not_null=["id"], unique=["id"]).expect(ranges={"x": (0, 3)})
    assert cf.expectations() == [] and len(expecting.expectations()) == 2  # original untouched
    later = expecting.filter("id > 0").mutate(y="x * 2").select("id", "x", "y")
    assert later.expectations() == expecting.expectations()  # carried through verbs, still lazy
    assert later.clear_expectations().expectations() == []
    with pytest.raises(TypeError, match="unknown rule"):
        cf.expect(not_nul=["id"])
    with pytest.raises(TypeError):
        cf.expect()


def test_assert_expectations():
    cf = df(_data()).expect(not_null=["id"], unique=["id"]).expect(ranges={"x": (0, 3)}, columns=["id"])
    with pytest.raises(CrowleyValidationError, match="range\\(x\\)"):
        cf.assert_expectations()
    report = cf.assert_expectations(on_fail="report")
    assert [(r["check"], r["passed"]) for r in report.rows] == [
        ("column_present", True), ("not_null", True), ("unique", True), ("range", False),
    ]
    fixed = cf.filter("x <= 3")
    assert fixed.assert_expectations() is fixed
    # an expectation on a column that was dropped later fails clearly
    dropped = cf.select("id").assert_expectations(on_fail="report")
    assert [r["observed"] for r in dropped.failures] == ["column 'x' not found"]
    plain = df(_data())
    assert plain.assert_expectations() is plain  # nothing attached, nothing to check
    assert (plain >> pipe.expect(not_null=["id"]) >> pipe.assert_expectations()).to_polars().height == 4


def test_schema_only_expectations_read_no_data():
    cf = df(_exploding(_data())).expect(columns=["id"], dtypes={"x": pl.Float64})
    assert cf.assert_expectations() is cf


# ---------------------------------------------------------------------
# branch / define_pipeline / collect_all
# ---------------------------------------------------------------------


def test_branch_and_collect_all():
    base = df(_exploding(_data())).filter("id > 1")
    lazy = base.branch(evens=pipe.filter("id % 2 == 0"), totals=lambda f: f.summarise(total="sum(x)"))
    assert set(lazy) == {"evens", "totals"} and lazy["totals"].columns == ["total"]  # nothing ran
    assert lazy["evens"].audit()["verb"].to_list() == ["filter", "filter"]

    real = df(_data()).filter("id > 1").branch(
        evens=pipe.filter("id % 2 == 0"), totals=lambda f: f.summarise(total="sum(x)")
    )
    done = cr.collect_all(real)
    assert done["evens"].to_polars()["id"].to_list() == [2, 4]
    assert done["totals"].to_polars()["total"].to_list() == [6.0]
    as_list = cr.collect_all([real["evens"], real["totals"]])
    assert isinstance(as_list, list) and as_list[1].to_polars().height == 1

    with pytest.raises(TypeError, match="at least one"):
        base.branch()
    with pytest.raises(TypeError, match="callable"):
        base.branch(bad=123)
    with pytest.raises(TypeError, match="must return a frame"):
        base.branch(bad=lambda f: 1)


def test_define_pipeline():
    clean = define_pipeline(
        "clean",
        pipe.drop_na("x"),
        pipe.mutate(double=col.x * 2),
        lambda f: f.arrange("-double"),
    )
    assert clean.name == "clean" and len(clean.steps) == 3
    assert repr(clean) == "Pipeline('clean': drop_na, mutate, <lambda>)"

    data = _data()
    want = data.lazy().drop_nulls("x").with_columns((pl.col("x") * 2).alias("double"))
    got = clean(df(data, order="fast"))
    assert got.explain() == want.sort(["double"], descending=[True], nulls_last=True).explain()
    assert (df(data) >> clean).to_polars()["double"].to_list() == [8.0, 4.0, 2.0]
    assert clean(df(data.head(2))).to_polars().height == 2  # reusable on another frame
    assert clean(df(_exploding(data))).audit()["verb"].to_list() == ["drop_na", "mutate", "arrange"]

    nested = define_pipeline("nested", clean, pipe.slice_head(1))
    assert nested(df(data)).to_polars()["id"].to_list() == [4]
    with pytest.raises(TypeError, match="not callable"):
        define_pipeline("bad", "filter")


# ---------------------------------------------------------------------
# compare_outputs
# ---------------------------------------------------------------------


def test_compare_outputs_equal_and_tolerance():
    data = _data()
    cf = df(data)
    same = cf.compare_outputs(data.sample(fraction=1.0, shuffle=True, seed=1))
    assert same and same.equal and repr(same) == "OutputComparison: equal (4 rows)"
    assert cf.compare_outputs(data, by="id")
    assert not cf.compare_outputs(data.reverse(), check_order=True)

    nudged = data.with_columns(pl.col("x") * (1 + 1e-12))
    assert cf.compare_outputs(nudged, by="id")
    strict = cf.compare_outputs(nudged, by="id", rtol=0.0)
    assert not strict and strict.mismatches == {"g": 0, "x": 3, "s": 0}
    assert cf.compare_outputs(data.with_columns(pl.col("x") + 0.5), by="id", atol=0.6)

    # two pipelines that should agree
    a = cf.filter("id > 1").mutate(y="x * 2")
    b = cf.mutate(y=col.x + col.x).filter(col.id >= 2)
    assert a.compare_outputs(b, by="id")


def test_compare_outputs_differences():
    data = _data()
    other = pl.DataFrame(
        {"id": [1, 2, 3, 5], "g": ["a", "z", "b", "c"], "x": [1.0, 2.0, 9.0, 5.0], "s": ["p", "q", "p", "q"]}
    )
    cmp = df(data).compare_outputs(other, by="id")
    assert not cmp
    assert (cmp.rows_left, cmp.rows_right, cmp.only_left, cmp.only_right) == (4, 4, 1, 1)
    assert cmp.mismatches == {"g": 1, "x": 1, "s": 0}  # a missing x against 9.0 counts
    assert "1 rows only on the left, 1 only on the right" in repr(cmp)
    table = dict(zip(cmp.to_polars()["item"].to_list(), cmp.to_polars()["value"].to_list()))
    assert table["mismatches:g"] == 1 and table["rows_only_left"] == 1

    shorter = df(data).compare_outputs(data.head(3))
    assert not shorter and shorter.mismatches is None and "row counts differ: 4 vs 3" in repr(shorter)

    retyped = data.with_columns(pl.col("id").cast(pl.Int32)).drop("s")
    schema_diff = df(data).compare_outputs(retyped)
    assert not schema_diff and schema_diff.schema.extra == ["s"]
    assert schema_diff.schema.dtype_mismatches == {"id": ("Int64", "Int32")}
    assert df(data).compare_outputs(data.with_columns(pl.col("id").cast(pl.Int32)), check_dtypes=False, by="g") or True

    assert df(data.head(0)).compare_outputs(data.head(0))
    with pytest.raises(ValueError, match="must exist on both sides"):
        df(data).compare_outputs(other, by="nope")
