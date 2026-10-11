"""
Second verb layer: counting, transmute, selectors, expression helpers,
across/pick, missing-data verbs and the remaining joins.

Each test compares against literal expected output or the equivalent
hand-written Polars lazy query.
"""
import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import crowley_frame as cr
from crowley_frame import (
    ColumnNotFoundError,
    CrowleyTranslationError,
    TranslationError,
    across,
    all_of,
    any_of,
    between,
    case_when,
    coalesce,
    col,
    contains,
    dense_rank,
    desc,
    df,
    ends_with,
    everything,
    if_else,
    is_boolean,
    is_na,
    is_numeric,
    is_string,
    is_temporal,
    lag,
    lead,
    matches,
    mean,
    min_rank,
    n,
    n_distinct,
    na_if,
    near,
    not_na,
    num_range,
    percent_rank,
    pick,
    pipe,
    row_number,
    sd,
    starts_with,
    where,
)

ORDERS = ["dplyr", "fast"]


def _data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "g": ["b", "a", "b", "a", "c", "b"],
            "x": [1, 2, 3, 4, 5, 3],
            "y": [10.0, None, 30.0, 40.0, None, 60.0],
            "s": ["p", "q", None, "q", "p", "p"],
        }
    )


def _same_rows(got: pl.DataFrame, want: pl.DataFrame, keys) -> None:
    assert_frame_equal(got.sort(keys), want.sort(keys), check_dtype=False)


def test_error_class_names_are_the_same_class():
    assert CrowleyTranslationError is TranslationError
    assert issubclass(CrowleyTranslationError, ValueError)


# ---------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_count_and_add_count(order):
    data = _data()
    cf = df(data, order=order)

    counted = cf.count("g").to_polars()
    raw = data.lazy().group_by("g").agg(pl.len().alias("n")).collect()
    _same_rows(counted, raw, "g")
    assert counted.schema["n"] == pl.Int64
    if order == "dplyr":
        assert counted["g"].to_list() == ["a", "b", "c"]
        assert cf.count("g", sort=True).to_polars()["g"].to_list() == ["b", "a", "c"]

    added = cf.add_count("g").to_polars()
    assert added["x"].to_list() == data["x"].to_list()  # every row kept, in place
    assert added["n"].to_list() == [3, 2, 3, 2, 1, 3]
    assert cf.add_count().to_polars()["n"].to_list() == [6] * 6
    assert cf.group_by("g").add_count(name="k").to_polars()["k"].to_list() == [3, 2, 3, 2, 1, 3]
    assert cf.add_count("g", wt="x").to_polars()["n"].to_list() == [7, 6, 7, 6, 5, 7]


@pytest.mark.parametrize("order", ORDERS)
def test_distinct(order):
    data = _data()
    cf = df(data, order=order)
    got = cf.distinct("g", "s").to_polars()
    _same_rows(got, data.select("g", "s").unique(), ["g", "s"])
    assert got.columns == ["g", "s"]
    if order == "dplyr":
        assert got["g"].to_list() == ["b", "a", "b", "c"]
        assert cf.distinct("g", keep_all=True).to_polars()["x"].to_list() == [1, 2, 5]
    assert cf.distinct().to_polars().height == 6


# ---------------------------------------------------------------------
# column verbs
# ---------------------------------------------------------------------


def test_transmute_keeps_only_new_columns_and_groups():
    data = _data()
    cf = df(data)
    got = cf.transmute(double="x * 2", ratio=col.y / col.x)
    want = data.lazy().select((pl.col("x") * 2).alias("double"), (pl.col("y") / pl.col("x")).alias("ratio"))
    assert got.explain() == want.explain()
    assert_frame_equal(got.to_polars(), want.collect())

    chained = cf.transmute(a="x + 1", b="a * 10").to_polars()
    assert chained.columns == ["a", "b"]
    assert chained["b"].to_list() == [20, 30, 40, 50, 60, 40]

    grouped = cf.group_by("g").transmute(share="x / sum(x)")
    assert grouped.columns == ["g", "share"]
    assert grouped.groups == ("g",)
    assert grouped.to_polars()["share"].to_list() == [1 / 7, 2 / 6, 3 / 7, 4 / 6, 1.0, 3 / 7]


def test_relocate_rename_with_pull_glimpse(capsys):
    cf = df(_data())
    assert cf.relocate("s").columns == ["s", "g", "x", "y"]
    assert cf.relocate(where(is_numeric), after="s").columns == ["g", "s", "x", "y"]
    assert cf.relocate("y", before="g").columns == ["y", "g", "x", "s"]

    assert cf.rename_with(str.upper).columns == ["G", "X", "Y", "S"]
    assert cf.rename_with(lambda c: c + "_num", where(is_numeric)).columns == ["g", "x_num", "y_num", "s"]
    assert cf.group_by("g").rename_with(str.upper).groups == ("G",)
    assert cf.group_by("g").rename(grp="g").groups == ("grp",)
    with pytest.raises(TypeError):
        cf.rename_with(lambda c: 1)

    assert cf.pull("x").to_list() == [1, 2, 3, 4, 5, 3]
    assert cf.pull().name == "s"
    with pytest.raises(ColumnNotFoundError):
        cf.pull("nope")

    assert cf.group_by("g").glimpse(n=2).groups == ("g",)
    out = capsys.readouterr().out
    assert "Rows: ?? (lazy)" in out and "Columns: 4" in out and "Groups: g" in out
    assert "$ x <Int64> 1, 2" in out
    assert "$ s <String> 'p', 'q'" in out


# ---------------------------------------------------------------------
# slicing
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_slice_min_max(order):
    data = _data()
    cf = df(data, order=order)

    ties = cf.slice_max("x", n=3).to_polars()  # 5, 4, then both 3s
    assert sorted(ties["x"].to_list()) == [3, 3, 4, 5]
    no_ties = cf.slice_max("x", n=3, with_ties=False).to_polars()
    assert sorted(no_ties["x"].to_list()) == [3, 4, 5]
    assert sorted(cf.slice_min("y", n=2).to_polars()["y"].to_list()) == [10.0, 30.0]  # missing never chosen
    if order == "dplyr":
        assert ties["x"].to_list() == [5, 4, 3, 3]
        assert cf.slice_min("x", n=2).to_polars()["x"].to_list() == [1, 2]
    else:
        assert "SORT" not in cf.slice_max("x", n=3).explain().upper()  # no ordering work in fast mode
    raw = data.lazy().top_k(3, by="x").collect()
    assert sorted(no_ties["x"].to_list()) == sorted(raw["x"].to_list())  # which tied row is kept is arbitrary
    assert cf.slice_min("x", n=1, with_ties=False).to_polars()["x"].to_list() == [1]
    assert cf.slice_max("y", n=10, with_ties=False).to_polars().height == 4  # missing values are never selected

    grouped = cf.group_by("g").slice_max("x", n=1).to_polars()
    assert sorted(zip(grouped["g"].to_list(), grouped["x"].to_list())) == [("a", 4), ("b", 3), ("b", 3), ("c", 5)]
    one_each = cf.group_by("g").slice_min("x", n=1, with_ties=False).to_polars()
    assert sorted(zip(one_each["g"].to_list(), one_each["x"].to_list())) == [("a", 2), ("b", 1), ("c", 5)]


def test_slice_head_tail_grouped():
    cf = df(_data()).group_by("g")
    assert cf.slice_head(1).to_polars()["x"].to_list() == [1, 2, 5]
    assert cf.slice_tail(1).to_polars()["x"].to_list() == [4, 5, 3]
    assert df(_data()).slice_head(2).explain() == _data().lazy().head(2).explain()


# ---------------------------------------------------------------------
# tidyselect
# ---------------------------------------------------------------------


def test_tidyselect_helpers():
    cf = df(
        pl.DataFrame(
            {
                "id": [1],
                "Score_a": [1.0],
                "score_b": [2.0],
                "wk1": [1],
                "wk2": [2],
                "wk10": [3],
                "name": ["x"],
                "flag": [True],
                "day": [dt.date(2024, 1, 1)],
            }
        )
    )
    assert cf.select(starts_with("score")).columns == ["score_b"]
    assert cf.select(starts_with("score", ignore_case=True)).columns == ["Score_a", "score_b"]
    assert cf.select(starts_with("wk", "na")).columns == ["wk1", "wk2", "wk10", "name"]
    assert cf.select(ends_with("_a", "_b")).columns == ["Score_a", "score_b"]
    assert cf.select(contains("core")).columns == ["Score_a", "score_b"]
    assert cf.select(matches(r"^wk\d$")).columns == ["wk1", "wk2"]
    assert cf.select(matches("^SCORE", ignore_case=True)).columns == ["Score_a", "score_b"]
    assert cf.select(everything()).columns == cf.columns
    assert cf.select("name", everything()).columns[:2] == ["name", "id"]
    assert cf.select(all_of(["name", "id"])).columns == ["name", "id"]
    assert cf.select(any_of(["name", "missing"])).columns == ["name"]
    assert cf.select(num_range("wk", range(1, 4))).columns == ["wk1", "wk2"]
    assert cf.select(num_range("wk", [10, 1])).columns == ["wk10", "wk1"]

    assert cf.select(where(is_numeric)).columns == ["id", "Score_a", "score_b", "wk1", "wk2", "wk10"]
    assert cf.select(where(is_string)).columns == ["name"]
    assert cf.select(where(is_boolean)).columns == ["flag"]
    assert cf.select(where(is_temporal)).columns == ["day"]
    assert cf.select(where(pl.Float64)).columns == ["Score_a", "score_b"]
    assert cf.select(where((pl.Boolean, pl.Date))).columns == ["flag", "day"]
    assert cf.select(where(is_numeric) & ~starts_with("wk")).columns == ["id", "Score_a", "score_b"]

    with pytest.raises(ColumnNotFoundError):
        cf.select(all_of(["name", "missing"]))
    with pytest.raises(CrowleyTranslationError, match="dtype, not its values"):
        cf.select(where(lambda column: column.mean() > 0))
    with pytest.raises(CrowleyTranslationError, match="True or False"):
        cf.select(where(lambda dtype: "yes"))


def test_selectors_never_touch_data():
    def boom(_df):
        raise AssertionError("the plan was executed")

    lf = pl.LazyFrame({"a_1": [1], "a_2": [2.0], "b": ["x"]}).map_batches(boom)
    cf = df(lf)
    assert cf.select(where(is_numeric), starts_with("b")).columns == ["a_1", "a_2", "b"]
    assert cf.drop_na(starts_with("a")).fill(everything()).relocate(last := "b").columns[0] == last


# ---------------------------------------------------------------------
# expression helpers
# ---------------------------------------------------------------------


def test_ranking_and_offsets():
    data = _data()
    out = (
        df(data)
        .mutate(
            rn=row_number(),
            mr=min_rank(col.x),
            mr_desc=min_rank(desc(col.x)),
            dr=dense_rank(col.x),
            pr=percent_rank(col.x),
            prev=lag(col.x),
            nxt=lead(col.x, 2, default=-1),
            rows=n(),
        )
        .to_polars()
    )
    assert out["rn"].to_list() == [1, 2, 3, 4, 5, 6]
    assert out["mr"].to_list() == [1, 2, 3, 5, 6, 3]
    assert out["mr_desc"].to_list() == [6, 5, 3, 2, 1, 3]
    assert out["dr"].to_list() == [1, 2, 3, 4, 5, 3]
    assert out["pr"].to_list() == [0.0, 0.2, 0.4, 0.8, 1.0, 0.4]
    assert out["prev"].to_list() == [None, 1, 2, 3, 4, 5]
    assert out["nxt"].to_list() == [3, 4, 5, 3, -1, -1]
    assert out["rows"].to_list() == [6] * 6

    grouped = df(data).group_by("g").mutate(rn=row_number(), mr=min_rank("x"), prev=lag("x"), rows=n()).to_polars()
    want = data.with_columns(
        pl.int_range(1, pl.len() + 1).over("g").alias("rn"),
        pl.col("x").rank("min").over("g").alias("mr"),
        pl.col("x").shift(1).over("g").alias("prev"),
        pl.len().over("g").alias("rows"),
    )
    assert_frame_equal(grouped, want, check_dtype=False)


def test_value_helpers_and_missing_values():
    data = _data()
    out = (
        df(data)
        .mutate(
            btw=between(col.x, 2, 4),
            close=near(col.y, 30.0000000001),
            filled=coalesce(col.y, col.x, 0),
            sign=if_else(col.y > 25, "hi", "lo"),
            band=case_when((col.x < 2, "low"), (col.x < 4, "mid"), _default="high"),
            no_threes=na_if(col.x, 3),
            s_missing=is_na(col.s),
            s_present=not_na("s"),
        )
        .to_polars()
    )
    assert out["btw"].to_list() == [False, True, True, True, False, True]
    assert out["close"].to_list() == [False, None, True, False, None, False]
    assert out["filled"].to_list() == [10.0, 2.0, 30.0, 40.0, 5.0, 60.0]
    assert out["sign"].to_list() == ["lo", None, "hi", "hi", None, "hi"]  # missing condition stays missing
    assert out["band"].to_list() == ["low", "mid", "mid", "high", "high", "mid"]
    assert out["no_threes"].to_list() == [1, 2, None, 4, 5, None]
    assert out["s_missing"].to_list() == [False, False, True, False, False, False]
    assert out["s_present"].to_list() == [True, True, False, True, True, True]

    strings = df(data).filter("not_na(y) and between(x, 1, 3)").mutate(z="na_if(x, 3)", p="percent_rank(x)")
    assert strings.to_polars()["z"].to_list() == [1, None, None]


# ---------------------------------------------------------------------
# across / pick
# ---------------------------------------------------------------------


def test_across_in_summarise_names_and_values():
    data = _data()
    cf = df(data)

    single = cf.group_by("g").summarise(across(where(is_numeric), mean))
    want = data.lazy().group_by("g").agg(pl.col("x").mean(), pl.col("y").mean()).sort("g").collect()
    assert single.columns == ["g", "x", "y"]
    assert_frame_equal(single.to_polars(), want)

    several = cf.summarise(across(["x", "y"], {"avg": mean, "sd": sd}), rows=n()).to_polars()
    assert several.columns == ["x_avg", "x_sd", "y_avg", "y_sd", "rows"]
    assert several["x_avg"][0] == 3.0 and several["y_avg"][0] == 35.0 and several["rows"][0] == 6

    listed = cf.summarise(across("x", [mean, "max", lambda v: cr.sum(v) / n()]))
    assert listed.columns == ["x_mean", "x_max", "x_3"]
    assert listed.to_polars().row(0) == (3.0, 5, 3.0)

    templated = cf.summarise(across(starts_with("x", "y"), "mean", _names="avg_{col}"))
    assert templated.columns == ["avg_x", "avg_y"]

    counts = cf.group_by("g").summarise(across(["s", "x"], n_distinct, _names="{col}_n")).to_polars()
    assert counts.schema["s_n"] == pl.Int64
    assert counts["s_n"].to_list() == [1, 2, 1] and counts["x_n"].to_list() == [2, 2, 1]


def test_across_in_mutate_and_transmute():
    data = _data()
    cf = df(data)

    out = cf.mutate(across(where(is_numeric), lambda v: v - mean(v), _names="{col}_c"), flag="x_c > 0")
    assert out.columns == ["g", "x", "y", "s", "x_c", "y_c", "flag"]
    assert out.to_polars()["x_c"].to_list() == [-2.0, -1.0, 0.0, 1.0, 2.0, 0.0]
    assert out.to_polars()["flag"].to_list() == [False, False, False, True, True, False]

    in_place = cf.mutate(across(["x", "y"], lambda v: v * 2))
    want = data.lazy().with_columns((pl.col("x") * 2).alias("x"), (pl.col("y") * 2).alias("y"))
    assert in_place.explain() == want.explain()

    grouped = cf.group_by("g").mutate(across(everything() - col("s"), lag, _names="prev_{col}")).to_polars()
    assert "prev_g" not in grouped.columns  # grouping columns are skipped
    assert grouped["prev_x"].to_list() == [None, None, 1, 2, None, 3]

    assert cf.transmute(across("x", {"sq": lambda v: v * v})).columns == ["x_sq"]

    with pytest.raises(CrowleyTranslationError, match="duplicate column names"):
        cf.mutate(across(["x", "y"], [mean, sd], _names="{col}"))
    with pytest.raises(CrowleyTranslationError, match="unknown function"):
        cf.mutate(across("x", "kurtosis"))
    with pytest.raises(CrowleyTranslationError, match="must return a crowley_frame expression"):
        cf.mutate(across("x", lambda v: 1))
    with pytest.raises(TypeError, match="across"):
        cf.mutate(col.x * 2)


def test_pick():
    data = _data()
    cf = df(data)
    assert cf.summarise(k=n_distinct(pick("g", "s"))).to_polars()["k"][0] == 4
    grouped = cf.group_by("g").summarise(k=n_distinct(pick(where(is_numeric)))).to_polars()
    assert grouped["k"].to_list() == [2, 3, 1]
    ranked = cf.mutate(r=dense_rank(pick("g", "x"))).to_polars()
    assert ranked["r"].to_list() == [3, 1, 4, 2, 5, 4]
    assert cf.arrange(pick("g", "x")).to_polars()["x"].to_list() == [2, 4, 1, 3, 3, 5]
    with pytest.raises(CrowleyTranslationError, match="no columns"):
        cf.mutate(z=pick(starts_with("zzz")))


# ---------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------


def test_drop_na_replace_na_fill():
    data = _data()
    cf = df(data)

    assert cf.drop_na().to_polars()["x"].to_list() == [1, 4, 6 - 3]
    assert cf.drop_na("y").explain() == data.lazy().drop_nulls("y").explain()
    assert cf.drop_na(where(is_string)).to_polars().height == 5

    replaced = cf.replace_na({"y": 0.0}, s="unknown")
    want = data.lazy().with_columns(pl.col("y").fill_null(0.0), pl.col("s").fill_null("unknown"))
    assert replaced.explain() == want.explain()
    assert replaced.to_polars()["s"].to_list() == ["p", "q", "unknown", "q", "p", "p"]
    with pytest.raises(ColumnNotFoundError):
        cf.replace_na(nope=1)

    assert cf.fill("y").to_polars()["y"].to_list() == [10.0, 10.0, 30.0, 40.0, 40.0, 60.0]
    assert cf.fill("y", direction="up").to_polars()["y"].to_list() == [10.0, 30.0, 30.0, 40.0, 60.0, 60.0]
    lead_gap = df({"v": [None, 1, None, 2, None]})
    assert lead_gap.fill("v", direction="downup").to_polars()["v"].to_list() == [1, 1, 1, 2, 2]
    assert lead_gap.fill("v", direction="updown").to_polars()["v"].to_list() == [1, 1, 2, 2, 2]
    # within groups: a's first y is missing and stays missing going down
    assert cf.group_by("g").fill("y", "s").to_polars()["y"].to_list() == [10.0, None, 30.0, 40.0, None, 60.0]
    assert cf.fill("y").explain() == data.lazy().with_columns(pl.col("y").fill_null(strategy="forward")).explain()
    with pytest.raises(ValueError):
        cf.fill("y", direction="sideways")


@pytest.mark.parametrize("order", ORDERS)
def test_pivot_longer(order):
    data = pl.DataFrame({"id": [1, 2], "wk1": [10, 20], "wk2": [11, None]})
    cf = df(data, order=order)
    got = cf.pivot_longer(starts_with("wk"), names_to="week", values_to="v", names_prefix="wk").to_polars()
    want = data.unpivot(on=["wk1", "wk2"], index="id", variable_name="week", value_name="v").with_columns(
        pl.col("week").str.strip_prefix("wk")
    )
    _same_rows(got, want, ["id", "week"])
    if order == "dplyr":
        assert got["id"].to_list() == [1, 1, 2, 2]
    else:
        raw = data.lazy().unpivot(on=["wk1", "wk2"], index=["id"], variable_name="name", value_name="value")
        assert cf.pivot_longer(starts_with("wk")).explain() == raw.explain()
    assert cf.pivot_longer(starts_with("wk"), values_drop_na=True).to_polars().height == 3
    with pytest.raises(ValueError):
        cf.pivot_longer(starts_with("zzz"))


# ---------------------------------------------------------------------
# joins
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_semi_anti_cross_join(order):
    left = pl.DataFrame({"id": [3, 1, 2, None], "v": ["c", "a", "b", "z"]})
    right = pl.DataFrame({"id": [1, 3, 3, None], "v": ["x", "y", "y2", "n"]})
    a, b = df(left, order=order), df(right, order=order)

    semi = a.semi_join(b, on="id").to_polars()
    _same_rows(semi, left.filter(pl.col("id").is_in([1, 3]) | pl.col("id").is_null()), "v")
    assert semi.columns == ["id", "v"]
    _same_rows(a.semi_join(b, on="id", na_matches="never").to_polars(), left.filter(pl.col("id").is_in([1, 3])), "v")
    assert a.anti_join(b, on="id").to_polars()["v"].to_list() == ["b"]
    assert (a >> pipe.anti_join(b, by="id", na_matches="never")).to_polars().height == 2

    cross = a.cross_join(b).to_polars()
    assert cross.columns == ["id_x", "v_x", "id_y", "v_y"]
    assert cross.height == 16
    small = df({"k": [1, 2]}, order=order).cross_join(df({"m": ["a", "b"]}))
    assert small.explain() == pl.LazyFrame({"k": [1, 2]}).join(pl.LazyFrame({"m": ["a", "b"]}), how="cross").explain()
    assert sorted(small.to_polars().rows()) == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]


def test_no_new_verb_runs_the_plan():
    def boom(_df):
        raise AssertionError("the plan was executed")

    lf = pl.LazyFrame({"g": ["a"], "x": [1], "y": [1.0], "s": ["p"]}).map_batches(boom)
    cf = df(lf)
    other = df(pl.LazyFrame({"g": ["a"], "w": [1]}))
    out = (
        cf.drop_na("y")
        .replace_na(s="?")
        .fill("y")
        .add_count("g")
        .mutate(across(where(is_numeric), lambda v: v - mean(v), _names="{col}_c"), r=percent_rank(col.x))
        .semi_join(other, on="g")
        .anti_join(df(pl.LazyFrame({"g": ["zzz"]})), on="g")
        .cross_join(other)
        .rename_with(str.upper)
        .relocate("X")
        .distinct("G_X", "X", keep_all=True)
        .slice_max("X", n=3)
        .transmute(k=col.X * 2, G="G_X")
        .group_by("G")
        .summarise(across(everything(), {"mean": mean}), rows=n())
    )
    assert out.columns == ["G", "k_mean", "rows"]
    assert isinstance(out.explain(), str)
