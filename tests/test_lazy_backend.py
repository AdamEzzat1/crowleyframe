import polars as pl
import pytest
from polars.testing import assert_frame_equal

import crowley_frame as cr
from crowley_frame import (
    ColumnNotFoundError,
    Frame,
    TranslationError,
    case_when,
    col,
    desc,
    df,
    if_else,
    lag,
    mean,
    n,
    pipe,
)


def _data():
    return pl.DataFrame(
        {
            "g": ["b", "a", "b", "a", "c"],
            "x": [1, 2, 3, 4, 5],
            "y": [10.0, None, 30.0, 40.0, 50.0],
        }
    )


class _Exploding:
    """Stands in for a data source: any attempt to read it fails the test."""

    def __init__(self):
        self.lf = pl.LazyFrame({"g": ["a"], "x": [1], "y": [1.0]}).map_batches(self._boom)

    @staticmethod
    def _boom(_df):
        raise AssertionError("the plan was executed")


# ---------------------------------------------------------------------
# laziness
# ---------------------------------------------------------------------


def test_verbs_do_not_materialise():
    cf = df(_Exploding().lf)
    out = (
        cf.select("g", "x", "y")
        .filter("x > 0")
        .mutate(z="x * 2")
        .rename(k="g")
        .group_by("k")
        .summarise(m=mean(col.z), n=n())
        .arrange("-m")
    )
    assert isinstance(out, Frame)
    assert out.columns == ["k", "m", "n"]
    with pytest.raises(Exception, match="the plan was executed"):
        out.to_polars()


def test_fast_mode_emits_the_hand_written_plan():
    data = _data().lazy()
    cf = df(data, order="fast")

    ours = (
        cf.filter(col.x > 1)
        .mutate(z=col.x * 2)
        .group_by("g")
        .summarise(m=mean(col.z))
        .arrange("g")
    )
    theirs = (
        data.filter(pl.col("x") > 1)
        .with_columns((pl.col("x") * 2).alias("z"))
        .group_by(["g"])
        .agg(pl.col("z").mean().alias("m"))
        .sort(["g"], descending=[False], nulls_last=True)
    )
    assert ours.explain() == theirs.explain()
    assert cf.select("g", "x").explain() == data.select(["g", "x"]).explain()


def test_collect_returns_frame_and_explain_is_text():
    cf = df(_data()).filter("x > 2")
    assert "FILTER" in cf.explain().upper() or "SELECTION" in cf.explain().upper()
    collected = cf.collect()
    assert isinstance(collected, Frame)
    assert collected.to_polars().height == 3
    assert (cf >> pipe.collect()).to_polars().height == 3


# ---------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------


def test_string_and_object_expressions_agree():
    cf = df(_data())
    a = cf.mutate(r="x / sum(x)", big="x > 2 and not is_na(y)").to_polars()
    b = cf.mutate(r=col.x / cr.sum(col.x), big=(col.x > 2) & ~cr.is_na(col.y)).to_polars()
    assert_frame_equal(a, b)
    assert a["big"].to_list() == [False, False, True, True, True]


def test_grouped_mutate_and_filter_use_groups():
    cf = df(_data()).group_by("g")
    out = cf.mutate(share="x / sum(x)", prev=lag(col.x)).to_polars()
    assert out["share"].to_list() == [0.25, 2 / 6, 0.75, 4 / 6, 1.0]
    assert out["prev"].to_list() == [None, None, 1, 2, None]

    top = cf.filter("x == max(x)").to_polars()
    assert top["x"].to_list() == [3, 4, 5]


def test_mutate_can_use_columns_created_in_the_same_call():
    out = df(_data()).mutate(a="x + 1", b="a * 10").to_polars()
    assert out["b"].to_list() == [20, 30, 40, 50, 60]


def test_case_when_and_if_else():
    out = (
        df(_data())
        .mutate(
            size=case_when((col.x < 2, "small"), (col.x < 4, "mid"), _default="big"),
            has_y=if_else(cr.is_na(col.y), "no", "yes"),
        )
        .to_polars()
    )
    assert out["size"].to_list() == ["small", "mid", "mid", "big", "big"]
    assert out["has_y"].to_list() == ["yes", "no", "yes", "yes", "yes"]


def test_lag_default_does_not_overwrite_real_missing_values():
    out = df({"x": [3, None, 1, 2]}).mutate(p=lag("x", default=0)).to_polars()
    assert out["p"].to_list() == [0, 3, None, 1]


def test_summarise_forms_and_ungrouped():
    cf = df(_data())
    grouped = cf.group_by("g").summarise(
        a=("x", "mean"), b="mean(x)", c=mean(col.x), rows=n()
    ).to_polars()
    assert grouped["g"].to_list() == ["a", "b", "c"]
    assert grouped["a"].to_list() == grouped["b"].to_list() == grouped["c"].to_list() == [3.0, 2.0, 5.0]
    assert grouped["rows"].to_list() == [2, 2, 1]

    total = cf.summarize(total="sum(x)", rows=n()).to_polars()
    assert total.to_dicts() == [{"total": 15, "rows": 5}]


def test_na_rm_false_propagates_missing():
    cf = df(_data())
    out = cf.summarise(skip=mean(col.y), strict=mean(col.y, na_rm=False)).to_polars()
    assert out["skip"][0] == 32.5
    assert out["strict"][0] is None


# ---------------------------------------------------------------------
# ordering modes
# ---------------------------------------------------------------------


def test_dplyr_order_guarantees():
    cf = df(_data(), order="dplyr")
    assert cf.group_by("g").summarise(m="mean(x)").to_polars()["g"].to_list() == ["a", "b", "c"]
    assert cf.count("g").to_polars()["g"].to_list() == ["a", "b", "c"]
    assert cf.distinct("g").to_polars()["g"].to_list() == ["b", "a", "c"]
    assert cf.distinct("g", keep_all=True).to_polars()["x"].to_list() == [1, 2, 5]
    # stable sort: ties keep input order; missing values last
    assert cf.arrange("g").to_polars()["x"].to_list() == [2, 4, 1, 3, 5]
    assert cf.arrange(desc(col.y)).to_polars()["y"].to_list() == [50.0, 40.0, 30.0, 10.0, None]


def test_fast_order_same_rows():
    data = _data()
    slow = df(data, order="dplyr").group_by("g").summarise(m="mean(x)", k=n()).to_polars()
    fast = df(data, order="fast").group_by("g").summarise(m="mean(x)", k=n()).to_polars()
    assert_frame_equal(slow, fast.sort("g"))


def test_set_order_changes_default():
    assert cr.options.order == "dplyr"
    try:
        cr.set_order("fast")
        plan = df(_data()).group_by("g").summarise(m="mean(x)").explain()
        assert "SORT" not in plan.upper()
    finally:
        cr.set_order("dplyr")
    with pytest.raises(ValueError):
        cr.set_order("random")


def test_joins_keep_left_order_and_match_missing_keys():
    left = df({"id": [3, None, 1, 2], "v": [30, 0, 10, 20]})
    right = df({"id": [1, 2, None], "w": ["one", "two", "missing"]})
    out = left.inner_join(right, on="id").to_polars()
    assert out["v"].to_list() == [0, 10, 20]
    assert out["w"].to_list() == ["missing", "one", "two"]
    assert left.inner_join(right, on="id", na_matches="never").to_polars()["v"].to_list() == [10, 20]
    assert left.anti_join(right, on="id").to_polars()["v"].to_list() == [30]
    assert (left >> pipe.left_join(right, on="id")).to_polars().height == 4


# ---------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------


def test_selector_algebra_and_strictness():
    cf = df({"id": [1], "score_a": [1.0], "score_b": [2.0], "name": ["x"]})
    assert cf.select(~col.starts_with("score")).columns == ["id", "name"]
    assert cf.select(col.where_numeric() - col("id")).columns == ["score_a", "score_b"]
    assert cf.select(col.ends_with("_b") | col.where_string()).columns == ["score_b", "name"]
    assert cf.select(col.any_of(["name", "nope"]), col.last_col()).columns == ["name"]
    assert cf.select().columns == []
    with pytest.raises(ColumnNotFoundError, match="did you mean 'name'"):
        cf.select("nmae")
    with pytest.raises(KeyError):
        cf.select(col.all_of(["id", "nope"]))


# ---------------------------------------------------------------------
# translation errors instead of slow fallbacks
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo hi')",
        "x.__class__",
        "[c for c in x]",
        "(lambda: 1)()",
        "().__class__.__base__.__subclasses__()",
        "np.sqrt(x)",
        "nosuchfn(x)",
        "x > 1 & y > 1",
        "x +",
    ],
)
def test_untranslatable_strings_raise(text):
    cf = df(_data())
    with pytest.raises(TranslationError):
        cf.filter(text)
    with pytest.raises(TranslationError):
        cf.mutate(z=text)


def test_unknown_column_in_expression_raises_at_the_verb():
    cf = df(_data())
    with pytest.raises(ColumnNotFoundError, match="did you mean"):
        cf.mutate(z="xx + 1")
    with pytest.raises(KeyError):
        cf.filter(col.nope > 1)
    with pytest.raises(KeyError):
        cf.group_by("nope")
    with pytest.raises(KeyError):
        cf.rename(a="nope")
    with pytest.raises(KeyError):
        cf.arrange("nope")


def test_python_callables_are_not_silently_run_per_group():
    cf = df(_data()).group_by("g")
    with pytest.raises(TranslationError, match="callables"):
        cf.summarise(m=("x", lambda s: s.mean()))
    with pytest.raises(TranslationError, match="unknown summary function"):
        cf.summarise(m=("x", "kurtosis"))


def test_python_truthiness_and_builtins_fail_loudly():
    with pytest.raises(TypeError, match="ambiguous"):
        bool(col.x > 1)
    with pytest.raises(TypeError, match="builtins"):
        sum(col.x)


def test_untranslated_verb_options_raise():
    cf = df({"k": ["a-b-c", "d-e"]})
    with pytest.raises(TranslationError):
        cf.separate("k", into=["p", "q"], sep="-", fill="left")
    with pytest.raises(TranslationError):
        cf.separate("k", into=["p", "q"], sep="[-_]", extra="merge")
    merged = cf.separate("k", into=["p", "q"], sep="-", extra="merge").to_polars()
    assert merged["q"].to_list() == ["b-c", "e"]


# ---------------------------------------------------------------------
# reshaping
# ---------------------------------------------------------------------


def test_pivot_longer_is_row_major_in_dplyr_mode_and_unite_keeps_literal_strings():
    wide = df({"id": [1, 2], "a": [10, 20], "b": [11, 21]})
    long = wide.pivot_longer(col("a"), col("b")).to_polars()
    assert long.columns == ["id", "name", "value"]
    assert long["id"].to_list() == [1, 1, 2, 2]
    assert long["value"].to_list() == [10, 11, 20, 21]

    united = df({"a": ["None", "x"], "b": ["nan", "z"]}).unite("u", ["a", "b"]).to_polars()
    assert united["u"].to_list() == ["None_nan", "x_z"]


def test_slice_sample_draws_whole_rows():
    cf = df({"x": list(range(100)), "y": [i * 10 for i in range(100)]})
    out = cf.slice_sample(n=20).to_polars()
    assert out.height == 20
    assert (out["y"] == out["x"] * 10).all()
