"""Stage 2: safer joins and key integrity."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from crowley_frame import CrowleyTranslationError, CrowleyValidationError, df, pipe
from crowley_frame.keys import DIAGNOSTIC_COLUMNS


def _orders() -> pl.DataFrame:
    return pl.DataFrame({"cust": [1, 1, 2, 3, None, 9], "amount": [10, 20, 30, 40, 50, 60]})


def _customers() -> pl.DataFrame:
    return pl.DataFrame({"cust": [1, 2, 3, 4, None], "name": ["a", "b", "c", "d", "n"]})


def _dupes() -> pl.DataFrame:
    return pl.DataFrame({"cust": [1, 1, 2], "name": ["a", "a2", "b"]})


def _exploding(data: pl.DataFrame) -> pl.LazyFrame:
    def boom(_df):
        raise AssertionError("the plan was executed")

    return data.lazy().map_batches(boom, schema=data.schema)


# ---------------------------------------------------------------------
# check_keys
# ---------------------------------------------------------------------


def test_check_keys():
    customers = df(_customers().drop_nulls())
    assert customers.check_keys("cust") is customers
    assert customers.check_keys("cust", other=_orders()) is customers

    report = df(_orders()).check_keys("cust", on_fail="report")
    failed = {r["check"]: r["observed"] for r in report.failures}
    assert failed == {"key_not_null": "1 rows with a missing key", "key_unique": "1 duplicate rows"}
    assert df(_orders()).check_keys("cust", allow_nulls=True, allow_duplicates=True) is not None
    with pytest.raises(CrowleyValidationError, match="key_unique"):
        df(_orders()).check_keys("cust", allow_nulls=True)

    multi = df(pl.DataFrame({"a": [1, 1, 1], "b": ["x", "y", "y"]}))
    assert multi.check_keys(["a", "b"], on_fail="report").failures[0]["observed"] == "1 duplicate rows"
    assert not multi.check_keys(["a", "zzz"], on_fail="report")

    text_keys = _customers().with_columns(pl.col("cust").cast(pl.String))
    report = customers.check_keys("cust", other=text_keys, on_fail="report")
    assert [r["check"] for r in report.failures] == ["key_dtype"]
    assert not customers.check_keys("cust", other=_orders(), other_by="nope", on_fail="report")
    assert df(_orders().head(0)).check_keys("cust") is not None  # an empty frame has valid keys


def test_check_keys_schema_checks_read_no_data():
    cf = df(_exploding(_orders()))
    assert cf.check_keys("cust", other=_customers(), allow_nulls=True, allow_duplicates=True) is cf
    with pytest.raises(Exception, match="the plan was executed"):
        cf.check_keys("cust")


# ---------------------------------------------------------------------
# validate_relationship
# ---------------------------------------------------------------------


def test_validate_relationship():
    orders, customers = df(_orders()), df(_customers())
    assert orders.validate_relationship(customers, by="cust", relationship="many-to-one") is orders
    assert customers.validate_relationship(orders, by="cust", relationship="one-to-many") is customers
    assert orders.validate_relationship(customers, relationship="m:m") is orders  # natural keys, no check needed

    with pytest.raises(CrowleyValidationError, match="left_keys_unique"):
        orders.validate_relationship(customers, by="cust", relationship="one-to-one")
    report = orders.validate_relationship(_dupes(), by="cust", relationship="one-to-one", on_fail="report")
    assert [r["check"] for r in report.failures] == ["left_keys_unique", "right_keys_unique"]
    renamed = _customers().rename({"cust": "customer_id"})
    assert orders.validate_relationship(renamed, left_on="cust", right_on="customer_id") is orders
    with pytest.raises(ValueError, match="relationship must be"):
        orders.validate_relationship(customers, by="cust", relationship="some-to-many")
    with pytest.raises(KeyError):
        orders.validate_relationship(customers, by="nope")

    lazy = df(_exploding(_orders()))
    assert lazy.validate_relationship(_exploding(_customers()), by="cust", relationship="many-to-many") is lazy


# ---------------------------------------------------------------------
# join_diagnostics
# ---------------------------------------------------------------------


@pytest.mark.parametrize("na_matches", ["na", "never"])
def test_join_diagnostics_predicts_real_join_sizes(na_matches):
    orders, customers = _orders(), _customers()
    diag = df(orders).join_diagnostics(customers, by="cust", na_matches=na_matches).to_polars()
    assert diag.columns == DIAGNOSTIC_COLUMNS and diag.height == 1
    row = diag.row(0, named=True)

    nulls_equal = na_matches == "na"
    kw = {"join_nulls": nulls_equal} if "join_nulls" in pl.DataFrame.join.__code__.co_varnames else {"nulls_equal": nulls_equal}
    for how in ("inner", "left", "full"):
        assert row[f"{how}_join_rows"] == orders.join(customers, on="cust", how=how, **kw).height
    semi = orders.join(customers, on="cust", how="semi", **kw).height
    assert row["left_matched_rows"] == semi and row["left_unmatched_rows"] == orders.height - semi
    assert row["left_rows"] == 6 and row["right_rows"] == 5
    assert row["left_null_keys"] == 1 and row["right_null_keys"] == 1
    assert row["left_duplicate_rows"] == 1 and row["right_duplicate_rows"] == 0
    assert row["right_unmatched_rows"] == (1 if nulls_equal else 2)
    assert row["left_match_rate"] == pytest.approx(semi / 6)
    assert row["row_multiplication"] == 1.0


def test_join_diagnostics_row_multiplication_and_errors():
    row = df(_orders()).join_diagnostics(_dupes(), by="cust").to_polars().row(0, named=True)
    assert row["right_duplicate_rows"] == 1
    assert row["left_join_rows"] == _orders().join(_dupes(), on="cust", how="left").height == 8
    assert row["row_multiplication"] == pytest.approx(8 / 6)

    multi_l = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "y"], "v": [1, 2, 3]})
    multi_r = pl.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"], "w": [1, 2, 3]})
    row = df(multi_l).join_diagnostics(multi_r, by=["a", "b"]).to_polars().row(0, named=True)
    assert row["inner_join_rows"] == multi_l.join(multi_r, on=["a", "b"]).height == 3

    empty = df(_orders().head(0)).join_diagnostics(_customers(), by="cust").to_polars().row(0, named=True)
    assert empty["left_rows"] == 0 and empty["inner_join_rows"] == 0 and empty["right_unmatched_rows"] == 5

    text_keys = _customers().with_columns(pl.col("cust").cast(pl.String))
    with pytest.raises(CrowleyValidationError, match="join key dtypes differ"):
        df(_orders()).join_diagnostics(text_keys, by="cust")
    with pytest.raises(ValueError):
        df(pl.DataFrame({"a": [1]})).join_diagnostics(pl.DataFrame({"b": [1]}))


# ---------------------------------------------------------------------
# safe_join
# ---------------------------------------------------------------------


@pytest.mark.parametrize("order", ["dplyr", "fast"])
def test_safe_join_without_checks_is_a_plain_lazy_join(order):
    left = df(_exploding(_orders()), order=order)
    right = df(_exploding(_customers()), order=order)
    out = left.safe_join(right, by="cust")
    assert out.columns == ["cust", "amount", "name"]
    assert out.explain() == left.left_join(right, on="cust").explain()

    real = df(_orders(), order=order).safe_join(_customers(), by="cust", how="inner").to_polars()
    want = df(_orders(), order=order).inner_join(_customers(), on="cust").to_polars()
    assert_frame_equal(real.sort("amount"), want.sort("amount"))
    if order == "dplyr":
        assert real["amount"].to_list() == [10, 20, 30, 40, 50]


def test_safe_join_eager_checks():
    orders, customers = df(_orders()), df(_customers())
    ok = orders.safe_join(customers, by="cust", relationship="many-to-one").to_polars()
    assert ok.height == 6 and ok["name"].to_list() == ["a", "a", "b", "c", "n", None]

    with pytest.raises(CrowleyValidationError, match="right keys are not unique \\(1 duplicate rows\\)"):
        orders.safe_join(_dupes(), by="cust", relationship="many-to-one")
    with pytest.raises(CrowleyValidationError, match="left keys are not unique") as err:
        orders.safe_join(customers, by="cust", relationship="one-to-one")
    assert err.value.report["left_duplicate_rows"] == 1

    with pytest.raises(CrowleyValidationError, match="1 of 6 left rows have no match"):
        orders.safe_join(customers, by="cust", unmatched="error")
    with pytest.raises(CrowleyValidationError, match="2 of 6 left rows"):
        orders.safe_join(customers, by="cust", unmatched="error", na_matches="never")
    with pytest.warns(UserWarning, match="1 of 6 left rows have no match"):
        warned = orders.safe_join(customers, by="cust", unmatched="warn")
    assert warned.to_polars().height == 6
    assert (orders >> pipe.safe_join(customers, by="cust", relationship="m:1")).to_polars().height == 6

    for bad in ({"how": "semi"}, {"unmatched": "drop"}, {"check": "later"}, {"relationship": "1-1"}):
        with pytest.raises(ValueError):
            orders.safe_join(customers, by="cust", **bad)


def test_safe_join_lazy_check_defers_to_polars():
    left, right = df(_exploding(_orders())), df(_exploding(_dupes()))
    out = left.safe_join(right, by="cust", relationship="many-to-one", check="lazy")
    assert out.columns == ["cust", "amount", "name"]  # nothing ran

    with pytest.raises(pl.exceptions.ComputeError, match="validation"):
        df(_orders()).safe_join(_dupes(), by="cust", relationship="many-to-one", check="lazy").to_polars()
    fine = df(_orders()).safe_join(_customers(), by="cust", relationship="many-to-one", check="lazy").to_polars()
    assert fine.height == 6
    with pytest.raises(CrowleyTranslationError, match="cannot check unmatched rows"):
        left.safe_join(right, by="cust", unmatched="error", check="lazy")
