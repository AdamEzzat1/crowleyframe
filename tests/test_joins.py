import pandas as pd
from pandas.testing import assert_frame_equal

from crowley_frame import df


def test_left_join_basic():
    left = df(pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]}))
    right = df(pd.DataFrame({"id": [2, 3, 4], "y": [200, 300, 400]}))

    out = left.left_join(right, on="id").to_pandas()

    expected = pd.merge(
        left.to_pandas(),
        right.to_pandas(),
        how="left",
        on="id",
        suffixes=("_x", "_y"),
    )

    assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_inner_join_basic():
    left = df(pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]}))
    right = df(pd.DataFrame({"id": [2, 3, 4], "y": [200, 300, 400]}))

    out = left.inner_join(right, on="id").to_pandas()

    expected = pd.merge(
        left.to_pandas(),
        right.to_pandas(),
        how="inner",
        on="id",
        suffixes=("_x", "_y"),
    )

    assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_join_suffixes_when_columns_overlap():
    left = df(pd.DataFrame({"id": [1, 2], "val": [10, 20]}))
    right = df(pd.DataFrame({"id": [2, 3], "val": [200, 300]}))

    out = left.left_join(right, on="id", suffixes=("_left", "_right")).to_pandas()

    expected = pd.merge(
        left.to_pandas(),
        right.to_pandas(),
        how="left",
        on="id",
        suffixes=("_left", "_right"),
    )

    assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))
    assert "val_left" in out.columns
    assert "val_right" in out.columns


def test_join_nan_key_behavior_matches_pandas():
    # Lock down whatever pandas does on THIS pandas version.
    # (Some versions treat NaN keys as matchable; others don't.)
    left_pdf = pd.DataFrame({"id": [1.0, float("nan"), 3.0], "x": [10, 20, 30]})
    right_pdf = pd.DataFrame({"id": [float("nan"), 3.0], "y": [200, 300]})

    left = df(left_pdf)
    right = df(right_pdf)

    out_left = left.left_join(right, on="id").to_pandas()
    out_inner = left.inner_join(right, on="id").to_pandas()

    expected_left = pd.merge(left_pdf, right_pdf, how="left", on="id", suffixes=("_x", "_y"))
    expected_inner = pd.merge(left_pdf, right_pdf, how="inner", on="id", suffixes=("_x", "_y"))

    assert_frame_equal(out_left.reset_index(drop=True), expected_left.reset_index(drop=True))
    assert_frame_equal(out_inner.reset_index(drop=True), expected_inner.reset_index(drop=True))
