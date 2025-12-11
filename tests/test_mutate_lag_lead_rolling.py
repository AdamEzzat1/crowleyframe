import pandas as pd
from pandas.testing import assert_frame_equal

from crowley_frame import df, lag, lead, rolling_mean


def test_mutate_string_expression():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    cf = df(pdf)

    out = cf.mutate(z="x + y").to_pandas()

    expected = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
            "z": [11, 22, 33],
        }
    )

    assert_frame_equal(out.reset_index(drop=True), expected)


def test_mutate_lag_and_lead():
    pdf = pd.DataFrame({"val": [10, 20, 30]})
    cf = df(pdf)

    out = cf.mutate(
        val_lag=lag("val", 1),
        val_lead=lead("val", 1),
    ).to_pandas()

    expected = pd.DataFrame(
        {
            "val": [10, 20, 30],
            "val_lag": [float("nan"), 10.0, 20.0],
            "val_lead": [20.0, 30.0, float("nan")],
        }
    )

    # use almost_equal-ish compare, since NaNs need special handling
    assert_frame_equal(out.reset_index(drop=True), expected)


def test_mutate_rolling_mean():
    pdf = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
    cf = df(pdf)

    out = cf.mutate(
        roll=rolling_mean("val", window=2, min_periods=1),
    ).to_pandas()

    expected = pd.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0],
            "roll": [1.0, 1.5, 2.5, 3.5],
        }
    )

    assert_frame_equal(out.reset_index(drop=True), expected)
