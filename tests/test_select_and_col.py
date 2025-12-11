import pandas as pd
from pandas.testing import assert_frame_equal

from crowley_frame import df, col


def test_select_exact_and_starts_with():
    pdf = pd.DataFrame(
        {
            "user_id": [1, 2],
            "user_score": [5, 7],
            "other": [0, 0],
        }
    )
    cf = df(pdf)

    out = cf.select(col("user_id"), col.starts_with("user_")).to_pandas()

    expected = pd.DataFrame(
        {
            "user_id": [1, 2],
            "user_score": [5, 7],
        }
    )

    assert_frame_equal(out.reset_index(drop=True), expected)


def test_select_where_numeric_and_where_string():
    pdf = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "score": [10.0, 20.0, 30.0],
            "group": ["a", "b", "a"],
        }
    )
    cf = df(pdf)

    num_out = cf.select(col.where_numeric()).to_pandas()
    str_out = cf.select(col.where_string()).to_pandas()

    expected_num_cols = ["user_id", "score"]
    expected_str_cols = ["group"]

    assert list(num_out.columns) == expected_num_cols
    assert list(str_out.columns) == expected_str_cols
