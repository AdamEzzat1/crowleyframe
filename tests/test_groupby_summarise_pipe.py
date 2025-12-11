import pandas as pd
from pandas.testing import assert_frame_equal

from crowley_frame import df, pipe


def test_groupby_summarise_pipe():
    pdf = pd.DataFrame(
        {
            "user_id": [1, 2, 1],
            "user_score": [5, 7, 9],
        }
    )
    cf = df(pdf)

    out = (
        cf
        >> pipe.group_by("user_id")
        >> pipe.summarise(
            mean_score=("user_score", "mean"),
            n=("user_score", "count"),
        )
    ).to_pandas()

    expected = pd.DataFrame(
        {
            "user_id": [1, 2],
            "mean_score": [7.0, 7.0],
            "n": [2, 1],
        }
    )

    # Sort by user_id to avoid any grouping-order surprises
    out_sorted = out.sort_values("user_id").reset_index(drop=True)
    expected_sorted = expected.sort_values("user_id").reset_index(drop=True)

    assert_frame_equal(out_sorted, expected_sorted)
