import pandas as pd
from pandas.testing import assert_frame_equal

from crowley_frame import df


def test_count_by_column_with_prop_and_sort():
    pdf = pd.DataFrame(
        {
            "grp": ["a", "a", "b", "b", "b"],
        }
    )
    cf = df(pdf)

    out = cf.count("grp", sort=True, prop=True).to_pandas()

    # Expect grp b (3 rows), then grp a (2 rows)
    expected = pd.DataFrame(
        {
            "grp": ["b", "a"],
            "n": [3, 2],
            "prop": [3 / 5, 2 / 5],
        }
    )

    # Sorting should already be correct because sort=True, but make it explicit
    out_sorted = out.sort_values("grp").reset_index(drop=True)
    expected_sorted = expected.sort_values("grp").reset_index(drop=True)

    assert_frame_equal(out_sorted, expected_sorted)


def test_count_without_columns_counts_rows():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    cf = df(pdf)

    out = cf.count().to_pandas()

    # One row, column 'n' == number of rows
    assert list(out.columns) == ["n"]
    assert out.loc[0, "n"] == 3
