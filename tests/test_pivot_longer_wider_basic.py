import pandas as pd
from pandas.testing import assert_frame_equal

from crowley_frame import df, col


def test_pivot_longer_basic():
    pdf = pd.DataFrame(
        {
            "id": [1, 2],
            "year_2023": [10, 20],
            "year_2024": [30, 40],
        }
    )
    cf = df(pdf)

    out = cf.pivot_longer(
        cols=[col.starts_with("year_")],
        names_to="year",
        values_to="value",
    ).to_pandas()

    expected = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "year": ["year_2023", "year_2024", "year_2023", "year_2024"],
            "value": [10, 30, 20, 40],
        }
    )

    out_sorted = out.sort_values(["id", "year"]).reset_index(drop=True)
    expected_sorted = expected.sort_values(["id", "year"]).reset_index(drop=True)

    assert_frame_equal(out_sorted, expected_sorted)


def test_pivot_wider_basic():
    long_pdf = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "year": ["year_2023", "year_2024", "year_2023", "year_2024"],
            "value": [10, 30, 20, 40],
        }
    )
    cf_long = df(long_pdf)

    wide = cf_long.pivot_wider(
        names_from="year",
        values_from="value",
    ).to_pandas()

    expected = pd.DataFrame(
        {
            "id": [1, 2],
            "year_2023": [10, 20],
            "year_2024": [30, 40],
        }
    )

    wide_sorted = wide.sort_values("id").reset_index(drop=True)
    expected_sorted = expected.sort_values("id").reset_index(drop=True)

    assert_frame_equal(wide_sorted, expected_sorted)
