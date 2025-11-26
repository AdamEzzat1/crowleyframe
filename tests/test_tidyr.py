from crowleyframe import df, col


def test_pivot_longer_basic():
    # One row per id for clean, non-aggregating behavior
    data = {
        "id": [1, 2],
        "year_2023": [10, 30],
        "year_2024": [11, 31],
    }
    cf = df(data)

    long = (
        cf
        .pivot_longer(
            col.matches(r"^year_"),
            names_to="year",
            values_to="value",
        )
        .to_pandas()
        .sort_values(["id", "year"])
        .reset_index(drop=True)
    )

    assert list(long.columns) == ["id", "year", "value"]
    # 2 ids × 2 year columns = 4 rows
    assert long["id"].tolist() == [1, 1, 2, 2]
    assert long["year"].tolist() == ["year_2023", "year_2024", "year_2023", "year_2024"]
    assert long["value"].tolist() == [10, 11, 30, 31]


def test_pivot_wider_roundtrip():
    data = {
        "id": [1, 2],
        "year_2023": [10, 30],
        "year_2024": [11, 31],
    }
    cf = df(data)

    long = cf.pivot_longer(
        col.matches(r"^year_"),
        names_to="year",
        values_to="value",
    )

    wide = (
        long
        .pivot_wider(
            names_from="year",
            values_from="value",
            values_fill=None,
        )
        .to_pandas()
        .sort_values(["id"])
        .reset_index(drop=True)
    )

    # Roundtrip: we should recover original structure (maybe column order differs)
    assert "id" in wide.columns
    assert "year_2023" in wide.columns
    assert "year_2024" in wide.columns

    wide_sorted = wide.sort_values("id").reset_index(drop=True)

    assert wide_sorted["id"].tolist() == [1, 2]
    assert wide_sorted["year_2023"].tolist() == [10, 30]
    assert wide_sorted["year_2024"].tolist() == [11, 31]
