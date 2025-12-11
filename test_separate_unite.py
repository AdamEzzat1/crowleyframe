from crowley_frame import df


def test_separate_basic():
    data = {
        "id": [1, 2, 3],
        "coords": ["1,2", "10,20", "5,7"],
    }
    cf = df(data)

    out = (
        cf
        .separate("coords", into=["x", "y"], sep=",")
        .to_pandas()
        .sort_values("id")
        .reset_index(drop=True)
    )

    assert list(out.columns) == ["id", "x", "y"]
    assert out["id"].tolist() == [1, 2, 3]
    assert out["x"].tolist() == ["1", "10", "5"]
    assert out["y"].tolist() == ["2", "20", "7"]


def test_unite_basic():
    data = {
        "first": ["Ada", "Bob", "Charlie"],
        "last": ["Lovelace", "Smith", "Brown"],
    }
    cf = df(data)

    out = (
        cf
        .unite("full_name", ["first", "last"], sep=" ")
        .to_pandas()
    )

    assert list(out.columns) == ["full_name"]
    assert out["full_name"].tolist() == [
        "Ada Lovelace",
        "Bob Smith",
        "Charlie Brown",
    ]


def test_unite_na_behavior():
    data = {
        "first": ["Ada", None, "Charlie"],
        "last": ["Lovelace", "Smith", None],
    }
    cf = df(data)

    # Default: na_rm=False => if any NA in row, full result is NA
    out1 = cf.unite("full_name", ["first", "last"], sep=" ").to_pandas()
    assert out1["full_name"].isna().tolist() == [False, True, True]

    # na_rm=True => ignore NAs and join non-null parts
    out2 = cf.unite("full_name", ["first", "last"], sep=" ", na_rm=True).to_pandas()
    assert out2["full_name"].tolist() == [
        "Ada Lovelace",
        "Smith",
        "Charlie",
    ]
