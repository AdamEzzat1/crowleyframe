import math

import pandas as pd
from crowley_frame import df


def test_slice_head_tail_basic():
    data = {
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
    }
    cf = df(data)

    head = cf.slice_head(3).to_pandas()
    tail = cf.slice_tail(2).to_pandas()

    assert head["x"].tolist() == [1, 2, 3]
    assert head["y"].tolist() == [10, 20, 30]

    assert tail["x"].tolist() == [4, 5]
    assert tail["y"].tolist() == [40, 50]


def test_slice_sample_n_and_prop():
    data = {"x": list(range(10))}
    cf = df(data)

    # sample fixed n
    s1 = cf.slice_sample(n=3, random_state=42).to_pandas()
    assert len(s1) == 3
    assert set(s1["x"]).issubset(set(range(10)))

    # sample by proportion
    s2 = cf.slice_sample(prop=0.5, random_state=42).to_pandas()
    # 0.5 * 10 = 5, but pandas could give exactly 5; we just check it's plausible
    assert 3 <= len(s2) <= 7
    assert set(s2["x"]).issubset(set(range(10)))

    # argument validation
    try:
        cf.slice_sample()
        assert False, "Expected ValueError when neither n nor prop provided"
    except ValueError:
        pass

    try:
        cf.slice_sample(n=3, prop=0.5)
        assert False, "Expected ValueError when both n and prop provided"
    except ValueError:
        pass


def test_slice_max_min_basic():
    data = {
        "x": [1, 5, 3, 9, 2],
        "y": ["a", "b", "c", "d", "e"],
    }
    cf = df(data)

    max2 = cf.slice_max("x", n=2).to_pandas()
    min2 = cf.slice_min("x", n=2).to_pandas()

    # nlargest/nsmallest return sorted by that column
    assert max2["x"].tolist() == [9, 5]
    assert min2["x"].tolist() == [1, 2]

    # column existence error
    try:
        cf.slice_max("z", n=1)
        assert False, "Expected KeyError for missing column"
    except KeyError:
        pass

    try:
        cf.slice_min("z", n=1)
        assert False, "Expected KeyError for missing column"
    except KeyError:
        pass
