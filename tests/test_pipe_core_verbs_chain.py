import pandas as pd
import pytest

from crowley_frame import df, pipe


def test_pipe_chain_core_verbs_parenthesized_to_pandas():
    pdf = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3],
            "user_score": [5, 7, 9, 7],
            "other": [0, 0, 1, 1],
        }
    )
    cf = df(pdf)

    out = (
        (
            cf
            >> pipe.filter("user_score >= 7")
            >> pipe.arrange("-user_score")
            >> pipe.rename(user="user_id", score="user_score")
            >> pipe.relocate("score", before="user")
            >> pipe.distinct("user", keep_all=True)
        )
        .to_pandas()
        .reset_index(drop=True)
    )

    # After filter >= 7, rows are (2,7,0), (1,9,1), (3,7,1)
    # arrange -user_score => (1,9,1) first, then 2 and 3 with score 7
    # rename => user_id->user, user_score->score
    # relocate => score before user
    # distinct("user", keep_all=True) keeps the first row per user, with all columns
    assert list(out.columns) == ["score", "user", "other"]
    assert out["user"].tolist() == [1, 2, 3]
    assert out["score"].tolist() == [9, 7, 7]
    assert out["other"].tolist() == [1, 0, 1]


def test_pipe_chain_without_parentheses_to_pandas_is_wrong_precedence():
    """
    This test documents Python precedence: `.to_pandas()` binds to the
    last pipe op result (a function) unless the chain is parenthesized.
    We *expect* an AttributeError here.
    """
    pdf = pd.DataFrame(
        {
            "user_id": [1, 2, 1],
            "user_score": [5, 7, 9],
        }
    )
    cf = df(pdf)

    with pytest.raises(AttributeError):
        # Intentionally missing parentheses around the chain
        (
            cf
            >> pipe.filter("user_score >= 7")
            >> pipe.arrange("-user_score")
            >> pipe.distinct("user").to_pandas()  # binds here, not to the whole chain
        )


def test_pipe_namespace_has_expected_verbs():
    # This catches “stale installed pipe.py” problems early.
    for name in ["group_by", "summarise", "filter", "arrange", "select", "mutate", "rename", "relocate", "distinct"]:
        assert hasattr(pipe, name), f"pipe.{name} is missing (did you rebuild/install?)"
