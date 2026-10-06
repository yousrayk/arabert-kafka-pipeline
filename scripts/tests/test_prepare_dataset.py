import pandas as pd

from prepare_dataset import map_labels, split


def astd_frame():
    # ASTD class ids: 0 Neutral, 1 Objective, 2 Positive, 3 Negative
    return pd.DataFrame({"tweet": ["محايد", "خبر", "رائع", "سيء", "رائع", "  "], "label": [0, 1, 2, 3, 2, 0]})


def test_objective_folds_into_neutral_by_default_and_dedupes():
    df = map_labels(astd_frame(), objective="neutral")
    assert list(df["label"]) == ["neutral", "neutral", "positive", "negative"]


def test_objective_can_be_dropped():
    df = map_labels(astd_frame(), objective="drop")
    assert "خبر" not in set(df["text"])
    assert set(df["label"]) == {"neutral", "positive", "negative"}


def test_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"id": range(300), "text": [f"t{i}" for i in range(300)],
                       "label": ["neutral"] * 200 + ["negative"] * 60 + ["positive"] * 40})
    train, val, test = split(df, seed=1)
    assert (len(train), len(val), len(test)) == (210, 30, 60)
    assert set(train["id"]) & set(test["id"]) == set()
    assert test["label"].value_counts()["positive"] == 8  # 40 * 0.2
