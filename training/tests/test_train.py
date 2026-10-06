"""Data-loading tests for train.py. They import torch/transformers (but
download nothing), so they skip in environments without the training
dependencies."""
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("arabert")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import LABEL2ID, load_split  # noqa: E402


@pytest.fixture
def split_csv(tmp_path):
    df = pd.DataFrame({"id": range(100), "text": [f"نص {i}" for i in range(100)],
                       "label": ["neutral"] * 70 + ["negative"] * 20 + ["positive"] * 10})
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def test_load_split_maps_labels_to_ids_and_preprocesses(split_csv):
    texts, labels = load_split(split_csv, str.upper, None, seed=0)
    assert len(texts) == len(labels) == 100
    assert set(labels) == set(LABEL2ID.values())


def test_subsample_is_stratified(split_csv):
    texts, labels = load_split(split_csv, lambda t: t, 50, seed=0)
    assert len(labels) == 50
    counts = pd.Series(labels).value_counts()
    assert counts[LABEL2ID["neutral"]] == 35
    assert counts[LABEL2ID["negative"]] == 10
    assert counts[LABEL2ID["positive"]] == 5
