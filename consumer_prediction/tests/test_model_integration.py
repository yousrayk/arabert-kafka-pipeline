"""Runs the real fine-tuned checkpoint end to end (preprocess -> AraBERT).
Skipped unless weights exist at MODEL_PATH — see conftest.py and README's
"Model weights" section."""
import os

import pytest

MODEL_PATH = os.environ["MODEL_PATH"]
pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(MODEL_PATH, "config.json")),
                                reason="fine-tuned weights not available at MODEL_PATH")


@pytest.fixture(scope="module")
def classifier():
    from app.classifier import SentimentClassifier

    clf = SentimentClassifier(MODEL_PATH, settle_seconds=0)
    assert clf.reload_if_changed(force=True)
    return clf


@pytest.fixture(scope="module")
def preprocess():
    from app.preprocess import ArabicPreprocessor

    return ArabicPreprocessor()


def test_outputs_one_of_three_labels_with_valid_probabilities(classifier, preprocess):
    [result] = classifier.predict([preprocess("الخدمة كانت ممتازة والموظفين محترمين")])
    assert result["label"] in ("negative", "neutral", "positive")
    assert 0.0 <= result["confidence"] <= 1.0
    assert sum(result["scores"].values()) == pytest.approx(1.0, abs=1e-3)


def test_clear_cases_get_the_obvious_polarity(classifier, preprocess):
    texts = ["أسوأ خدمة في حياتي، حرامية ونصابين", "شكرا جزيلا، تجربة رائعة وأنصح بها الجميع"]
    labels = [r["label"] for r in classifier.predict([preprocess(t) for t in texts])]
    assert labels == ["negative", "positive"]
