import json

from app.pipeline import Incoming, process_batch
from app.transforms import DLQ_TOPIC, PREDICTIONS_TOPIC


def raw(id_, text="نص", **extra):
    return Incoming(json.dumps({"schema_version": 1, "id": id_, "text": text, **extra},
                               ensure_ascii=False).encode("utf-8"),
                    {"topic": "reviews.raw", "partition": 0, "offset": 1})


class FakeClassifier:
    model_version = "fake@1"

    def __init__(self, poison=None):
        self.calls = []
        self.poison = poison

    def predict(self, texts):
        self.calls.append(list(texts))
        if self.poison is not None and self.poison in texts:
            raise RuntimeError("CUDA-style blowup on this input")
        return [{"label": "negative" if "سيء" in t else "positive", "confidence": 0.9,
                 "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9}} for t in texts]


def identity(text):
    return text


def test_valid_batch_produces_one_prediction_per_message_in_order():
    clf = FakeClassifier()
    out = process_batch([raw("a", "رائع"), raw("b", "سيء جدا")], clf, identity)
    assert [o.topic for o in out] == [PREDICTIONS_TOPIC, PREDICTIONS_TOPIC]
    assert [o.key for o in out] == ["a", "b"]
    assert [o.payload["label"] for o in out] == ["positive", "negative"]
    assert all(o.payload["model_version"] == "fake@1" for o in out)


def test_inference_runs_once_per_batch():
    clf = FakeClassifier()
    process_batch([raw(str(i)) for i in range(8)], clf, identity)
    assert len(clf.calls) == 1 and len(clf.calls[0]) == 8


def test_classifier_sees_preprocessed_text():
    clf = FakeClassifier()
    process_batch([raw("a", "  رائع  ")], clf, lambda t: f"<{t.strip()}>")
    assert clf.calls == [["<رائع>"]]


def test_malformed_messages_go_to_dlq_without_blocking_the_rest():
    clf = FakeClassifier()
    bad = Incoming(b"{not json", {"topic": "reviews.raw", "partition": 1, "offset": 9})
    out = process_batch([raw("a"), bad, raw("b")], clf, identity)
    predictions = [o for o in out if o.topic == PREDICTIONS_TOPIC]
    dlq = [o for o in out if o.topic == DLQ_TOPIC]
    assert [p.key for p in predictions] == ["a", "b"]
    assert len(dlq) == 1
    assert dlq[0].payload["original_payload"] == "{not json"
    assert dlq[0].payload["source"] == {"topic": "reviews.raw", "partition": 1, "offset": 9}


def test_preprocessing_failure_goes_to_dlq():
    def flaky(text):
        if text == "boom":
            raise ValueError("preprocess failed")
        return text

    out = process_batch([raw("a", "boom"), raw("b", "ok")], FakeClassifier(), flaky)
    assert {o.topic for o in out if o.key == "b"} == {PREDICTIONS_TOPIC}
    assert [o.topic for o in out if o.key is None] == [DLQ_TOPIC]


def test_batched_inference_failure_isolates_the_poison_message():
    clf = FakeClassifier(poison="poison")
    out = process_batch([raw("a", "ok"), raw("b", "poison"), raw("c", "ok")], clf, identity)
    assert [o.key for o in out if o.topic == PREDICTIONS_TOPIC] == ["a", "c"]
    dlq = [o for o in out if o.topic == DLQ_TOPIC]
    assert len(dlq) == 1 and "poison" in dlq[0].payload["original_payload"]


def test_empty_batch_produces_nothing():
    assert process_batch([], FakeClassifier(), identity) == []
