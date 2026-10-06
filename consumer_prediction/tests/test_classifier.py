import json
import os

import pytest

from app.classifier import SentimentClassifier


class FakeBackend:
    def __init__(self, labels=("negative", "neutral", "positive"), probs=(0.1, 0.2, 0.7), tag=""):
        self.labels = list(labels)
        self.probs = list(probs)
        self.tag = tag

    def predict_proba(self, texts):
        return [self.probs for _ in texts]


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def write_weights(model_dir, mtime, content="v1"):
    model_dir.mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        path = model_dir / name
        path.write_text(content)
        os.utime(path, (mtime, mtime))


@pytest.fixture
def setup(tmp_path):
    model_dir = tmp_path / "arabert_sentiment"
    clock = Clock()
    loads = []

    def load_fn(path):
        backend = load_fn.next_backend or FakeBackend(tag=f"load-{len(loads)}")
        load_fn.next_backend = None
        loads.append(backend)
        return backend

    load_fn.next_backend = None
    clf = SentimentClassifier(str(model_dir), load_fn=load_fn, check_interval=5, settle_seconds=2, clock=clock)
    return model_dir, clock, loads, load_fn, clf


def test_not_ready_until_weights_appear(setup):
    model_dir, clock, loads, _, clf = setup
    assert clf.reload_if_changed(force=True) is False
    assert not clf.ready
    with pytest.raises(RuntimeError):
        clf.predict(["نص"])

    write_weights(model_dir, clock.now - 10)
    assert clf.reload_if_changed(force=True) is True
    assert clf.ready and len(loads) == 1


def test_predict_returns_label_confidence_and_scores(setup):
    model_dir, clock, _, _, clf = setup
    write_weights(model_dir, clock.now - 10)
    clf.reload_if_changed(force=True)
    [result] = clf.predict(["المنتج رائع"])
    assert result["label"] == "positive"
    assert result["confidence"] == pytest.approx(0.7)
    assert result["scores"] == pytest.approx({"negative": 0.1, "neutral": 0.2, "positive": 0.7})


def test_hot_reloads_when_weights_change(setup):
    model_dir, clock, loads, _, clf = setup
    write_weights(model_dir, clock.now - 10)
    clf.reload_if_changed(force=True)
    v1 = clf.model_version

    clock.now += 60
    write_weights(model_dir, clock.now - 10, content="v2-bigger")
    clf.predict(["نص"])  # predict triggers the check
    assert len(loads) == 2
    assert clf.model_version != v1


def test_no_reload_when_nothing_changed(setup):
    model_dir, clock, loads, _, clf = setup
    write_weights(model_dir, clock.now - 10)
    clf.reload_if_changed(force=True)
    clock.now += 60
    clf.predict(["نص"])
    assert len(loads) == 1


def test_checks_are_throttled_to_the_interval(setup):
    model_dir, clock, loads, _, clf = setup
    write_weights(model_dir, clock.now - 10)
    clf.reload_if_changed(force=True)

    write_weights(model_dir, clock.now - 5, content="v2-bigger")
    clock.now += 1  # within the 5s interval: not even looked at
    assert clf.reload_if_changed() is False
    clock.now += 5
    assert clf.reload_if_changed() is True


def test_waits_for_files_to_settle_before_loading(setup):
    model_dir, clock, loads, _, clf = setup
    write_weights(model_dir, clock.now - 0.5)  # still being copied
    assert clf.reload_if_changed(force=True) is False
    clock.now += 3
    assert clf.reload_if_changed(force=True) is True


def test_failed_load_keeps_serving_previous_model(setup):
    model_dir, clock, loads, load_fn, clf = setup
    write_weights(model_dir, clock.now - 10)
    clf.reload_if_changed(force=True)
    good_version = clf.model_version

    def broken(path):
        raise OSError("truncated safetensors")

    clf._load_fn = broken
    clock.now += 60
    write_weights(model_dir, clock.now - 10, content="corrupt!!")
    assert clf.reload_if_changed(force=True) is False
    assert clf.model_version == good_version
    assert clf.predict(["نص"])[0]["label"] == "positive"


def test_failed_fingerprint_is_not_retried_until_files_change(setup):
    model_dir, clock, loads, _, clf = setup
    attempts = []

    def broken(path):
        attempts.append(path)
        raise OSError("bad")

    clf._load_fn = broken
    write_weights(model_dir, clock.now - 10)
    clf.reload_if_changed(force=True)
    clf.reload_if_changed(force=True)
    assert len(attempts) == 1


def test_rejects_model_with_unexpected_labels(setup):
    model_dir, clock, _, load_fn, clf = setup
    load_fn.next_backend = FakeBackend(labels=("benign", "jailbreak"), probs=(0.5, 0.5))
    write_weights(model_dir, clock.now - 10)
    assert clf.reload_if_changed(force=True) is False
    assert not clf.ready


def test_model_version_prefers_training_meta(setup):
    model_dir, clock, _, _, clf = setup
    write_weights(model_dir, clock.now - 10)
    meta = model_dir / "training_meta.json"
    meta.write_text(json.dumps({"model_version": "arabertv02-astd-20261005"}))
    os.utime(meta, (clock.now - 10, clock.now - 10))
    clf.reload_if_changed(force=True)
    assert clf.model_version == "arabertv02-astd-20261005"


def test_model_version_falls_back_to_folder_and_mtime(setup):
    model_dir, clock, _, _, clf = setup
    write_weights(model_dir, 999_000)
    clf.reload_if_changed(force=True)
    assert clf.model_version == "arabert_sentiment@999000"
