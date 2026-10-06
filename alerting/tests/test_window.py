import json

import pytest

from app.main import make_handler
from app.transforms import ALERTS_TOPIC, build_alert_payload, parse_prediction
from app.window import SlidingWindow, SpikeDetector


def test_window_counts_only_events_inside_the_window():
    w = SlidingWindow(10)
    w.add(0, True)
    w.add(5, False)
    state = w.add(10, False)  # t=0 is now exactly 10s old -> out
    assert (state.total, state.negative) == (2, 0)
    assert state.window_start == 5 and state.window_end == 10


def test_window_share():
    w = SlidingWindow(60)
    for t, neg in [(0, True), (1, True), (2, False), (3, False)]:
        state = w.add(t, neg)
    assert state.negative_share == 0.5


def test_late_event_inside_window_is_counted_in_order():
    w = SlidingWindow(10)
    w.add(100, False)
    w.add(105, False)
    state = w.add(102, True)  # arrived late from another partition
    assert (state.total, state.negative) == (3, 1)
    state = w.add(112.5, False)  # evicts 100 and 102, keeps 105
    assert (state.total, state.negative) == (2, 0)


def test_event_older_than_window_is_dropped():
    w = SlidingWindow(10)
    w.add(100, False)
    state = w.add(80, True)
    assert (state.total, state.negative) == (1, 0)


def feed(detector, start, labels, step=1.0):
    results = []
    for i, label in enumerate(labels):
        r = detector.observe(start + i * step, label)
        if r:
            results.append((start + i * step, r[0], round(r[1].negative_share, 3)))
    return results


def test_no_alert_below_min_count_even_at_100_percent_negative():
    d = SpikeDetector(window_seconds=60, threshold=0.4, min_count=20)
    assert feed(d, 0, ["negative"] * 19) == []


def test_fires_once_when_share_crosses_threshold():
    d = SpikeDetector(window_seconds=60, threshold=0.4, min_count=10)
    events = feed(d, 0, ["positive"] * 10 + ["negative"] * 10)
    # 7 negatives out of 17 = 0.41 -> crossing; then silent while it stays high
    assert len(events) == 1
    assert events[0][1] == "firing"
    assert events[0][2] >= 0.4


def test_hysteresis_prevents_flapping_around_threshold():
    d = SpikeDetector(window_seconds=1000, threshold=0.5, rearm_below=0.3, min_count=4)
    # oscillate right around 50%
    labels = ["negative", "positive"] * 20
    events = feed(d, 0, labels)
    assert [e[1] for e in events] == ["firing"]


def test_resolves_and_rearms_when_share_drops():
    d = SpikeDetector(window_seconds=20, threshold=0.5, rearm_below=0.2, min_count=10)
    events = feed(d, 0, ["negative"] * 20 + ["positive"] * 20 + ["negative"] * 20)
    assert [e[1] for e in events] == ["firing", "resolved", "firing"]


def test_window_slides_so_old_negatives_stop_counting():
    d = SpikeDetector(window_seconds=10, threshold=0.5, rearm_below=0.3, min_count=5)
    feed(d, 0, ["negative"] * 10)
    assert d.firing
    events = feed(d, 100, ["positive"] * 5)  # long gap: old negatives slid out
    assert [e[1] for e in events] == ["resolved"]


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        SpikeDetector(threshold=0.3, rearm_below=0.5)


def test_alert_payload_matches_contract():
    d = SpikeDetector(window_seconds=60, threshold=0.4, min_count=2)
    d.observe(1_767_225_600, "negative")
    status, state = d.observe(1_767_225_601, "negative")
    payload = build_alert_payload(status, state, 60, 0.4, 2)
    assert payload["schema_version"] == 1
    assert payload["type"] == "negative_spike"
    assert payload["status"] == "firing"
    assert payload["window_start"] == "2026-01-01T00:00:00+00:00"
    assert payload["window_end"] == "2026-01-01T00:00:01+00:00"
    assert (payload["total"], payload["negative"], payload["negative_share"]) == (2, 2, 1.0)
    assert payload["threshold"] == 0.4 and payload["window_seconds"] == 60
    assert "alert_id" in payload and "emitted_at" in payload


def test_parse_prediction_reads_event_time_and_label():
    value = json.dumps({"schema_version": 1, "id": "a", "label": "negative",
                        "predicted_at": "2026-01-01T00:00:00+00:00"}).encode()
    assert parse_prediction(value) == (1_767_225_600.0, "negative", "a")


class Msg:
    def __init__(self, value):
        self._v = value

    def value(self):
        return self._v


def pred_msg(i, label):
    return Msg(json.dumps({"schema_version": 1, "id": str(i), "label": label,
                           "predicted_at": f"2026-01-01T00:00:{i:02d}+00:00"}).encode())


def test_handler_emits_alert_messages_and_skips_malformed():
    handle = make_handler(SpikeDetector(window_seconds=60, threshold=0.5, min_count=4))
    out = handle([Msg(b"garbage")] + [pred_msg(i, "negative") for i in range(5)])
    assert len(out) == 1
    assert out[0].topic == ALERTS_TOPIC
    assert out[0].payload["status"] == "firing"
