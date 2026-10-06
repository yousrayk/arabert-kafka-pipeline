import json

import pandas as pd
import pytest

from app.data import AlertFeed, alert_intervals, bucket_counts, kpis, load_joined


def row(id_, label, t, latency=100.0):
    return {"id": id_, "text": "نص", "label": label, "confidence": 0.9, "model_version": "v1",
            "produced_at": f"2026-01-01T00:00:{t:02d}+00:00", "predicted_at": f"2026-01-01T00:00:{t:02d}+00:00",
            "joined_at": f"2026-01-01T00:00:{t:02d}+00:00", "end_to_end_ms": latency}


def write_jsonl(path, rows, trailing=""):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(trailing)


def test_load_joined_missing_file_is_empty(tmp_path):
    assert load_joined(str(tmp_path / "nope.jsonl")).empty


def test_load_joined_dedupes_by_id_and_skips_torn_line(tmp_path):
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, [row("a", "negative", 1), row("a", "negative", 1), row("b", "positive", 2)],
                trailing='{"id": "c", "lab')
    df = load_joined(str(path))
    assert list(df["id"]) == ["a", "b"]
    assert str(df["predicted_at"].dt.tz) == "UTC"


def test_bucket_counts_long_form_with_share_and_zero_filled_gaps(tmp_path):
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, [row("a", "negative", 1), row("b", "positive", 2), row("c", "negative", 3),
                       row("d", "neutral", 35)])
    counts = bucket_counts(load_joined(str(path)), "10s")
    buckets = counts["bucket"].drop_duplicates().dt.second.tolist()
    assert buckets == [0, 10, 20, 30]  # 10s and 20s are empty but present
    first = counts[counts["bucket"].dt.second == 0].set_index("label")
    assert first.loc["negative", "count"] == 2 and first.loc["positive", "count"] == 1
    assert first.loc["negative", "negative_share"] == pytest.approx(2 / 3)
    empty = counts[counts["bucket"].dt.second == 10]
    assert empty["count"].sum() == 0 and (empty["negative_share"] == 0).all()


def test_bucket_counts_empty():
    assert bucket_counts(load_joined("does-not-exist")).empty


def test_kpis(tmp_path):
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, [row(str(i), "negative" if i < 3 else "neutral", i, latency=float(i * 10))
                       for i in range(10)])
    k = kpis(load_joined(str(path)))
    assert k["total"] == 10
    assert k["negative_share"] == pytest.approx(0.3)
    assert k["positive_share"] == 0.0
    assert k["p95_latency_ms"] == pytest.approx(pd.Series([i * 10.0 for i in range(10)]).quantile(0.95))
    assert k["model_versions"] == ["v1"]


def alert(status, t, share=0.5):
    return {"status": status, "window_end": f"2026-01-01T00:{t:02d}:00+00:00", "negative_share": share,
            "threshold": 0.4, "window_seconds": 60}


def test_alert_intervals_pairs_firing_with_resolved():
    df = alert_intervals([alert("firing", 1, 0.45), alert("resolved", 3), alert("firing", 7)])
    assert len(df) == 2
    assert df.loc[0, "start"].minute == 1 and df.loc[0, "end"].minute == 3
    assert df.loc[0, "peak_share"] == 0.45
    assert pd.isna(df.loc[1, "end"])  # still firing


def test_alert_intervals_empty():
    assert alert_intervals([]).empty


class FakeMsg:
    def __init__(self, value=None, error=None):
        self._v, self._e = value, error

    def value(self):
        return self._v

    def error(self):
        return self._e


class FakeConsumer:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    def poll(self, timeout):
        return self.msgs.pop(0) if self.msgs else None


def test_alert_feed_collects_alerts_and_records_errors():
    feed = AlertFeed(FakeConsumer([FakeMsg(json.dumps(alert("firing", 1)).encode()), FakeMsg(b"junk"),
                                   FakeMsg(error="broker down"), None]))
    for _ in range(4):
        feed.poll_once()
    assert [a["status"] for a in feed.snapshot()] == ["firing"]
    assert feed.error == "broker down"
