import csv
import json

import pytest

from eval import classification_metrics, evaluate, percentile, performance


def test_perfect_predictions():
    m = classification_metrics(["negative", "neutral", "positive"], ["negative", "neutral", "positive"])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_macro_f1_matches_hand_computation():
    gold = ["negative", "negative", "neutral", "neutral", "positive", "positive"]
    pred = ["negative", "neutral", "neutral", "neutral", "positive", "negative"]
    m = classification_metrics(gold, pred)
    # negative: P=1/2 R=1/2 F1=.5 ; neutral: P=2/3 R=1 F1=.8 ; positive: P=1 R=1/2 F1=2/3
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["per_class"]["neutral"]["f1"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3, abs=1e-4)
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_class_never_predicted_scores_zero_not_crash():
    m = classification_metrics(["positive", "negative"], ["negative", "negative"])
    assert m["per_class"]["positive"]["f1"] == 0.0


def test_percentile_interpolates_like_numpy():
    assert percentile(list(range(101)), 0.95) == 95.0          # exact rank
    assert percentile([0.0, 10.0], 0.95) == 9.5                 # interpolated between ranks
    assert percentile([], 0.95) is None
    assert percentile([7.0], 0.95) == 7.0


def joined_row(i, label, source_id, produced_s, predicted_s, joined_s):
    t = lambda s: f"2026-01-01T00:00:{s:06.3f}+00:00"
    return {"id": f"id-{i}", "source_id": source_id, "label": label, "model_version": "v1",
            "produced_at": t(produced_s), "predicted_at": t(predicted_s), "joined_at": t(joined_s),
            "end_to_end_ms": round((joined_s - produced_s) * 1000, 1)}


def test_performance_rates_and_latency():
    rows = [joined_row(i, "neutral", str(i), i * 1.0, i * 1.0 + 0.1, i * 1.0 + 0.2) for i in range(11)]
    p = performance(rows)
    assert p["offered_rate_per_sec"] == pytest.approx(11 / 10)
    assert p["throughput_per_sec"] == pytest.approx(11 / 10.2, abs=0.01)
    assert p["produce_to_predict_ms"]["p95"] == pytest.approx(100, abs=1)
    assert p["produce_to_join_ms"]["p50"] == pytest.approx(200, abs=1)


def test_evaluate_matches_by_source_id_and_dedupes(tmp_path):
    labels = tmp_path / "replay.csv"
    with open(labels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label"])
        w.writerows([["astd-1", "x", "negative"], ["astd-2", "y", "positive"], ["astd-3", "z", "neutral"]])
    joined = tmp_path / "joined.jsonl"
    rows = [joined_row(1, "negative", "astd-1", 0, 0.1, 0.2), joined_row(1, "negative", "astd-1", 0, 0.1, 0.2),
            joined_row(2, "neutral", "astd-2", 1, 1.1, 1.2), joined_row(9, "neutral", "smoke-test", 2, 2.1, 2.2)]
    joined.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    report = evaluate(str(joined), str(labels))
    assert report["coverage"] == {"expected": 3, "joined": 3, "matched": 2, "missing": 1}
    assert report["quality"]["accuracy"] == 0.5
    assert report["model_versions"] == ["v1"]


def test_evaluate_can_score_a_single_tagged_run(tmp_path):
    labels = tmp_path / "replay.csv"
    labels.write_text("id,text,label\nastd-1,x,negative\n", encoding="utf-8")
    run_a = {**joined_row(1, "negative", "astd-1", 0, 0.1, 0.2), "source": "capacity"}
    run_b = {**joined_row(2, "positive", "astd-1", 5, 5.1, 5.2), "source": "latency"}
    joined = tmp_path / "joined.jsonl"
    joined.write_text(json.dumps(run_a) + "\n" + json.dumps(run_b) + "\n", encoding="utf-8")
    assert evaluate(str(joined), str(labels), source="capacity")["quality"]["accuracy"] == 1.0
    assert evaluate(str(joined), str(labels), source="latency")["quality"]["accuracy"] == 0.0
