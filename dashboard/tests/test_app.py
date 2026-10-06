"""Renders the whole Streamlit page headlessly with AppTest, so a script
error (bad column, chart spec, API misuse) fails a test instead of the demo."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


def run_app(monkeypatch, path):
    monkeypatch.setenv("JOINED_RESULTS_PATH", str(path))
    # Nothing listens here: the alert feed must degrade, not crash the page.
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")
    monkeypatch.syspath_prepend(str(Path(APP).parents[1]))
    return AppTest.from_file(APP, default_timeout=30).run()


def test_renders_empty_state(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "missing.jsonl")
    assert not at.exception
    assert any("No joined results yet" in i.value for i in at.info)


def test_renders_charts_and_kpis_with_data(monkeypatch, tmp_path):
    path = tmp_path / "joined.jsonl"
    labels = ["negative", "neutral", "positive"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(60):
            ts = f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00"
            f.write(json.dumps({"id": str(i), "text": "نص", "label": labels[i % 3], "confidence": 0.8,
                                "model_version": "v1", "produced_at": ts, "predicted_at": ts, "joined_at": ts,
                                "end_to_end_ms": 120.0 + i}, ensure_ascii=False) + "\n")
    at = run_app(monkeypatch, path)
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Reviews"] == "60"
    assert metrics["Negative"] == "33%"
    assert len(at.get("vega_lite_chart")) == 2
