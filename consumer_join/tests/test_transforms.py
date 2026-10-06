from datetime import datetime, timezone

import pytest

from app.transforms import build_joined_payload, parse


def test_build_joined_payload_matches_contract():
    raw = {"schema_version": 1, "id": "abc-123", "text": "الخدمة سيئة", "source": "astd",
           "source_id": "astd-7", "produced_at": "2026-01-01T00:00:00+00:00"}
    prediction = {"schema_version": 1, "id": "abc-123", "label": "negative", "confidence": 0.91,
                  "scores": {"negative": 0.91, "neutral": 0.06, "positive": 0.03},
                  "model_version": "arabertv02-astd-1", "predicted_at": "2026-01-01T00:00:00.200000+00:00"}
    joined_at = datetime(2026, 1, 1, 0, 0, 0, 350000, tzinfo=timezone.utc)

    payload = build_joined_payload(raw, prediction, joined_at)

    assert payload == {
        "schema_version": 1,
        "id": "abc-123",
        "text": "الخدمة سيئة",
        "source": "astd",
        "source_id": "astd-7",
        "label": "negative",
        "confidence": 0.91,
        "scores": {"negative": 0.91, "neutral": 0.06, "positive": 0.03},
        "model_version": "arabertv02-astd-1",
        "produced_at": "2026-01-01T00:00:00+00:00",
        "predicted_at": "2026-01-01T00:00:00.200000+00:00",
        "joined_at": "2026-01-01T00:00:00.350000+00:00",
        "end_to_end_ms": 350.0,
    }


def test_missing_timestamps_give_null_latency_not_a_crash():
    payload = build_joined_payload({"id": "a", "text": "x"},
                                   {"id": "a", "label": "neutral", "confidence": 0.5, "model_version": "v"})
    assert payload["end_to_end_ms"] is None


@pytest.mark.parametrize("value", [b"{bad", b'"str"', b'{"schema_version": 1}', b'{"schema_version": 7, "id": "a"}'])
def test_parse_rejects_contract_violations(value):
    with pytest.raises(ValueError):
        parse(value)
