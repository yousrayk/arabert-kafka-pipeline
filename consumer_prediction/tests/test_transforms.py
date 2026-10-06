import json

import pytest

from app.transforms import build_dlq_payload, build_prediction_payload, parse_raw


def raw_bytes(**overrides):
    msg = {"schema_version": 1, "id": "abc-123", "text": "الخدمة ممتازة", "produced_at": "2026-01-01T00:00:00+00:00"}
    msg.update(overrides)
    return json.dumps(msg, ensure_ascii=False).encode("utf-8")


def test_parse_raw_accepts_valid_message():
    assert parse_raw(raw_bytes())["text"] == "الخدمة ممتازة"


@pytest.mark.parametrize("value", [
    b"\xff\xfe not utf8",
    b'{"truncated": ',
    b'["a", "list"]',
    raw_bytes(schema_version=99),
    raw_bytes(id=""),
    raw_bytes(text="   "),
    raw_bytes(text=None),
])
def test_parse_raw_rejects_contract_violations(value):
    with pytest.raises(ValueError):
        parse_raw(value)


def test_build_prediction_payload_matches_contract():
    payload = build_prediction_payload("abc-123", "negative", 0.987654,
                                       {"negative": 0.987654, "neutral": 0.01, "positive": 0.002346},
                                       "arabertv02-astd@1", 12.3456)
    assert payload["schema_version"] == 1
    assert payload["id"] == "abc-123"
    assert payload["label"] == "negative"
    assert payload["confidence"] == 0.9877
    assert payload["scores"] == {"negative": 0.9877, "neutral": 0.01, "positive": 0.0023}
    assert payload["model_version"] == "arabertv02-astd@1"
    assert payload["inference_ms"] == 12.35
    assert "predicted_at" in payload


def test_build_dlq_payload_matches_contract():
    source = {"topic": "reviews.raw", "partition": 2, "offset": 17}
    payload = build_dlq_payload(b'{"bad": "json"', "boom", source=source)
    assert payload["schema_version"] == 1
    assert payload["stage"] == "consumer_prediction"
    assert payload["original_payload"] == '{"bad": "json"'
    assert payload["error"] == "boom"
    assert payload["source"] == source
    assert "failed_at" in payload


def test_dlq_payload_survives_undecodable_bytes():
    payload = build_dlq_payload(b"\xff\xfe", "bad bytes")
    json.dumps(payload)  # must still be serializable
