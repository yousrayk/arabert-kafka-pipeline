import json
from datetime import datetime, timezone

SCHEMA_VERSION = 1
SUPPORTED_RAW_VERSIONS = {1}
RAW_TOPIC = "reviews.raw"
PREDICTIONS_TOPIC = "reviews.predictions"
DLQ_TOPIC = "reviews.dlq"


def parse_raw(value: bytes) -> dict:
    """Validates a reviews.raw message against its contract; anything that
    doesn't conform raises ValueError and is routed to the DLQ."""
    try:
        raw = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"not valid UTF-8 JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError("payload is not a JSON object")
    if raw.get("schema_version") not in SUPPORTED_RAW_VERSIONS:
        raise ValueError(f"unsupported schema_version {raw.get('schema_version')!r}")
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("missing id")
    if not isinstance(raw.get("text"), str) or not raw["text"].strip():
        raise ValueError("missing text")
    return raw


def build_prediction_payload(raw_id: str, label: str, confidence: float, scores: dict,
                             model_version: str, inference_ms: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": raw_id,
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "model_version": model_version,
        "inference_ms": round(inference_ms, 2),
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }


def build_dlq_payload(original_payload: bytes | str, error: str, stage: str = "consumer_prediction",
                      source: dict | None = None) -> dict:
    if isinstance(original_payload, bytes):
        original_payload = original_payload.decode("utf-8", errors="replace")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "error": error,
        "original_payload": original_payload,
        # topic/partition/offset of the failed message, so it can be found
        # and replayed once the cause is fixed.
        "source": source,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
