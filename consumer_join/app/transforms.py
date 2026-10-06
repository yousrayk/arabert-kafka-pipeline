import json
from datetime import datetime, timezone

SCHEMA_VERSION = 1
RAW_TOPIC = "reviews.raw"
PREDICTIONS_TOPIC = "reviews.predictions"
DLQ_TOPIC = "reviews.dlq"
SUPPORTED_VERSIONS = {1}


def parse(value: bytes) -> dict:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"not valid UTF-8 JSON: {e}") from e
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str) or not payload["id"]:
        raise ValueError("payload is not an object with an id")
    if payload.get("schema_version") not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported schema_version {payload.get('schema_version')!r}")
    return payload


def _ms_between(start_iso: str | None, end: datetime) -> float | None:
    if not start_iso:
        return None
    try:
        return round((end - datetime.fromisoformat(start_iso)).total_seconds() * 1000, 1)
    except ValueError:
        return None


def build_joined_payload(raw: dict, prediction: dict, joined_at: datetime | None = None) -> dict:
    joined_at = joined_at or datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": raw["id"],
        "text": raw["text"],
        "source": raw.get("source"),
        "source_id": raw.get("source_id"),
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "scores": prediction.get("scores"),
        "model_version": prediction["model_version"],
        "produced_at": raw.get("produced_at"),
        "predicted_at": prediction.get("predicted_at"),
        "joined_at": joined_at.isoformat(),
        # produce -> join wall time: the end-to-end latency the eval reports.
        "end_to_end_ms": _ms_between(raw.get("produced_at"), joined_at),
    }


def build_dlq_payload(original_payload, error: str, source: dict | None = None) -> dict:
    if isinstance(original_payload, bytes):
        original_payload = original_payload.decode("utf-8", errors="replace")
    elif not isinstance(original_payload, str):
        original_payload = json.dumps(original_payload, ensure_ascii=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "consumer_join",
        "error": error,
        "original_payload": original_payload,
        "source": source,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
