import json
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = 1
PREDICTIONS_TOPIC = "reviews.predictions"
ALERTS_TOPIC = "alerts.negative_spike"


def parse_prediction(value: bytes) -> tuple[float, str, str]:
    """Returns (event time as epoch seconds, label, id)."""
    p = json.loads(value.decode("utf-8"))
    if p.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version {p.get('schema_version')!r}")
    ts = datetime.fromisoformat(p["predicted_at"]).timestamp()
    return ts, p["label"], p["id"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def build_alert_payload(status: str, state, window_seconds: float, threshold: float, min_count: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "alert_id": str(uuid.uuid4()),
        "type": "negative_spike",
        "status": status,  # "firing" | "resolved"
        "window_start": _iso(state.window_start),
        "window_end": _iso(state.window_end),
        "window_seconds": window_seconds,
        "total": state.total,
        "negative": state.negative,
        "negative_share": round(state.negative_share, 4),
        "threshold": threshold,
        "min_count": min_count,
        "emitted_at": datetime.now(timezone.utc).isoformat(),
    }
