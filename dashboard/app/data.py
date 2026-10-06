"""Pure data shaping for the dashboard — no Streamlit, no Kafka — so it is
unit-testable on its own."""
import json
import os
import threading

import pandas as pd

LABELS = ["negative", "neutral", "positive"]
ALERTS_TOPIC = "alerts.negative_spike"


def load_joined(path: str) -> pd.DataFrame:
    """Reads the consumer_join JSONL sink. The sink is at-least-once (a
    join replayed after a restart can be written twice), so rows are
    deduplicated by id. A torn last line (mid-write) is skipped."""
    cols = ["id", "text", "label", "confidence", "model_version", "produced_at",
            "predicted_at", "joined_at", "end_to_end_ms"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df = df.drop_duplicates(subset="id", keep="first")
    for col in ("produced_at", "predicted_at", "joined_at"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df.sort_values("predicted_at").reset_index(drop=True)


def bucket_counts(df: pd.DataFrame, freq: str = "30s") -> pd.DataFrame:
    """Long-form counts per (time bucket, label), plus the negative share per
    bucket — what the stacked bars and the share line are drawn from.
    Empty buckets inside the range are kept as zeros so gaps read as gaps."""
    if df.empty:
        return pd.DataFrame(columns=["bucket", "label", "count", "total", "negative_share"])
    wide = (df.assign(bucket=df["predicted_at"].dt.floor(freq))
              .pivot_table(index="bucket", columns="label", values="id", aggfunc="count", fill_value=0)
              .reindex(columns=LABELS, fill_value=0))
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=freq, tz="UTC")
    wide = wide.reindex(full, fill_value=0)
    wide.index.name = "bucket"
    total = wide.sum(axis=1)
    share = (wide["negative"] / total.where(total > 0)).fillna(0.0)
    long = wide.reset_index().melt(id_vars="bucket", var_name="label", value_name="count")
    long = long.merge(pd.DataFrame({"bucket": wide.index, "total": total.values,
                                    "negative_share": share.values}), on="bucket")
    return long.sort_values(["bucket", "label"]).reset_index(drop=True)


def kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    shares = df["label"].value_counts(normalize=True) if total else pd.Series(dtype=float)
    latency = pd.to_numeric(df["end_to_end_ms"], errors="coerce").dropna()
    return {
        "total": total,
        **{f"{label}_share": float(shares.get(label, 0.0)) for label in LABELS},
        "p50_latency_ms": float(latency.quantile(0.5)) if len(latency) else None,
        "p95_latency_ms": float(latency.quantile(0.95)) if len(latency) else None,
        "model_versions": sorted(df["model_version"].dropna().unique().tolist()) if total else [],
    }


def alert_intervals(alerts: list[dict]) -> pd.DataFrame:
    """Pairs each "firing" alert with the next "resolved" one into an
    interval to shade on the timeline; a spike still in progress has no
    end yet (end = NaT, drawn up to "now")."""
    rows, open_row = [], None
    for a in sorted(alerts, key=lambda a: a["window_end"]):
        if a["status"] == "firing" and open_row is None:
            open_row = {"start": a["window_end"], "end": None, "peak_share": a["negative_share"],
                        "threshold": a["threshold"], "window_seconds": a["window_seconds"]}
        elif a["status"] == "resolved" and open_row is not None:
            open_row["end"] = a["window_end"]
            rows.append(open_row)
            open_row = None
    if open_row is not None:
        rows.append(open_row)
    df = pd.DataFrame(rows, columns=["start", "end", "peak_share", "threshold", "window_seconds"])
    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df


class AlertFeed:
    """Tails alerts.negative_spike on a background thread for the lifetime
    of the Streamlit server (created once via st.cache_resource). It reads
    the topic from the beginning under a throwaway group id and never
    commits, so every dashboard process sees the full alert history."""

    def __init__(self, consumer):
        self._consumer = consumer
        self._alerts: list[dict] = []
        self._lock = threading.Lock()
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "AlertFeed":
        self._consumer.subscribe([ALERTS_TOPIC])
        self._thread.start()
        return self

    def poll_once(self, timeout: float = 1.0) -> None:
        msg = self._consumer.poll(timeout)
        if msg is None:
            return
        if msg.error() is not None:
            self.error = str(msg.error())
            return
        try:
            alert = json.loads(msg.value().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        with self._lock:
            self._alerts.append(alert)

    def _run(self) -> None:
        while True:
            self.poll_once()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._alerts)
