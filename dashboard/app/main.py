import os
import uuid
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from app.data import AlertFeed, LABELS, alert_intervals, bucket_counts, kpis, load_joined

JOINED_PATH = os.environ.get("JOINED_RESULTS_PATH", "/data/joined_results.jsonl")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Sentiment is a polarity, so it uses a diverging encoding: two opposed
# hues with a neutral gray midpoint (validated for CVD separation and
# contrast on both light and dark surfaces). Order = stacking order.
COLORS = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
INK_MUTED = "#898781"

st.set_page_config(page_title="Arabic sentiment stream", layout="wide")


@st.cache_resource
def alert_feed() -> AlertFeed | None:
    try:
        from confluent_kafka import Consumer

        consumer = Consumer({
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"dashboard-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })
        return AlertFeed(consumer).start()
    except Exception as e:  # dashboard still useful without the alert feed
        st.warning(f"Alert feed unavailable: {e}")
        return None


@st.cache_data(ttl=2)
def cached_joined(path: str, mtime: float, size: int) -> pd.DataFrame:
    return load_joined(path)


def joined_frame() -> pd.DataFrame:
    try:
        stat = os.stat(JOINED_PATH)
        return cached_joined(JOINED_PATH, stat.st_mtime, stat.st_size)
    except FileNotFoundError:
        return load_joined(JOINED_PATH)


def pct(x: float) -> str:
    return f"{x:.0%}"


def volume_chart(counts: pd.DataFrame) -> alt.Chart:
    order = {label: i for i, label in enumerate(LABELS)}
    data = counts.assign(order=counts["label"].map(order))
    return alt.Chart(data).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("bucket:T", title=None),
        y=alt.Y("count:Q", title="Reviews per bucket", stack="zero"),
        color=alt.Color("label:N", title="Sentiment",
                        scale=alt.Scale(domain=LABELS, range=[COLORS[l] for l in LABELS]),
                        legend=alt.Legend(orient="top", direction="horizontal")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("bucket:T", title="Bucket", format="%H:%M:%S"),
                 alt.Tooltip("label:N", title="Sentiment"),
                 alt.Tooltip("count:Q", title="Reviews"),
                 alt.Tooltip("total:Q", title="Bucket total")],
    ).properties(height=260)


def share_chart(counts: pd.DataFrame, intervals: pd.DataFrame, threshold: float | None) -> alt.Chart:
    share = counts[counts["label"] == "negative"][["bucket", "negative_share", "total"]]
    layers = []
    if not intervals.empty:
        spans = intervals.assign(end=intervals["end"].fillna(share["bucket"].max()))
        layers.append(alt.Chart(spans).mark_rect(color=COLORS["negative"], opacity=0.12).encode(
            x="start:T", x2="end:T",
            tooltip=[alt.Tooltip("start:T", title="Spike from", format="%H:%M:%S"),
                     alt.Tooltip("end:T", title="to", format="%H:%M:%S"),
                     alt.Tooltip("peak_share:Q", title="Share at trigger", format=".0%")]))
    line = alt.Chart(share).mark_line(color=COLORS["negative"], strokeWidth=2).encode(
        x=alt.X("bucket:T", title=None),
        y=alt.Y("negative_share:Q", title="Negative share", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])))
    points = line.mark_circle(size=40, color=COLORS["negative"], opacity=0).encode(
        opacity=alt.value(0),
        tooltip=[alt.Tooltip("bucket:T", title="Bucket", format="%H:%M:%S"),
                 alt.Tooltip("negative_share:Q", title="Negative share", format=".0%"),
                 alt.Tooltip("total:Q", title="Reviews")])
    layers += [line, points]
    if threshold is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
            color=INK_MUTED, strokeDash=[4, 4]).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=200)


st.title("Arabic review sentiment — live")
st.caption(f"Joined results: `{JOINED_PATH}` · alerts: `alerts.negative_spike` on `{BOOTSTRAP}`")

with st.sidebar:
    st.header("View")
    bucket = st.selectbox("Bucket size", ["10s", "30s", "1min", "5min"], index=1)
    span = st.selectbox("Time range", ["Last 5 minutes", "Last 30 minutes", "Last 2 hours", "All"], index=3)
    refresh = st.toggle("Auto-refresh (5s)", value=True)

feed = alert_feed()


@st.fragment(run_every=5 if refresh else None)
def live():
    df = joined_frame()
    alerts = feed.snapshot() if feed else []
    if feed and feed.error:
        st.warning(f"Alert feed error: {feed.error}")
    if df.empty:
        st.info("No joined results yet. Start the stack and run the gateway: "
                "`docker compose --profile replay up gateway`.")
        return

    minutes = {"Last 5 minutes": 5, "Last 30 minutes": 30, "Last 2 hours": 120}.get(span)
    if minutes:
        df = df[df["predicted_at"] >= df["predicted_at"].max() - timedelta(minutes=minutes)]

    k = kpis(df)
    cols = st.columns(5)
    cols[0].metric("Reviews", f"{k['total']:,}")
    cols[1].metric("Negative", pct(k["negative_share"]))
    cols[2].metric("Neutral", pct(k["neutral_share"]))
    cols[3].metric("Positive", pct(k["positive_share"]))
    cols[4].metric("p95 end-to-end", f"{k['p95_latency_ms']:,.0f} ms" if k["p95_latency_ms"] else "—")

    counts = bucket_counts(df, bucket)
    intervals = alert_intervals(alerts)
    threshold = alerts[-1]["threshold"] if alerts else None
    st.subheader("Sentiment over time")
    st.altair_chart(volume_chart(counts), width="stretch")
    st.subheader("Negative share")
    st.caption("Shaded spans: negative-spike alerts (firing → resolved). Dashed line: alert threshold. "
               "Per-bucket share here; the alert itself uses a sliding window.")
    st.altair_chart(share_chart(counts, intervals, threshold), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Alerts ({sum(a['status'] == 'firing' for a in alerts)} fired)")
        if alerts:
            table = pd.DataFrame(alerts)[["status", "window_end", "negative", "total", "negative_share",
                                          "threshold"]].iloc[::-1]
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"negative_share": st.column_config.NumberColumn(format="percent"),
                                        "threshold": st.column_config.NumberColumn(format="percent")})
        else:
            st.write("No alerts.")
    with right:
        st.subheader("Latest reviews")
        latest = df.sort_values("predicted_at", ascending=False).head(15)[["label", "confidence", "text"]]
        st.dataframe(latest, hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                  format="%.2f")})
    st.caption("Model version(s): " + ", ".join(k["model_versions"]))


live()
