"""Scores a pipeline run against the dataset's gold labels.

    python scripts/eval.py                                   # defaults below
    python scripts/eval.py --joined data/joined_results.jsonl --labels data/replay.csv --out data/eval_report.json

Quality: accuracy, macro-F1, per-class precision/recall/F1 and the
confusion matrix, matching joined results to gold labels by source_id (the
label never travels through the pipeline).

Performance, from the timestamps each stage stamps on the message:
- offered rate   = messages / (last produced_at - first produced_at)
- throughput     = messages / (last joined_at - first produced_at)
- latency        = produce -> predict and produce -> join, p50/p95/p99.
To measure capacity rather than the gateway's pacing, replay with
RATE_PER_SEC=0 (as fast as possible) — see README "Results".
"""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]


def load_joined(path: str) -> list[dict]:
    """Deduplicated by id (the sink is at-least-once); torn lines skipped."""
    seen, rows = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") not in seen:
                seen.add(r.get("id"))
                rows.append(r)
    return rows


def load_gold(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["label"] for row in csv.DictReader(f)}


def classification_metrics(gold: list[str], pred: list[str], labels=LABELS) -> dict:
    n = len(gold)
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in labels if g != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": tp + fn}
    return {
        "n": n,
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / n, 4) if n else 0.0,
        "macro_f1": round(sum(c["f1"] for c in per_class.values()) / len(labels), 4),
        "per_class": per_class,
        "confusion_matrix": {"rows=gold, cols=pred": labels, "matrix": [[matrix[g][p] for p in labels] for g in labels]},
    }


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (same as numpy's default)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def _ts(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def performance(rows: list[dict]) -> dict:
    produced = [t for t in (_ts(r.get("produced_at")) for r in rows) if t]
    joined = [t for t in (_ts(r.get("joined_at")) for r in rows) if t]
    to_predict = [(_ts(r["predicted_at"]) - _ts(r["produced_at"])).total_seconds() * 1000
                  for r in rows if r.get("predicted_at") and r.get("produced_at")]
    to_join = [r["end_to_end_ms"] for r in rows if r.get("end_to_end_ms") is not None]
    out: dict = {"messages": len(rows)}
    if len(produced) > 1:
        span = (max(produced) - min(produced)).total_seconds()
        out["offered_rate_per_sec"] = round(len(rows) / span, 2) if span > 0 else None
    if produced and joined:
        span = (max(joined) - min(produced)).total_seconds()
        out["throughput_per_sec"] = round(len(rows) / span, 2) if span > 0 else None
        out["wall_seconds"] = round(span, 1)
    for name, values in (("produce_to_predict_ms", to_predict), ("produce_to_join_ms", to_join)):
        out[name] = {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
                     "p99": percentile(values, 0.99), "max": round(max(values), 1) if values else None}
    return out


def evaluate(joined_path: str, labels_path: str, source: str | None = None) -> dict:
    rows = load_joined(joined_path)
    if source:
        rows = [r for r in rows if r.get("source") == source]
    gold = load_gold(labels_path)
    matched = [r for r in rows if r.get("source_id") in gold]
    report = {
        "coverage": {"expected": len(gold), "joined": len(rows), "matched": len(matched),
                     "missing": len(set(gold) - {r.get("source_id") for r in matched})},
        "model_versions": sorted({r.get("model_version") for r in matched if r.get("model_version")}),
        "quality": classification_metrics([gold[r["source_id"]] for r in matched], [r["label"] for r in matched]),
        "performance": performance(matched),
    }
    return report


def to_markdown(report: dict) -> str:
    q, p, c = report["quality"], report["performance"], report["coverage"]
    lines = [
        f"Matched {c['matched']} / {c['expected']} dataset rows ({c['missing']} missing) · "
        f"model: {', '.join(report['model_versions']) or 'n/a'}",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {q['accuracy']:.4f} |",
        f"| Macro-F1 | {q['macro_f1']:.4f} |",
    ]
    for label, m in q["per_class"].items():
        lines.append(f"| F1 {label} (n={m['support']}) | {m['f1']:.4f} |")
    lines += [
        f"| Offered rate | {p.get('offered_rate_per_sec')} msg/s |",
        f"| Throughput (end to end) | {p.get('throughput_per_sec')} msg/s |",
        f"| Produce→predict p50 / p95 | {p['produce_to_predict_ms']['p50']} / {p['produce_to_predict_ms']['p95']} ms |",
        f"| Produce→join p50 / p95 / p99 | {p['produce_to_join_ms']['p50']} / {p['produce_to_join_ms']['p95']} / "
        f"{p['produce_to_join_ms']['p99']} ms |",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined_results.jsonl")
    parser.add_argument("--labels", default="data/replay.csv")
    parser.add_argument("--source", default=None,
                        help="only score rows with this `source` tag (gateway RUN_TAG), e.g. one run of several")
    parser.add_argument("--out", default="data/eval_report.json")
    args = parser.parse_args()
    report = evaluate(args.joined, args.labels, args.source)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(to_markdown(report))
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
