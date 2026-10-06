import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gateway")


@dataclass(frozen=True)
class Review:
    source_id: str
    text: str


def load_reviews(csv_path: str) -> list[Review]:
    """Reads the replay file written by scripts/prepare_dataset.py. Only id
    and text are read — the gold label stays in the file for scripts/eval.py
    and is never sent into the pipeline."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [Review(row["id"], row["text"]) for row in csv.DictReader(f) if row["text"].strip()]


def paced(items: Iterable, rate_per_sec: float, clock: Callable[[], float] = time.monotonic,
          sleep_fn: Callable[[float], None] = time.sleep) -> Iterator:
    """Yields items at rate_per_sec against an absolute schedule (item i is
    due at start + i/rate), so a slow send is caught up on instead of drifting
    the overall rate down. rate_per_sec <= 0 means as fast as possible."""
    start = clock()
    for i, item in enumerate(items):
        if rate_per_sec > 0:
            delay = start + i / rate_per_sec - clock()
            if delay > 0:
                sleep_fn(delay)
        yield item


def wait_for_health(session, health_url: str, attempts: int = 60, sleep_fn=time.sleep) -> None:
    for _ in range(attempts):
        try:
            if session.get(health_url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        sleep_fn(1)
    raise RuntimeError(f"producer not healthy at {health_url}")


def replay(reviews: list[Review], url: str, rate_per_sec: float, source: str = "dataset",
           session=None, clock=time.monotonic, sleep_fn=time.sleep) -> dict:
    """POSTs every review to the producer at the configured rate — this is
    what drives the whole pipeline end to end (see ARCHITECTURE.md)."""
    session = session or requests.Session()
    stats = {"sent": 0, "failed": 0}
    started = clock()
    for review in paced(reviews, rate_per_sec, clock=clock, sleep_fn=sleep_fn):
        body = {"text": review.text, "source": source, "source_id": review.source_id}
        try:
            resp = session.post(url, json=body, timeout=10)
        except requests.RequestException as e:
            stats["failed"] += 1
            logger.error(json.dumps({"event": "send_failed", "source_id": review.source_id, "error": str(e)}))
            continue
        if resp.status_code == 200:
            stats["sent"] += 1
        else:
            stats["failed"] += 1
            logger.error(json.dumps({"event": "send_failed", "source_id": review.source_id,
                                     "status": resp.status_code, "body": resp.text[:200]}))
        if stats["sent"] % 100 == 0 and stats["sent"]:
            elapsed = clock() - started
            logger.info(json.dumps({"event": "progress", **stats,
                                    "achieved_rate": round(stats["sent"] / elapsed, 1) if elapsed else None}))
    return stats


def run():
    csv_path = os.environ.get("REPLAY_CSV", "/data/replay.csv")
    url = os.environ.get("PRODUCER_URL", "http://producer:5000/send")
    rate = float(os.environ.get("RATE_PER_SEC", "20"))
    limit = int(os.environ.get("MAX_MESSAGES", "0"))
    source = os.environ.get("DATASET_SOURCE", "astd")

    reviews = load_reviews(csv_path)
    if limit > 0:
        reviews = reviews[:limit]
    session = requests.Session()
    wait_for_health(session, url.rsplit("/", 1)[0] + "/health")
    logger.info(json.dumps({"event": "replay_start", "count": len(reviews), "rate_per_sec": rate}))
    stats = replay(reviews, url, rate, source=source, session=session)
    logger.info(json.dumps({"event": "replay_complete", **stats}))


if __name__ == "__main__":
    run()
