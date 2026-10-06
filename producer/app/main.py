import json
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException

from app.models import IncomingReview
from app.publisher import KafkaPublisher, Publisher, PublishError

SCHEMA_VERSION = 1
RAW_TOPIC = "reviews.raw"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("producer")

app = FastAPI(title="reviews producer")
_publisher: Publisher | None = None


def get_publisher() -> Publisher:
    global _publisher
    if _publisher is None:
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        _publisher = KafkaPublisher(bootstrap)
    return _publisher


def build_raw_message(review: IncomingReview) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        "text": review.text,
        "source": review.source,
        "source_id": review.source_id,
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }


# Plain `def` (not async): FastAPI runs it in its threadpool, so the blocking
# wait for the broker ack in publish() doesn't stall the event loop.
@app.post("/send")
def send(review: IncomingReview, publisher: Publisher = Depends(get_publisher)):
    raw = build_raw_message(review)
    try:
        # Keyed by id: every message about one review (raw, prediction) lands
        # on the same partition number — see ARCHITECTURE.md "Co-partitioning".
        publisher.publish(RAW_TOPIC, raw["id"], raw)
    except PublishError as e:
        logger.error(json.dumps({"event": "publish_failed", "id": raw["id"], "error": str(e)}))
        raise HTTPException(status_code=503, detail="could not publish to Kafka") from e
    logger.info(json.dumps({"event": "published", "id": raw["id"], "topic": RAW_TOPIC}))
    return {"id": raw["id"]}


@app.get("/health")
def health():
    return {"status": "ok"}
