import json
import logging
import os
import signal
from dataclasses import dataclass

from app.kafka_io import consumer_config, producer_config, run_loop
from app.transforms import ALERTS_TOPIC, PREDICTIONS_TOPIC, build_alert_payload, parse_prediction
from app.window import SpikeDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("alerting")


@dataclass
class Outgoing:
    topic: str
    key: str | None
    payload: dict


def make_handler(detector: SpikeDetector):
    """Malformed predictions are logged and skipped rather than sent to a
    DLQ: consumer_prediction produced them, so they're already valid by
    contract, and an alerting hiccup shouldn't pollute the review DLQ."""

    def handle(messages) -> list[Outgoing]:
        out = []
        for m in messages:
            try:
                ts, label, _ = parse_prediction(m.value())
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(json.dumps({"event": "skip_malformed", "error": str(e)}))
                continue
            result = detector.observe(ts, label)
            if result:
                status, state = result
                alert = build_alert_payload(status, state, detector.window.window_seconds,
                                            detector.threshold, detector.min_count)
                logger.info(json.dumps({"event": "alert", **alert}))
                out.append(Outgoing(ALERTS_TOPIC, "negative_spike", alert))
        return out

    return handle


def run():
    from confluent_kafka import Consumer, Producer

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    detector = SpikeDetector(
        window_seconds=float(os.environ.get("WINDOW_SECONDS", "60")),
        threshold=float(os.environ.get("NEGATIVE_SHARE_THRESHOLD", "0.4")),
        rearm_below=float(os.environ["REARM_BELOW"]) if os.environ.get("REARM_BELOW") else None,
        min_count=int(os.environ.get("MIN_COUNT", "20")),
    )
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    consumer = Consumer(consumer_config(bootstrap, "alerting"))
    producer = Producer(producer_config(bootstrap))
    consumer.subscribe([PREDICTIONS_TOPIC])
    logger.info(json.dumps({"event": "started", "window_seconds": detector.window.window_seconds,
                            "threshold": detector.threshold, "rearm_below": detector.rearm_below,
                            "min_count": detector.min_count}))
    try:
        run_loop(consumer, producer, make_handler(detector), batch_size=200, should_stop=lambda: stop["flag"])
    finally:
        consumer.close()


if __name__ == "__main__":
    run()
