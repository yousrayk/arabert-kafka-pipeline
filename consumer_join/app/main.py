import json
import logging
import os
import signal

from app.join_buffer import JoinBuffer
from app.kafka_io import consumer_config, producer_config, publish_all
from app.processor import JoinProcessor
from app.transforms import PREDICTIONS_TOPIC, RAW_TOPIC

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("consumer_join")


class JsonlSink:
    """Appends joined results as JSON lines — a real destination instead of
    a dead-end print. flush() per batch, so the commit that follows never
    acknowledges input whose output isn't on disk yet."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, rows: list[dict]) -> None:
        for row in rows:
            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def join_loop(consumer, producer, processor: JoinProcessor, sink, should_stop,
              batch_size: int = 200, poll_timeout: float = 1.0, commit_fn=None) -> None:
    last_committed: dict = {}
    while not should_stop():
        messages = [m for m in consumer.consume(num_messages=batch_size, timeout=poll_timeout)
                    if m.error() is None]
        joined, dlq = processor.handle(messages)
        dlq += processor.evict()
        if joined:
            sink.write(joined)
        if dlq:
            publish_all(producer, dlq)
        # Only commit once outputs are durable, and only up to what the
        # OffsetTracker says is no longer needed by the buffer.
        offsets = processor.offsets.committable()
        changed = {tp: o for tp, o in offsets.items() if last_committed.get(tp) != o}
        if changed:
            commit_fn(changed)
            last_committed.update(changed)


def run():
    from confluent_kafka import Consumer, Producer, TopicPartition

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    output_path = os.environ.get("JOINED_RESULTS_PATH", "/data/joined_results.jsonl")
    ttl = float(os.environ.get("JOIN_TTL_SECONDS", "120"))

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    config = consumer_config(bootstrap, "consumer-join")
    # Range assignment gives one instance the *same* partition numbers of
    # both topics — with co-partitioned topics (same key, same partition
    # count), that's every message for a given id. See ARCHITECTURE.md.
    config["partition.assignment.strategy"] = "range"
    consumer = Consumer(config)
    producer = Producer(producer_config(bootstrap))
    processor = JoinProcessor(JoinBuffer(ttl_seconds=ttl, dedupe_seconds=max(600.0, 5 * ttl)))
    sink = JsonlSink(output_path)

    def commit(offsets):
        consumer.commit(offsets=[TopicPartition(t, p, o) for (t, p), o in offsets.items()], asynchronous=False)

    def on_revoke(_consumer, _partitions):
        # The in-memory buffer can't follow partitions to another instance.
        processor.reset()

    consumer.subscribe([RAW_TOPIC, PREDICTIONS_TOPIC], on_revoke=on_revoke)
    logger.info(json.dumps({"event": "started", "ttl_seconds": ttl, "output": output_path}))
    try:
        join_loop(consumer, producer, processor, sink, lambda: stop["flag"], commit_fn=commit)
    finally:
        logger.info(json.dumps({"event": "stopped", **processor.stats, "buffered": len(processor.buffer)}))
        consumer.close()
        sink.close()


if __name__ == "__main__":
    run()
