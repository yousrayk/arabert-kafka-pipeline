import json
import threading
from typing import Protocol


class PublishError(Exception):
    pass


class Publisher(Protocol):
    def publish(self, topic: str, key: str, payload: dict) -> None: ...


class KafkaPublisher:
    """Real publisher, backed by one confluent-kafka Producer shared across
    requests. publish() blocks until the broker acknowledges the write
    (acks=all), so a 200 from /send means the review is durably in Kafka.

    Requests are served from FastAPI's threadpool, so many publish() calls
    are in flight at once and linger.ms batches them together — blocking per
    request doesn't serialize throughput. A background thread drives
    poll(), which is what fires the delivery callbacks.
    """

    def __init__(self, bootstrap_servers: str, timeout_seconds: float = 10.0):
        from confluent_kafka import Producer

        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "acks": "all",
                "enable.idempotence": True,
                "linger.ms": 5,
                "compression.type": "lz4",
            }
        )
        self._timeout = timeout_seconds
        self._running = True
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    def _poll_loop(self) -> None:
        while self._running:
            self._producer.poll(0.1)

    def publish(self, topic: str, key: str, payload: dict) -> None:
        done = threading.Event()
        result: dict = {}

        def on_delivery(err, _msg):
            result["err"] = err
            done.set()

        try:
            self._producer.produce(
                topic,
                key=key.encode("utf-8"),
                value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                on_delivery=on_delivery,
            )
        except BufferError as e:
            raise PublishError(f"local producer queue full: {e}") from e

        if not done.wait(self._timeout):
            raise PublishError(f"delivery to {topic} not confirmed within {self._timeout}s")
        if result["err"] is not None:
            raise PublishError(str(result["err"]))

    def close(self) -> None:
        self._running = False
        self._producer.flush(self._timeout)


class FakePublisher:
    """In-memory publisher for unit tests — no Kafka broker needed."""

    def __init__(self, fail_with: Exception | None = None):
        self.published: list[tuple[str, str, dict]] = []
        self.fail_with = fail_with

    def publish(self, topic: str, key: str, payload: dict) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.published.append((topic, key, payload))
