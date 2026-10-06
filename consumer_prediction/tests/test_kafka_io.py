import json

import pytest

from app.kafka_io import DeliveryError, publish_all, run_loop
from app.pipeline import Outgoing


class FakeMessage:
    def __init__(self, value, error=None):
        self._value, self._error = value, error

    def value(self):
        return self._value

    def error(self):
        return self._error


class FakeConsumer:
    def __init__(self, batches, log):
        self.batches = list(batches)
        self.log = log

    def consume(self, num_messages, timeout):
        return self.batches.pop(0) if self.batches else []

    def commit(self, asynchronous):
        self.log.append("commit")


class FakeProducer:
    def __init__(self, log, fail=False):
        self.log = log
        self.fail = fail
        self.pending = []
        self.sent = []

    def produce(self, topic, key, value, on_delivery):
        self.pending.append((topic, key, value, on_delivery))

    def poll(self, timeout):
        return 0

    def flush(self, timeout):
        for topic, key, value, cb in self.pending:
            cb("broker said no" if self.fail else None, None)
            self.sent.append((topic, key, json.loads(value)))
        self.pending = []
        self.log.append("flush")
        return 0


def stop_after(n):
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > n
    return should_stop


def test_commits_only_after_outputs_are_flushed():
    log = []
    consumer = FakeConsumer([[FakeMessage(b"1"), FakeMessage(b"2")]], log)
    producer = FakeProducer(log)
    handle = lambda msgs: [Outgoing("out", m.value().decode(), {"v": m.value().decode()}) for m in msgs]

    run_loop(consumer, producer, handle, batch_size=10, should_stop=stop_after(1))

    assert log == ["flush", "commit"]
    assert [(t, k) for t, k, _ in producer.sent] == [("out", b"1"), ("out", b"2")]


def test_delivery_failure_raises_and_does_not_commit():
    log = []
    consumer = FakeConsumer([[FakeMessage(b"1")]], log)
    producer = FakeProducer(log, fail=True)
    handle = lambda msgs: [Outgoing("out", "k", {})]

    with pytest.raises(DeliveryError):
        run_loop(consumer, producer, handle, batch_size=10, should_stop=stop_after(1))
    assert "commit" not in log


def test_consume_errors_are_skipped_not_handled():
    log, seen = [], []
    consumer = FakeConsumer([[FakeMessage(None, error="partition EOF"), FakeMessage(b"ok")]], log)

    def handle(msgs):
        seen.extend(m.value() for m in msgs)
        return []

    run_loop(consumer, FakeProducer(log), handle, batch_size=10, should_stop=stop_after(1))
    assert seen == [b"ok"]


def test_empty_poll_does_not_commit():
    log = []
    run_loop(FakeConsumer([], log), FakeProducer(log), lambda m: [], batch_size=10, should_stop=stop_after(3))
    assert log == []


def test_publish_all_serializes_arabic_as_utf8_and_keyless_dlq():
    log = []
    producer = FakeProducer(log)
    publish_all(producer, [Outgoing("reviews.dlq", None, {"text": "سيء"})])
    topic, key, payload = producer.sent[0]
    assert key is None and payload == {"text": "سيء"}
