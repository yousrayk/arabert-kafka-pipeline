import json

from app.join_buffer import JoinBuffer
from app.main import join_loop
from app.processor import JoinProcessor, OffsetTracker
from app.transforms import DLQ_TOPIC


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Msg:
    def __init__(self, topic, partition, offset, payload):
        self._t, self._p, self._o = topic, partition, offset
        self._v = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()

    def topic(self):
        return self._t

    def partition(self):
        return self._p

    def offset(self):
        return self._o

    def value(self):
        return self._v

    def error(self):
        return None


def raw(id_, offset, partition=0):
    return Msg("reviews.raw", partition, offset, {"schema_version": 1, "id": id_, "text": "نص",
                                                  "produced_at": "2026-01-01T00:00:00+00:00"})


def pred(id_, offset, partition=0):
    return Msg("reviews.predictions", partition, offset, {"schema_version": 1, "id": id_, "label": "negative",
                                                          "confidence": 0.8, "model_version": "v1"})


def make(ttl=60):
    clock = Clock()
    return JoinProcessor(JoinBuffer(ttl_seconds=ttl, clock=clock)), clock


def test_offset_tracker_commits_past_consumed_when_nothing_held():
    t = OffsetTracker()
    t.consumed(("a", 0), 4)
    assert t.committable() == {("a", 0): 5}


def test_offset_tracker_never_commits_past_a_held_offset():
    t = OffsetTracker()
    for o in range(3, 8):
        t.consumed(("a", 0), o)
    t.hold(("a", 0), 4)
    t.hold(("a", 0), 6)
    assert t.committable() == {("a", 0): 4}
    t.release(("a", 0), 4)
    assert t.committable() == {("a", 0): 6}
    t.release(("a", 0), 6)
    assert t.committable() == {("a", 0): 8}


def test_joined_pair_is_emitted_and_offsets_advance_past_both():
    proc, _ = make()
    joined, dlq = proc.handle([raw("x", 10), pred("x", 20)])
    assert [j["id"] for j in joined] == ["x"] and dlq == []
    assert proc.offsets.committable() == {("reviews.raw", 0): 11, ("reviews.predictions", 0): 21}


def test_unjoined_message_pins_its_partition_offset():
    proc, _ = make()
    proc.handle([raw("waiting", 10), raw("x", 11), pred("x", 20)])
    assert proc.offsets.committable()[("reviews.raw", 0)] == 10


def test_ttl_expiry_goes_to_dlq_and_unpins_offset():
    proc, clock = make(ttl=60)
    proc.handle([raw("orphan", 10)])
    clock.now = 61
    [dlq] = proc.evict()
    assert dlq.topic == DLQ_TOPIC
    assert dlq.payload["stage"] == "consumer_join"
    assert dlq.payload["error"].startswith("join_ttl_expired")
    assert json.loads(dlq.payload["original_payload"])["id"] == "orphan"
    assert dlq.payload["source"] == {"topic": "reviews.raw", "partition": 0, "offset": 10}
    assert proc.offsets.committable()[("reviews.raw", 0)] == 11
    assert proc.stats["expired"] == 1


def test_malformed_message_goes_to_dlq_and_is_not_held():
    proc, _ = make()
    joined, dlq = proc.handle([Msg("reviews.raw", 0, 3, b"{oops")])
    assert joined == [] and len(dlq) == 1
    assert proc.offsets.committable() == {("reviews.raw", 0): 4}


def test_duplicate_delivery_is_counted_not_rejoined():
    proc, _ = make()
    proc.handle([raw("x", 1), pred("x", 1)])
    joined, _ = proc.handle([pred("x", 2)])
    assert joined == [] and proc.stats["duplicates"] == 1


def test_reset_drops_state():
    proc, _ = make()
    proc.handle([raw("x", 1)])
    proc.reset()
    assert len(proc.buffer) == 0 and proc.offsets.committable() == {}


class FakeConsumer:
    def __init__(self, batches):
        self.batches = list(batches)

    def consume(self, num_messages, timeout):
        return self.batches.pop(0) if self.batches else []


class FakeSink:
    def __init__(self, log):
        self.rows, self.log = [], log

    def write(self, rows):
        self.rows += rows
        self.log.append("sink")


class FakeProducer:
    def __init__(self, log):
        self.log, self.sent = log, []

    def produce(self, topic, key, value, on_delivery):
        self.sent.append(topic)
        on_delivery(None, None)

    def poll(self, timeout):
        return 0

    def flush(self, timeout):
        self.log.append("dlq")
        return 0


def stop_after(n):
    state = {"n": 0}

    def should_stop():
        state["n"] += 1
        return state["n"] > n
    return should_stop


def test_loop_writes_sink_before_committing_and_evicts_on_empty_polls():
    log, commits = [], []
    proc, clock = make(ttl=60)
    consumer = FakeConsumer([[raw("x", 0), pred("x", 0), raw("orphan", 1)], []])
    sink, producer = FakeSink(log), FakeProducer(log)

    def commit(offsets):
        commits.append(dict(offsets))
        log.append("commit")
        clock.now = 120  # time passes before the next (empty) poll

    join_loop(consumer, producer, proc, sink, stop_after(2), commit_fn=commit)

    assert [r["id"] for r in sink.rows] == ["x"]
    assert log == ["sink", "commit", "dlq", "commit"]
    assert commits[0][("reviews.raw", 0)] == 1        # orphan pinned
    assert commits[1] == {("reviews.raw", 0): 2}      # released after TTL -> DLQ
    assert producer.sent == [DLQ_TOPIC]
