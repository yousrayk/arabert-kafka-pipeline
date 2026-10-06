from app.join_buffer import PREDICTION, RAW, JoinBuffer

RAW_MSG = {"id": "id-1", "text": "ممتاز"}
PRED = {"id": "id-1", "label": "positive", "confidence": 0.9, "model_version": "v1"}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_no_join_until_both_halves_present():
    buf = JoinBuffer()
    assert buf.add(RAW, RAW_MSG).kind == "pending"
    assert len(buf) == 1


def test_join_when_prediction_arrives_after_raw():
    buf = JoinBuffer()
    buf.add(RAW, RAW_MSG, meta="raw-meta")
    outcome = buf.add(PREDICTION, PRED, meta="pred-meta")
    assert outcome.kind == "joined"
    assert outcome.pair == (RAW_MSG, PRED)
    assert outcome.released == ["raw-meta"]
    assert len(buf) == 0


def test_join_when_raw_arrives_after_prediction_keeps_pair_order():
    buf = JoinBuffer()
    buf.add(PREDICTION, PRED, meta="pred-meta")
    outcome = buf.add(RAW, RAW_MSG)
    assert outcome.pair == (RAW_MSG, PRED)
    assert outcome.released == ["pred-meta"]


def test_different_ids_do_not_cross_join():
    buf = JoinBuffer()
    buf.add(RAW, {"id": "id-1", "text": "واحد"})
    buf.add(RAW, {"id": "id-2", "text": "اثنان"})
    outcome = buf.add(PREDICTION, {**PRED, "id": "id-2"})
    assert outcome.pair[0]["text"] == "اثنان"
    assert len(buf) == 1


def test_redelivered_half_after_join_is_a_duplicate_not_a_new_pending_entry():
    buf = JoinBuffer()
    buf.add(RAW, RAW_MSG)
    buf.add(PREDICTION, PRED)
    # at-least-once: the prediction is delivered again
    assert buf.add(PREDICTION, PRED).kind == "duplicate"
    assert len(buf) == 0  # nothing left to time out as "unmatched"


def test_redelivered_half_while_pending_is_a_duplicate():
    buf = JoinBuffer()
    buf.add(RAW, RAW_MSG)
    assert buf.add(RAW, RAW_MSG).kind == "duplicate"
    assert len(buf) == 1


def test_ttl_evicts_unmatched_entries_with_their_meta():
    clock = Clock()
    buf = JoinBuffer(ttl_seconds=60, clock=clock)
    buf.add(RAW, RAW_MSG, meta=("reviews.raw", 0, 5))
    clock.now = 59
    assert buf.evict_expired() == []
    clock.now = 60
    [evicted] = buf.evict_expired()
    assert (evicted.side, evicted.id, evicted.meta) == (RAW, "id-1", ("reviews.raw", 0, 5))
    assert len(buf) == 0


def test_ttl_evicts_only_the_expired_prefix():
    clock = Clock()
    buf = JoinBuffer(ttl_seconds=10, clock=clock)
    buf.add(RAW, {"id": "old", "text": "x"})
    clock.now = 5
    buf.add(PREDICTION, {**PRED, "id": "orphan-pred"})
    buf.add(RAW, {"id": "new", "text": "y"})
    clock.now = 12
    assert [e.id for e in buf.evict_expired()] == ["old"]
    clock.now = 15
    assert sorted(e.id for e in buf.evict_expired()) == ["new", "orphan-pred"]


def test_late_half_after_eviction_waits_again_instead_of_joining():
    clock = Clock()
    buf = JoinBuffer(ttl_seconds=10, clock=clock)
    buf.add(RAW, RAW_MSG)
    clock.now = 10
    buf.evict_expired()
    assert buf.add(PREDICTION, PRED).kind == "pending"


def test_dedupe_memory_is_bounded_by_its_own_ttl():
    clock = Clock()
    buf = JoinBuffer(ttl_seconds=10, dedupe_seconds=100, clock=clock)
    buf.add(RAW, RAW_MSG)
    buf.add(PREDICTION, PRED)
    clock.now = 100
    buf.evict_expired()
    # forgotten: a very late redelivery is buffered (and would expire to DLQ)
    assert buf.add(PREDICTION, PRED).kind == "pending"
