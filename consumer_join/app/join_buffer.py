import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

RAW, PREDICTION = "raw", "prediction"


@dataclass
class Outcome:
    kind: str                      # "pending" | "joined" | "duplicate"
    pair: tuple[dict, dict] | None = None   # (raw, prediction) when joined
    released: list[Any] = field(default_factory=list)  # metas no longer held


@dataclass
class Evicted:
    side: str
    id: str
    payload: dict
    meta: Any


class JoinBuffer:
    """In-memory stream-stream join of reviews.raw and reviews.predictions
    on id, with TTL eviction.

    - Whichever half arrives first waits in the buffer; the second half
      completes the join and both are dropped from it.
    - A half that waits longer than `ttl_seconds` is evicted (returned by
      evict_expired) so a prediction that never comes — its raw went to the
      DLQ, say — can't grow memory without bound.
    - Joined ids are remembered for `dedupe_seconds`, so a redelivered half
      (at-least-once upstream) is recognized as a duplicate instead of
      waiting out the TTL and being reported as unmatched.

    Each entry carries an opaque `meta` (the Kafka coordinates, in practice)
    that is handed back when the entry leaves the buffer, which is how the
    caller knows which offsets are safe to commit.
    """

    def __init__(self, ttl_seconds: float = 60.0, dedupe_seconds: float = 600.0,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self.dedupe = dedupe_seconds
        self._clock = clock
        # id -> (arrived_at, payload, meta); insertion order == arrival order,
        # so expiry only ever has to look at the front.
        self._pending = {RAW: OrderedDict(), PREDICTION: OrderedDict()}
        self._joined: OrderedDict[str, float] = OrderedDict()

    def __len__(self) -> int:
        return len(self._pending[RAW]) + len(self._pending[PREDICTION])

    def add(self, side: str, payload: dict, meta: Any = None) -> Outcome:
        id_ = payload["id"]
        other = PREDICTION if side == RAW else RAW
        if id_ in self._joined or id_ in self._pending[side]:
            return Outcome("duplicate")
        if id_ in self._pending[other]:
            _, other_payload, other_meta = self._pending[other].pop(id_)
            self._joined[id_] = self._clock()
            pair = (payload, other_payload) if side == RAW else (other_payload, payload)
            return Outcome("joined", pair, [other_meta])
        self._pending[side][id_] = (self._clock(), payload, meta)
        return Outcome("pending")

    def evict_expired(self) -> list[Evicted]:
        now = self._clock()
        evicted = []
        for side, entries in self._pending.items():
            while entries:
                id_, (arrived, payload, meta) = next(iter(entries.items()))
                if now - arrived < self.ttl:
                    break
                entries.popitem(last=False)
                evicted.append(Evicted(side, id_, payload, meta))
        while self._joined and now - next(iter(self._joined.values())) >= self.dedupe:
            self._joined.popitem(last=False)
        return evicted
