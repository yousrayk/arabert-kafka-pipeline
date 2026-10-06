import json
import logging
from collections import defaultdict
from dataclasses import dataclass

from app.join_buffer import PREDICTION, RAW, JoinBuffer
from app.transforms import (DLQ_TOPIC, PREDICTIONS_TOPIC, RAW_TOPIC, build_dlq_payload,
                            build_joined_payload, parse)

logger = logging.getLogger("consumer_join")

SIDE_BY_TOPIC = {RAW_TOPIC: RAW, PREDICTIONS_TOPIC: PREDICTION}


@dataclass
class Outgoing:
    topic: str
    key: str | None
    payload: dict


class OffsetTracker:
    """Decides which offsets are safe to commit for a stateful consumer.

    Committing the latest consumed offset would be wrong here: a message
    sitting unjoined in the in-memory buffer would be lost on a crash. So
    per partition, the commit point is the *oldest offset still held in the
    buffer* — or one past the last consumed offset when nothing is held.
    After a restart, the consumer re-reads from there and the buffer is
    rebuilt (re-joins of already-written pairs are possible: the sink is
    at-least-once, readers dedupe by id).
    """

    def __init__(self):
        self._next: dict[tuple[str, int], int] = {}
        self._held: dict[tuple[str, int], set[int]] = defaultdict(set)

    def consumed(self, tp: tuple[str, int], offset: int) -> None:
        self._next[tp] = max(self._next.get(tp, 0), offset + 1)

    def hold(self, tp: tuple[str, int], offset: int) -> None:
        self._held[tp].add(offset)

    def release(self, tp: tuple[str, int], offset: int) -> None:
        self._held[tp].discard(offset)

    def committable(self) -> dict[tuple[str, int], int]:
        return {tp: (min(self._held[tp]) if self._held[tp] else nxt) for tp, nxt in self._next.items()}


class JoinProcessor:
    def __init__(self, buffer: JoinBuffer):
        self.buffer = buffer
        self.offsets = OffsetTracker()
        self.stats = {"joined": 0, "duplicates": 0, "expired": 0, "malformed": 0}

    def reset(self) -> None:
        """Drops all join state — on partition revocation. Safe because the
        committed offsets never pass a buffered message: whoever owns the
        partitions next re-reads from there and rebuilds the buffer."""
        self.buffer = JoinBuffer(self.buffer.ttl, self.buffer.dedupe)
        self.offsets = OffsetTracker()

    def handle(self, messages) -> tuple[list[dict], list[Outgoing]]:
        """Feeds consumed messages into the join. Returns (joined payloads
        for the sink, DLQ messages)."""
        joined, dlq = [], []
        for m in messages:
            tp, offset = (m.topic(), m.partition()), m.offset()
            self.offsets.consumed(tp, offset)
            source = {"topic": tp[0], "partition": tp[1], "offset": offset}
            try:
                payload = parse(m.value())
            except ValueError as e:
                self.stats["malformed"] += 1
                dlq.append(Outgoing(DLQ_TOPIC, None, build_dlq_payload(m.value(), str(e), source)))
                continue

            outcome = self.buffer.add(SIDE_BY_TOPIC[tp[0]], payload, meta=(tp, offset))
            if outcome.kind == "pending":
                self.offsets.hold(tp, offset)
            elif outcome.kind == "duplicate":
                self.stats["duplicates"] += 1
            else:
                for held_tp, held_offset in outcome.released:
                    self.offsets.release(held_tp, held_offset)
                joined.append(build_joined_payload(*outcome.pair))
                self.stats["joined"] += 1
        return joined, dlq

    def evict(self) -> list[Outgoing]:
        """TTL eviction — called every loop iteration, including empty
        polls, so stale halves expire even when traffic stops."""
        dlq = []
        for e in self.buffer.evict_expired():
            tp, offset = e.meta
            self.offsets.release(tp, offset)
            self.stats["expired"] += 1
            missing = PREDICTION if e.side == RAW else RAW
            logger.warning(json.dumps({"event": "join_expired", "id": e.id, "have": e.side}))
            dlq.append(Outgoing(DLQ_TOPIC, None, build_dlq_payload(
                e.payload, f"join_ttl_expired: no {missing} for id within {self.buffer.ttl}s",
                {"topic": tp[0], "partition": tp[1], "offset": offset})))
        return dlq
