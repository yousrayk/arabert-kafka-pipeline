from collections import deque
from dataclasses import dataclass


@dataclass
class WindowState:
    window_start: float
    window_end: float
    total: int
    negative: int

    @property
    def negative_share(self) -> float:
        return self.negative / self.total if self.total else 0.0


class SlidingWindow:
    """Exact sliding window over the last `window_seconds` of event time
    (a prediction's predicted_at). The window's right edge is the latest
    timestamp seen, not the wall clock, so it behaves the same live and on
    a replayed backlog, and tolerates the small reordering that comes from
    reading three partitions at once (late events still count while they
    are inside the window; events already outside it are dropped).
    Counts are maintained incrementally, so each event is O(1) amortized.
    """

    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, bool]] = deque()
        self._negative = 0
        self._latest = float("-inf")

    def add(self, ts: float, is_negative: bool) -> WindowState:
        self._latest = max(self._latest, ts)
        cutoff = self._latest - self.window_seconds
        if ts > cutoff:
            # Keep the deque ordered by inserting late events in place;
            # they are rare and land near the right end.
            if self._events and ts < self._events[-1][0]:
                items = list(self._events)
                i = len(items)
                while i > 0 and items[i - 1][0] > ts:
                    i -= 1
                items.insert(i, (ts, is_negative))
                self._events = deque(items)
            else:
                self._events.append((ts, is_negative))
            self._negative += is_negative
        while self._events and self._events[0][0] <= cutoff:
            _, neg = self._events.popleft()
            self._negative -= neg
        start = self._events[0][0] if self._events else self._latest
        return WindowState(start, self._latest, len(self._events), self._negative)


class SpikeDetector:
    """Fires once when the negative share crosses `threshold` (with at
    least `min_count` predictions in the window, so three negatives out of
    four at startup don't page anyone), then stays quiet until the share
    falls back below `rearm_below` — hysteresis, so a share oscillating
    around the threshold produces one alert, not one per message. Emits a
    "resolved" event when it re-arms."""

    def __init__(self, window_seconds: float = 60, threshold: float = 0.4,
                 rearm_below: float | None = None, min_count: int = 20):
        if rearm_below is None:
            rearm_below = round(threshold * 0.8, 4)
        if not 0 <= rearm_below <= threshold <= 1:
            raise ValueError("need 0 <= rearm_below <= threshold <= 1")
        self.window = SlidingWindow(window_seconds)
        self.threshold = threshold
        self.rearm_below = rearm_below
        self.min_count = min_count
        self.firing = False

    def observe(self, ts: float, label: str) -> tuple[str, WindowState] | None:
        state = self.window.add(ts, label == "negative")
        if state.total < self.min_count:
            return None
        if not self.firing and state.negative_share >= self.threshold:
            self.firing = True
            return "firing", state
        if self.firing and state.negative_share < self.rearm_below:
            self.firing = False
            return "resolved", state
        return None
