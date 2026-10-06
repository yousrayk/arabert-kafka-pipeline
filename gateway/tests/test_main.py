import csv

import requests

from app.main import Review, load_reviews, paced, replay, wait_for_health


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        writer.writerows(rows)


def test_load_reviews_reads_id_and_text_but_not_label(tmp_path):
    path = tmp_path / "replay.csv"
    write_csv(path, [["1", "خدمة رائعة", "positive"], ["2", "تجربة سيئة", "negative"]])
    reviews = load_reviews(str(path))
    assert reviews == [Review("1", "خدمة رائعة"), Review("2", "تجربة سيئة")]


def test_load_reviews_skips_blank_text(tmp_path):
    path = tmp_path / "replay.csv"
    write_csv(path, [["1", "   ", "neutral"], ["2", "نص", "neutral"]])
    assert [r.source_id for r in load_reviews(str(path))] == ["2"]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_paced_spaces_items_at_the_configured_rate():
    clock = FakeClock()
    times = [clock() for _ in paced(range(5), rate_per_sec=10, clock=clock, sleep_fn=clock.sleep)]
    assert [round(t, 6) for t in times] == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_paced_catches_up_after_a_slow_item_instead_of_drifting():
    clock = FakeClock()
    times = []
    for i in paced(range(4), rate_per_sec=10, clock=clock, sleep_fn=clock.sleep):
        times.append(round(clock(), 6))
        if i == 0:
            clock.now += 0.25  # first send took 250ms — longer than 2 slots
    # items 1 and 2 are overdue, so they go immediately; item 3 is back on schedule
    assert times == [0.0, 0.25, 0.25, 0.3]


def test_paced_zero_rate_never_sleeps():
    sleeps = []
    assert list(paced(range(3), 0, sleep_fn=sleeps.append)) == [0, 1, 2]
    assert sleeps == []


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = list(statuses or [])

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        status = self.statuses.pop(0) if self.statuses else 200
        if isinstance(status, Exception):
            raise status
        return FakeResponse(status, {"id": f"id-{len(self.calls)}"})

    def get(self, url, timeout):
        status = self.statuses.pop(0) if self.statuses else 200
        if isinstance(status, Exception):
            raise status
        return FakeResponse(status)


def test_replay_posts_every_review_with_provenance_and_no_label():
    session = FakeSession()
    clock = FakeClock()
    reviews = [Review("1", "ممتاز"), Review("2", "سيء")]

    stats = replay(reviews, "http://producer:5000/send", rate_per_sec=0, source="astd",
                   session=session, clock=clock, sleep_fn=clock.sleep)

    assert stats == {"sent": 2, "failed": 0}
    assert [c[1] for c in session.calls] == [
        {"text": "ممتاز", "source": "astd", "source_id": "1"},
        {"text": "سيء", "source": "astd", "source_id": "2"},
    ]
    assert all(c[0] == "http://producer:5000/send" for c in session.calls)


def test_replay_counts_failures_and_keeps_going():
    session = FakeSession(statuses=[503, requests.ConnectionError("down"), 200])
    reviews = [Review(str(i), "نص") for i in range(3)]
    stats = replay(reviews, "http://x/send", rate_per_sec=0, session=session)
    assert stats == {"sent": 1, "failed": 2}
    assert len(session.calls) == 3


def test_wait_for_health_retries_until_ok():
    session = FakeSession(statuses=[requests.ConnectionError("not up"), 503, 200])
    sleeps = []
    wait_for_health(session, "http://x/health", sleep_fn=sleeps.append)
    assert len(sleeps) == 2
