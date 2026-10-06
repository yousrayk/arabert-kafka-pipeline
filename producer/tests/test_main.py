from fastapi.testclient import TestClient

from app.main import RAW_TOPIC, app, get_publisher
from app.publisher import FakePublisher, PublishError

fake = FakePublisher()
app.dependency_overrides[get_publisher] = lambda: fake

client = TestClient(app)


def test_send_publishes_versioned_message_to_raw_topic():
    resp = client.post("/send", json={"text": "الخدمة ممتازة", "source": "astd", "source_id": "42"})
    assert resp.status_code == 200
    body = resp.json()

    topic, key, payload = fake.published[-1]
    assert topic == RAW_TOPIC
    assert payload["schema_version"] == 1
    assert payload["id"] == body["id"]
    assert payload["text"] == "الخدمة ممتازة"
    assert payload["source"] == "astd"
    assert payload["source_id"] == "42"
    assert "produced_at" in payload


def test_message_is_keyed_by_id():
    resp = client.post("/send", json={"text": "سيء"})
    topic, key, payload = fake.published[-1]
    assert key == payload["id"] == resp.json()["id"]


def test_provenance_fields_are_optional():
    resp = client.post("/send", json={"text": "عادي"})
    assert resp.status_code == 200
    _, _, payload = fake.published[-1]
    assert payload["source"] is None
    assert payload["source_id"] is None


def test_ids_are_unique_per_send():
    a = client.post("/send", json={"text": "واحد"}).json()["id"]
    b = client.post("/send", json={"text": "واحد"}).json()["id"]
    assert a != b


def test_rejects_empty_text():
    assert client.post("/send", json={"text": ""}).status_code == 422


def test_rejects_missing_text_field():
    assert client.post("/send", json={}).status_code == 422


def test_rejects_oversized_text():
    assert client.post("/send", json={"text": "ا" * 5001}).status_code == 422


def test_returns_503_when_kafka_unavailable():
    failing = FakePublisher(fail_with=PublishError("broker down"))
    app.dependency_overrides[get_publisher] = lambda: failing
    try:
        resp = client.post("/send", json={"text": "مرحبا"})
        assert resp.status_code == 503
    finally:
        app.dependency_overrides[get_publisher] = lambda: fake


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
