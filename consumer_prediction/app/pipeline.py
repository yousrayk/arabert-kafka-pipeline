import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

from app.transforms import (DLQ_TOPIC, PREDICTIONS_TOPIC, build_dlq_payload,
                            build_prediction_payload, parse_raw)

logger = logging.getLogger("consumer_prediction")


@dataclass
class Incoming:
    value: bytes
    source: dict | None = None  # {"topic", "partition", "offset"}


@dataclass
class Outgoing:
    topic: str
    key: str | None
    payload: dict


def _dlq(item: Incoming, error: Exception) -> Outgoing:
    logger.warning(json.dumps({"event": "dlq", "error": str(error), "source": item.source}))
    return Outgoing(DLQ_TOPIC, None, build_dlq_payload(item.value, f"{type(error).__name__}: {error}",
                                                       source=item.source))


def process_batch(batch: list[Incoming], classifier, preprocess: Callable[[str], str],
                  clock: Callable[[], float] = time.perf_counter) -> list[Outgoing]:
    """Turns a batch of reviews.raw messages into reviews.predictions
    messages, routing every per-message failure to reviews.dlq so one bad
    message never blocks or crashes the stream.

    Inference runs once for the whole batch (much better CPU throughput
    than one forward pass per message). If that batched call fails, the
    batch is retried one message at a time to isolate the poison message(s),
    and only those go to the DLQ.
    """
    out: list[Outgoing] = []
    parsed: list[tuple[Incoming, dict, str]] = []
    for item in batch:
        try:
            raw = parse_raw(item.value)
            parsed.append((item, raw, preprocess(raw["text"])))
        except Exception as e:
            out.append(_dlq(item, e))
    if not parsed:
        return out

    start = clock()
    try:
        results = classifier.predict([text for _, _, text in parsed])
        per_message_ms = (clock() - start) * 1000 / len(parsed)
        timed = [(r, per_message_ms) for r in results]
    except Exception:
        logger.exception("Batched inference failed; retrying per message to isolate the failure.")
        timed = []
        for item, _, text in parsed:
            start = clock()
            try:
                timed.append((classifier.predict([text])[0], (clock() - start) * 1000))
            except Exception as e:
                timed.append((None, 0.0))
                out.append(_dlq(item, e))

    for (item, raw, _), (result, ms) in zip(parsed, timed):
        if result is None:
            continue
        payload = build_prediction_payload(raw["id"], result["label"], result["confidence"],
                                           result["scores"], classifier.model_version, ms)
        out.append(Outgoing(PREDICTIONS_TOPIC, raw["id"], payload))
    return out
