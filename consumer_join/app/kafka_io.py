"""Consume -> process -> produce -> commit loop with at-least-once delivery:
offsets are committed only after every output of the batch has been
acknowledged by the broker, so a crash anywhere before the commit means the
batch is re-read and re-processed rather than lost. (Downstream consumers
therefore have to tolerate duplicates — see consumer_join's dedup.)
"""
import json
import logging
from typing import Callable

logger = logging.getLogger("kafka_io")


def consumer_config(bootstrap: str, group_id: str) -> dict:
    return {
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    }


def producer_config(bootstrap: str) -> dict:
    return {"bootstrap.servers": bootstrap, "acks": "all", "enable.idempotence": True, "linger.ms": 5}


class DeliveryError(RuntimeError):
    pass


def publish_all(producer, outgoing, flush_timeout: float = 30.0) -> None:
    """Produces every message and blocks until all are acknowledged; raises
    if any failed, so the caller doesn't commit the input offsets."""
    errors = []

    def on_delivery(err, _msg):
        if err is not None:
            errors.append(err)

    for o in outgoing:
        producer.produce(
            o.topic,
            key=o.key.encode("utf-8") if o.key else None,
            value=json.dumps(o.payload, ensure_ascii=False).encode("utf-8"),
            on_delivery=on_delivery,
        )
        producer.poll(0)
    remaining = producer.flush(flush_timeout)
    if remaining:
        raise DeliveryError(f"{remaining} message(s) not delivered within {flush_timeout}s")
    if errors:
        raise DeliveryError(f"{len(errors)} delivery error(s), first: {errors[0]}")


def run_loop(consumer, producer, handle: Callable[[list], list], batch_size: int,
             should_stop: Callable[[], bool], poll_timeout: float = 1.0) -> None:
    while not should_stop():
        messages = consumer.consume(num_messages=batch_size, timeout=poll_timeout)
        if not messages:
            continue
        good = []
        for m in messages:
            if m.error() is not None:
                logger.error(json.dumps({"event": "consume_error", "error": str(m.error())}))
            else:
                good.append(m)
        if good:
            publish_all(producer, handle(good))
        consumer.commit(asynchronous=False)
