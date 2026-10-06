import json
import logging
import os
import signal
import time

from app.classifier import SentimentClassifier
from app.kafka_io import consumer_config, producer_config, run_loop
from app.pipeline import Incoming, process_batch
from app.preprocess import ArabicPreprocessor
from app.transforms import RAW_TOPIC

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("consumer_prediction")


def run():
    from confluent_kafka import Consumer, Producer

    model_path = os.environ.get("MODEL_PATH", "/model")
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))

    if os.environ.get("TORCH_THREADS"):
        import torch
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    classifier = SentimentClassifier(
        model_path,
        check_interval=float(os.environ.get("RELOAD_CHECK_SECONDS", "5")),
        settle_seconds=float(os.environ.get("RELOAD_SETTLE_SECONDS", "2")),
    )
    # Don't subscribe before a model is available: messages simply wait in
    # reviews.raw (nothing is lost) until weights are dropped into MODEL_PATH.
    while not classifier.reload_if_changed(force=True):
        if stop["flag"]:
            return
        logger.info(json.dumps({"event": "waiting_for_model", "model_path": model_path}))
        time.sleep(5)

    preprocess = ArabicPreprocessor()

    def handle(messages):
        batch = [Incoming(m.value(), {"topic": m.topic(), "partition": m.partition(), "offset": m.offset()})
                 for m in messages]
        out = process_batch(batch, classifier, preprocess)
        logger.info(json.dumps({"event": "batch_processed", "in": len(batch), "out": len(out),
                                "model_version": classifier.model_version}))
        return out

    consumer = Consumer(consumer_config(bootstrap, "consumer-prediction"))
    producer = Producer(producer_config(bootstrap))
    consumer.subscribe([RAW_TOPIC])
    try:
        run_loop(consumer, producer, handle, batch_size, lambda: stop["flag"])
    finally:
        consumer.close()
        logger.info("Consumer stopped.")


if __name__ == "__main__":
    run()
