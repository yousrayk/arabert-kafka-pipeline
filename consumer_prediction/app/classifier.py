import json
import logging
import os
import time
from typing import Callable, Protocol

logger = logging.getLogger("consumer_prediction")

LABELS = ("negative", "neutral", "positive")
META_FILE = "training_meta.json"


class Backend(Protocol):
    labels: list[str]

    def predict_proba(self, texts: list[str]) -> list[list[float]]: ...


class HFBackend:
    """A fine-tuned AutoModelForSequenceClassification saved with
    save_pretrained(). Label names come from the checkpoint's own id2label,
    so the class order is whatever training wrote, not an assumption here."""

    def __init__(self, model_path: str, max_length: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._max_length = max_length
        id2label = self._model.config.id2label
        self.labels = [str(id2label[i]).lower() for i in range(len(id2label))]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        enc = self._tokenizer(texts, padding=True, truncation=True,
                              max_length=self._max_length, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self._model(**enc).logits
        return self._torch.softmax(logits, dim=-1).tolist()


def _files(model_path: str) -> list[str]:
    return [os.path.join(model_path, f) for f in sorted(os.listdir(model_path))
            if os.path.isfile(os.path.join(model_path, f))]


class SentimentClassifier:
    """Serves a model from a directory and hot-swaps it when the files there
    change — drop new weights in and the next batch uses them, no restart
    (same live-update behavior as the reference repo). On top of that:

    - the directory is checked at most every `check_interval` seconds, not
      on every message;
    - a change is only picked up once the newest file is `settle_seconds`
      old, so a half-copied checkpoint isn't loaded mid-write;
    - the new model is fully loaded *before* the swap, and if loading fails
      (corrupt files, wrong label set) the old model keeps serving and that
      exact set of files isn't retried until it changes again.
    """

    def __init__(self, model_path: str, load_fn: Callable[[str], Backend] = HFBackend,
                 check_interval: float = 5.0, settle_seconds: float = 2.0,
                 clock: Callable[[], float] = time.time):
        self.model_path = model_path
        self._load_fn = load_fn
        self._check_interval = check_interval
        self._settle_seconds = settle_seconds
        self._clock = clock
        self._backend: Backend | None = None
        self._fingerprint = None
        self._failed_fingerprint = None
        self._last_check = float("-inf")
        self.model_version: str | None = None

    @property
    def ready(self) -> bool:
        return self._backend is not None

    def _fingerprint_now(self):
        if not os.path.isdir(self.model_path):
            return None
        stats = [(os.path.basename(f), os.path.getsize(f), os.path.getmtime(f)) for f in _files(self.model_path)]
        return tuple(stats) or None

    def _version_for(self, fingerprint) -> str:
        meta_path = os.path.join(self.model_path, META_FILE)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    version = json.load(f).get("model_version")
                if version:
                    return str(version)
            except (OSError, ValueError):
                pass
        newest = max(mtime for _, _, mtime in fingerprint)
        return f"{os.path.basename(self.model_path.rstrip('/'))}@{int(newest)}"

    def reload_if_changed(self, force: bool = False) -> bool:
        """Returns True if a new model was swapped in."""
        now = self._clock()
        if not force and now - self._last_check < self._check_interval:
            return False
        self._last_check = now

        fingerprint = self._fingerprint_now()
        if fingerprint is None or fingerprint in (self._fingerprint, self._failed_fingerprint):
            return False
        newest = max(mtime for _, _, mtime in fingerprint)
        if now - newest < self._settle_seconds:
            return False  # still being written; look again next check

        try:
            backend = self._load_fn(self.model_path)
            unknown = set(backend.labels) - set(LABELS)
            if unknown:
                raise ValueError(f"model has unexpected labels {sorted(unknown)}; expected {LABELS}")
        except Exception as e:
            self._failed_fingerprint = fingerprint
            logger.error(json.dumps({"event": "model_load_failed", "model_path": self.model_path,
                                     "error": str(e), "serving": self.model_version}))
            return False

        previous = self.model_version
        self._backend, self._fingerprint = backend, fingerprint
        self.model_version = self._version_for(fingerprint)
        logger.info(json.dumps({"event": "model_loaded", "model_version": self.model_version,
                                "previous": previous}))
        return True

    def predict(self, texts: list[str]) -> list[dict]:
        self.reload_if_changed()
        if self._backend is None:
            raise RuntimeError(f"no model loaded from {self.model_path}")
        labels = self._backend.labels
        results = []
        for probs in self._backend.predict_proba(texts):
            best = max(range(len(probs)), key=probs.__getitem__)
            results.append({
                "label": labels[best],
                "confidence": float(probs[best]),
                "scores": {label: float(p) for label, p in zip(labels, probs)},
            })
        return results
