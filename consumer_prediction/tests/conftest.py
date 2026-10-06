import os
from pathlib import Path

# Convenience default for local (non-Docker) test runs: weights produced by
# training/train.py land in ./models/arabert_sentiment. In Docker, MODEL_PATH
# is set explicitly via docker-compose to the bind-mounted path instead.
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "arabert_sentiment")
os.environ.setdefault("MODEL_PATH", DEFAULT_MODEL_PATH)
