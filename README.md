# Arabic sentiment pipeline (Kafka + AraBERT)

Real-time sentiment analysis for Arabic text. Reviews are published to Kafka, classified as
negative / neutral / positive by a fine-tuned AraBERT v02 model, joined back with their text, and shown
on a dashboard. An alert is raised when the share of negative reviews spikes.

Based on the structure of [Jailbreak_Detection](https://github.com/yousrayk/Jailbreak_Detection), using
Kafka (KRaft, no Zookeeper) instead of Pulsar.

## Services

| Service | Role |
|---|---|
| `producer` | FastAPI `POST /send`, publishes to `reviews.raw` |
| `consumer_prediction` | AraBERT classification, writes to `reviews.predictions`; failures go to `reviews.dlq`. Reloads the model when the weights change |
| `consumer_join` | Joins raw messages with their predictions by id, writes `data/joined_results.jsonl` |
| `alerting` | Writes to `alerts.negative_spike` when the negative share in a 60 s window passes 0.4 |
| `dashboard` | Streamlit page: sentiment over time and alerts |
| `gateway` | Replays the test set into the producer |

## Setup

Needs Docker and Python 3.12+.

```bash
pip install pandas pyarrow scikit-learn
python scripts/prepare_dataset.py
```

This downloads [ASTD](https://huggingface.co/datasets/arbml/ASTD) (~10k Egyptian-dialect tweets) and
writes train/val/test splits to `data/`. The "objective" class is merged into neutral.

**Model weights** are not in the repo. Train them with
[`training/finetune_arabert_colab.ipynb`](training/finetune_arabert_colab.ipynb) on a Colab GPU, or locally
with `python training/train.py`, and put the result in `models/arabert_sentiment/`.

## Run

```bash
docker compose up -d --build
docker compose --profile replay up gateway
```

- Dashboard: http://localhost:8501
- API docs (send your own text): http://localhost:5000/docs
- Results: `data/joined_results.jsonl`

Useful variables: `RATE_PER_SEC` (default 20, `0` = no limit), `MAX_MESSAGES`, `REPLAY_FILE`
(`spike.csv` contains only negatives, to trigger an alert), `ALERT_THRESHOLD`.

Evaluate a run:

```bash
python scripts/eval.py
```

## Tests

```bash
cd producer && python -m pytest -q    # same for each service folder
```

## Results

Measured on a laptop CPU (no GPU). The model used here was trained on only 1,200 tweets for 2 epochs,
so accuracy is low. A full training run on a GPU should do much better.

| Metric | Value |
|---|---|
| Accuracy | 0.52 |
| Macro-F1 | 0.43 |
| Throughput | ~14 msg/s |
| Latency at 3 msg/s (p50 / p95) | 1.8 s / 5.4 s |

The bottleneck is model inference on CPU.
