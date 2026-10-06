"""Fine-tunes aubmindlab/bert-base-arabertv02 for 3-class Arabic sentiment
(negative / neutral / positive) and saves a checkpoint consumer_prediction
can serve — and hot-reload, if it's written into the served directory.

    python scripts/prepare_dataset.py                      # once: data/astd/*.csv
    python training/train.py                               # GPU recommended
    python training/train.py --max-train-samples 1500 --epochs 1   # CPU smoke run

Text goes through the same ArabicPreprocessor the serving path uses
(imported from consumer_prediction/app/preprocess.py), so there's no
train/serve skew in cleaning.

Note on the original notebook (NLP_for_Sentiment_Analysis.ipynb): it trained
a binary TF/Keras head on bert-base-arabert (v1, which expects Farasa
segmentation) at max_len 120. This script moves to PyTorch (what the
consumer serves), v02 (no Farasa/Java at inference), three classes, and
class-weighted loss, because ASTD's neutral class dominates.
"""
import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consumer_prediction"))
from app.preprocess import MODEL_NAME, ArabicPreprocessor  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load_split(path: Path, preprocess, limit: int | None, seed: int) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    if limit and len(df) > limit:
        # Stratified: the same fraction of every class, so the subsample keeps
        # the full split's label distribution.
        df = df.groupby("label").sample(frac=limit / len(df), random_state=seed)
    return [preprocess(t) for t in df["text"].astype(str)], [LABEL2ID[label] for label in df["label"]]


def batches(tokenizer, texts, labels, batch_size, max_length, shuffle):
    def collate(items):
        enc = tokenizer([t for t, _ in items], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in items])
        return enc

    return DataLoader(list(zip(texts, labels)), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.inference_mode()
def predict(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    preds, gold = [], []
    for batch in loader:
        labels = batch.pop("labels")
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        preds += logits.argmax(-1).cpu().tolist()
        gold += labels.tolist()
    return gold, preds


def metrics(gold, preds) -> dict:
    return {
        "accuracy": round(accuracy_score(gold, preds), 4),
        "macro_f1": round(f1_score(gold, preds, average="macro"), 4),
        "per_class": classification_report(gold, preds, labels=range(len(LABELS)), target_names=LABELS,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, preds, labels=range(len(LABELS))).tolist(),
    }


def install(staging: Path, out: Path) -> None:
    """Copies files into the served directory instead of swapping the
    directory itself — a Docker bind mount pins the original directory, so
    replacing it would hide the new files from consumer_prediction. The
    consumer's settle window covers the copy being non-atomic."""
    out.mkdir(parents=True, exist_ok=True)
    for f in staging.iterdir():
        shutil.copy2(f, out / f.name)
    shutil.rmtree(staging)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data" / "astd"))
    p.add_argument("--out", default=str(ROOT / "models" / "arabert_sentiment"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="stratified subsample of train, for quick CPU runs")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data_dir)

    preprocess = ArabicPreprocessor()
    train_x, train_y = load_split(data / "train.csv", preprocess, args.max_train_samples, args.seed)
    val_x, val_y = load_split(data / "val.csv", preprocess, None, args.seed)
    test_x, test_y = load_split(data / "test.csv", preprocess, None, args.seed)
    print(f"device={device} train={len(train_x)} val={len(val_x)} test={len(test_x)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID).to(device)

    train_loader = batches(tokenizer, train_x, train_y, args.batch_size, args.max_length, shuffle=True)
    val_loader = batches(tokenizer, val_x, val_y, 64, args.max_length, shuffle=False)
    test_loader = batches(tokenizer, test_x, test_y, 64, args.max_length, shuffle=False)

    # Inverse-frequency class weights: ASTD is ~3/4 neutral once Objective is
    # folded in, and unweighted training mostly learns to say "neutral".
    counts = np.bincount(train_y, minlength=len(LABELS))
    weights = None if args.no_class_weights else torch.tensor(
        len(train_y) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    out = Path(args.out)
    staging = out.parent / (out.name + ".staging")
    best_f1, history = -1.0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}", flush=True)

        val = metrics(*predict(model, val_loader, device))
        history.append({"epoch": epoch, "train_loss": round(running / len(train_loader), 4),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        print(json.dumps(history[-1]))
        if val["macro_f1"] > best_f1:  # keep the best epoch by val macro-F1
            best_f1 = val["macro_f1"]
            if staging.exists():
                shutil.rmtree(staging)
            model.save_pretrained(staging)
            tokenizer.save_pretrained(staging)

    best = AutoModelForSequenceClassification.from_pretrained(staging).to(device)
    test_gold, test_pred = predict(best, test_loader, device)
    test = metrics(test_gold, test_pred)
    print(f"TEST accuracy={test['accuracy']} macro_f1={test['macro_f1']}")
    print(classification_report(test_gold, test_pred, target_names=LABELS, zero_division=0))

    stamp = datetime.now(timezone.utc)
    meta = {
        "model_version": f"arabertv02-astd-{stamp:%Y%m%d%H%M%S}",
        "base_model": MODEL_NAME,
        "labels": LABELS,
        "trained_at": stamp.isoformat(),
        "train_seconds": round(time.time() - started),
        "device": device,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data_dir", "out")},
        "train_size": len(train_x),
        "history": history,
        "test": test,
    }
    (staging / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    install(staging, out)
    print(f"saved {meta['model_version']} -> {out}")


if __name__ == "__main__":
    main()
