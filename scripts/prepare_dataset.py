"""Downloads ASTD (Arabic Sentiment Tweets Dataset, Nabil et al. 2015) from
the Hugging Face Hub and writes stratified train/val/test splits.

    python scripts/prepare_dataset.py            # -> data/astd/{train,val,test}.csv, data/replay.csv

ASTD has four labels; the pipeline is 3-class, so "Objective" (factual,
no opinion) is folded into neutral by default — --objective drop removes
those rows instead. The test split is also written to data/replay.csv, which
the gateway replays: the pipeline is evaluated on tweets the model never saw.
data/spike.csv holds just the test negatives, for demonstrating alerting.
"""
import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ASTD_URL = "https://huggingface.co/datasets/arbml/ASTD/resolve/main/data/train-00000-of-00001.parquet"
# Class ids as declared in the dataset card's ClassLabel feature.
ASTD_LABELS = {0: "neutral", 1: "objective", 2: "positive", 3: "negative"}
LABELS = ["negative", "neutral", "positive"]


def map_labels(df: pd.DataFrame, objective: str) -> pd.DataFrame:
    out = pd.DataFrame({"text": df["tweet"].astype(str).str.strip(), "label": df["label"].map(ASTD_LABELS)})
    if objective == "drop":
        out = out[out["label"] != "objective"]
    else:
        out.loc[out["label"] == "objective", "label"] = "neutral"
    out = out[out["text"] != ""].drop_duplicates(subset="text")
    assert set(out["label"]) <= set(LABELS), set(out["label"])
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, rest = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=2 / 3, stratify=rest["label"], random_state=seed)
    return train, val, test  # 70 / 10 / 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--objective", choices=["neutral", "drop"], default="neutral")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with urllib.request.urlopen(ASTD_URL, timeout=120) as resp:
        raw = pd.read_parquet(io.BytesIO(resp.read()))
    df = map_labels(raw, args.objective)
    df.insert(0, "id", [f"astd-{i}" for i in range(len(df))])

    out = Path(args.out)
    (out / "astd").mkdir(parents=True, exist_ok=True)
    train, val, test = split(df, args.seed)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / "astd" / f"{name}.csv", index=False, encoding="utf-8")
        print(f"{name:5s} {len(part):5d}  {part['label'].value_counts().to_dict()}")
    test.to_csv(out / "replay.csv", index=False, encoding="utf-8")
    # Negative-only slice of the same test split: replaying it after (or
    # during) a normal replay drives the negative share up, which is how the
    # alerting path is demonstrated — ASTD's natural ~17% never crosses 0.4.
    spike = test[test["label"] == "negative"]
    spike.to_csv(out / "spike.csv", index=False, encoding="utf-8")
    print(f"replay file: {out / 'replay.csv'} | spike file ({len(spike)} negatives): {out / 'spike.csv'}")


if __name__ == "__main__":
    main()
