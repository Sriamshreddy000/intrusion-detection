"""Apply the time-based train/test split (the methodology notes methodology #1).

Never a random shuffle: everything before `train_end` is train, everything
on/after it is test. The cutoff lives in configs/split.yaml so it's a
tracked decision, not a magic number buried in code.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="path to user-day label parquet")
    parser.add_argument("--split-config", required=True, help="path to configs/split.yaml")
    parser.add_argument("--out", required=True, help="output parquet path (adds a 'split' column)")
    args = parser.parse_args()

    with open(args.split_config) as f:
        cfg = yaml.safe_load(f)
    train_end = pd.Timestamp(cfg["train_end"])

    labels = pd.read_parquet(args.labels)
    labels["date"] = pd.to_datetime(labels["date"])
    labels["split"] = pd.Series(
        pd.Categorical(
            ["train" if d < train_end else "test" for d in labels["date"]],
            categories=["train", "test"],
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_path, index=False)

    for split_name in ["train", "test"]:
        subset = labels[labels["split"] == split_name]
        print(f"--- {split_name} ---", file=sys.stderr)
        print(f"date range: {subset['date'].min().date()} to {subset['date'].max().date()}", file=sys.stderr)
        print(f"user-days: {len(subset)}", file=sys.stderr)
        for col in ["label_event_day", "label_window"]:
            pos = int(subset[col].sum())
            print(
                f"  {col}: positives={pos}  rate={pos / len(subset):.6f}  "
                f"distinct_malicious_users={subset.loc[subset[col] == 1, 'user'].nunique()}",
                file=sys.stderr,
            )
        print(file=sys.stderr)

    train_users = set(labels.loc[(labels["split"] == "train") & (labels["label_event_day"] == 1), "user"])
    test_users = set(labels.loc[(labels["split"] == "test") & (labels["label_event_day"] == 1), "user"])
    print(f"malicious users (event-day) in train only: {len(train_users - test_users)}", file=sys.stderr)
    print(f"malicious users (event-day) in test only: {len(test_users - train_users)}", file=sys.stderr)
    print(f"malicious users (event-day) in both (straddle the cutoff): {len(train_users & test_users)}", file=sys.stderr)

    print(f"wrote split label table: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
