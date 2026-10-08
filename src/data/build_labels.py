"""Build the user-day label table for CERT r4.2.

Unit of analysis is one user-day (the methodology notes methodology #5): a row exists
for every (user, calendar_date) pair where that user had at least one
logon event — logon.csv is the base, since "logons precede other PC
activity" per the dataset readme, so a day with any activity always has a
logon row for that user.

Two label definitions are kept side by side, because they measure
different things and published CERT benchmarks mostly use the window
convention:

- label_event_day: malicious only on a day where the insider has an
  actual logged malicious act in their detail observables file
  (answers/r4.2-<scenario>/<details>). Stricter, less label noise. This
  is the PRIMARY label for training/evaluation.
- label_window: malicious on every calendar day inside the insider's
  full [start, end] campaign window from insiders.csv, regardless of
  whether an act was logged that specific day. Kept for reporting
  against published benchmarks, which mostly use this convention.

label_event_day is always a subset of label_window for a given insider
(event timestamps fall inside [start, end] by construction).

Detail files are NOT purely the insider's own rows — e.g. an email
scenario's detail file includes the counterpart's reply emails too. Rows
are filtered to the `user` field (column index 3, constant across all
row types: type,id,date,user,pc,...) matching the actual insider's
username before extracting event dates, so correspondents don't get
mislabeled as malicious.

No identity features are emitted here — this table is label-only
(user, date, label_event_day, label_window, scenario). Per-user identity
stays as a join key for building features later, never as a model input
itself (the methodology notes #6).
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def load_insiders(answers_dir: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(answers_dir / "insiders.csv", dtype=str)
    df = df[df["dataset"] == dataset].copy()
    df["start"] = pd.to_datetime(df["start"], format="%m/%d/%Y %H:%M:%S")
    df["end"] = pd.to_datetime(df["end"], format="%m/%d/%Y %H:%M:%S")
    df["scenario"] = df["scenario"].astype(int)
    return df[["scenario", "details", "user", "start", "end"]]


def load_active_user_days(release_dir: Path) -> pd.DataFrame:
    logon = pd.read_csv(
        release_dir / "logon.csv",
        usecols=["date", "user"],
        dtype={"user": "category"},
    )
    logon["date"] = pd.to_datetime(logon["date"], format="%m/%d/%Y %H:%M:%S").dt.date
    active = logon.drop_duplicates(subset=["user", "date"])[["user", "date"]]
    return active.reset_index(drop=True)


def event_days_for_insider(answers_dir: Path, scenario: int, details: str, user: str) -> set:
    detail_path = answers_dir / f"r4.2-{scenario}" / details
    dates = set()
    with open(detail_path, newline="") as f:
        for row in csv.reader(f):
            if row[3] == user:
                dates.add(pd.to_datetime(row[2], format="%m/%d/%Y %H:%M:%S").date())
    return dates


def build_label_table(active: pd.DataFrame, insiders: pd.DataFrame, answers_dir: Path) -> pd.DataFrame:
    active = active.copy()
    active["label_event_day"] = 0
    active["label_window"] = 0
    active["scenario"] = pd.NA

    active = active.set_index(["user", "date"])

    for row in insiders.itertuples(index=False):
        window_dates = pd.date_range(row.start.normalize(), row.end.normalize(), freq="D").date
        window_idx = [(row.user, d) for d in window_dates if (row.user, d) in active.index]
        if window_idx:
            active.loc[window_idx, "label_window"] = 1
            active.loc[window_idx, "scenario"] = row.scenario

        event_dates = event_days_for_insider(answers_dir, row.scenario, row.details, row.user)
        event_idx = [(row.user, d) for d in event_dates if (row.user, d) in active.index]
        if event_idx:
            active.loc[event_idx, "label_event_day"] = 1
            active.loc[event_idx, "scenario"] = row.scenario

    active = active.reset_index()
    return active


def report(labels: pd.DataFrame, label_col: str, title: str):
    total = len(labels)
    malicious = int(labels[label_col].sum())
    distinct_users = labels.loc[labels[label_col] == 1, "user"].nunique()

    print(f"--- {title} ({label_col}) ---", file=sys.stderr)
    print(f"total user-days: {total}", file=sys.stderr)
    print(f"malicious user-days: {malicious}", file=sys.stderr)
    print(f"positive rate: {malicious / total:.6f}", file=sys.stderr)
    print(f"distinct malicious users: {distinct_users}", file=sys.stderr)

    by_scenario_users = (
        labels.loc[labels[label_col] == 1]
        .groupby("scenario")["user"]
        .nunique()
        .rename("distinct_malicious_users")
    )
    by_scenario_days = (
        labels.loc[labels[label_col] == 1]
        .groupby("scenario")
        .size()
        .rename("malicious_user_days")
    )
    print("by scenario (users):", file=sys.stderr)
    print(by_scenario_users.to_string(), file=sys.stderr)
    print("by scenario (user-days):", file=sys.stderr)
    print(by_scenario_days.to_string(), file=sys.stderr)
    print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="path to data/raw")
    parser.add_argument("--release", default="r4.2")
    parser.add_argument("--dataset-tag", default="4.2")
    parser.add_argument("--out", required=True, help="output parquet path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    release_dir = data_root / args.release
    answers_dir = data_root / "answers"

    insiders = load_insiders(answers_dir, args.dataset_tag)
    active = load_active_user_days(release_dir)
    labels = build_label_table(active, insiders, answers_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_path, index=False)

    report(labels, "label_event_day", "event-day labeling (primary)")
    report(labels, "label_window", "window labeling (benchmark comparison)")

    print(f"wrote label table: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
