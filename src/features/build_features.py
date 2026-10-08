"""Build the user-day feature table: one row per (user, date), joined to labels.

Three feature families (see FEATURE_DESCRIPTIONS at the bottom for the
full list with one-line descriptions each):

1. Raw daily counts — straight per-table counts for that user-day, plus
   active_hours (the span between the first and last logged event across
   all five tables that day).
2. Per-user relative features — each raw count from family 1 expressed
   as a ratio and z-score against that user's own trailing 30-day
   mean/std (src/features/rolling.py). No-future-leakage is enforced
   there via closed="left" time windows — see that module's docstring.
3. Timing and novelty — off-hours/weekend flags, deviation of today's
   first-logon time from the user's own trailing baseline, first-ever
   use of a PC (built from a strictly-prior running set), and days since
   this user's last active day.

No identity features: `user` is kept only as a join/grouping key through
this pipeline. It — and raw `date` — must be dropped before anything is
fed to a model (the methodology notes methodology #6). PC names and email addresses
are consumed only to derive counts/flags; they never appear as columns
in the output.

Table aggregation results are cached under --cache-dir as parquet, since
re-scanning http.csv (14.5GB) and email.csv (1.36GB) on every run would
be wasteful. Pass --force to rebuild the cache.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# python -I doesn't add the script's own directory to sys.path, so the
# sibling imports below need it added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregates import (
    aggregate_device,
    aggregate_email,
    aggregate_file,
    aggregate_http,
    aggregate_logon,
)
from rolling import add_mode_deviation_features, add_ratio_features, add_zscore_features

RAW_COUNT_COLS = [
    "n_file_copies",
    "n_emails_sent",
    "n_emails_external",
    "n_usb_connects",
    "n_http_visits",
    "n_logons",
    "active_hours",
]

# Discrete integer counts: 76-94% of rows have an exactly-zero trailing std
# for these (see the methodology notes), so a mean/std z-score divides by near-zero
# constantly. Use mode-deviation instead (no division at all).
DISCRETE_COUNT_COLS = [
    "n_file_copies",
    "n_emails_sent",
    "n_emails_external",
    "n_usb_connects",
    "n_http_visits",
    "n_logons",
]

# Continuous (never exactly repeat day to day, so "mode" isn't meaningful):
# kept as a mean/std z-score, but std_floor/zscore_clip must be chosen on a
# validation slice of TRAIN (see src/models/tune_zscore_floor.py) and passed
# in as CLI args — never hardcoded here, never tuned against test.
CONTINUOUS_ZSCORE_COLS = ["active_hours", "first_logon_hour"]

FEATURE_DESCRIPTIONS = {
    # Family 1 — raw daily counts
    "n_file_copies": "Count of files copied to removable media that day.",
    "n_emails_sent": "Count of emails sent (as the acting user) that day.",
    "n_emails_external": "Count of those emails with >=1 recipient outside @dtaa.com.",
    "n_usb_connects": "Count of USB/removable-device connect events that day.",
    "n_http_visits": "Count of web page visits that day.",
    "n_logons": "Count of logon events that day (can be >1: multiple machines/sessions).",
    "active_hours": "Hours between the first and last logged event (any table) that day.",
    # Family 2 — per-user relative
    **{
        f"{c}_ratio_30d": f"{c} divided by this user's own trailing 30-day mean (prior days only)."
        for c in RAW_COUNT_COLS
    },
    **{
        f"{c}_diff_from_mode_30d": f"{c} minus this user's own trailing 30-day mode (most common prior value)."
        for c in DISCRETE_COUNT_COLS
    },
    **{
        f"{c}_differs_from_mode_30d": f"1 if {c} today differs at all from this user's trailing 30-day mode, else 0."
        for c in DISCRETE_COUNT_COLS
    },
    **{
        f"{c}_zscore_30d": f"{c} as a z-score against this user's own trailing 30-day mean/std (std floor+clip tuned on a train-internal validation slice)."
        for c in CONTINUOUS_ZSCORE_COLS
    },
    # Family 3 — timing & novelty
    "first_logon_hour": "Hour of day (fractional) of the first logon event, if any that day.",
    "last_logon_hour": "Hour of day (fractional) of the LAST logon event that day (catches a second, later session).",
    "is_after_hours": "First logon before 06:00 or after 20:00 (first-session-only; see any_session_after_hours).",
    "any_session_after_hours": "ANY logon that day (not just the first) is before 06:00 or after 20:00.",
    "is_weekend": "Date falls on Saturday or Sunday.",
    "is_new_pc_today": "Used at least one PC today never seen in this user's strictly-prior history.",
    "days_since_last_activity": "Calendar days since this user's previous active user-day (NaN on a user's first observed day).",
}


def load_or_build(name, builder, path, cache_dir: Path, force: bool) -> pd.DataFrame:
    cache_path = cache_dir / f"{name}.parquet"
    if cache_path.exists() and not force:
        print(f"  [cache hit] {name} <- {cache_path}", file=sys.stderr)
        return pd.read_parquet(cache_path)
    print(f"  [building] {name} <- {path}", file=sys.stderr)
    df = builder(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def build_new_pc_flags(logon_path: str) -> pd.DataFrame:
    """is_new_pc_today, enforced causally: a day's flag only compares against
    the set of PCs seen on that user's strictly-earlier days (the running
    `seen` set is updated AFTER a day's flag is computed, never before)."""
    logon = pd.read_csv(logon_path, usecols=["date", "user", "pc"])
    logon["date"] = pd.to_datetime(logon["date"], format="%m/%d/%Y %H:%M:%S").dt.normalize()

    daily_pcs = (
        logon.groupby(["user", "date"])["pc"]
        .agg(lambda s: frozenset(s.unique()))
        .reset_index(name="pcs_today")
        .sort_values(["user", "date"])
    )

    seen: dict[str, frozenset] = {}
    flags = np.empty(len(daily_pcs), dtype=bool)
    for i, row in enumerate(daily_pcs.itertuples(index=False)):
        prior = seen.get(row.user, frozenset())
        flags[i] = bool(row.pcs_today - prior)
        seen[row.user] = prior | row.pcs_today

    daily_pcs["is_new_pc_today"] = flags
    return daily_pcs[["user", "date", "is_new_pc_today"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="path to data/raw")
    parser.add_argument("--release", default="r4.2")
    parser.add_argument("--labels", required=True, help="path to the split label parquet")
    parser.add_argument("--cache-dir", default="data/interim/table_aggregates")
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true", help="rebuild cached table aggregates")
    parser.add_argument(
        "--std-floor", type=float, required=True,
        help="std floor for active_hours/first_logon_hour zscore — choose via tune_zscore_floor.py on a train-internal validation slice, never on test",
    )
    parser.add_argument("--zscore-clip", type=float, default=10.0)
    args = parser.parse_args()

    release_dir = Path(args.data_root) / args.release
    cache_dir = Path(args.cache_dir)

    labels = pd.read_parquet(args.labels)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()

    print("Aggregating source tables (cached after first run):", file=sys.stderr)
    logon_agg = load_or_build("logon", aggregate_logon, release_dir / "logon.csv", cache_dir, args.force)
    device_agg = load_or_build("device", aggregate_device, release_dir / "device.csv", cache_dir, args.force)
    file_agg = load_or_build("file", aggregate_file, release_dir / "file.csv", cache_dir, args.force)
    email_agg = load_or_build("email", aggregate_email, release_dir / "email.csv", cache_dir, args.force)
    http_agg = load_or_build("http", aggregate_http, release_dir / "http.csv", cache_dir, args.force)
    pc_flags = load_or_build("new_pc_flags", build_new_pc_flags, release_dir / "logon.csv", cache_dir, args.force)

    df = labels.copy()

    count_fill = {}
    for name, agg, cols in [
        ("logon", logon_agg, ["n_logons"]),
        ("device", device_agg, ["n_usb_connects"]),
        ("file", file_agg, ["n_file_copies"]),
        ("email", email_agg, ["n_emails_sent", "n_emails_external"]),
        ("http", http_agg, ["n_http_visits"]),
    ]:
        keep = ["user", "date"] + cols + ["min_ts", "max_ts"]
        renamed = agg[keep].rename(columns={"min_ts": f"{name}_min_ts", "max_ts": f"{name}_max_ts"})
        df = df.merge(renamed, on=["user", "date"], how="left")
        for c in cols:
            count_fill[c] = 0

    df = df.merge(
        logon_agg[["user", "date", "first_logon_ts", "last_logon_ts", "any_after_hours_logon"]],
        on=["user", "date"],
        how="left",
    )
    df = df.merge(pc_flags, on=["user", "date"], how="left")

    for c, fill in count_fill.items():
        df[c] = df[c].fillna(fill)
    df["is_new_pc_today"] = df["is_new_pc_today"].fillna(False)
    df["any_after_hours_logon"] = df["any_after_hours_logon"].astype("boolean").fillna(False).astype(bool)

    # active_hours: span across whichever tables had any activity that day.
    min_cols = [f"{n}_min_ts" for n in ["logon", "device", "file", "email", "http"]]
    max_cols = [f"{n}_max_ts" for n in ["logon", "device", "file", "email", "http"]]
    overall_min = df[min_cols].min(axis=1)
    overall_max = df[max_cols].max(axis=1)
    df["active_hours"] = (overall_max - overall_min).dt.total_seconds() / 3600.0

    # first_logon_hour / last_logon_hour: fractional hour of day of the first/last
    # 'Logon' event, if any.
    df["first_logon_hour"] = (
        df["first_logon_ts"].dt.hour
        + df["first_logon_ts"].dt.minute / 60.0
        + df["first_logon_ts"].dt.second / 3600.0
    )
    df["last_logon_hour"] = (
        df["last_logon_ts"].dt.hour
        + df["last_logon_ts"].dt.minute / 60.0
        + df["last_logon_ts"].dt.second / 3600.0
    )

    df["is_weekend"] = df["date"].dt.weekday >= 5
    # is_after_hours: first-logon-only (kept as-is, see known gap in the methodology notes).
    df["is_after_hours"] = (df["first_logon_hour"] < 6.0) | (df["first_logon_hour"] > 20.0)
    # any_session_after_hours: fix — checks every Logon event that day, not just
    # the first, so a normal morning login followed by an after-hours second
    # session (e.g. BIH0745) is caught.
    df["any_session_after_hours"] = df["any_after_hours_logon"]

    df = df.sort_values(["user", "date"]).reset_index(drop=True)
    df["days_since_last_activity"] = (
        df.groupby("user")["date"].diff().dt.days
    )

    # Family 2a: trailing-30-day ratio for all 7 raw counts (mean-based, unaffected
    # by the near-zero-std issue below — not touched by that investigation).
    df = add_ratio_features(df, RAW_COUNT_COLS, window="30D", min_periods=5)
    # Family 2b: mode-deviation for the 6 discrete counts — no division anywhere.
    df = add_mode_deviation_features(df, DISCRETE_COUNT_COLS, window="30D", min_periods=5)
    # Family 2c: mean/std zscore for the 2 continuous features only, with a
    # std floor/clip chosen externally on a train-internal validation slice.
    df = add_zscore_features(
        df, CONTINUOUS_ZSCORE_COLS, std_floor=args.std_floor, zscore_clip=args.zscore_clip,
        window="30D", min_periods=5,
    )

    drop_ts_cols = min_cols + max_cols + ["first_logon_ts", "last_logon_ts", "any_after_hours_logon"]
    df = df.drop(columns=drop_ts_cols)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    feature_cols = [c for c in df.columns if c not in ("user", "date", "split", "label_event_day", "label_window", "scenario")]
    print(f"\nwrote feature table: {out_path}", file=sys.stderr)
    print(f"rows: {len(df)}  feature columns: {len(feature_cols)}", file=sys.stderr)
    print("\nFeature list:", file=sys.stderr)
    for c in feature_cols:
        desc = FEATURE_DESCRIPTIONS.get(c, "(no description registered)")
        print(f"  {c}: {desc}", file=sys.stderr)


if __name__ == "__main__":
    main()
