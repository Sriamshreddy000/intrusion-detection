"""Honestly tune std_floor for the active_hours/first_logon_hour z-score.

TEST IS NEVER TOUCHED HERE. A validation slice is carved out of the END
of TRAIN (by default the last 2 months of 2010, right before the
2011-01-01 train/test cutoff), and the model is fit on the remaining,
earlier part of train. This mirrors the project's own time-based-split
rule one level down: tune on an earlier/later split within train, same
as train/test is split within the whole dataset.

This exists because the mean/std z-score for discrete count features
was found to be numerically unstable (see the methodology notes) and was replaced
with mode-deviation instead (no tunable constant, no division). The two
continuous features (active_hours, first_logon_hour) don't have a
meaningful "mode" (they're floats that rarely repeat exactly), so they
keep a z-score — but its std_floor must be chosen without ever looking
at test, which is what this script does.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from rolling import add_zscore_features
from train_baseline import NON_FEATURE_COLS


def fit_eval(train_df, val_df, feature_cols, target, seed=42):
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[target].astype(int)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df[target].astype(int)

    # scale_pos_weight=1: heavy reweighting destabilizes LightGBM on this few
    # positives — see train_baseline.py and the methodology notes for the investigation.
    scale_pos_weight = 1.0
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, num_leaves=15, learning_rate=0.05,
        min_child_samples=20, scale_pos_weight=scale_pos_weight, random_state=seed, verbosity=-1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="an already-built features.parquet (any placeholder std-floor)")
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--val-start", default="2010-11-01")
    parser.add_argument("--val-end", default="2011-01-01")
    parser.add_argument("--floor-candidates", default="0.05,0.1,0.2,0.3,0.5,1.0,2.0")
    parser.add_argument("--zscore-clip", type=float, default=10.0)
    # after picking, rebuild the real features.parquet with the winner:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--release", default="r4.2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--cache-dir", default="data/interim/table_aggregates")
    parser.add_argument("--final-out", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])
    continuous_cols = ["active_hours", "first_logon_hour"]
    base = df.drop(columns=[f"{c}_zscore_30d" for c in continuous_cols])

    train_mask = df["split"] == "train"
    tuning_train_mask = train_mask & (df["date"] < args.val_start)
    val_mask = train_mask & (df["date"] >= args.val_start) & (df["date"] < args.val_end)
    print(
        f"tuning_train: {tuning_train_mask.sum()} rows ({df.loc[tuning_train_mask, args.target].sum()} positive)  "
        f"validation: {val_mask.sum()} rows ({df.loc[val_mask, args.target].sum()} positive)",
        file=sys.stderr,
    )

    floor_candidates = [float(f) for f in args.floor_candidates.split(",")]
    results = []
    for floor in floor_candidates:
        with_z = add_zscore_features(base, continuous_cols, std_floor=floor, zscore_clip=args.zscore_clip)
        feature_cols = [c for c in with_z.columns if c not in NON_FEATURE_COLS]
        train_df = with_z[tuning_train_mask]
        val_df = with_z[val_mask]
        pr_auc = fit_eval(train_df, val_df, feature_cols, args.target)
        results.append((floor, pr_auc))
        print(f"  std_floor={floor:<6} validation PR-AUC={pr_auc:.4f}", file=sys.stderr)

    best_floor, best_pr_auc = max(results, key=lambda t: t[1])
    print(f"\nchosen std_floor={best_floor} (validation PR-AUC={best_pr_auc:.4f}), never evaluated against test", file=sys.stderr)

    print(f"\nrebuilding final features.parquet with std_floor={best_floor} ...", file=sys.stderr)
    subprocess.run(
        [
            sys.executable, "-I", str(Path(__file__).resolve().parent.parent / "features" / "build_features.py"),
            "--data-root", args.data_root, "--release", args.release,
            "--labels", args.labels, "--cache-dir", args.cache_dir,
            "--out", args.final_out, "--std-floor", str(best_floor), "--zscore-clip", str(args.zscore_clip),
        ],
        check=True,
    )
    print(f"wrote final features: {args.final_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
