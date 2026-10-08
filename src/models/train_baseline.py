"""Train LightGBM with class weighting on the time-based train split,
evaluate on test. One row per user-day in, one JSON metrics file out.

Evaluation is built around the operating-decision framing from the methodology notes,
not just an offline score:
- PR-AUC (average_precision_score) — the primary offline metric, since
  ROC-AUC is misleadingly high under this much imbalance.
- precision@k and catch rate (recall) at a stated analyst-capacity
  budget: k alerts/day * (distinct days in the test period) = a total
  alert budget for the whole test window, then the top-scoring user-days
  up to that budget are "alerted". This directly answers "at this
  budget, what fraction of intrusions do we catch, and what fraction of
  alerts are real."

`user` and `date` are dropped before fitting — they are join keys only,
never model inputs (the methodology notes methodology #6).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

NON_FEATURE_COLS = {"user", "date", "label_event_day", "label_window", "scenario", "split"}


def load_split(features_path: str, target: str):
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train = train[feature_cols].astype(float)
    y_train = train[target].astype(int)
    X_test = test[feature_cols].astype(float)
    y_test = test[target].astype(int)

    return X_train, y_train, X_test, y_test, test["date"], test["user"], feature_cols


def precision_at_k_budgets(
    y_test: pd.Series, scores: np.ndarray, test_dates: pd.Series, test_users: pd.Series, k_list: list[int]
) -> dict:
    """Day-level catch rate alone flatters long campaigns: a model that
    just keeps re-flagging the same already-identified insider racks up
    day-level recall without finding any new intrusion. distinct_users_caught
    is tracked as a primary metric alongside it for exactly that reason —
    it's the number a security lead actually wants ("how many insiders did
    we find", not "how many of their days did we flag")."""
    test_days = test_dates.dt.normalize().nunique()
    n_pos = int(y_test.sum())
    order = np.argsort(-scores)
    y_sorted = y_test.to_numpy()[order]

    user_if_positive = test_users.where(y_test == 1)
    user_sorted = user_if_positive.to_numpy()[order]
    n_malicious_users = len(set(test_users[y_test == 1]))

    out = {"test_days": int(test_days), "n_positives_test": n_pos, "n_malicious_users_test": n_malicious_users}
    for k in k_list:
        budget = min(k * test_days, len(scores))
        caught = int(y_sorted[:budget].sum())
        users_caught = {u for u in user_sorted[:budget] if u is not None and u == u}
        out[str(k)] = {
            "alerts_per_day": k,
            "alert_budget_total": int(budget),
            "alerts_caught": caught,
            "precision": caught / budget if budget else float("nan"),
            "catch_rate": caught / n_pos if n_pos else float("nan"),
            "distinct_users_caught": len(users_caught),
            "distinct_users_total": n_malicious_users,
            "distinct_user_catch_rate": len(users_caught) / n_malicious_users if n_malicious_users else float("nan"),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--k-list", default="50,100,200")
    parser.add_argument("--tag", required=True, help="identifier for this run, stored in the metrics file")
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    k_list = [int(k) for k in args.k_list.split(",")]

    X_train, y_train, X_test, y_test, test_dates, test_users, feature_cols = load_split(args.features, args.target)

    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    natural_scale_pos_weight = n_neg / n_pos  # reported, but NOT used — see below

    # scale_pos_weight=1 (no reweighting), not the natural ~340:1 ratio.
    # Investigated and confirmed on a train-internal validation slice: with
    # only 565-722 positives, heavy reweighting made LightGBM chase individual
    # rows rather than learn generalizable rules — validation PR-AUC went
    # from 0.0290 at the natural weight to 0.9874 at weight=1, and degraded
    # further as features were added at any weight >1. PR-AUC only needs
    # good ranking, not calibrated probabilities, so no reweighting is
    # needed for this evaluation. See the methodology notes for the full investigation.
    scale_pos_weight = 1.0

    model_params = dict(
        objective="binary",
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        verbosity=-1,
    )
    clf = lgb.LGBMClassifier(**model_params)
    clf.fit(X_train, y_train)

    scores = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, scores)
    k_results = precision_at_k_budgets(y_test, scores, test_dates, test_users, k_list)

    importances = sorted(
        zip(feature_cols, clf.feature_importances_.tolist()), key=lambda t: -t[1]
    )

    metrics = {
        "tag": args.tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "n_train": len(X_train),
        "n_positives_train": n_pos,
        "n_test": len(X_test),
        "scale_pos_weight": scale_pos_weight,
        "natural_scale_pos_weight": natural_scale_pos_weight,
        "model_params": model_params,
        "feature_cols": feature_cols,
        "pr_auc": pr_auc,
        "precision_at_k": k_results,
        "feature_importance": importances,
    }

    out_path = Path(args.out_metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"=== {args.tag} ===", file=sys.stderr)
    print(f"train: {len(X_train)} rows, {n_pos} positive ({target_rate(n_pos, len(X_train))})", file=sys.stderr)
    print(f"test:  {len(X_test)} rows, {int(y_test.sum())} positive ({target_rate(int(y_test.sum()), len(X_test))})", file=sys.stderr)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}", file=sys.stderr)
    print(f"PR-AUC: {pr_auc:.4f}", file=sys.stderr)
    print(
        f"test days: {k_results['test_days']}, test positives: {k_results['n_positives_test']}, "
        f"malicious users in test: {k_results['n_malicious_users_test']}",
        file=sys.stderr,
    )
    for k in k_list:
        r = k_results[str(k)]
        print(
            f"  k={k}/day -> budget={r['alert_budget_total']} alerts | "
            f"day_catch={r['alerts_caught']}/{k_results['n_positives_test']} ({r['catch_rate']:.3f}) | "
            f"users_caught={r['distinct_users_caught']}/{r['distinct_users_total']} ({r['distinct_user_catch_rate']:.3f}) | "
            f"precision={r['precision']:.4f}",
            file=sys.stderr,
        )
    print("\ntop 10 features by importance:", file=sys.stderr)
    for name, imp in importances[:10]:
        print(f"  {name}: {imp}", file=sys.stderr)
    print(f"\nwrote metrics: {out_path}", file=sys.stderr)


def target_rate(pos, total):
    return f"{pos/total:.4%}" if total else "n/a"


if __name__ == "__main__":
    main()
