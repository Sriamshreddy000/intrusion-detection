"""Full catch-rate-vs-alert-volume sweep, tightened operating points, and
per-scenario catch rate breakdown.

The earlier k=50/100/200 alerts/day table was too loose to be informative
(50/day over 137 test days = 6,850 alerts chasing 244 positives, ~8% of
all test user-days flagged) — catch rate was already flat across that
whole range. This sweeps a much finer, tighter grid so the actual bend
in the curve is visible, and breaks catch rate down by scenario, since
an aggregate number can hide one scenario being caught near-perfectly
while others are missed almost entirely.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from train_baseline import load_split

NAVY = "#1f3a5f"
MUTED = "#6b7280"


def fit_model(X_train, y_train, seed=42):
    # scale_pos_weight=1: heavy reweighting destabilizes LightGBM on this few
    # positives — see train_baseline.py and the methodology notes for the investigation.
    scale_pos_weight = 1.0
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)
    return clf, scale_pos_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--out-fig", required=True)
    parser.add_argument("--k-list", default="1,2,5,10,20,50,100,200")
    args = parser.parse_args()

    k_list = [int(k) for k in args.k_list.split(",")]

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])
    X_train, y_train, X_test, y_test, test_dates, test_users, feature_cols = load_split(args.features, args.target)
    clf, scale_pos_weight = fit_model(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    test_days = test_dates.dt.normalize().nunique()
    n_pos = int(y_test.sum())
    order = np.argsort(-scores)
    y_sorted = y_test.to_numpy()[order]

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    # scenario is non-null for any label_window==1 row, which is a superset of
    # label_event_day==1 rows — mask it to NaN wherever this isn't an actual
    # target-label positive, so scenario-breakdown counts can't exceed totals.
    scenario_if_positive = test_df["scenario"].where(test_df[args.target] == 1)
    scenario_sorted = scenario_if_positive.to_numpy()[order]
    scenario_totals = scenario_if_positive.value_counts().to_dict()

    # user is only non-null-meaningful here for positive rows — same masking
    # logic, so "distinct users caught" can't pick up a negative row's user.
    user_if_positive = test_df["user"].where(test_df[args.target] == 1)
    user_sorted = user_if_positive.to_numpy()[order]
    all_malicious_users = set(test_df.loc[test_df[args.target] == 1, "user"])
    n_malicious_users = len(all_malicious_users)

    # --- fine grid for the full curve ---
    max_budget = len(scores)
    grid_budgets = sorted(set(
        list(range(1, 51)) + list(range(55, 500, 5)) + list(range(500, max_budget, 50)) + [max_budget]
    ))
    grid_budgets = [b for b in grid_budgets if b <= max_budget]
    catch_rates, precisions, alerts_per_day = [], [], []
    for b in grid_budgets:
        caught = int(y_sorted[:b].sum())
        catch_rates.append(caught / n_pos)
        precisions.append(caught / b)
        alerts_per_day.append(b / test_days)

    # --- tightened operating points ---
    k_results = {
        "test_days": int(test_days),
        "n_positives_test": n_pos,
        "n_malicious_users_test": n_malicious_users,
        "scale_pos_weight": scale_pos_weight,
    }
    for k in k_list:
        budget = min(k * test_days, max_budget)
        caught = int(y_sorted[:budget].sum())
        by_scenario = {}
        for sc in [1.0, 2.0, 3.0]:
            sc_total = scenario_totals.get(sc, 0)
            sc_caught = int(((scenario_sorted[:budget] == sc)).sum())
            by_scenario[str(sc)] = {
                "total": int(sc_total),
                "caught": sc_caught,
                "catch_rate": sc_caught / sc_total if sc_total else None,
            }
        users_caught = {u for u in user_sorted[:budget] if u is not None and u == u}  # drop None/NaN
        k_results[str(k)] = {
            "alerts_per_day": k,
            "alert_budget_total": int(budget),
            "alerts_caught": caught,
            "precision": caught / budget if budget else float("nan"),
            "catch_rate": caught / n_pos if n_pos else float("nan"),
            "distinct_users_caught": len(users_caught),
            "distinct_users_total": n_malicious_users,
            "distinct_user_catch_rate": len(users_caught) / n_malicious_users if n_malicious_users else float("nan"),
            "by_scenario": by_scenario,
        }

    out_path = Path(args.out_metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, **k_results}, f, indent=2)

    print(f"=== {args.tag}: tightened operating points ===", file=sys.stderr)
    print(f"test_days={test_days}  n_positives_test={n_pos}  n_malicious_users_test={n_malicious_users}", file=sys.stderr)
    for k in k_list:
        r = k_results[str(k)]
        sc_str = ", ".join(
            f"s{sc}: {d['caught']}/{d['total']}" + (f" ({d['catch_rate']:.2f})" if d["catch_rate"] is not None else "")
            for sc, d in r["by_scenario"].items()
        )
        print(
            f"  k={k:>4}/day budget={r['alert_budget_total']:>6} "
            f"day_catch={r['alerts_caught']:>4}/{n_pos} ({r['catch_rate']:.3f})  "
            f"users_caught={r['distinct_users_caught']:>2}/{n_malicious_users} ({r['distinct_user_catch_rate']:.3f})  "
            f"precision={r['precision']:.4f}  [{sc_str}]",
            file=sys.stderr,
        )

    # --- plot: two single-axis panels, no dual-axis ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    ax.plot(alerts_per_day, catch_rates, color=NAVY, linewidth=2)
    ax.set_ylabel("Catch rate (recall)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{args.tag}: catch rate vs. alert volume")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    ax2 = axes[1]
    ax2.plot(alerts_per_day, precisions, color=MUTED, linewidth=2)
    ax2.set_ylabel("Precision")
    ax2.set_xlabel("Alerts per day (budget)")
    ax2.set_xscale("log")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    for k in k_list:
        axes[0].axvline(k, color="#d1d5db", linewidth=0.8, linestyle="--", zorder=0)

    fig.tight_layout()
    fig_path = Path(args.out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote figure: {fig_path}", file=sys.stderr)
    print(f"wrote metrics: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
