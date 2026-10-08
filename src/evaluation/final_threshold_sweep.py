"""Final threshold sweep for the writeup: catch rate vs. alert volume
across the full budget range, seed-swept (same 5-seed protocol as the
scale_pos_weight stability check and the realism stress test — 42, 1, 7,
123, 2024), on the final clean feature set (mode-deviation features,
clipped ratios, validation-tuned continuous z-scores, scale_pos_weight=1).

Two curves, both seed-swept with a mean line and a ±std band:
- Insiders caught (fraction of 18) — the primary decision metric. "Caught"
  means at least one of that insider's actual label_event_day==1 rows
  ranked within the alert budget; day-level catch rate alone flatters
  long campaigns (the methodology notes methodology — distinct-user tracking).
- Day-level catch rate (fraction of 244 positive user-days) — secondary,
  kept for comparability with earlier iteration-log entries.

k=1,2,5,10,20 alerts/day are marked on both curves; k=2/day (the chosen
operating point — see the methodology notes Realism Stress Test / damage-weighted
audit, where the signal saturates) is highlighted distinctly.
"""
import argparse
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
BAND = "#1f3a5f"
HIGHLIGHT = "#b45309"
K_MARKS = [1, 2, 5, 10, 20]
CHOSEN_K = 2


def run_one_seed(df, target, seed):
    X_train, y_train, X_test, y_test, test_dates, test_users, feature_cols = load_split_df(df, target)
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, num_leaves=15, learning_rate=0.05,
        min_child_samples=20, scale_pos_weight=1.0, random_state=seed, verbosity=-1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    test = df[df["split"] == "test"].reset_index(drop=True).copy()
    test["score"] = scores
    test_days = test["date"].dt.normalize().nunique()

    ranked = test.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    n_pos = int(ranked[target].sum())

    # day-level: cumulative count of positive rows within top-B
    y_sorted = ranked[target].to_numpy()
    day_cum_caught = np.cumsum(y_sorted)

    # insider-level: best (lowest) rank among each malicious user's positive rows
    pos_rows = ranked[ranked[target] == 1]
    best_rank_per_user = pos_rows.groupby("user")["rank"].min().sort_values().to_numpy()
    n_users = len(best_rank_per_user)

    max_budget = len(ranked)
    grid = sorted(set(
        list(range(1, 51)) + list(range(55, 500, 5)) + list(range(500, max_budget, 50)) + [max_budget]
    ))
    grid = np.array([b for b in grid if b <= max_budget])

    day_catch_rate = day_cum_caught[grid - 1] / n_pos
    insider_catch_rate = np.searchsorted(best_rank_per_user, grid, side="right") / n_users
    alerts_per_day = grid / test_days

    return alerts_per_day, day_catch_rate, insider_catch_rate, test_days, n_pos, n_users


def load_split_df(df, target):
    """Same as train_baseline.load_split but takes an already-loaded df,
    so it can be reused across seeds without re-reading the parquet."""
    from train_baseline import NON_FEATURE_COLS
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    X_train = train[feature_cols].astype(float)
    y_train = train[target].astype(int)
    X_test = test[feature_cols].astype(float)
    y_test = test[target].astype(int)
    return X_train, y_train, X_test, y_test, test["date"], test["user"], feature_cols


def interp_at(alerts_per_day, curve, k_list):
    return {k: float(np.interp(k, alerts_per_day, curve)) for k in k_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--seeds", default="42,1,7,123,2024")
    parser.add_argument("--out-fig", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])
    seeds = [int(s) for s in args.seeds.split(",")]

    # common grid (alerts/day is continuous and seed-independent in its
    # definition, test_days is fixed, so budgets map to the same alerts/day
    # grid across seeds — interpolate each seed's curve onto a shared grid)
    common_grid = np.unique(np.concatenate([
        np.linspace(0.05, 1, 20), np.linspace(1, 5, 40), np.linspace(5, 50, 46), np.linspace(50, 300, 26),
    ]))

    day_curves, insider_curves = [], []
    day_at_k, insider_at_k = [], []
    for seed in seeds:
        apd, day_rate, insider_rate, test_days, n_pos, n_users = run_one_seed(df, args.target, seed)
        day_curves.append(np.interp(common_grid, apd, day_rate))
        insider_curves.append(np.interp(common_grid, apd, insider_rate))
        day_at_k.append(interp_at(apd, day_rate, K_MARKS))
        insider_at_k.append(interp_at(apd, insider_rate, K_MARKS))
        print(f"seed={seed}: " + ", ".join(f"k={k}: day={day_at_k[-1][k]:.3f} insiders={insider_at_k[-1][k]:.3f}" for k in K_MARKS), file=sys.stderr)

    day_curves = np.array(day_curves)
    insider_curves = np.array(insider_curves)
    day_mean, day_std = day_curves.mean(axis=0), day_curves.std(axis=0)
    insider_mean, insider_std = insider_curves.mean(axis=0), insider_curves.std(axis=0)

    print(f"\n=== mean ± std at marked operating points (n={n_users} insiders, n_pos={n_pos}, test_days={test_days}) ===", file=sys.stderr)
    for k in K_MARKS:
        d_vals = np.array([d[k] for d in day_at_k])
        i_vals = np.array([d[k] for d in insider_at_k])
        marker = " <-- chosen" if k == CHOSEN_K else ""
        print(
            f"k={k:>3}/day: day_catch={d_vals.mean():.3f}±{d_vals.std():.3f}  "
            f"insiders_catch={i_vals.mean():.3f}±{i_vals.std():.3f}{marker}",
            file=sys.stderr,
        )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for ax, mean, std, ylabel in [
        (axes[0], insider_mean, insider_std, "Insiders caught (of 18)"),
        (axes[1], day_mean, day_std, "Day-level catch rate (of 244)"),
    ]:
        ax.plot(common_grid, mean, color=NAVY, linewidth=2, label="mean (5 seeds)")
        ax.fill_between(common_grid, mean - std, mean + std, color=BAND, alpha=0.18, label="±1 std")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

        for k in K_MARKS:
            y = np.interp(k, common_grid, mean)
            yerr = np.interp(k, common_grid, std)
            if k == CHOSEN_K:
                ax.errorbar([k], [y], yerr=[yerr], fmt="o", color=HIGHLIGHT, markersize=10,
                            capsize=5, zorder=5, label=f"chosen: k={k}/day")
            else:
                ax.errorbar([k], [y], yerr=[yerr], fmt="o", color=NAVY, markersize=6, capsize=4, zorder=4)

    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].set_title("Final threshold sweep: catch rate vs. alert volume (5-seed mean ± std)")
    axes[1].set_xlabel("Alerts per day (budget, log scale)")

    fig.tight_layout()
    fig_path = Path(args.out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote figure: {fig_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
