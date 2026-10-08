"""Realism stress test: how much of the 0.9840 test PR-AUC depends on
negative user-days being almost perfectly constant (a simulation artifact
— see the methodology notes methodology #7), rather than genuine behavioral signal?

Injects integer jitter into the 6 discrete raw counts on NEGATIVE
(label_event_day==0) user-days only — malicious days are left untouched,
since the point is to ask "if normal users varied day to day the way real
ones would, would the model still work," not to make the attacks harder
to see. All downstream derived features (ratio_30d, diff_from_mode_30d,
differs_from_mode_30d) are recomputed from the jittered counts — jittering
the raw column alone without recomputing its baselines would leave the
test comparing a noisy raw count against a baseline computed from the
clean history, which isn't the question being asked.

This is a FINAL robustness characterization of the already-chosen model,
not a hyperparameter search — there is nothing left to choose based on
the result, so unlike the earlier debugging investigation, it is run
through to test. The scale_pos_weight=1 finding and feature set are
fixed inputs here, not being re-tuned.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from rolling import add_mode_deviation_features, add_ratio_features
from train_baseline import NON_FEATURE_COLS

DISCRETE_COUNT_COLS = [
    "n_file_copies", "n_emails_sent", "n_emails_external",
    "n_usb_connects", "n_http_visits", "n_logons",
]


def jitter_negatives(df: pd.DataFrame, jitter: int, target: str, seed: int) -> pd.DataFrame:
    """Add uniform integer jitter in [-jitter, jitter] to the 6 discrete
    counts, on negative rows only, clipped at 0. Then recompute ratio_30d
    and mode-deviation features from the jittered counts — everything
    else (timing features, continuous zscore, labels) is untouched."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    if jitter > 0:
        neg_mask = out[target] == 0
        for col in DISCRETE_COUNT_COLS:
            noise = rng.integers(-jitter, jitter + 1, size=neg_mask.sum())
            out.loc[neg_mask, col] = (out.loc[neg_mask, col] + noise).clip(lower=0)

    # drop the old derived columns, recompute from (possibly jittered) raw counts
    old_derived = [c for c in out.columns if c.endswith("_ratio_30d") and c.replace("_ratio_30d", "") in DISCRETE_COUNT_COLS]
    old_derived += [c for c in out.columns if "_mode_30d" in c]
    out = out.drop(columns=old_derived)
    out = add_ratio_features(out, DISCRETE_COUNT_COLS, window="30D", min_periods=5)
    out = add_mode_deviation_features(out, DISCRETE_COUNT_COLS, window="30D", min_periods=5)
    return out


def fit_eval(df: pd.DataFrame, target: str, seed: int = 42, exclude_cols: list[str] = None):
    exclude_cols = exclude_cols or []
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c not in exclude_cols]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train, y_train = train[feature_cols].astype(float), train[target].astype(int)
    X_test, y_test = test[feature_cols].astype(float), test[target].astype(int)

    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, num_leaves=15, learning_rate=0.05,
        min_child_samples=20, scale_pos_weight=1.0, random_state=seed, verbosity=-1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, scores)

    test_df = test.copy()
    test_df["score"] = scores
    test_days = test_df["date"].dt.normalize().nunique()
    budget = 2 * test_days
    ranked = test_df.sort_values("score", ascending=False)
    alerted_users = set(ranked.head(budget)["user"])
    malicious_users = set(test_df.loc[test_df[target] == 1, "user"])
    insiders_caught = len(malicious_users & alerted_users)
    return pr_auc, insiders_caught, len(malicious_users)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--jitter-levels", default="0,1,2,3")
    parser.add_argument("--seeds", default="42,1,7,123,2024")
    parser.add_argument("--out-fig", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])

    jitter_levels = [int(j) for j in args.jitter_levels.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    # Each seed drives BOTH the jitter draw and the model's random_state, so
    # a "run" is a single reproducible (noise realization, model fit) pair —
    # this is what "run-to-run variance" means here, not just model variance
    # on fixed noise. Same seed list used for the scale_pos_weight stability
    # check earlier, for consistency.
    summary = []
    for j in jitter_levels:
        pr_aucs, catches = [], []
        for seed in seeds:
            jittered = jitter_negatives(df, j, args.target, seed=seed)
            pr_auc, caught, total = fit_eval(jittered, args.target, seed=seed)
            pr_aucs.append(pr_auc)
            catches.append(caught)
        pr_aucs, catches = np.array(pr_aucs), np.array(catches)
        summary.append((j, pr_aucs, catches, total))
        print(
            f"jitter=±{j}: PR-AUC mean={pr_aucs.mean():.4f} std={pr_aucs.std():.4f} "
            f"[{', '.join(f'{v:.4f}' for v in pr_aucs)}]  "
            f"insiders_caught mean={catches.mean():.2f} std={catches.std():.2f} "
            f"[{', '.join(str(v) for v in catches)}] (of {total})",
            file=sys.stderr,
        )

    # without differs_from_mode / diff_from_mode at all, on the CLEAN (jitter=0) data
    differs_diff_cols = [c for c in df.columns if "_mode_30d" in c]
    no_mode_pr_aucs, no_mode_catches = [], []
    for seed in seeds:
        pr_auc, caught, total_nm = fit_eval(df, args.target, seed=seed, exclude_cols=differs_diff_cols)
        no_mode_pr_aucs.append(pr_auc)
        no_mode_catches.append(caught)
    no_mode_pr_aucs, no_mode_catches = np.array(no_mode_pr_aucs), np.array(no_mode_catches)
    print(
        f"\nwithout any mode-deviation features (clean data): "
        f"PR-AUC mean={no_mode_pr_aucs.mean():.4f} std={no_mode_pr_aucs.std():.4f}  "
        f"insiders_caught mean={no_mode_catches.mean():.2f} std={no_mode_catches.std():.2f} (of {total_nm})",
        file=sys.stderr,
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    js = [r[0] for r in summary]
    pr_means = [r[1].mean() for r in summary]
    pr_stds = [r[1].std() for r in summary]
    catch_means = [r[2].mean() for r in summary]
    catch_stds = [r[2].std() for r in summary]

    axes[0].errorbar(js, pr_means, yerr=pr_stds, marker="o", color="#1f3a5f", linewidth=2, capsize=4)
    axes[0].axhline(no_mode_pr_aucs.mean(), color="#b45309", linestyle="--", linewidth=1.5, label="no mode-deviation features (jitter=0)")
    axes[0].set_ylabel("Test PR-AUC (mean ± std, 5 seeds)")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Realism stress test: negative-day count jitter (5-seed sweep)")

    axes[1].errorbar(js, catch_means, yerr=catch_stds, marker="o", color="#1f3a5f", linewidth=2, capsize=4)
    axes[1].axhline(no_mode_catches.mean(), color="#b45309", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("Insiders caught @ 2/day (mean ± std, of 18)")
    axes[1].set_xlabel("Jitter magnitude on negative-day counts (±N)")
    axes[1].set_ylim(0, 19)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig_path = Path(args.out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote figure: {fig_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
