"""Single-feature ablation: train the same model config on ONE feature at
a time, report test PR-AUC for each. If any single feature approaches the
full-model PR-AUC on its own, that's a sign the model is reading one
obvious tell rather than combining weak behavioral signals — worth
knowing before trusting the aggregate number.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
from sklearn.metrics import average_precision_score

from train_baseline import load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, test_dates, test_users, feature_cols = load_split(args.features, args.target)
    # scale_pos_weight=1: heavy reweighting destabilizes LightGBM on this few
    # positives — see train_baseline.py and the methodology notes for the investigation.
    scale_pos_weight = 1.0

    results = []
    for col in feature_cols:
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            verbosity=-1,
        )
        clf.fit(X_train[[col]], y_train)
        scores = clf.predict_proba(X_test[[col]])[:, 1]
        pr_auc = average_precision_score(y_test, scores)
        results.append((col, pr_auc))

    results.sort(key=lambda t: -t[1])
    print(f"{'feature':<32} {'test PR-AUC (single feature)':>30}", file=sys.stderr)
    for col, pr_auc in results:
        print(f"{col:<32} {pr_auc:>30.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
