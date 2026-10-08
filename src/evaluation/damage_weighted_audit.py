"""Damage-weighted coverage: CERT has no dollar values for its red-team
scenarios, so no monetary damage figure is invented here. Severity is an
explicit ORDINAL PROXY, not a calibrated cost estimate, assigned by
qualitative reading of answers/scenarios.txt:

    severity_weight = scenario number itself (1, 2, or 3)

  - Scenario 1 (weight 1): single wikileaks upload via removable drive,
    one bounded event.
  - Scenario 2 (weight 2): sustained thumb-drive theft over a job-hunting
    campaign spanning weeks to months — ongoing, larger-volume exposure.
  - Scenario 3 (weight 3): keylogger theft of a supervisor's credentials,
    used to send an impersonated, alarm-triggering mass email — the only
    scenario with an organization-wide trust/reputational consequence on
    top of the data exposure, judged the most severe of the three.

This ranking is a judgment call, stated as such — a reasonable analyst
could weight sustained volume (scenario 2) above a single high-trust
breach (scenario 3). It is NOT a quantity we fit or calibrate; it is
fixed before looking at any catch-rate numbers, specifically so the
weighting can't be reverse-engineered to flatter the result.

Weighting unit is the INSIDER (one campaign = one incident), not the
user-day — the methodology notes's "assign each incident/attack a severity weight"
means one weight per incident, not one per day of it. A model that
catches a sustained scenario-2 campaign on only one of its 30+ days
still "caught the incident" for this purpose; day-level granularity is
handled separately by the day-level catch rate already reported
elsewhere (src/evaluation/threshold_sweep.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_baseline import load_split

SEVERITY_WEIGHTS = {1: 1, 2: 2, 3: 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", default="label_event_day")
    parser.add_argument("--k-list", default="1,2,5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    k_list = [int(k) for k in args.k_list.split(",")]

    X_train, y_train, X_test, y_test, test_dates, test_users, feature_cols = load_split(args.features, args.target)
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, num_leaves=15, learning_rate=0.05,
        min_child_samples=20, scale_pos_weight=1.0, random_state=args.seed, verbosity=-1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    df = pd.read_parquet(args.features)
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[df["split"] == "test"].reset_index(drop=True).copy()
    test_df["score"] = scores
    test_days = test_df["date"].dt.normalize().nunique()

    # one severity weight per distinct insider (their scenario)
    insider_scenario = (
        test_df.loc[test_df[args.target] == 1, ["user", "scenario"]]
        .drop_duplicates()
        .set_index("user")["scenario"]
    )
    insider_severity = insider_scenario.map(SEVERITY_WEIGHTS)
    total_severity = insider_severity.sum()
    total_insiders = len(insider_severity)

    print("Severity weights by scenario: 1->1, 2->2, 3->3 (ordinal proxy, see module docstring)", file=sys.stderr)
    print(f"Test insiders by scenario: {insider_scenario.value_counts().sort_index().to_dict()}", file=sys.stderr)
    print(f"Total severity-weighted mass in test: {total_severity} (across {total_insiders} insiders)\n", file=sys.stderr)

    ranked = test_df.sort_values("score", ascending=False)
    for k in k_list:
        budget = min(k * test_days, len(test_df))
        alerted_users = set(ranked.head(budget)["user"])
        caught = insider_severity.index.isin(alerted_users)
        n_caught = int(caught.sum())
        severity_caught = insider_severity[caught].sum()

        unweighted_rate = n_caught / total_insiders
        weighted_rate = severity_caught / total_severity

        by_scenario = {}
        for sc in [1, 2, 3]:
            sc_insiders = insider_scenario[insider_scenario == sc].index
            sc_total = len(sc_insiders)
            sc_caught = sum(1 for u in sc_insiders if u in alerted_users)
            by_scenario[sc] = (sc_caught, sc_total)

        print(
            f"k={k:>3}/day budget={budget:>6}: "
            f"unweighted={n_caught}/{total_insiders} ({unweighted_rate:.3f})  "
            f"severity-weighted={severity_caught}/{total_severity} ({weighted_rate:.3f})  "
            f"gap={weighted_rate - unweighted_rate:+.3f}  "
            f"by_scenario={ {sc: f'{c}/{t}' for sc, (c, t) in by_scenario.items()} }",
            file=sys.stderr,
        )

    missed_at_2 = insider_severity.index[~insider_severity.index.isin(set(ranked.head(2 * test_days)["user"]))]
    if len(missed_at_2):
        print(f"\nInsiders missed at k=2/day: {list(missed_at_2)}, scenarios: {insider_scenario[missed_at_2].to_dict()}", file=sys.stderr)
    else:
        print("\nNo insiders missed at k=2/day (0 severity at risk, weighted or not).", file=sys.stderr)


if __name__ == "__main__":
    main()
