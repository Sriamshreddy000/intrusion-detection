# Insider Threat Detection on CERT r4.2

Flags employee-days that look like insider activity in organizational activity logs — logons, file copies to removable media, email, USB device connects, web traffic. Framed deliberately as **rare-event detection under an analyst alert budget**, not as a leaderboard score: the deliverable is an operating point a security team could actually staff, plus an honest account of how much of the result survives contact with realistic data.

## The problem

CERT r4.2 is 18 months of synthetic activity logs for ~1,000 employees. Aggregated to one row per user per day:

| | |
|---|---|
| User-days | 330,452 |
| Malicious user-days | 966 (0.292%) |
| Distinct insiders | 70, across 3 attack scenarios |

At a 0.3% base rate, two conventional defaults are useless:

- **Accuracy** — predicting "benign" for every row scores 99.7% and finds nothing.
- **A 0.5 decision threshold** — arbitrary. The threshold is the decision variable, so it is chosen here against a stated review capacity, not left at a library default.

Evaluation uses PR-AUC rather than ROC-AUC, which is flattering at this imbalance because the false-positive denominator is enormous.

## Approach

**One row = one user-day.** The ground truth marks *people over periods*, not individual events. A single logon is never suspicious; "copied 400 files at 02:00" only exists once a day is aggregated. Labeling at event level would mark thousands of ordinary clicks as malicious.

Two label definitions are carried side by side. `label_event_day` (primary) marks only days with a logged malicious act — 966 positives. `label_window` marks every calendar day inside an insider's campaign window — 1,364 positives, and the convention most published CERT work uses, kept here for comparability.

**Time-based split, never random.** Train on 2010 (246,423 user-days, 722 positives, 56 insiders); test on 2011-01-01 → 2011-05-17 (84,029 user-days, 244 positives, 18 insiders). Positive rate is 0.293% in train and 0.290% in test, so neither side got an easier deal. A random shuffle would let future behavior inform a day's own baseline — a score that could never be reproduced in deployment, where Monday is scored on Monday.

**Every baseline looks strictly backwards.** Trailing windows use pandas' time-based rolling with `closed="left"`, which excludes the current row's own timestamp structurally rather than by a `shift()` that can be forgotten. Verified against hand-computed examples before running on the full data.

**No identity features.** No user ID, PC name, email address, or department. Four insiders appear in both the train and test periods, so an identity feature would let the model memorize *who* is malicious and recognize them again — a near-perfect score with no ability to catch an employee who goes rogue for the first time. All signal is relative to each person's own history.

**Features (35 columns), in three families:**

1. Raw daily counts — file copies, emails sent, external emails, USB connects, HTTP visits, logons, active hours.
2. Per-user relative — each count against that user's own trailing 30-day baseline.
3. Timing and novelty — after-hours and weekend flags, first-ever use of a machine, deviation from that user's usual start time, gap since last activity.

Model is LightGBM. It is the least interesting choice in the project, which is intentional.

## Results

Test set, final model. "Insiders caught" means at least one of that person's malicious days ranked inside the alert budget — the number a security lead would actually ask for. Day-level catch rate is reported alongside it, but it flatters long campaigns: one insider flagged on 23 separate days counts as 23 catches and one person found.

| Alerts/day | Total alerts | Insiders caught | Day-level catch | Precision |
|---|---|---|---|---|
| 1 | 137 | 11 / 18 | 56.1% | 100% |
| **2** | **274** | **18 / 18** | **96.7%** | **86.1%** |
| 5 | 685 | 18 / 18 | 99.2% | 35.3% |
| 20 | 2,740 | 18 / 18 | 99.6% | 8.9% |

> At a 2-alerts/day budget — 274 alerts across the 137-day test period, reviewable by a single analyst — the model surfaces every one of the 18 insiders present in test at 86% precision. The curve is flat from there to 20 alerts/day, so additional budget buys nothing but review labor.

Seed-swept across 5 seeds, that operating point is 97.8% ± 2.7% of insiders. Test PR-AUC is 0.9840, against a validation mean of 0.9867 ± 0.0029 — close enough to rule out overfitting to the split.

`reports/figures/final_threshold_sweep.png` plots the full catch-rate-vs-alert-volume curve with ±1 std bands.

## What the headline number doesn't say

**0.98 is an upper bound on a dataset that is easier than reality, and I measured by how much.**

The model's strongest feature asks whether a count differs at all from that user's trailing mode. It works extraordinarily well because **94% of simulated user-days have an exactly-zero trailing standard deviation** for web visits — most synthetic employees do precisely the same number of things every single day. Real employees do not behave like clockwork, so a feature that fires on "anything changed" would fire on everyone.

To quantify the dependence, small integer jitter was injected into the raw counts of **benign days only** — leaving malicious days untouched — so that normal users vary the way real ones do. Every downstream baseline feature was recomputed, the model retrained with a fixed config, and each level run across 5 seeds:

| Jitter on benign counts | Test PR-AUC | Insiders caught @ 2/day |
|---|---|---|
| none (as simulated) | 0.978 ± 0.009 | 18.0 ± 0.0 |
| ±1 | 0.891 ± 0.027 | 16.8 ± 0.4 |
| ±2 | 0.863 ± 0.019 | 16.0 ± 0.9 |
| ±3 | 0.871 ± 0.012 | 15.8 ± 1.2 |

The three jittered levels overlap heavily and are not distinguishable from each other — no trend should be read into which magnitude scored higher. What is clearly separated is *any* jitter versus none. **The credible range once normal behavior varies realistically is roughly 0.86–0.92 PR-AUC and 15–17 of 18 insiders**, not 0.9840. Degradation is front-loaded: even ±1 breaks the artificial constancy the feature depends on.

This is a property of the dataset's count features broadly, not one feature family. Dropping the mode-deviation features entirely on clean data gives 0.876 ± 0.143 — a standard deviation so large that the ablation cannot confidently say whether they help, hurt, or are neutral.

## Blind spot found

A severity-weighted audit, using scenario number as an explicit ordinal proxy (CERT publishes no monetary damage figures, so none were invented): 1 = single upload, 2 = sustained thumb-drive theft, 3 = keylogger and credential impersonation.

At the tightest budget, severity-weighted coverage (70.0%) exceeds unweighted (61.1%) — but not for a reassuring reason. **All 10 sustained-theft insiders are caught before the single highest-severity one is found at all.** The model prioritizes high-volume, long-running campaigns over singular high-trust breaches. At 2 alerts/day the distinction disappears and both reach 100%.

The miss audit at the chosen operating point is too thin to generalize from: 8 missed user-days, 0 distinct insiders missed — every miss belongs to someone already caught on 2–32 other days. Reported as such rather than forced into a pattern.

## Two findings worth recording

**Class weighting actively destroyed ranking performance.** Setting `scale_pos_weight` to the natural 340:1 ratio is the reflexive move for imbalanced data. Here it was the single largest problem in the project. With only ~722 positive training examples, heavy upweighting made the booster build rules around individual rows instead of generalizable splits — and performance degraded as features were added, at every weight except 1, because each extra column was another way to isolate a single row. Validation PR-AUC across 5 seeds:

| `scale_pos_weight` | Mean | Std |
|---|---|---|
| 1 | 0.9867 | 0.0029 |
| 5 | 0.5155 | 0.2198 |
| 20 | 0.0741 | 0.0567 |

Weight 5 swings from 0.29 to 0.93 on random seed alone. PR-AUC is a ranking metric — it needs correct relative ordering, not calibrated probabilities — so reweighting bought nothing and cost stability. `scale_pos_weight=1.0` is hardcoded, with the natural ratio still recorded in the metrics output.

**A mean/std z-score was the wrong tool for these counts.** With a trailing std of zero on most rows, `(x - mean) / (std + eps)` divides by near-zero and produced values in the millions. Test PR-AUC for that formulation swung between 0.05 and 0.92 depending on an arbitrary denominator floor — a knob, not a signal. Flooring the denominator at a principled value destroyed performance entirely, which is how the 94%-constant artifact was found: the original bug had been accidentally working as a change detector. Replaced with trailing-mode deviation, which involves no division at all.

Every suspicious number in this project turned out to have a mechanism behind it. Finding them is most of what the code history is.

## What I'd do next

- **An unsupervised component.** Isolation forest or autoencoder reconstruction error, to catch attack shapes the labels never covered. A supervised model can only recognize what it has seen labeled, which is the wrong assumption for an adversary who adapts.
- **r6.2.** The sparser release — 5 insiders among 3,995 users — is the harsher test. r4.2 was chosen for a workable number of positives.
- **Tiered routing instead of one cutoff.** Auto-escalate above a high score, human review in the middle band, ignore below, plus hard rules that force review of high-stakes events regardless of model score. A single threshold is the wrong shape for a system where the worst miss is catastrophic.
- **Label-delay realism.** Real insider labels arrive weeks or months after the fact. This evaluation assumes labels are available the moment a campaign ends.

## Repo layout

```
src/
  data/build_labels.py          user-day label table from the answer keys
  data/time_split.py            applies the date cutoff in configs/split.yaml
  features/aggregates.py        chunked per-table daily aggregates
  features/rolling.py           trailing baselines, closed="left" guarantee
  features/build_features.py    assembles the 35-column feature table
  models/train_baseline.py      training + operating-point evaluation
  models/ablation.py            single-feature PR-AUC, one column at a time
  models/tune_zscore_floor.py   constant tuning on a train-internal slice
  evaluation/threshold_sweep.py        catch rate vs alert volume
  evaluation/final_threshold_sweep.py  seed-swept final curve
  evaluation/realism_stress_test.py    noise injection on benign days
  evaluation/damage_weighted_audit.py  severity-weighted coverage
reports/metrics/   per-iteration JSON, one file per run
reports/figures/   curves and stress-test plots
configs/split.yaml train/test cutoff as a tracked decision
```

## Running it

Requires Python 3.12+ and `pip install -r requirements.txt`. The dataset is not included — download CERT r4.2 and `answers.tar.bz2` from [CMU KiltHub](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) and extract both into `data/raw/`.

```bash
# 1. labels and split
python src/data/build_labels.py --data-root data/raw --out data/interim/user_day_labels.parquet
python src/data/time_split.py --labels data/interim/user_day_labels.parquet \
  --split-config configs/split.yaml --out data/interim/user_day_labels_split.parquet

# 2. features — first pass with a placeholder floor, then tuned on a train-internal
#    validation slice, which writes the final table
python src/features/build_features.py --data-root data/raw \
  --labels data/interim/user_day_labels_split.parquet \
  --out data/processed/features_tmp.parquet --std-floor 1.0
python src/models/tune_zscore_floor.py --features data/processed/features_tmp.parquet \
  --data-root data/raw --labels data/interim/user_day_labels_split.parquet \
  --final-out data/processed/features.parquet

# 3. train and evaluate
python src/models/train_baseline.py --features data/processed/features.parquet \
  --k-list 1,2,5,10,20 --tag final --out-metrics reports/metrics/final.json

# 4. the analyses that matter
python src/evaluation/final_threshold_sweep.py --features data/processed/features.parquet \
  --out-fig reports/figures/final_threshold_sweep.png
python src/evaluation/realism_stress_test.py --features data/processed/features.parquet \
  --out-fig reports/figures/realism_stress_test.png
python src/evaluation/damage_weighted_audit.py --features data/processed/features.parquet
```

The first feature build scans a 14.5GB table and takes a while; table aggregates are cached under `data/interim/` afterwards, so later runs only recompute the rolling baselines.

## Note on the data

CERT r4.2 is synthetic — red-team scenarios injected into simulated organizational activity. It is the standard public benchmark for this task because no comparable real-world labeled dataset exists, and it is described by its own authors as a dense dataset with deliberately strong signal. The stress test above exists because that matters for how the headline number should be read.
