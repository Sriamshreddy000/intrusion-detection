"""Per-user trailing baselines with a hard no-future-leakage guarantee.

GUARANTEE: for the row scored on day D, every baseline below only sees
that same user's rows with date STRICTLY LESS THAN D. This is enforced
structurally, not by convention — pandas' time-based `Rolling` window is
given `closed="left"`, which excludes the window's right edge (the
current row's own timestamp) by construction. There is no separate
shift() step to forget, and no risk of a centered or right-closed window
silently including the current or a future day: closed="left" is the
only thing that decides the window boundary, and it's set once, here.

The window is calendar-time based ("30D"), not a row-count based window,
so it correctly spans gaps (weekends, vacations, days with no activity
at all) rather than quietly looking further back in elapsed time for
sparser users.

TWO DIFFERENT BASELINE MECHANISMS ARE USED, DELIBERATELY:

1. add_mode_deviation_features — for discrete count features
   (n_file_copies, n_emails_sent, n_emails_external, n_usb_connects,
   n_http_visits, n_logons). A mean/std z-score was tried first and
   discarded: in this dataset, 76-94% of rows have an exactly-zero
   trailing std for these features (most users do almost exactly the
   same number of X per day), so (raw-mean)/(std+eps) divides by
   near-zero constantly, producing a numerically unstable feature whose
   test PR-AUC swung from 0.05 to 0.92 depending on an arbitrary
   denominator-floor constant — see the methodology notes for the investigation.
   This mechanism instead compares the raw value against the trailing
   MODE (most common prior value) and reports a magnitude and a binary
   "did it differ at all" flag. No division anywhere, so there is no
   near-zero-denominator failure mode left to be unstable about.

2. add_zscore_features — for the two continuous features (active_hours,
   first_logon_hour), where "mode" isn't a meaningful concept (continuous
   values essentially never repeat exactly). Keeps the mean/std z-score,
   but with a tunable std floor and a clip — std_floor/zscore_clip are
   NOT hardcoded here; they are chosen by tune_zscore_floor.py on a
   validation slice carved out of TRAIN, never on test, and passed in by
   the caller (build_features.py).
"""
import numpy as np
import pandas as pd

EPS = 1e-6


def _trailing_rolling(df: pd.DataFrame, col: str, user_col: str, date_col: str, window: str, min_periods: int):
    df = df.sort_values([user_col, date_col]).reset_index(drop=True)
    indexed = df.set_index(date_col)
    grouped = indexed.groupby(user_col, sort=False)
    return df, grouped[col].rolling(window, closed="left", min_periods=min_periods)


def add_ratio_features(
    df: pd.DataFrame,
    value_cols: list[str],
    user_col: str = "user",
    date_col: str = "date",
    window: str = "30D",
    min_periods: int = 5,
    ratio_clip: float = 10.0,
) -> pd.DataFrame:
    """Adds `{col}_ratio_30d` = min(raw / (trailing_mean + EPS), ratio_clip).

    Same near-zero-mean blowup as the original z-score bug lives here too
    (found during the instability investigation — n_file_copies_ratio_30d
    reached 27,000,000, n_usb_connects_ratio_30d reached 10,000,000, from
    dividing by a trailing mean of essentially 0). Clipped unconditionally;
    confirmed on a train-internal validation slice not to regress
    performance (0.9874 unclipped vs 0.9888 clipped, both at
    scale_pos_weight=1 — see the methodology notes for the full investigation).
    """
    out = df.copy()
    for col in value_cols:
        sorted_df, roll = _trailing_rolling(df, col, user_col, date_col, window, min_periods)
        trailing_mean = roll.mean().reset_index(drop=True)
        raw = sorted_df[col].reset_index(drop=True)
        ratio = (raw / (trailing_mean + EPS)).clip(upper=ratio_clip)
        out = out.merge(
            sorted_df[[user_col, date_col]].assign(**{f"{col}_ratio_30d": ratio.values}),
            on=[user_col, date_col],
            how="left",
        )
    return out


def add_mode_deviation_features(
    df: pd.DataFrame,
    value_cols: list[str],
    user_col: str = "user",
    date_col: str = "date",
    window: str = "30D",
    min_periods: int = 5,
) -> pd.DataFrame:
    """Adds `{col}_diff_from_mode_30d` (signed: raw - trailing mode) and
    `{col}_differs_from_mode_30d` (1 if raw != trailing mode, else 0).

    Trailing mode = the most common value this user took on in the
    strictly-prior 30-day window (ties broken by pandas' Series.mode(),
    which returns the smallest tied value first). NaN in either output
    means "not enough trailing history yet" (same min_periods rule as
    the ratio/zscore features) — left as NaN, not imputed.
    """
    out = df.copy()
    for col in value_cols:
        sorted_df, roll = _trailing_rolling(df, col, user_col, date_col, window, min_periods)

        def _mode_or_nan(window_vals):
            if len(window_vals) == 0:
                return np.nan
            m = window_vals.mode()
            return m.iloc[0] if len(m) else np.nan

        trailing_mode = roll.apply(_mode_or_nan, raw=False).reset_index(drop=True)
        raw = sorted_df[col].reset_index(drop=True)

        diff = raw - trailing_mode
        differs = (raw != trailing_mode).astype(float)
        differs[trailing_mode.isna()] = np.nan

        out = out.merge(
            sorted_df[[user_col, date_col]].assign(
                **{f"{col}_diff_from_mode_30d": diff.values, f"{col}_differs_from_mode_30d": differs.values}
            ),
            on=[user_col, date_col],
            how="left",
        )
    return out


def add_zscore_features(
    df: pd.DataFrame,
    value_cols: list[str],
    std_floor: float,
    zscore_clip: float,
    user_col: str = "user",
    date_col: str = "date",
    window: str = "30D",
    min_periods: int = 5,
) -> pd.DataFrame:
    """Adds `{col}_zscore_30d` = clip((raw - trailing_mean) / max(trailing_std, std_floor), -clip, clip).

    std_floor and zscore_clip are deliberately required arguments, not
    defaults baked in here — see tune_zscore_floor.py. They must be
    chosen on a validation slice carved out of TRAIN, never on test.
    """
    out = df.copy()
    for col in value_cols:
        sorted_df, roll = _trailing_rolling(df, col, user_col, date_col, window, min_periods)
        trailing_mean = roll.mean().reset_index(drop=True)
        trailing_std = roll.std().reset_index(drop=True)
        raw = sorted_df[col].reset_index(drop=True)

        safe_std = trailing_std.clip(lower=std_floor)
        zscore = ((raw - trailing_mean) / safe_std).clip(-zscore_clip, zscore_clip)

        out = out.merge(
            sorted_df[[user_col, date_col]].assign(**{f"{col}_zscore_30d": zscore.values}),
            on=[user_col, date_col],
            how="left",
        )
    return out
