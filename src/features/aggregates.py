"""Per-(user, date) aggregates from each raw CERT table.

All five source tables are read in chunks (pd.read_csv(chunksize=...)),
even the small ones, so this stays memory-safe regardless of table size —
http.csv alone is ~14.5GB. Each function only reads the columns it needs
(usecols) to avoid materializing the huge free-text `content`/`url`
columns, which dominate file size but aren't used by these features.

Each function returns one row per (user, date) that appears in that
table, with a count and the min/max event timestamp that day. The
timestamps feed into the cross-table "active span" used for
active_hours in build_features.py — they are not used for any
leakage-sensitive computation themselves.
"""
import pandas as pd

DATE_FMT = "%m/%d/%Y %H:%M:%S"
DTAA_DOMAIN = "@dtaa.com"
CHUNKSIZE = 1_000_000


def _finalize(chunks: list[pd.DataFrame], count_cols: list[str]) -> pd.DataFrame:
    full = pd.concat(chunks, ignore_index=True)
    agg = {c: "sum" for c in count_cols}
    agg["min_ts"] = "min"
    agg["max_ts"] = "max"
    out = full.groupby(["user", "date"], as_index=False).agg(agg)
    return out


def aggregate_logon(path) -> pd.DataFrame:
    """Returns user, date, n_logons, first_logon_ts, last_logon_ts,
    any_after_hours_logon, min_ts, max_ts.

    n_logons counts only 'Logon' activity rows. min_ts/max_ts span both
    Logon and Logoff rows (used for the cross-table active-hours span).
    first_logon_ts/last_logon_ts are the earliest/latest 'Logon'
    timestamps that day. any_after_hours_logon checks EVERY 'Logon' row
    that day against the after-hours window (before 06:00 or after
    20:00), not just the first — a user who logs in normally in the
    morning and returns for a second, after-hours session must still be
    flagged, which first_logon_hour alone misses (confirmed against
    BIH0745: normal 08:04 login, actual malicious session at 20:15).
    any_after_hours_logon is OR-combined across chunks via max(), since
    booleans as 0/1 make max() equivalent to OR.
    """
    chunks = []
    for chunk in pd.read_csv(path, usecols=["date", "user", "activity"], chunksize=CHUNKSIZE):
        ts = pd.to_datetime(chunk["date"], format=DATE_FMT)
        day = ts.dt.normalize()
        chunk = chunk.assign(_ts=ts, date=day)

        logons = chunk[chunk["activity"] == "Logon"].copy()
        hour = logons["_ts"].dt.hour + logons["_ts"].dt.minute / 60.0
        logons["_after_hours"] = (hour < 6.0) | (hour > 20.0)
        logon_agg = logons.groupby(["user", "date"], as_index=False).agg(
            n_logons=("_ts", "count"),
            first_logon_ts=("_ts", "min"),
            last_logon_ts=("_ts", "max"),
            any_after_hours_logon=("_after_hours", "max"),
        )
        span_agg = chunk.groupby(["user", "date"], as_index=False)["_ts"].agg(min_ts="min", max_ts="max")
        merged = span_agg.merge(logon_agg, on=["user", "date"], how="left")
        chunks.append(merged)

    full = pd.concat(chunks, ignore_index=True)
    out = full.groupby(["user", "date"], as_index=False).agg(
        n_logons=("n_logons", "sum"),
        first_logon_ts=("first_logon_ts", "min"),
        last_logon_ts=("last_logon_ts", "max"),
        any_after_hours_logon=("any_after_hours_logon", "max"),
        min_ts=("min_ts", "min"),
        max_ts=("max_ts", "max"),
    )
    return out


def aggregate_device(path) -> pd.DataFrame:
    """Returns user, date, n_usb_connects, min_ts, max_ts.

    n_usb_connects counts only 'Connect' rows. min_ts/max_ts span both
    Connect and Disconnect rows.
    """
    chunks = []
    for chunk in pd.read_csv(path, usecols=["date", "user", "activity"], chunksize=CHUNKSIZE):
        ts = pd.to_datetime(chunk["date"], format=DATE_FMT)
        day = ts.dt.normalize()
        chunk = chunk.assign(_ts=ts, date=day)

        connects = chunk[chunk["activity"] == "Connect"]
        connect_agg = connects.groupby(["user", "date"], as_index=False)["_ts"].agg(n_usb_connects="count")
        span_agg = chunk.groupby(["user", "date"], as_index=False)["_ts"].agg(min_ts="min", max_ts="max")
        merged = span_agg.merge(connect_agg, on=["user", "date"], how="left")
        chunks.append(merged)

    full = pd.concat(chunks, ignore_index=True)
    out = full.groupby(["user", "date"], as_index=False).agg(
        n_usb_connects=("n_usb_connects", "sum"),
        min_ts=("min_ts", "min"),
        max_ts=("max_ts", "max"),
    )
    return out


def aggregate_file(path) -> pd.DataFrame:
    """Returns user, date, n_file_copies, min_ts, max_ts. Every row is one file copy."""
    chunks = []
    for chunk in pd.read_csv(path, usecols=["date", "user"], chunksize=CHUNKSIZE):
        ts = pd.to_datetime(chunk["date"], format=DATE_FMT)
        day = ts.dt.normalize()
        g = chunk.assign(_ts=ts, date=day).groupby(["user", "date"], as_index=False)["_ts"].agg(
            n_file_copies="count", min_ts="min", max_ts="max"
        )
        chunks.append(g)
    return _finalize(chunks, ["n_file_copies"])


def aggregate_http(path) -> pd.DataFrame:
    """Returns user, date, n_http_visits, min_ts, max_ts.

    Only date/user are read — url/content are the large text columns and
    aren't needed for these features. This is the single biggest file
    (~14.5GB); keeping usecols minimal is what makes chunked reading of
    it tractable.
    """
    chunks = []
    for chunk in pd.read_csv(path, usecols=["date", "user"], chunksize=CHUNKSIZE):
        ts = pd.to_datetime(chunk["date"], format=DATE_FMT)
        day = ts.dt.normalize()
        g = chunk.assign(_ts=ts, date=day).groupby(["user", "date"], as_index=False)["_ts"].agg(
            n_http_visits="count", min_ts="min", max_ts="max"
        )
        chunks.append(g)
    return _finalize(chunks, ["n_http_visits"])


def _has_external_recipient(series: pd.Series) -> pd.Series:
    """True if to/cc/bcc (semicolon-joined addresses, NaN-safe) contains any non-@dtaa.com address."""
    filled = series.fillna("")
    return filled.apply(
        lambda s: any(not addr.lower().endswith(DTAA_DOMAIN) for addr in s.split(";") if addr)
    )


def aggregate_email(path) -> pd.DataFrame:
    """Returns user, date, n_emails_sent, n_emails_external, min_ts, max_ts.

    n_emails_sent counts every row (each row is one sent email, `user` is
    the sending employee). n_emails_external counts rows where at least
    one address in to/cc/bcc does not end in @dtaa.com — recipients
    decide "external", not the `from` field (employees sometimes send
    from personal webmail per the dataset readme, so `from` is not a
    reliable internal/external signal).
    """
    chunks = []
    for chunk in pd.read_csv(path, usecols=["date", "user", "to", "cc", "bcc"], chunksize=CHUNKSIZE):
        ts = pd.to_datetime(chunk["date"], format=DATE_FMT)
        day = ts.dt.normalize()
        combined = (
            chunk["to"].fillna("") + ";" + chunk["cc"].fillna("") + ";" + chunk["bcc"].fillna("")
        )
        is_external = _has_external_recipient(combined)
        chunk = chunk.assign(_ts=ts, date=day, is_external=is_external)

        sent_agg = chunk.groupby(["user", "date"], as_index=False)["_ts"].agg(
            n_emails_sent="count", min_ts="min", max_ts="max"
        )
        ext_agg = (
            chunk[chunk["is_external"]]
            .groupby(["user", "date"], as_index=False)["_ts"]
            .agg(n_emails_external="count")
        )
        merged = sent_agg.merge(ext_agg, on=["user", "date"], how="left")
        chunks.append(merged)

    full = pd.concat(chunks, ignore_index=True)
    out = full.groupby(["user", "date"], as_index=False).agg(
        n_emails_sent=("n_emails_sent", "sum"),
        n_emails_external=("n_emails_external", "sum"),
        min_ts=("min_ts", "min"),
        max_ts=("max_ts", "max"),
    )
    return out
