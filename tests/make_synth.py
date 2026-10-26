"""Synthetic SPEDIA-like logs with (a) prevalence drift, (b) a source-shortcut that flips over time,
(c) a label-leaking column. ONLY for smoke-testing code paths - never report results from it."""
import numpy as np
import pandas as pd


def make(n=9000, users=60, seed=0):
    r = np.random.default_rng(seed)
    t = np.sort(r.uniform(0, 40 * 24 * 3600, n))
    ts = pd.Timestamp("2025-03-01") + pd.to_timedelta(t, unit="s")
    prev = 0.15 + 0.5 * (t / t.max())                               # prevalence drifts 15% -> 65%
    y = (r.random(n) < prev).astype(int)
    u = r.integers(0, users, n)
    evt = np.where(y == 1, r.choice(["ssh", "ftp", "cmd", "http", "mail", "file"], n, p=[.3, .25, .2, .1, .05, .1]),
                   r.choice(["ssh", "ftp", "cmd", "http", "mail", "file"], n, p=[.1, .05, .2, .35, .2, .1]))
    hour = np.where(y == 1, r.choice(24, n, p=np.r_[np.full(7, .06), np.full(12, .02), np.full(5, .06)] / (np.r_[np.full(7, .06), np.full(12, .02), np.full(5, .06)].sum())), r.integers(8, 19, n))
    ts = ts.normalize() + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(r.integers(0, 3600, n), unit="s")
    order = np.argsort(ts.values)
    nbytes = np.where(y == 1, r.lognormal(9, 1.2, n), r.lognormal(7.5, 1.0, n))
    early = t < t.max() * 0.5
    src = np.where(early, np.where(y == 1, "exercise", "simulated"), r.choice(["exercise", "simulated"], n))   # shortcut flips
    cmd = np.where(y == 1, r.choice(["scp -r /data x@10.0.0.%d" % i for i in range(30)], n), r.choice(["ls", "cd", "cat f", "vim a", "grep x"], n))
    df = pd.DataFrame({"Timestamp": ts, "UserName": ["u%02d" % i for i in u], "EventType": evt,
                       "Host": r.choice(["h%d" % i for i in range(12)], n), "Bytes": nbytes,
                       "Command": cmd, "Source": src, "AnomalyType": np.where(y == 1, "exfil", "none"), "Anomaly": y})
    return df.iloc[order].reset_index(drop=True)
