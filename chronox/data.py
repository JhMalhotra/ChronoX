"""Dataset loading. SPEDIA (auto-detecting schema) and CERT r4.2 (unified event table)."""
import glob
import os
import re

import numpy as np
import pandas as pd

_TRUE = {"1", "true", "yes", "y", "anomaly", "malicious", "attack", "anomalous", "t"}


def _to_binary(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    return s.astype("string").str.strip().str.lower().isin(_TRUE).astype(int)


def _parse_ts(s):
    if pd.api.types.is_numeric_dtype(s):
        med = float(pd.to_numeric(s, errors="coerce").median())
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed").dt.tz_convert(None)


def _detect_ts(df, hint=None):
    if hint:
        return hint
    best, rate = None, 0.0
    for c in df.columns:
        if re.search(r"time|date|stamp|(^|_)ts$", c, re.I):
            r = _parse_ts(df[c].head(2000)).notna().mean()
            if r > rate:
                best, rate = c, r
    if best is None or rate < 0.9:
        raise ValueError("Could not detect a timestamp column; set Cfg.ts_col.")
    return best


def _detect_user(df, hint=None):
    if hint:
        return hint
    for c in df.columns:
        if re.search(r"user|account|employee|actor|usr", c, re.I):
            return c
    return None


def load_spedia(path, label_col="Anomaly", ts_col=None, user_col=None, drop_cols=(), max_rows=None):
    df = pd.read_csv(path, low_memory=False, nrows=max_rows)
    ts_col, user_col = _detect_ts(df, ts_col), _detect_user(df, user_col)
    out = df.drop(columns=[c for c in (label_col, ts_col, user_col, *drop_cols) if c and c in df.columns]).copy()
    out.insert(0, "y", _to_binary(df[label_col]).values)
    out.insert(0, "user", df[user_col].astype(str).values if user_col else "all")
    out.insert(0, "ts", _parse_ts(df[ts_col]).values)
    out = out.dropna(subset=["ts"]).reset_index(drop=True)
    out.attrs["ts_col"], out.attrs["user_col"] = ts_col, user_col
    return out


def load_cert_r42(root, answers_dir, n_benign_users=150, seed=0):
    """CERT r4.2 -> unified event table. Column layout follows the public release; verify on your copy."""
    fmt = "%m/%d/%Y %H:%M:%S"
    parts = []
    spec = {
        "logon": ["id", "date", "user", "pc", "activity"],
        "device": ["id", "date", "user", "pc", "activity"],
        "file": ["id", "date", "user", "pc", "filename", "content"],
        "http": ["id", "date", "user", "pc", "url", "content"],
        "email": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
    }
    for name, cols in spec.items():
        p = os.path.join(root, f"{name}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, usecols=lambda c: c in cols, low_memory=False)
        d["ts"] = pd.to_datetime(d["date"], format=fmt, errors="coerce")
        act = d["activity"].astype(str) if "activity" in d else ""
        d["event_type"] = name + ("_" + act if "activity" in d else "")
        d["host"] = d["pc"].astype(str)
        d["size"] = pd.to_numeric(d["size"], errors="coerce") if "size" in d else np.nan
        d["content_len"] = d["content"].astype(str).str.len() if "content" in d else np.nan
        d["url_len"] = d["url"].astype(str).str.len() if "url" in d else np.nan
        parts.append(d[["id", "ts", "user", "event_type", "host", "size", "content_len", "url_len"]])
    df = pd.concat(parts, ignore_index=True)
    bad = set()
    for f in glob.glob(os.path.join(answers_dir, "**", "*.csv"), recursive=True):
        if os.path.basename(f).startswith("insiders"):
            continue
        with open(f, errors="ignore") as fh:
            for line in fh:
                bits = line.strip().split(",")
                if len(bits) > 2:
                    bad.add(bits[1].strip("{} "))
    df["y"] = df["id"].astype(str).str.strip("{} ").isin(bad).astype(int)
    insiders = set(df.loc[df.y == 1, "user"])
    rng = np.random.default_rng(seed)
    others = sorted(set(df["user"]) - insiders)
    keep = insiders | set(rng.choice(others, size=min(n_benign_users, len(others)), replace=False))
    df = df[df["user"].isin(keep)].drop(columns="id").dropna(subset=["ts"]).reset_index(drop=True)
    return df


def load(cfg):
    if cfg.dataset == "spedia":
        return load_spedia(cfg.data_path, cfg.label_col, cfg.ts_col, cfg.user_col, cfg.drop_cols, cfg.max_rows)
    if cfg.dataset == "cert":
        return load_cert_r42(cfg.cert_root, cfg.cert_answers, cfg.cert_users)
    raise ValueError(cfg.dataset)


def dataset_overview(dfs):
    """Tabular overview of instances, users, time span, prevalence per dataset."""
    rows = []
    for name, d in dfs.items():
        q = d["ts"].quantile([0.6, 0.8]).values
        rows.append({
            "dataset": name, "instances": len(d), "users": d["user"].nunique(),
            "raw_columns": d.shape[1] - 3, "start": d["ts"].min(), "end": d["ts"].max(),
            "anomaly_rate": round(d["y"].mean(), 4),
            "rate_first60%": round(d.loc[d.ts <= q[0], "y"].mean(), 4),
            "rate_60-80%": round(d.loc[(d.ts > q[0]) & (d.ts <= q[1]), "y"].mean(), 4),
            "rate_last20%": round(d.loc[d.ts > q[1], "y"].mean(), 4),
        })
    return pd.DataFrame(rows)
