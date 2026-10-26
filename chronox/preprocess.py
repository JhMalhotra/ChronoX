"""Leakage-safe preprocessing: sessionisation, user-relative baselines, leakage/shortcut audits, PSI."""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def build_features(df, cfg, train_end_ts):
    """All features use only (a) the row itself, (b) the same user's PAST, (c) statistics fitted on rows <= train_end_ts."""
    d = df.sort_values("ts", kind="stable").reset_index(drop=True)
    tr = (d["ts"] <= train_end_ts).values
    deny = re.compile(cfg.deny_regex, re.I)
    F, G, dropped = {}, {}, []
    user = d["user"].astype(str)

    hr = d["ts"].dt.hour + d["ts"].dt.minute / 60.0
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    dow = d["ts"].dt.day_of_week
    F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    F["is_weekend"] = (dow >= 5).astype(float)
    F["off_hours"] = ((hr < 7) | (hr > 19)).astype(float)
    for k in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "off_hours"):
        G[k] = "time"

    g = d.groupby(user)
    n_prev = g.cumcount()
    gap_min = g["ts"].diff().dt.total_seconds().div(60)
    if cfg.use_user_baselines:
        F["user_event_count"] = np.log1p(n_prev)
        F["gap_prev_min"] = np.log1p(gap_min.fillna(60 * 24 * 30).clip(upper=60 * 24 * 30))
        G["user_event_count"] = G["gap_prev_min"] = "user_baseline"
    if cfg.use_sessions:
        new = gap_min.isna() | (gap_min > cfg.session_gap_min)
        sid = new.cumsum()
        F["sess_pos"] = np.log1p(d.groupby(sid).cumcount())
        F["sess_elapsed_min"] = np.log1p((d["ts"] - d.groupby(sid)["ts"].transform("min")).dt.total_seconds() / 60)
        G["sess_pos"] = G["sess_elapsed_min"] = "session"
        d["session_id"] = sid.values

    num_cols = []
    for c in d.columns:
        if c in ("user", "ts", "y", "session_id") or deny.search(c):
            if c not in ("user", "ts", "y", "session_id"):
                dropped.append((c, "name matches leakage deny-list"))
            continue
        if c in cfg.drop_cols:
            continue
        s = d[c]
        if pd.api.types.is_bool_dtype(s):
            F[c], G[c] = s.astype(float), "raw_num"
        elif pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce").astype(float)
            if x.nunique() > 1:
                F[c], G[c] = x, "raw_num"
                num_cols.append(c)
        else:
            s = s.astype("string").fillna("NA")
            if s.nunique() > 0.9 * len(s):
                dropped.append((c, "ID-like (cardinality > 90%)"))
                continue
            freq = s[tr].value_counts(normalize=True)
            F[c + "__rarity"] = -np.log(s.map(freq).fillna(0.0).astype(float) + 1e-6)
            G[c + "__rarity"] = "cat_rarity"
            if s.str.len().mean() > 20:
                F[c + "__len"], G[c + "__len"] = s.str.len().astype(float), "raw_num"
            if cfg.use_user_baselines:
                F[c + "__user_new"] = (~(user + "||" + s).duplicated()).astype(float)
                G[c + "__user_new"] = "user_baseline"

    if cfg.use_user_baselines:
        for c in sorted(num_cols, key=lambda k: -F[k].std())[:8]:
            x = F[c].fillna(0.0)
            cs = x.groupby(user).cumsum() - x
            cs2 = (x ** 2).groupby(user).cumsum() - x ** 2
            n = n_prev.replace(0, np.nan)
            mu = cs / n
            var = (cs2 / n - mu ** 2).clip(lower=0)
            z = ((x - mu) / np.sqrt(var + 1e-6)).where(n_prev >= 5, 0.0).clip(-10, 10)
            F[c + "__uz"], G[c + "__uz"] = z.fillna(0.0), "user_baseline"

    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr].median()).fillna(0.0).astype("float32")
    meta = d[["user", "ts", "y"] + (["session_id"] if "session_id" in d else [])].copy()
    return X, meta, G, dropped


def leakage_scan(X, y, train_mask, thr=0.97):
    """Flag features that alone almost perfectly separate classes on the training period."""
    bad = []
    yt = y[train_mask]
    if len(np.unique(yt)) < 2:
        return bad
    for c in X.columns:
        try:
            a = roc_auc_score(yt, X.loc[train_mask, c].values)
        except ValueError:
            continue
        if max(a, 1 - a) >= thr:
            bad.append((c, f"single-feature AUC={a:.3f}"))
    return bad


def adversarial_audit(X, ts, y, train_end, val_end, target_auc=0.70, max_drop=4, seed=0, n=16000, feat_auc=0.75):
    """CLASS-CONDITIONAL adversarial validation. Only benign (y=0) rows are used: benign behaviour should be roughly
    stationary, so a feature that lets a model tell the train period from the later period on benign rows is a
    time/source artefact. Using all rows would confound label-prior shift with covariate shift and delete the
    genuinely predictive features. Iteratively drop the most period-revealing feature until AUC <= target."""
    rng = np.random.default_rng(seed)
    ben = np.asarray(y) == 0
    a_idx = np.where((ts <= train_end) & ben)[0]
    b_idx = np.where((ts > train_end) & (ts <= val_end) & ben)[0]
    if len(a_idx) < 200 or len(b_idx) < 200:
        return list(X.columns), [], np.nan
    k = n // 2
    idx = np.r_[rng.choice(a_idx, min(k, len(a_idx)), replace=False), rng.choice(b_idx, min(k, len(b_idx)), replace=False)]
    lab = (ts[idx] > train_end).astype(int)
    keep, log, auc = list(X.columns), [], np.nan
    # stage 1: univariate -- a single benign feature that alone reveals the period is a time/source artefact
    for c in list(keep):
        a = roc_auc_score(lab, X.iloc[idx][c].values)
        if max(a, 1 - a) >= feat_auc and len(keep) > 5:
            log.append((c, f"univariate adversarial AUC={max(a, 1 - a):.3f}"))
            keep.remove(c)
    # stage 2: multivariate -- iteratively drop the most period-revealing feature
    n_multi = 0
    for _ in range(max_drop + 1):
        rf = RandomForestClassifier(100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        p = cross_val_predict(rf, X.iloc[idx][keep].values, lab, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(lab, p)
        if auc <= target_auc or n_multi >= max_drop or len(keep) <= 5:
            break
        rf.fit(X.iloc[idx][keep].values, lab)
        worst = keep[int(np.argmax(rf.feature_importances_))]
        log.append((worst, f"multivariate adversarial AUC={auc:.3f}"))
        keep.remove(worst)
        n_multi += 1
    return keep, log, auc


def psi(a, b, bins=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    ea = np.clip(np.histogram(a, bins=qs)[0] / len(a), 1e-4, None)
    eb = np.clip(np.histogram(np.clip(b, qs[0], qs[-1]), bins=qs)[0] / len(b), 1e-4, None)
    return float(np.sum((eb - ea) * np.log(eb / ea)))
