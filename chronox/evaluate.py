"""Rolling-origin evaluation, metrics (mean ± SD), model overview, and the experiment driver."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from . import featopt, models, preprocess, xai
from .models import HAS_TORCH


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    two = len(np.unique(y)) == 2
    return {
        "AUC": roc_auc_score(y, p) if two else np.nan,
        "PR-AUC": average_precision_score(y, p) if two else np.nan,
        "Recall": recall_score(y, pred, zero_division=0),
        "Accuracy": accuracy_score(y, pred),
        "BalAcc": balanced_accuracy_score(y, pred) if two else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred) if two else np.nan,
        "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
    }


def prepare(df, cfg, seed=0):
    """Build features, run audits and (optionally) feature optimisation using ONLY the early 'dev' region."""
    ts = df["ts"].sort_values().values
    n = len(df)
    dev_end = pd.Timestamp(ts[int(n * cfg.dev_frac) - 1])
    train_end = pd.Timestamp(ts[int(n * cfg.dev_frac * 0.75) - 1])
    X, meta, groups, dropped = preprocess.build_features(df, cfg, train_end)
    y, tsv = meta["y"].values, meta["ts"].values
    trm = (meta["ts"] <= train_end).values
    log = {"dropped": list(dropped), "ga_history": [], "pareto": None, "adv_auc": np.nan}
    keep = list(X.columns)
    if cfg.use_leak_audit:
        bad = preprocess.leakage_scan(X, y, trm, cfg.leak_auc)
        keep = [c for c in keep if c not in {b[0] for b in bad}]
        log["dropped"] += bad
    if cfg.use_adv_audit:
        keep, adv_log, adv_auc = preprocess.adversarial_audit(X[keep], tsv, y, train_end, dev_end, cfg.adv_target_auc, cfg.adv_max_drop, seed, feat_auc=cfg.adv_feature_auc)
        log["dropped"] += adv_log
        log["adv_auc"] = adv_auc
    if cfg.use_featopt:
        dev = np.where(meta["ts"] <= dev_end)[0]
        keep, hist, par = featopt.optimise_features(X.iloc[dev][keep], y[dev], cfg, seed)
        log["ga_history"], log["pareto"] = hist, par
    log["n_features_total"], log["n_features_selected"], log["groups"] = X.shape[1], len(keep), groups
    return X, meta, keep, log


def rolling_eval(X, meta, keep, cfg, df_name="", collect_last=False):
    """Expanding-window rolling-origin: each block is predicted by models trained only on earlier data."""
    y, ts = meta["y"].values, meta["ts"].values
    n = len(y)
    start = int(n * cfg.dev_frac)
    edges = np.linspace(start, n, cfg.n_blocks + 1).astype(int)
    A = X[keep].values
    res, last = [], None
    for b in range(cfg.n_blocks):
        te = np.arange(edges[b], edges[b + 1])
        pre = np.arange(0, edges[b])
        cut = int(len(pre) * 0.8)
        tr, va = pre[:cut], pre[cut:]
        for seed in cfg.seeds:
            if len(np.unique(y[tr])) < 2:
                continue
            ens = models.ChronoEnsemble(cfg, seed).fit(A[tr], y[tr], ts[tr], A[va], y[va], ts[va])
            for name, m in ens.members.items():
                p = m.predict_proba(A[te])[:, 1]
                thr = models.best_threshold(m.predict_proba(A[va])[:, 1], y[va])[0]
                res.append({"block": b, "seed": seed, "model": name, **metrics(y[te], p, thr),
                            "fit_s": ens.fit_time[name], "block_prev": y[te].mean()})
            t0 = time.time()
            q = ens.predict_proba(A[te])
            res.append({"block": b, "seed": seed, "model": "CHRONOX", **metrics(y[te], q, ens.thr),
                        "fit_s": sum(ens.fit_time.values()), "block_prev": y[te].mean(), "infer_s": time.time() - t0})
            if collect_last and b == cfg.n_blocks - 1 and seed == cfg.seeds[0]:
                last = {"ens": ens, "tr": tr, "va": va, "te": te, "q": q}
    return pd.DataFrame(res), last


def summarise(res, cols=("AUC", "PR-AUC", "Recall", "Accuracy", "F1", "MCC", "Brier")):
    rows = []
    for m, g in res.groupby("model"):
        r = {"model": m, "n_runs": len(g)}
        for c in cols:
            r[c] = f"{g[c].mean():.3f} ± {g[c].std(ddof=0):.3f}"
        r["_auc"] = g["AUC"].mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("_auc", ascending=False).drop(columns="_auc").reset_index(drop=True)


def run_experiment(df, cfg, collect_xai=None):
    """Full pipeline for one config. Returns dict with detection results, logs and (optionally) XAI results."""
    collect_xai = cfg.run_xai if collect_xai is None else collect_xai
    X, meta, keep, log = prepare(df, cfg, seed=cfg.seeds[0])
    res, last = rolling_eval(X, meta, keep, cfg, collect_last=collect_xai)
    out = {"results": res, "summary": summarise(res), "log": log, "keep": keep, "X": X, "meta": meta}
    if collect_xai and last is not None:
        out["xai"] = run_xai(X, meta, keep, last, cfg)
    return out


def run_xai(X, meta, keep, last, cfg):
    A = X[keep].values.astype("float32")
    y = meta["y"].values
    ens, tr, te, q = last["ens"], last["tr"], last["te"], last["q"]
    pred = (q >= ens.thr).astype(int)
    alerts = np.where(pred == 1)[0]
    if len(alerts) == 0:
        return None
    rng = np.random.default_rng(0)
    pick = rng.choice(alerts, min(cfg.n_explain, len(alerts)), replace=False)
    rows = te[pick]
    users = meta["user"].values
    codes = pd.factorize(users)[0]
    user_rows = {}
    for r in tr:
        user_rows.setdefault(users[r], []).append(r)
    user_rows = {u: np.array(v) for u, v in user_rows.items()}
    base = np.median(A[tr][y[tr] == 0], axis=0) if (y[tr] == 0).any() else np.median(A[tr], axis=0)
    cnn = None
    if cfg.use_cnn and cfg.xai_gradcam and HAS_TORCH:
        w = models.recency_weights(meta["ts"].values[tr]) if cfg.use_recency else None
        cnn = models.SeqCNN(A, codes, cfg, 0).fit(tr, y, w)
        res_cnn = metrics(y[te], cnn.predict_proba(te), 0.5)
    else:
        res_cnn = None
    ctx = xai.ExplainContext(ens.predict_raw, A, tr, users, keep, base, cnn, user_rows)
    df, summ, store = xai.explain_alerts(ctx, rows, cfg, y_true=y[rows], y_pred=pred[pick])
    return {"per_alert": df, "summary": summ, "store": store, "names": keep, "cnn_metrics": res_cnn,
            "trust_auc": summ.attrs.get("trust_auc", np.nan), "has_gradcam": cnn is not None}
