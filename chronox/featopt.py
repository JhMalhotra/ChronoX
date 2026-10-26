"""Stability-aware multi-objective feature optimisation (genetic search).

fitness(S) = mean_AUC(S) - lam * std_AUC(S) - mu * |S|/d - nu * mean_PSI(S)
AUC is measured over rolling-origin temporal folds, so a subset that only works in one era is penalised.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .preprocess import psi


def _folds(n, k=4):
    edges = np.linspace(0, n, k + 2).astype(int)           # blocks over time-sorted rows
    return [(np.arange(0, edges[i]), np.arange(edges[i], edges[i + 1])) for i in range(1, k + 1)]


def optimise_features(X, y, cfg, seed=0):
    """X: DataFrame sorted by time (dev region only). Returns (selected names, history, pareto DataFrame)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    names, d = list(X.columns), X.shape[1]
    if len(X) > cfg.ga_rows:                                  # temporally-ordered subsample
        keep = np.sort(rng.choice(len(X), cfg.ga_rows, replace=False))
        X, y = X.iloc[keep], y[keep]
    A = X.values
    folds = _folds(len(A), 3)
    half = len(A) // 2
    psis = np.array([psi(A[:half, j], A[half:, j]) for j in range(d)])
    cache = {}

    def evaluate(mask):
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        cols = np.where(mask)[0]
        if len(cols) < min(cfg.min_features, d):
            return (-1.0, 0.0, 0.0, float(len(cols)), 1.0)
        aucs = []
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=40, max_depth=4, random_state=seed)
            m.fit(A[tr][:, cols], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(A[te][:, cols])[:, 1]))
        mu_, sd_ = (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (0.5, 0.0)
        pen = float(np.clip(psis[cols].mean() / 0.5, 0, 1))
        fit = mu_ - cfg.lam_std * sd_ - cfg.mu_sparse * len(cols) / d - cfg.nu_psi * pen
        cache[key] = (fit, mu_, sd_, len(cols), pen)
        return cache[key]

    pop = rng.random((cfg.ga_pop, d)) < 0.5
    pop[0] = True
    hist = []
    for gen in range(cfg.ga_gen):
        fits = np.array([evaluate(m)[0] for m in pop])
        order = np.argsort(-fits)
        hist.append({"gen": gen, "best_fitness": float(fits[order[0]]), "mean_fitness": float(fits.mean()),
                     "best_k": int(pop[order[0]].sum())})
        new = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new) < cfg.ga_pop:
            i = max(rng.integers(0, cfg.ga_pop, 3), key=lambda t: fits[t])
            j = max(rng.integers(0, cfg.ga_pop, 3), key=lambda t: fits[t])
            child = np.where(rng.random(d) < 0.5, pop[i], pop[j])
            flip = rng.random(d) < 1.5 / d
            new.append(np.where(flip, ~child, child))
        pop = np.array(new)
    best = max(cache.items(), key=lambda kv: kv[1][0])[0]
    best_mask = np.frombuffer(best, dtype=bool)
    rows = [{"fitness": v[0], "auc_mean": v[1], "auc_sd": v[2], "k": v[3], "psi_pen": v[4], "mask": k}
            for k, v in cache.items() if v[3] > 0]
    df = pd.DataFrame(rows)
    pareto = []
    for i, r in df.iterrows():                                # non-dominated on (auc_mean up, auc_sd down, k down, psi down)
        dom = ((df.auc_mean >= r.auc_mean) & (df.auc_sd <= r.auc_sd) & (df.k <= r.k) & (df.psi_pen <= r.psi_pen) &
               ((df.auc_mean > r.auc_mean) | (df.auc_sd < r.auc_sd) | (df.k < r.k) | (df.psi_pen < r.psi_pen))).any()
        if not dom:
            pareto.append(i)
    par = df.loc[pareto].sort_values("auc_mean", ascending=False).drop(columns="mask").head(15).reset_index(drop=True)
    return [n for n, m in zip(names, best_mask) if m], hist, par
