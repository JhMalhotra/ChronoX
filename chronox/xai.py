"""RAFE: Reliability-Aware Fused Explanations.

Three attribution sources for one alert, all expressed over the same selected-feature space:
  (1) interventional Shapley values (sampling estimator; exact TreeSHAP optional via `shap`)
  (2) BC-LIME: LIME whose perturbations are *replaced by the user's own past behaviour* (on-manifold baseline)
  (3) TF-Grad-CAM: temporal Grad-CAM x input-gradient from the 1D-CNN (needs torch)
Each source gets a per-alert reliability  r = faithfulness x stability  (measured on the deployed decision function),
sources are fused by reliability-weighted normalised importance, and a per-alert TRUST score
(= inter-source rank agreement x mean reliability) flags alerts whose explanation should not be relied upon.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ explainers
def sampling_shapley(f, x, bg, n_perm, rng):
    d = len(x)
    rows, perms = [], []
    for b in bg[rng.integers(0, len(bg), n_perm)]:
        p = rng.permutation(d)
        cur = b.copy()
        rows.append(cur.copy())
        for j in p:
            cur[j] = x[j]
            rows.append(cur.copy())
        perms.append(p)
    pr = f(np.asarray(rows)).reshape(n_perm, d + 1)
    phi = np.zeros(d)
    for k, p in enumerate(perms):
        phi[p] += np.diff(pr[k])
    return phi / n_perm


def tree_shap(model, X):
    """Optional exact TreeSHAP for a single tree model (RF/ET/XGB). Returns None when `shap` is unavailable."""
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        return sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv
    except Exception:
        return None


def bc_lime(f, x, pool, n, rng, width=0.5):
    d = len(x)
    A = (rng.random((n, d)) < rng.random((n, 1))).astype(float)
    A[0] = 1.0
    B = pool[rng.integers(0, len(pool), n)]
    Z = np.where(A == 1, x, B)
    p = np.clip(f(Z), 1e-4, 1 - 1e-4)
    t = np.log(p / (1 - p))
    w = np.exp(-((1 - A.mean(1)) ** 2) / width ** 2)
    m = Ridge(alpha=1.0).fit(A, t, sample_weight=w)
    return m.coef_, float(m.score(A, t, sample_weight=w))


# ------------------------------------------------------------------ explanation quality metrics
def _curve(f, x, base, order, mode):
    d = len(x)
    rows = []
    cur = x.copy() if mode == "del" else base.copy()
    rows.append(cur.copy())
    for j in order:
        cur[j] = base[j] if mode == "del" else x[j]
        rows.append(cur.copy())
    p = f(np.asarray(rows))
    return float(np.trapezoid(p, dx=1.0 / d)), p


def faithfulness(f, x, attr, base, k):
    order = np.argsort(-np.abs(attr))
    del_auc, p = _curve(f, x, base, order, "del")
    ins_auc, _ = _curve(f, x, base, order, "ins")
    x2 = x.copy(); x2[order[:k]] = base[order[:k]]
    p0 = f(x[None])[0]
    comp = float(np.clip((p0 - f(x2[None])[0]) / max(p0, 1e-6), 0, 1))
    return comp, del_auc, ins_auc


def jaccard_topk(a, b, k):
    sa, sb = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    return len(sa & sb) / len(sa | sb)


def sparsity(a, k):
    m = np.abs(a)
    return float(np.sort(m)[::-1][:k].sum() / (m.sum() + 1e-12))


def _norm(a):
    m = np.abs(a)
    return m / (m.sum() + 1e-12)


def rafe_fuse(attrs, rel, use_rel=True):
    names = list(attrs)
    A = np.stack([_norm(attrs[n]) for n in names])
    w = np.array([rel[n] for n in names]) if use_rel else np.ones(len(names))
    w = w / (w.sum() + 1e-12)
    fused = (w[:, None] * A).sum(0)
    if len(names) > 1:
        cs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = spearmanr(A[i], A[j])[0]
                cs.append(0.0 if np.isnan(r) else r)
        agree = float(np.mean(cs))
    else:
        agree = 1.0
    trust = max(agree, 0.0) * float(np.mean([rel[n] for n in names]))
    return fused, w, agree, trust


# ------------------------------------------------------------------ pipeline
class ExplainContext:
    def __init__(self, f, Xs_all, train_rows, users, names, base, cnn=None, user_rows=None, seed=0):
        self.f, self.Xs, self.names, self.base, self.cnn = f, Xs_all, names, base, cnn
        self.users, self.user_rows = users, user_rows
        rng = np.random.default_rng(seed)
        self.train_rows = train_rows
        self.bg = Xs_all[rng.choice(train_rows, min(100, len(train_rows)), replace=False)]
        self.gpool = Xs_all[rng.choice(train_rows, min(500, len(train_rows)), replace=False)]

    def pool(self, row, cfg):
        if not cfg.xai_bc:
            return self.gpool
        r = self.user_rows.get(self.users[row])
        return self.Xs[r] if r is not None and len(r) >= 20 else self.gpool


def explain_alerts(ctx, rows, cfg, y_true=None, y_pred=None, seed=0):
    """Explain `rows` (global row ids). Returns (per-alert DataFrame, summary DataFrame, store of attributions)."""
    rng = np.random.default_rng(seed)
    d = ctx.Xs.shape[1]
    k = int(min(cfg.topk, max(1, d // 3)))
    use_cam = bool(cfg.xai_gradcam and ctx.cnn is not None)
    cam1 = cam2 = None
    if use_cam:
        S1, cam1 = ctx.cnn.explain(rows, 0.0)
        S2, _ = ctx.cnn.explain(rows, 0.05, seed + 1)
        G1, G2 = S1.sum(1), S2.sum(1)
    recs, store = [], []
    for i, r in enumerate(rows):
        x = ctx.Xs[r]
        pool = ctx.pool(r, cfg)
        t0 = time.time(); a_sh = sampling_shapley(ctx.f, x, ctx.bg, cfg.shap_perms, rng); t_sh = time.time() - t0
        a_sh2 = sampling_shapley(ctx.f, x, ctx.bg, cfg.shap_perms, rng)
        t0 = time.time(); a_li, r2 = bc_lime(ctx.f, x, pool, cfg.lime_samples, rng); t_li = time.time() - t0
        a_li2, _ = bc_lime(ctx.f, x, pool, cfg.lime_samples, rng)
        src = {"shapley": (a_sh, a_sh2), "bc_lime": (a_li, a_li2)}
        if use_cam:
            src["tf_gradcam"] = (G1[i], G2[i])
        rel, row = {}, {"row": int(r)}
        for n, (a, a2) in src.items():
            comp, dl, ins = faithfulness(ctx.f, x, a, ctx.base, k)
            stab = jaccard_topk(a, a2, k)
            rel[n] = comp * stab + 1e-3
            row.update({f"{n}_comp": comp, f"{n}_del": dl, f"{n}_ins": ins, f"{n}_stab": stab, f"{n}_sparse": sparsity(a, k)})
        row["bc_lime_r2"] = r2
        row["shapley_time"], row["bc_lime_time"] = t_sh, t_li
        fused, w, agree, trust = rafe_fuse({n: a for n, (a, _) in src.items()}, rel, cfg.xai_reliability)
        comp, dl, ins = faithfulness(ctx.f, x, fused, ctx.base, k)
        row.update({"fused_comp": comp, "fused_del": dl, "fused_ins": ins, "fused_sparse": sparsity(fused, k),
                    "agreement": agree, "trust": trust})
        for n, wi in zip(src, w):
            row[f"w_{n}"] = float(wi)
        if cam1 is not None:
            row["cam_peak_pos"] = int(np.argmax(cam1[i]))
        recs.append(row)
        store.append({"row": int(r), "fused": fused, "shapley": a_sh, "bc_lime": a_li,
                      "tf_gradcam": G1[i] if use_cam else None, "heat": S1[i] if use_cam else None})
    df = pd.DataFrame(recs)
    if y_true is not None and y_pred is not None:
        df["correct"] = (np.asarray(y_true) == np.asarray(y_pred)).astype(int)
    out = []
    for n in ["shapley", "bc_lime", "tf_gradcam", "fused"]:
        if f"{n}_comp" not in df:
            continue
        out.append({"explainer": n,
                    **{m: f"{df[f'{n}_{m}'].mean():.3f} ± {df[f'{n}_{m}'].std():.3f}"
                       for m in ("comp", "del", "ins", "sparse") },
                    "stability(top-k Jaccard)": f"{df[f'{n}_stab'].mean():.3f} ± {df[f'{n}_stab'].std():.3f}" if f"{n}_stab" in df else "n/a (fusion of the above)"})
    summ = pd.DataFrame(out).rename(columns={"comp": "comprehensiveness↑", "del": "deletion-AUC↓", "ins": "insertion-AUC↑", "sparse": "top-k mass (sparsity)↑"})
    if "correct" in df and df["correct"].nunique() == 2:
        summ.attrs["trust_auc"] = float(roc_auc_score(df["correct"], df["trust"]))
    return df, summ, store
