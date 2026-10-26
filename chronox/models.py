"""Model zoo, stability-weighted ensemble, calibration, EM prior-shift adaptation, 1D-CNN with Grad-CAM."""
import time
from collections import deque

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def make_models(seed, n_est=200):
    z = {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5)),
        "RF": RandomForestClassifier(n_est, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "ET": ExtraTreesClassifier(n_est, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "HGB": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((64, 32), early_stopping=True, max_iter=100, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        z["XGB"] = XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.08, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=seed, eval_metric="logloss")
    except Exception:
        pass
    return z


def fit_member(m, X, y, w=None):
    if w is not None:
        try:
            if hasattr(m, "steps"):
                return m.fit(X, y, **{m.steps[-1][0] + "__sample_weight": w})
            return m.fit(X, y, sample_weight=w)
        except (TypeError, ValueError):
            pass
    return m.fit(X, y)


def recency_weights(ts, floor=0.15):
    t = ts.astype("int64").astype(float)
    age = (t.max() - t) / max(t.max() - t.min(), 1.0)
    w = np.exp(-2.0 * age)
    w = np.clip(w / w.max(), floor, 1.0)
    return w / w.mean()


def block_aucs(p, y, ts, k=3):
    order = np.argsort(ts)
    out = []
    for b in np.array_split(order, k):
        if len(np.unique(y[b])) == 2:
            out.append(roc_auc_score(y[b], p[b]))
    return (float(np.mean(out)), float(np.std(out))) if out else (0.5, 0.0)


def em_prior(p, pi_tr, iters=100, tol=1e-6):
    """Saerens et al. (2002) EM label-shift adaptation of calibrated posteriors (changes accuracy/recall, not AUC)."""
    pi_tr = float(np.clip(pi_tr, 0.01, 0.99))
    pi = pi_tr
    q = p
    for _ in range(iters):
        a, b = p * (pi / pi_tr), (1 - p) * ((1 - pi) / (1 - pi_tr))
        q = a / (a + b + 1e-12)
        new = float(q.mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return q, pi


def best_threshold(p, y):
    cand = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    best, bt = -1.0, 0.5
    for t in cand:
        pred = p >= t
        tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), ((~pred) & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, bt = f1, float(t)
    return bt, float(best)


class ChronoEnsemble:
    """Stability-weighted ensemble: member weight = softmax((mean_blockAUC - std_blockAUC)/T) on a chronological
    validation window; recency-weighted training; isotonic calibration; optional EM prior-shift adaptation."""

    def __init__(self, cfg, seed=0):
        self.cfg, self.seed = cfg, seed

    def fit(self, X, y, ts, Xv, yv, tsv):
        cfg = self.cfg
        w = recency_weights(ts) if cfg.use_recency else None
        self.members = make_models(self.seed, cfg.n_estimators)
        self.fit_time, self.val_scores = {}, {}
        for n, m in self.members.items():
            t0 = time.time()
            fit_member(m, X, y, w)
            self.fit_time[n] = time.time() - t0
            self.val_scores[n] = block_aucs(m.predict_proba(Xv)[:, 1], yv, tsv)
        names = list(self.members)
        score = np.array([self.val_scores[n][0] - self.val_scores[n][1] for n in names])
        if cfg.use_ensemble:
            e = np.exp((score - score.max()) / 0.05)
            self.w = dict(zip(names, e / e.sum()))
        else:
            self.w = {n: float(n == names[int(np.argmax(score))]) for n in names}
        raw = self.predict_raw(Xv)
        self.pi_val = float(np.clip(yv.mean(), 0.01, 0.99))
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, yv) if len(np.unique(yv)) == 2 else None
        thr_val, f1_val = best_threshold(self._cal(raw), yv)
        # Under prior shift the empirical val-optimal threshold is stale. For calibrated posteriors the F1-optimal
        # threshold is F1*/2 (Lipton et al., 2014), independent of prevalence -> use it with EM-adapted posteriors.
        self.thr = max(f1_val / 2.0, 0.05) if cfg.use_prior_shift else thr_val
        return self

    def predict_raw(self, X):
        return sum(self.w[n] * m.predict_proba(X)[:, 1] for n, m in self.members.items() if self.w[n] > 0)

    def _cal(self, raw):
        return np.clip(self.iso.predict(raw), 1e-4, 1 - 1e-4) if self.iso is not None else raw

    def predict_proba(self, X):
        q = self._cal(self.predict_raw(X))
        if self.cfg.use_prior_shift:
            q, self.pi_hat = em_prior(q, self.pi_val)
        return q


# --------------------------------------------------------------------------- sequence model + Grad-CAM
def window_index(user_codes, L):
    """idx[i, :] = positions of user's last L events ending at i (chronological, left-padded with -1)."""
    N = len(user_codes)
    idx = -np.ones((N, L), dtype=np.int64)
    recent = {}
    for i, u in enumerate(user_codes):
        dq = recent.setdefault(u, deque(maxlen=L))
        dq.append(i)
        idx[i, L - len(dq):] = list(dq)
    return idx


if HAS_TORCH:
    class _Net(nn.Module):
        def __init__(self, f, c=48):
            super().__init__()
            self.features = nn.Sequential(nn.Conv1d(f, c, 3, padding=1), nn.ReLU(), nn.Conv1d(c, c, 3, padding=1), nn.ReLU())
            self.head = nn.Linear(c, 1)

        def forward(self, x):
            a = self.features(x)
            return self.head(a.mean(2)).squeeze(1), a


class SeqCNN:
    """1D-CNN over each user's last L events. Its Grad-CAM gives *when* (event position) evidence."""

    def __init__(self, Xs, user_codes, cfg, seed=0):
        if not HAS_TORCH:
            raise RuntimeError("torch not installed")
        self.cfg, self.seed = cfg, seed
        torch.manual_seed(seed)
        self.mu = self.sd = None
        self.idx = window_index(user_codes, cfg.seq_len)
        self.Xs = Xs.astype("float32")

    def _scale(self, rows_fit):
        self.mu = self.Xs[rows_fit].mean(0)
        self.sd = self.Xs[rows_fit].std(0) + 1e-6
        Z = (self.Xs - self.mu) / self.sd
        self.Zp = torch.tensor(np.vstack([Z, np.zeros((1, Z.shape[1]), "float32")]))   # last row = padding

    def _batch(self, rows):
        i = self.idx[rows]
        i = np.where(i < 0, len(self.Xs), i)
        return self.Zp[torch.tensor(i)].permute(0, 2, 1)         # B, F, L

    def fit(self, rows_tr, y, w=None):
        self._scale(rows_tr)
        self.net = _Net(self.Xs.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        yt = torch.tensor(y[rows_tr], dtype=torch.float32)
        wt = torch.tensor(w if w is not None else np.ones(len(rows_tr)), dtype=torch.float32)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.cfg.cnn_epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(len(rows_tr)), max(len(rows_tr) // 256, 1)):
                logit, _ = self.net(self._batch(rows_tr[b]))
                loss = (nn.functional.binary_cross_entropy_with_logits(logit, yt[b], reduction="none") * wt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, rows):
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in np.array_split(rows, max(len(rows) // 1024, 1)):
                out.append(torch.sigmoid(self.net(self._batch(b))[0]).numpy())
        return np.concatenate(out)

    def explain(self, rows, noise=0.0, seed=0):
        """Temporal-Feature Grad-CAM: S[b,t,f] = GradCAM_t * |x_tf * dlogit/dx_tf|, normalised to sum 1 per alert."""
        self.net.eval()
        g = torch.Generator().manual_seed(seed)
        x = self._batch(rows).clone()
        if noise > 0:
            x = x + noise * torch.randn(x.shape, generator=g)
        x.requires_grad_(True)
        logit, A = self.net(x)
        A.retain_grad()
        logit.sum().backward()
        alpha = A.grad.mean(2, keepdim=True)                        # channel weights (Grad-CAM)
        cam = torch.relu((alpha * A).sum(1))                        # B, L
        cam = cam / (cam.sum(1, keepdim=True) + 1e-9)
        ig = (x.grad * x).abs()                                     # B, F, L
        S = ig * cam[:, None, :]
        S = S / (S.sum((1, 2), keepdim=True) + 1e-9)
        return S.permute(0, 2, 1).detach().numpy(), cam.detach().numpy()
