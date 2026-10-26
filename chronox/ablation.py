"""Ablation study: toggle one CHRONOX component at a time and re-run the full protocol."""
import dataclasses

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .evaluate import run_experiment

DETECTION = {
    "Full CHRONOX": {},
    "- leakage/shortcut audit": {"use_leak_audit": False, "use_adv_audit": False},
    "- adversarial audit only": {"use_adv_audit": False},
    "- user-relative baselines": {"use_user_baselines": False},
    "- sessionisation": {"use_sessions": False},
    "- feature optimisation (GA)": {"use_featopt": False},
    "- recency weighting": {"use_recency": False},
    "- prior-shift (EM) adaptation": {"use_prior_shift": False},
    "- stability-weighted ensemble": {"use_ensemble": False},
    "Baseline (repo-style: no audit/GA/user/session/recency/EM/ensemble)": {
        "use_leak_audit": False, "use_adv_audit": False, "use_featopt": False, "use_user_baselines": False,
        "use_sessions": False, "use_recency": False, "use_prior_shift": False, "use_ensemble": False},
}
XAI = {
    "Full RAFE": {},
    "- BC-LIME (global-pool LIME instead)": {"xai_bc": False},
    "- TF-Grad-CAM source": {"xai_gradcam": False},
    "- reliability weighting (equal weights)": {"xai_reliability": False},
}


def run_ablation(df, cfg, which="detection"):
    table = DETECTION if which == "detection" else XAI
    rows, full_auc = [], None
    for name, patch in table.items():
        c = dataclasses.replace(cfg, **patch, run_xai=(which == "xai"))
        out = run_experiment(df, c)
        if which == "detection":
            g = out["results"].query("model == 'CHRONOX'")
            r = {"variant": name, "features": out["log"]["n_features_selected"]}
            for m in ("AUC", "PR-AUC", "Recall", "Accuracy", "F1", "MCC"):
                r[m] = f"{g[m].mean():.3f} ± {g[m].std(ddof=0):.3f}"
            vec = g.sort_values(["block", "seed"])["AUC"].values
            if full_auc is None:
                full_auc, r["Δ AUC vs full"], r["Wilcoxon p"] = vec, "-", "-"
            else:
                r["Δ AUC vs full"] = f"{np.nanmean(vec) - np.nanmean(full_auc):+.3f}"
                try:
                    r["Wilcoxon p"] = f"{wilcoxon(full_auc, vec).pvalue:.3f}" if len(vec) == len(full_auc) and len(vec) >= 6 else "n/a"
                except ValueError:
                    r["Wilcoxon p"] = "n/a"
        else:
            x = out.get("xai")
            if x is None:
                continue
            s = x["summary"].set_index("explainer")
            tgt = "fused"
            r = {"variant": name, "comprehensiveness↑": s.loc[tgt, "comprehensiveness↑"], "deletion-AUC↓": s.loc[tgt, "deletion-AUC↓"],
                 "insertion-AUC↑": s.loc[tgt, "insertion-AUC↑"], "trust→correctness AUC": f"{x['trust_auc']:.3f}"}
        rows.append(r)
    return pd.DataFrame(rows)
