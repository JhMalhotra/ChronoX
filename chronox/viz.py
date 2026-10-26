import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ga_convergence(hist, path):
    if not hist:
        return
    g = [h["gen"] for h in hist]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(g, [h["best_fitness"] for h in hist], label="best"); ax.plot(g, [h["mean_fitness"] for h in hist], label="mean")
    ax.set_xlabel("generation"); ax.set_ylabel("stability-aware fitness"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def drift_plot(res, path):
    g = res.groupby(["block", "model"])["AUC"].mean().unstack()
    prev = res.groupby("block")["block_prev"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m in [c for c in ("CHRONOX", "RF", "HGB", "LR") if c in g]:
        ax.plot(g.index, g[m], marker="o", label=m)
    ax.set_xlabel("rolling-origin test block (time →)"); ax.set_ylabel("AUC"); ax.set_ylim(0, 1.02)
    ax2 = ax.twinx(); ax2.bar(prev.index, prev.values, alpha=0.15, color="gray"); ax2.set_ylabel("anomaly prevalence")
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fused_bar(store_item, names, path, k=10):
    f = store_item["fused"]; o = np.argsort(-f)[:k][::-1]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh([names[i] for i in o], f[o]); ax.set_xlabel("RAFE fused importance"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def gradcam_heatmap(store_item, names, path, k=12):
    H = store_item.get("heat")
    if H is None:
        return
    top = np.argsort(-H.sum(0))[:k]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    im = ax.imshow(H[:, top].T, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(top))); ax.set_yticklabels([names[i] for i in top], fontsize=7)
    ax.set_xlabel("event position in user's last-L window (right = the alert)"); fig.colorbar(im, label="TF-Grad-CAM")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
