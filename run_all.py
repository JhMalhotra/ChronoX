"""python run_all.py --dataset spedia --data-path data/logs_SPEDIA_annotated_en.csv [--quick] [--no-ablation]"""
import argparse, dataclasses, json, os
import numpy as np
import pandas as pd
from chronox import ablation, data, evaluate, viz
from chronox.config import Cfg


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="spedia"); ap.add_argument("--data-path"); ap.add_argument("--cert-root")
    ap.add_argument("--cert-answers"); ap.add_argument("--out", default="outputs"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-ablation", action="store_true"); ap.add_argument("--max-rows", type=int)
    a = ap.parse_args()
    cfg = Cfg(dataset=a.dataset, out_dir=a.out, seeds=tuple(range(a.seeds)), n_blocks=a.blocks, max_rows=a.max_rows)
    if a.data_path: cfg = dataclasses.replace(cfg, data_path=a.data_path)
    if a.cert_root: cfg = dataclasses.replace(cfg, cert_root=a.cert_root, cert_answers=a.cert_answers or os.path.join(a.cert_root, "answers"))
    if a.quick:
        cfg = dataclasses.replace(cfg, seeds=(0, 1), n_blocks=3, n_estimators=60, ga_pop=8, ga_gen=4, ga_rows=3000, n_explain=20, shap_perms=8, lime_samples=200)
    os.makedirs(cfg.out_dir, exist_ok=True)
    O = lambda n: os.path.join(cfg.out_dir, n)
    rep = [f"# CHRONOX report — dataset: {cfg.dataset}\n"]

    df = data.load(cfg)
    ov = data.dataset_overview({cfg.dataset: df}); ov.to_csv(O("table1_dataset_overview.csv"), index=False)
    rep += ["## Table 1. Dataset overview\n", md(ov), ""]

    out = evaluate.run_experiment(df, cfg)
    res, log = out["results"], out["log"]
    res.to_csv(O("raw_results_per_block_seed.csv"), index=False)
    out["summary"].to_csv(O("table2_detection_mean_sd.csv"), index=False)
    mo = res.groupby("model").agg(features=("AUC", lambda s: log["n_features_selected"]), fit_seconds=("fit_s", "mean")).reset_index()
    mo["train_instances(avg)"] = int(np.mean([len(df) * (cfg.dev_frac + i * (1 - cfg.dev_frac) / cfg.n_blocks) for i in range(cfg.n_blocks)]))
    mo["raw_features"] = log["n_features_total"]
    mo.to_csv(O("table3_model_overview.csv"), index=False)
    rep += [f"## Table 2. Detection (rolling-origin; mean ± SD over {cfg.n_blocks} blocks × {len(cfg.seeds)} seeds)\n", md(out["summary"]), "",
            "## Table 3. Model overview (instances / features / cost)\n", md(mo.round(3)), "",
            f"## Preprocessing audit\nFeatures: {log['n_features_total']} built → {log['n_features_selected']} selected. Benign adversarial AUC after audit: {log['adv_auc']:.3f}\n",
            md(pd.DataFrame(log["dropped"], columns=["removed feature", "reason"])) if log["dropped"] else "_nothing removed_", ""]
    if log["pareto"] is not None:
        # Keep only subsets with <=10 features and select top 10 by AUC mean
        pareto_filtered = log["pareto"][log["pareto"]["k"] <= 10].sort_values("auc_mean", ascending=False).head(10)
        pareto_filtered.to_csv(O("table4_feature_pareto.csv"), index=False)
        rep += ["## Table 4. Feature-subset Pareto front (<=10 features, top 10)\n", md(pareto_filtered.round(3)), ""]
    viz.ga_convergence(log["ga_history"], O("fig_ga_convergence.png")); viz.drift_plot(res, O("fig_drift_auc_vs_prevalence.png"))

    x = out.get("xai")
    if x:
        x["per_alert"].drop(columns=[c for c in x["per_alert"] if c == "heat"], errors="ignore").to_csv(O("xai_per_alert.csv"), index=False)
        x["summary"].to_csv(O("table5_explanation_quality.csv"), index=False)
        rep += ["## Table 5. Explanation quality (on the deployed decision function)\n", md(x["summary"]),
                f"\nTrust→correctness AUC: **{x['trust_auc']:.3f}**; Grad-CAM source active: **{x['has_gradcam']}**; CNN metrics: {x['cnn_metrics']}\n"]
        viz.fused_bar(x["store"][0], x["names"], O("fig_fused_importance.png")); viz.gradcam_heatmap(x["store"][0], x["names"], O("fig_tf_gradcam.png"))
        # ---- XAI Dashboard generation ----

        dash_path = O("xai_dashboard.html")
        with open(dash_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>XAI Dashboard</title>")
            f.write("<style>body{font-family:Arial,sans-serif;margin:20px;}h1{color:#2c3e50;}h2{color:#34495e;}img{max-width:100%;height:auto;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f2f2f2;}</style>")
            f.write("</head><body>")
            f.write("<h1>Chronox XAI Dashboard</h1>")
            f.write("<h2>Explanation Quality Summary</h2>")
            f.write("<pre>" + str(x["summary"]) + "</pre>")
            f.write("<h2>Feature Importance (Fused)</h2>")
            f.write("<img src='fig_fused_importance.png' alt='Fused importance'>")
            f.write("<h2>Grad‑CAM Heatmap</h2>")
            f.write("<img src='fig_tf_gradcam.png' alt='Grad‑CAM heatmap'>")
            try:
                alerts_df = pd.read_csv(O("xai_per_alert.csv"))
                f.write("<h2>Per‑Alert Explanations</h2>")
                f.write(alerts_df.to_html(index=False, classes='table'))
            except Exception as e:
                f.write("<p>Could not load per‑alert CSV: " + str(e) + "</p>")
            f.write("</body></html>")

    if not a.no_ablation:
        ab = ablation.run_ablation(df, dataclasses.replace(cfg, seeds=cfg.seeds[:2]), "detection"); ab.to_csv(O("table6_ablation_detection.csv"), index=False)
        rep += ["## Table 6. Ablation — detection (CHRONOX ensemble)\n", md(ab), ""]
        ax = ablation.run_ablation(df, dataclasses.replace(cfg, seeds=cfg.seeds[:1]), "xai"); ax.to_csv(O("table7_ablation_xai.csv"), index=False)
        rep += ["## Table 7. Ablation — explanation (RAFE)\n", md(ax), ""]
    open(O("REPORT.md"), "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
