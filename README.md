# CHRONOX — drift-robust, explanation-reliable insider-threat detection

Built on top of `mahimag07/exai-insider-threat-detection` (SPEDIA, RF/ET/LR + SHAP dashboard).
Motivation from that repo's own `RUN_RESULTS_SUMMARY.md`: random-split AUC 0.9635, but **chronological test AUC 0.4536**
with anomaly prevalence moving 18% → 76% → 80%. CHRONOX is designed around that failure.

## Quick start
```bash
pip install -r requirements.txt
cp -r chronox run_all.py <your-repo>/            # or run from this folder
python run_all.py --dataset spedia --data-path data/logs_SPEDIA_annotated_en.csv          # full run + ablations
python run_all.py --dataset spedia --data-path data/logs_SPEDIA_annotated_en.csv --quick  # ~minutes, sanity check
# optional second dataset (CERT r4.2: logon/device/file/http/email.csv + answers/):
python run_all.py --dataset cert --cert-root data/cert_r4.2
python tests/test_units.py
```
Outputs land in `outputs/`: `REPORT.md` (all tables), `table1…table7*.csv`, `raw_results_per_block_seed.csv`, figures.

## Layout
| file | role |
|---|---|
| `chronox/data.py` | SPEDIA loader (auto-detects timestamp/user columns), CERT r4.2 loader, dataset overview table |
| `chronox/preprocess.py` | sessionisation, user-relative baselines, leakage scan, class-conditional adversarial audit, PSI |
| `chronox/featopt.py` | stability-aware genetic feature optimisation + Pareto front |
| `chronox/models.py` | model zoo, stability-weighted ensemble, isotonic calibration, EM prior-shift, 1D-CNN + Grad-CAM |
| `chronox/xai.py` | RAFE: Shapley + BC-LIME + TF-Grad-CAM, reliability fusion, trust score, explanation metrics |
| `chronox/evaluate.py` | rolling-origin protocol, metrics, experiment driver |
| `chronox/ablation.py` | detection and explanation ablations (with Wilcoxon tests) |
