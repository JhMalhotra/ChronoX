# CHRONOX methodology and novelty statement

## 1. Problem (from the repo's own measurements)
Random stratified split: ROC-AUC 0.9635. Chronological split: ROC-AUC 0.4536 (<0.5 = ranking *inverted*), PR-AUC 0.8385 only because prevalence is 80%.
AUC is invariant to label-prior shift, so an AUC below 0.5 implies **concept/covariate shift or a shortcut that stopped holding**, not just a prevalence change. Likely suspects (to be confirmed by the audits below): source-identifying fields (SPEDIA mixes three generators), time-trending fields, and annotation columns.

## 2. Contributions
**C1 Leakage- and shortcut-aware preprocessing.** (a) name deny-list for annotation/label-like columns; (b) single-feature leakage scan; (c) *class-conditional* adversarial validation: only benign rows are used to ask "can a model tell the training period from the later period?" — benign behaviour should be stationary, and using all rows would confuse label-prior shift with covariate shift and delete the genuinely predictive features (observed in my first prototype). Stage 1 drops features that reveal the period on their own (univariate AUC ≥ 0.75), stage 2 iteratively drops the most revealing remaining feature until benign period-AUC ≤ 0.70.
**C2 Leakage-safe behavioural features.** Sessionisation (gap rule), cyclical time encodings, *user-relative* baselines computed from the same user's strict past (expanding mean/std z-scores, "first time this user did X"), category rarity fitted on the training window only.
**C3 Stability-aware feature optimisation.** Genetic search maximising
`fitness(S) = mean_AUC_rolling(S) − λ·sd_AUC_rolling(S) − μ·|S|/d − ν·mean_PSI(S)`; the Pareto front over (AUC↑, SD↓, |S|↓, PSI↓) is reported. Searched on the earliest 50% of the timeline only.
**C4 Drift-robust detector.** Recency-weighted training; ensemble weights `softmax((mean_blockAUC − sd_blockAUC)/T)` on a chronological validation window; isotonic calibration; EM label-shift adaptation of posteriors on the unlabeled test window (Saerens 2002) with the F1-optimal posterior threshold F1*/2 (Lipton 2014) so the decision actually moves with prevalence.
**C5 RAFE — Reliability-Aware Fused Explanations.** For each alert, three attributions in one feature space:
 1. interventional Shapley values (sampling estimator; exact TreeSHAP baseline optional);
 2. **BC-LIME**: LIME whose perturbations substitute the *user's own past events* (on-manifold, user-specific baseline) instead of global noise;
 3. **TF-Grad-CAM**: Grad-CAM over the 1D-CNN's last conv layer along the user's last-L event window × |input·gradient| → an (event position × feature) heat-map: *when* and *what*.
 Reliability `r_s = comprehensiveness_s × stability_s` (measured against the deployed decision function; stability = top-k Jaccard under re-sampling/noise). Fused importance `Σ_s w_s·|â_s|`, `w_s = r_s/Σr`. **Trust** = max(mean pairwise Spearman agreement, 0) × mean r. Trust is validated by whether it predicts that an alert is correct (trust→correctness AUC).
**Honest framing of novelty.** From the ~10 searches I ran I did not find prior insider-threat work combining C1's class-conditional audit with C3–C5, but that is not a systematic review. Do one before writing "first". Individual ingredients (SHAP/LIME agreement, Grad-CAM for cyber data, session-based features, SHAP sparsity metrics) exist.

## 3. Evaluation protocol
* Rolling-origin: first 50% of time = development (audits, GA); remaining 50% split into K blocks; each block is predicted by models trained only on earlier data (last 20% of that history = validation for weights/calibration/threshold).
* Repeats: seeds × blocks; report mean ± SD of AUC, PR-AUC, recall, accuracy, F1, MCC, Brier (`table2`).
* Dataset overview, model overview (instances, raw/selected features, fit time) → `table1`, `table3`.
* Report **both** this and the repo's random-split number; the gap is itself a result.
* Datasets: SPEDIA (primary). CERT r4.2 (secondary, via loader). I deliberately do **not** pool them into one chronological timeline (different eras, different generators); evaluate separately and, if desired, train-on-one/test-on-other as a transfer test. SPEDIA already embeds CERT-derived synthetic events, so pooling would partly double-count.
* Explanation metrics (`table5`): comprehensiveness@k, deletion-AUC, insertion-AUC, top-k mass, stability, trust→correctness AUC.
* Ablation (`table6/7`): remove one component at a time; Wilcoxon signed-rank on paired (block, seed) AUCs vs the full model; plus a repo-style baseline row.

## 4. Literature used (2020–2026; verified via search snippets only — read the full papers before citing)
| ref | used for |
|---|---|
| Mateo-Muñoz et al., 2026, *A session-based insider threat detection approach leveraging the CERT and SPEDIA datasets*, Mach. Learn. Appl. 25:100946 | session-level detection; 12 models compared, XGBoost best; SPEDIA validation → C2, model zoo |
| *AI-Based Insider-Threat Detection: A Hybrid Explainable Framework with Automated Response and Privilege Containment*, Computers 15(7):426, 2026 | hybrid ensemble + SHAP on CERT → C4 baseline to compare against |
| Interpretable and robust intrusion detection: hybrid CBAM-CNN with XAI, Comput. Electr. Eng. 138:111348, 2026 | 1D-CNN + SHAP/LIME, descriptive-accuracy and sparsity metrics → C5 metrics |
| Hermosilla et al., 2025, Computers 14(5):160 | SHAP vs LIME for forensic IDS |
| Systematic review, XAI in IDS, Front. Artif. Intell. 8:1526221, 2025 | XAI overhead and gaps → motivation for fusion/trust |
| Abbasi et al., *Enhancing Detection of Anomaly-Based Intrusions in Smart Health Using Explainable AI* (year to confirm) | LIME–SHAP agreement across seeds → agreement term of trust |
| Facade, arXiv:2412.06700, 2024 | CERT benign realism and false-positive critique → threats to validity |
| Yuan & Wu, arXiv:2005.12433, 2020 | insider-threat DL survey, CERT scenarios |
| Hernandes Junior, Unesp thesis 2025; Diff-DDoS arXiv:2608.17796 | quantitative Grad-CAM statistics (saliency-mass on attacked cells) → TF-Grad-CAM evaluation |
| SPEDIA dataset, Zenodo (DOI 10.5281/zenodo.15525713) | primary data |
**Outside the requested window but used as original algorithm sources:** Saerens et al. 2002 (EM prior adaptation), Lipton et al. 2014 (F1*/2 threshold).

## 5. Threats to validity
* A leakage scan cannot tell a leaky field from a strongly informative one — in my synthetic test the audit *lowered* headline AUC because the removed field was a perfect (synthetic) signal. Review the `removed feature` table by hand; thresholds are in `Cfg`.
* Shapley/LIME/Grad-CAM explain different models (tabular ensemble vs CNN); fusion is a consensus over views, scored against the deployed tabular decision, not a single causal explanation.
* Grad-CAM is positive-evidence only; LIME/Shapley are signed. Fusion uses magnitudes.
* Explanation faithfulness metrics rely on a baseline (benign median); results depend on that choice.
* Labels are the dataset's `Anomaly` flag, not adjudicated malicious intent; SPEDIA is partly synthetic.
