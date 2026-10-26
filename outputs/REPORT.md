# CHRONOX report — dataset: spedia

## Table 1. Dataset overview

| dataset | instances | users | raw_columns | start | end | anomaly_rate | rate_first60% | rate_60-80% | rate_last20% |
|---|---|---|---|---|---|---|---|---|---|
| spedia | 72250 | 17 | 21 | 2025-03-06 11:46:28 | 2025-03-30 19:45:22.120000 | 0.575 | 0.425 | 0.6493 | 0.9504 |

## Table 2. Detection (rolling-origin; mean ± SD over 4 blocks × 3 seeds)

| model | n_runs | AUC | PR-AUC | Recall | Accuracy | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|
| HGB | 12 | 0.959 ± 0.020 | 0.991 ± 0.003 | 0.946 ± 0.013 | 0.924 ± 0.010 | 0.949 ± 0.012 | 0.701 ± 0.116 | 0.086 ± 0.022 |
| CHRONOX | 12 | 0.953 ± 0.030 | 0.989 ± 0.004 | 0.945 ± 0.031 | 0.935 ± 0.015 | 0.955 ± 0.019 | 0.702 ± 0.131 | 0.046 ± 0.011 |
| RF | 12 | 0.953 ± 0.026 | 0.990 ± 0.003 | 0.937 ± 0.023 | 0.914 ± 0.019 | 0.943 ± 0.015 | 0.677 ± 0.122 | 0.100 ± 0.031 |
| ET | 12 | 0.929 ± 0.049 | 0.983 ± 0.007 | 0.945 ± 0.035 | 0.897 ± 0.029 | 0.933 ± 0.021 | 0.621 ± 0.129 | 0.144 ± 0.046 |
| MLP | 12 | 0.858 ± 0.069 | 0.966 ± 0.016 | 0.869 ± 0.169 | 0.821 ± 0.139 | 0.877 ± 0.097 | 0.500 ± 0.207 | 0.228 ± 0.091 |
| LR | 12 | 0.775 ± 0.102 | 0.942 ± 0.030 | 0.838 ± 0.210 | 0.751 ± 0.160 | 0.827 ± 0.119 | 0.306 ± 0.213 | 0.284 ± 0.109 |

## Table 3. Model overview (instances / features / cost)

| model | features | fit_seconds | train_instances(avg) | raw_features |
|---|---|---|---|---|
| CHRONOX | 13 | 24.585 | 49671 | 54 |
| ET | 13 | 1.204 | 49671 | 54 |
| HGB | 13 | 3.421 | 49671 | 54 |
| LR | 13 | 0.087 | 49671 | 54 |
| MLP | 13 | 18.283 | 49671 | 54 |
| RF | 13 | 1.59 | 49671 | 54 |

## Preprocessing audit
Features: 54 built → 13 selected. Benign adversarial AUC after audit: 1.000

| removed feature | reason |
|---|---|
| id | ID-like (cardinality > 90%) |
| Agent_name__rarity | single-feature AUC=0.995 |
| Agent_name__len | single-feature AUC=0.994 |
| dow_cos | univariate adversarial AUC=0.942 |
| user_event_count | univariate adversarial AUC=0.900 |
| Content__rarity | multivariate adversarial AUC=1.000 |
| dow_sin | multivariate adversarial AUC=1.000 |
| Size_before__uz | multivariate adversarial AUC=1.000 |
| Level__uz | multivariate adversarial AUC=1.000 |

## Table 4. Feature-subset Pareto front (<=10 features, top 10)

| fitness | auc_mean | auc_sd | k | psi_pen |
|---|---|---|---|---|

## Table 5. Explanation quality (on the deployed decision function)

| explainer | comprehensiveness↑ | deletion-AUC↓ | insertion-AUC↑ | top-k mass (sparsity)↑ | stability(top-k Jaccard) |
|---|---|---|---|---|---|
| shapley | 0.731 ± 0.193 | 0.209 ± 0.049 | 0.748 ± 0.183 | 0.884 ± 0.058 | 0.595 ± 0.218 |
| bc_lime | 0.630 ± 0.299 | 0.280 ± 0.130 | 0.697 ± 0.212 | 0.724 ± 0.075 | 0.616 ± 0.222 |
| tf_gradcam | 0.722 ± 0.230 | 0.270 ± 0.074 | 0.690 ± 0.162 | 0.764 ± 0.029 | 0.967 ± 0.111 |
| fused | 0.746 ± 0.184 | 0.207 ± 0.052 | 0.751 ± 0.178 | 0.741 ± 0.046 | n/a (fusion of the above) |

Trust→correctness AUC: **0.772**; Grad-CAM source active: **True**; CNN metrics: {'AUC': 0.950588153224446, 'PR-AUC': 0.9958564385959442, 'Recall': 0.9894217207334274, 'Accuracy': 0.9767493356953055, 'BalAcc': 0.8802070435728205, 'Precision': 0.9859451862262825, 'Specificity': np.float64(0.7709923664122137), 'F1': 0.9876803942273847, 'MCC': 0.7817861787729484, 'Brier': 0.01941343955695629}

## Table 6. Ablation — detection (CHRONOX ensemble)

| variant | features | AUC | PR-AUC | Recall | Accuracy | F1 | MCC | Δ AUC vs full | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| Full CHRONOX | 13 | 0.953 ± 0.031 | 0.989 ± 0.004 | 0.945 ± 0.032 | 0.936 ± 0.015 | 0.955 ± 0.020 | 0.704 ± 0.130 | - | - |
| - leakage/shortcut audit | 14 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | +0.047 | 0.008 |
| - adversarial audit only | 21 | 0.960 ± 0.015 | 0.989 ± 0.007 | 0.950 ± 0.039 | 0.932 ± 0.039 | 0.953 ± 0.029 | 0.716 ± 0.095 | +0.007 | 0.383 |
| - user-relative baselines | 12 | 0.823 ± 0.091 | 0.959 ± 0.016 | 0.787 ± 0.216 | 0.747 ± 0.162 | 0.813 ± 0.121 | 0.378 ± 0.194 | -0.130 | 0.008 |
| - sessionisation | 12 | 0.946 ± 0.032 | 0.987 ± 0.009 | 0.968 ± 0.027 | 0.932 ± 0.032 | 0.957 ± 0.020 | 0.671 ± 0.169 | -0.007 | 0.547 |
| - feature optimisation (GA) | 46 | 0.977 ± 0.014 | 0.993 ± 0.007 | 0.984 ± 0.016 | 0.954 ± 0.032 | 0.970 ± 0.022 | 0.810 ± 0.076 | +0.024 | 0.078 |
| - recency weighting | 13 | 0.954 ± 0.030 | 0.989 ± 0.004 | 0.958 ± 0.018 | 0.934 ± 0.014 | 0.955 ± 0.016 | 0.690 ± 0.140 | +0.001 | 0.250 |
| - prior-shift (EM) adaptation | 13 | 0.953 ± 0.031 | 0.989 ± 0.004 | 0.948 ± 0.016 | 0.921 ± 0.011 | 0.948 ± 0.012 | 0.688 ± 0.123 | +0.000 | 0.219 |
| - stability-weighted ensemble | 13 | 0.957 ± 0.023 | 0.990 ± 0.004 | 0.960 ± 0.029 | 0.935 ± 0.017 | 0.955 ± 0.018 | 0.684 ± 0.135 | +0.004 | 0.641 |
| Baseline (repo-style: no audit/GA/user/session/recency/EM/ensemble) | 30 | 0.999 ± 0.002 | 0.998 ± 0.003 | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.999 ± 0.002 | 0.997 ± 0.003 | +0.046 | 0.008 |

## Table 7. Ablation — explanation (RAFE)

| variant | comprehensiveness↑ | deletion-AUC↓ | insertion-AUC↑ | trust→correctness AUC |
|---|---|---|---|---|
| Full RAFE | 0.746 ± 0.184 | 0.207 ± 0.052 | 0.751 ± 0.178 | 0.772 |
| - BC-LIME (global-pool LIME instead) | 0.756 ± 0.173 | 0.204 ± 0.049 | 0.753 ± 0.178 | 0.663 |
| - TF-Grad-CAM source | 0.735 ± 0.194 | 0.211 ± 0.054 | 0.747 ± 0.182 | 0.745 |
| - reliability weighting (equal weights) | 0.732 ± 0.194 | 0.212 ± 0.056 | 0.748 ± 0.183 | 0.772 |
