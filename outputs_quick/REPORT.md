# CHRONOX report — dataset: spedia

## Table 1. Dataset overview

| dataset | instances | users | raw_columns | start | end | anomaly_rate | rate_first60% | rate_60-80% | rate_last20% |
|---|---|---|---|---|---|---|---|---|---|
| spedia | 72250 | 17 | 21 | 2025-03-06 11:46:28 | 2025-03-30 19:45:22.120000 | 0.575 | 0.425 | 0.6493 | 0.9504 |

## Table 2. Detection (rolling-origin; mean ± SD over 3 blocks × 2 seeds)

| model | n_runs | AUC | PR-AUC | Recall | Accuracy | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|---|
| CHRONOX | 6 | 0.974 ± 0.008 | 0.991 ± 0.005 | 0.979 ± 0.011 | 0.948 ± 0.023 | 0.966 ± 0.017 | 0.812 ± 0.055 | 0.039 ± 0.015 |
| RF | 6 | 0.972 ± 0.011 | 0.992 ± 0.006 | 0.991 ± 0.005 | 0.945 ± 0.038 | 0.964 ± 0.027 | 0.817 ± 0.070 | 0.128 ± 0.037 |
| HGB | 6 | 0.961 ± 0.024 | 0.986 ± 0.015 | 0.980 ± 0.011 | 0.928 ± 0.054 | 0.954 ± 0.037 | 0.755 ± 0.122 | 0.090 ± 0.029 |
| MLP | 6 | 0.952 ± 0.015 | 0.986 ± 0.007 | 0.967 ± 0.023 | 0.916 ± 0.026 | 0.947 ± 0.019 | 0.706 ± 0.115 | 0.143 ± 0.080 |
| ET | 6 | 0.950 ± 0.022 | 0.987 ± 0.009 | 0.961 ± 0.034 | 0.908 ± 0.037 | 0.940 ± 0.028 | 0.690 ± 0.053 | 0.156 ± 0.039 |
| LR | 6 | 0.886 ± 0.011 | 0.961 ± 0.024 | 0.903 ± 0.056 | 0.850 ± 0.044 | 0.900 ± 0.038 | 0.509 ± 0.085 | 0.240 ± 0.033 |

## Table 3. Model overview (instances / features / cost)

| model | features | fit_seconds | train_instances(avg) | raw_features |
|---|---|---|---|---|
| CHRONOX | 28 | 7.233 | 48166 | 54 |
| ET | 28 | 0.217 | 48166 | 54 |
| HGB | 28 | 1.77 | 48166 | 54 |
| LR | 28 | 0.08 | 48166 | 54 |
| MLP | 28 | 4.852 | 48166 | 54 |
| RF | 28 | 0.314 | 48166 | 54 |

## Preprocessing audit
Features: 54 built → 28 selected. Benign adversarial AUC after audit: 1.000

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

## Table 4. Feature-subset Pareto front (top 15)

| fitness | auc_mean | auc_sd | k | psi_pen |
|---|---|---|---|---|
| 0.836 | 0.978 | 0.007 | 28 | 0.259 |
| 0.833 | 0.966 | 0.001 | 25 | 0.263 |
| 0.797 | 0.965 | 0.023 | 22 | 0.365 |
| 0.805 | 0.961 | 0.03 | 30 | 0.255 |
| 0.834 | 0.959 | 0.025 | 26 | 0.189 |
| 0.796 | 0.953 | 0.023 | 23 | 0.319 |
| 0.789 | 0.953 | 0.026 | 22 | 0.342 |
| 0.807 | 0.948 | 0.023 | 25 | 0.251 |
| 0.803 | 0.944 | 0.029 | 24 | 0.247 |
| 0.821 | 0.941 | 0.023 | 25 | 0.178 |
| 0.792 | 0.939 | 0.054 | 24 | 0.227 |
| 0.788 | 0.939 | 0.023 | 23 | 0.299 |
| 0.8 | 0.939 | 0.028 | 23 | 0.252 |
| 0.819 | 0.938 | 0.023 | 26 | 0.172 |
| 0.774 | 0.915 | 0.073 | 29 | 0.141 |

## Table 5. Explanation quality (on the deployed decision function)

| explainer | comprehensiveness↑ | deletion-AUC↓ | insertion-AUC↑ | top-k mass (sparsity)↑ | stability(top-k Jaccard) |
|---|---|---|---|---|---|
| shapley | 0.570 ± 0.242 | 0.287 ± 0.046 | 0.656 ± 0.210 | 0.685 ± 0.081 | 0.439 ± 0.177 |
| bc_lime | 0.240 ± 0.224 | 0.408 ± 0.107 | 0.563 ± 0.250 | 0.445 ± 0.040 | 0.326 ± 0.163 |
| tf_gradcam | 0.229 ± 0.261 | 0.446 ± 0.085 | 0.499 ± 0.218 | 0.561 ± 0.024 | 0.983 ± 0.075 |
| fused | 0.590 ± 0.253 | 0.295 ± 0.056 | 0.647 ± 0.211 | 0.565 ± 0.071 | n/a (fusion of the above) |

Trust→correctness AUC: **0.684**; Grad-CAM source active: **True**; CNN metrics: {'AUC': 0.9461640402922649, 'PR-AUC': 0.9942451282522411, 'Recall': 0.9835304424003504, 'Accuracy': 0.9723467862481315, 'BalAcc': 0.8761352371491384, 'Precision': 0.9872493844530426, 'Specificity': np.float64(0.7687400318979266), 'F1': 0.985386404528898, 'MCC': 0.7291047340324854, 'Brier': 0.02361704222857952}

## Table 6. Ablation — detection (CHRONOX ensemble)

| variant | features | AUC | PR-AUC | Recall | Accuracy | F1 | MCC | Δ AUC vs full | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| Full CHRONOX | 28 | 0.974 ± 0.008 | 0.991 ± 0.005 | 0.979 ± 0.011 | 0.948 ± 0.023 | 0.966 ± 0.017 | 0.812 ± 0.055 | - | - |
| - leakage/shortcut audit | 27 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 | +0.026 | 0.031 |
| - adversarial audit only | 32 | 0.915 ± 0.013 | 0.966 ± 0.020 | 0.956 ± 0.028 | 0.872 ± 0.069 | 0.918 ± 0.046 | 0.572 ± 0.085 | -0.059 | 0.031 |
| - user-relative baselines | 12 | 0.846 ± 0.030 | 0.945 ± 0.027 | 0.876 ± 0.112 | 0.844 ± 0.080 | 0.888 ± 0.066 | 0.449 ± 0.135 | -0.128 | 0.031 |
| - sessionisation | 23 | 0.980 ± 0.012 | 0.994 ± 0.002 | 0.979 ± 0.016 | 0.963 ± 0.016 | 0.975 ± 0.012 | 0.873 ± 0.018 | +0.006 | 0.219 |
| - feature optimisation (GA) | 46 | 0.972 ± 0.009 | 0.989 ± 0.008 | 0.922 ± 0.084 | 0.905 ± 0.054 | 0.933 ± 0.043 | 0.732 ± 0.028 | -0.003 | 0.562 |
| - recency weighting | 28 | 0.974 ± 0.010 | 0.992 ± 0.003 | 0.970 ± 0.016 | 0.950 ± 0.019 | 0.967 ± 0.015 | 0.819 ± 0.052 | -0.000 | 0.688 |
| - prior-shift (EM) adaptation | 28 | 0.974 ± 0.008 | 0.991 ± 0.005 | 0.982 ± 0.013 | 0.935 ± 0.037 | 0.958 ± 0.026 | 0.778 ± 0.075 | +0.000 | 1.000 |
| - stability-weighted ensemble | 28 | 0.975 ± 0.010 | 0.992 ± 0.004 | 0.994 ± 0.003 | 0.950 ± 0.035 | 0.967 ± 0.025 | 0.841 ± 0.056 | +0.001 | 0.688 |
| Baseline (repo-style: no audit/GA/user/session/recency/EM/ensemble) | 30 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.001 | 0.999 ± 0.002 | +0.026 | 0.031 |

## Table 7. Ablation — explanation (RAFE)

| variant | comprehensiveness↑ | deletion-AUC↓ | insertion-AUC↑ | trust→correctness AUC |
|---|---|---|---|---|
| Full RAFE | 0.590 ± 0.253 | 0.295 ± 0.056 | 0.647 ± 0.211 | 0.684 |
| - BC-LIME (global-pool LIME instead) | 0.564 ± 0.207 | 0.279 ± 0.051 | 0.664 ± 0.196 | 0.737 |
| - TF-Grad-CAM source | 0.581 ± 0.270 | 0.275 ± 0.041 | 0.659 ± 0.207 | 0.526 |
| - reliability weighting (equal weights) | 0.554 ± 0.301 | 0.315 ± 0.059 | 0.627 ± 0.234 | 0.684 |
