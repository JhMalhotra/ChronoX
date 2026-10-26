from dataclasses import dataclass, asdict


@dataclass
class Cfg:
    # ---- data ----
    dataset: str = "spedia"                      # spedia | cert
    data_path: str = "data/logs_SPEDIA_annotated_en.csv"
    cert_root: str = "data/cert_r4.2"
    cert_answers: str = "data/cert_r4.2/answers"
    cert_users: int = 150                        # benign users sampled (all insiders kept)
    label_col: str = "Anomaly"
    ts_col: str | None = None                    # auto-detect when None
    user_col: str | None = None
    drop_cols: tuple = ()
    max_rows: int | None = None
    # ---- preprocessing ----
    session_gap_min: int = 30
    deny_regex: str = r"anomal|label|annot|malic|mitre|tactic|techni|ttp|threat|attack"
    leak_auc: float = 0.97                       # single-feature AUC flagged as leakage
    adv_target_auc: float = 0.70
    adv_max_drop: int = 4
    adv_feature_auc: float = 0.75
    # ---- evaluation ----
    seeds: tuple = (0, 1, 2)
    n_blocks: int = 4                            # rolling-origin test blocks
    dev_frac: float = 0.5                        # earliest fraction used for audit + GA only
    n_estimators: int = 200
    # ---- detection toggles (ablation) ----
    use_sessions: bool = True
    use_user_baselines: bool = True
    use_leak_audit: bool = True
    use_adv_audit: bool = True
    use_featopt: bool = True
    use_recency: bool = True
    use_prior_shift: bool = True
    use_ensemble: bool = True
    # ---- feature optimisation ----
    ga_pop: int = 24
    ga_gen: int = 12
    ga_rows: int = 9000
    min_features: int = 8
    lam_std: float = 0.5
    mu_sparse: float = 0.10
    nu_psi: float = 0.30
    # ---- XAI ----
    run_xai: bool = True
    n_explain: int = 120
    shap_perms: int = 24
    lime_samples: int = 500
    topk: int = 5
    use_cnn: bool = True
    seq_len: int = 32
    cnn_epochs: int = 6
    xai_bc: bool = True                          # baseline-conditioned LIME (else global-pool LIME)
    xai_gradcam: bool = True
    xai_fusion: bool = True
    xai_reliability: bool = True                 # reliability weights (else equal weights)
    out_dir: str = "outputs"

    def asdict(self):
        return asdict(self)
