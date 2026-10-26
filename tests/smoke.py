import dataclasses, sys, time
sys.path.insert(0, ".")
import pandas as pd
from chronox.config import Cfg
from chronox import data, evaluate
from tests.make_synth import make

raw = make()
raw.to_csv("/tmp/synth.csv", index=False)
cfg = Cfg(data_path="/tmp/synth.csv", seeds=(0, 1), n_blocks=3, n_estimators=60, ga_pop=8, ga_gen=4, ga_rows=3000,
          n_explain=15, shap_perms=8, lime_samples=200, use_cnn=False)
df = data.load(cfg)
print(data.dataset_overview({"synthetic": df}).T)
t0 = time.time()
out = evaluate.run_experiment(df, cfg)
print("elapsed %.1fs" % (time.time() - t0))
print("dropped:", out["log"]["dropped"])
print("selected", out["log"]["n_features_selected"], "/", out["log"]["n_features_total"], "adv_auc", out["log"]["adv_auc"])
print(out["summary"].to_string())
x = out.get("xai")
if x:
    print(x["summary"].to_string()); print("trust_auc", x["trust_auc"])
