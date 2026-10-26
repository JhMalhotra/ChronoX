import sys; sys.path.insert(0,".")
import dataclasses, numpy as np
from chronox.config import Cfg
from chronox import data, evaluate, models
from tests.make_synth import make
make().to_csv("/tmp/synth.csv", index=False)
cfg = Cfg(data_path="/tmp/synth.csv", seeds=(0,), n_blocks=3, n_estimators=60, ga_pop=8, ga_gen=4, ga_rows=3000, use_cnn=False, run_xai=False)
df = data.load(cfg)
for ps in (True, False):
    c = dataclasses.replace(cfg, use_prior_shift=ps)
    out = evaluate.run_experiment(df, c)
    g = out["results"].query("model=='CHRONOX'")[["block","AUC","Recall","Accuracy","block_prev"]]
    print("prior_shift", ps); print(g.to_string(index=False))
print(out["log"]["dropped"]); print(out["log"]["n_features_selected"], out["log"]["adv_auc"])
