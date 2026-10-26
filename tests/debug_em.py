import sys; sys.path.insert(0,".")
import numpy as np
from chronox.config import Cfg
from chronox import data, evaluate, models
cfg = Cfg(data_path="/tmp/synth.csv", seeds=(0,), n_blocks=3, n_estimators=60, ga_pop=8, ga_gen=4, ga_rows=3000, use_cnn=False, run_xai=False)
df = data.load(cfg)
X, meta, keep, log = evaluate.prepare(df, cfg)
print("dropped:", [d[0] for d in log["dropped"]], "kept", len(keep), "adv", round(log["adv_auc"],3))
y, ts = meta.y.values, meta.ts.values; n=len(y); A=X[keep].values
edges=np.linspace(int(n*.5), n, 4).astype(int)
b=0; te=np.arange(edges[b],edges[b+1]); pre=np.arange(0,edges[b]); cut=int(len(pre)*.8); tr,va=pre[:cut],pre[cut:]
ens=models.ChronoEnsemble(cfg,0).fit(A[tr],y[tr],ts[tr],A[va],y[va],ts[va])
raw=ens.predict_raw(A[te]); cal=ens._cal(raw)
print("thr",ens.thr,"pi_val",ens.pi_val, "val prev", y[va].mean(), "test prev", y[te].mean())
print("raw quantiles", np.quantile(raw,[.1,.5,.9]).round(3), "cal quantiles", np.quantile(cal,[.1,.5,.9]).round(3))
q,pi=models.em_prior(cal,ens.pi_val); print("pi_hat",pi,"q quantiles",np.quantile(q,[.1,.5,.9]).round(3))
for name,pp in (("cal",cal),("q",q)):
    pred=pp>=ens.thr; print(name,"recall",(pred&(y[te]==1)).sum()/(y[te]==1).sum())
