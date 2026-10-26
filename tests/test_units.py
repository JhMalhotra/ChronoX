import sys; sys.path.insert(0, ".")
import numpy as np
from chronox import models, preprocess, xai

def test_window_index():
    idx = models.window_index(np.array([0, 1, 0, 0, 1]), 3)
    assert idx[3].tolist() == [0, 2, 3] and idx[4].tolist() == [-1, 1, 4] and idx[0].tolist() == [-1, -1, 0]

def test_em_prior_recovers_shift():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.7).astype(int)
    s = rng.normal(y * 2.0, 1.0)
    p_tr = lambda s_, pi: 1 / (1 + np.exp(-(2 * s_ - 2 + np.log(pi / (1 - pi)))))   # posterior for N(2,1) vs N(0,1)
    q, pi = models.em_prior(p_tr(s, 0.2), 0.2)
    assert abs(pi - 0.7) < 0.03

def test_psi_zero_on_same():
    a = np.random.default_rng(0).normal(size=5000)
    assert preprocess.psi(a, a) < 1e-6

def test_shapley_additivity():
    f = lambda X: X @ np.array([1.0, 2.0, 0.0])
    x, bg = np.array([1.0, 1.0, 1.0]), np.zeros((5, 3))
    phi = xai.sampling_shapley(f, x, bg, 50, np.random.default_rng(0))
    assert np.allclose(phi, [1, 2, 0], atol=1e-6)

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print(k, "ok")
