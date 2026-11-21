"""Stage F: Bayesian optimization behind the ask/tell interface.

A compact, transparent GP (RBF kernel) + Expected Improvement. This is the
direct ancestor of the curiosity method: later we keep THIS acquisition and swap
the GP surrogate for the dynamics ensemble (the key ablation).

  surrogate : GP posterior mean/variance over the objective as a function of u
  acquisition: EI for MINIMIZATION, maximized over a pool of feasible candidates

The objective is deterministic (fixed run_seed), so GP noise is tiny (jitter for
numerical stability). Hyperparameters are fixed for v0 (lengthscale is a knob);
BO in d=11 is genuinely hard, which is itself an informative baseline result.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr   # standard normal CDF

from breakpoint.scenarios.space import DIM, rejection_sample
from breakpoint.search.base import Searcher

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / _SQRT_2PI


class BOSearch(Searcher):
    def __init__(self, seed: int = 0, n_init: int = 15, lengthscale: float = 0.5,
                 sigma_f: float = 1.0, noise: float = 1e-6, n_candidates: int = 800):
        self.rng = np.random.default_rng(seed)
        self.n_init = n_init
        self.ls = lengthscale
        self.sf2 = sigma_f ** 2
        self.noise = noise
        self.n_cand = n_candidates
        self.X: list = []
        self.y: list = []

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return self.sf2 * np.exp(-0.5 * d2 / self.ls ** 2)

    def _candidates(self) -> np.ndarray:
        return np.array([rejection_sample(self.rng)[0] for _ in range(self.n_cand)])

    def ask(self) -> np.ndarray:
        if len(self.X) < self.n_init:
            u, _, _ = rejection_sample(self.rng)
            return u

        X = np.array(self.X)
        y = np.array(self.y)
        ymu, ysd = y.mean(), y.std() + 1e-9
        ys = (y - ymu) / ysd

        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        L = cho_factor(K, lower=True)
        alpha = cho_solve(L, ys)

        cand = self._candidates()
        Ks = self._kernel(cand, X)
        mu = Ks @ alpha
        v = solve_triangular(L[0], Ks.T, lower=True)
        var = np.maximum(self.sf2 - (v ** 2).sum(0), 1e-12)
        s = np.sqrt(var)

        fbest = ys.min()                 # minimization
        imp = fbest - mu
        z = imp / s
        ei = imp * ndtr(z) + s * _norm_pdf(z)
        ei[s < 1e-9] = 0.0
        return cand[int(np.argmax(ei))]

    def tell(self, u: np.ndarray, obj: float) -> None:
        self.X.append(np.asarray(u, dtype=float))
        self.y.append(float(obj))