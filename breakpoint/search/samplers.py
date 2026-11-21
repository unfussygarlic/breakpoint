"""Stage E: non-adaptive baselines -- random search and Latin-hypercube sampling.

Both only need ask(); tell() is inherited as a no-op. Both return FEASIBLE
scenarios (infeasible draws are rejection-replaced). For LHS this slightly
weakens the strict space-filling property -- we report how many were replaced.
"""
from __future__ import annotations

import numpy as np

from breakpoint.scenarios.space import DIM, rejection_sample, unnormalize, is_feasible
from breakpoint.search.base import Searcher


class RandomSearch(Searcher):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        u, _, _ = rejection_sample(self.rng)
        return u


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube design: one sample per stratum in each dimension."""
    out = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


class LatinHypercube(Searcher):
    def __init__(self, budget: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        design = _lhs(budget, DIM, self.rng)
        self.n_replaced = 0
        for i in range(budget):
            if not is_feasible(unnormalize(design[i])):
                design[i], _, _ = rejection_sample(self.rng)
                self.n_replaced += 1
        self.design = design
        self.i = 0

    def ask(self) -> np.ndarray:
        u = self.design[self.i]
        self.i += 1
        return u