"""Stage F: CMA-ES search behind the ask/tell interface (via the `cma` package).

CMA proposes a population per generation in the bounded box [0,1]^DIM; we hand
them out one at a time through ask(), buffer the objectives from tell(), and feed
a whole generation back to the optimizer once it is complete. Order is preserved
(the driver asks then tells sequentially), so objectives match proposals.

Minimizes whatever objective the driver passes (rho_safety for Setup 2).
"""
from __future__ import annotations

import cma
import numpy as np

from breakpoint.scenarios.space import DIM
from breakpoint.search.base import Searcher


class CMASearch(Searcher):
    def __init__(self, seed: int = 0, sigma0: float = 0.2, x0=None, popsize: int | None = None):
        x0 = [0.5] * DIM if x0 is None else list(x0)
        # NB: cma treats seed=0 as "random" -> non-reproducible. Offset so a
        # search seed of 0 maps to a real cma seed. Non-zero seeds are
        # reproducible and order-independent (verified).
        opts = {"bounds": [0.0, 1.0], "seed": int(seed) + 1, "verbose": -9}
        if popsize is not None:
            opts["popsize"] = popsize
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        self._gen: list = []
        self._objs: list = []
        self._i = 0

    def ask(self) -> np.ndarray:
        if self._i >= len(self._gen):
            self._gen = self.es.ask()
            self._objs = []
            self._i = 0
        u = np.clip(self._gen[self._i], 0.0, 1.0)
        self._i += 1
        return u

    def tell(self, u: np.ndarray, obj: float) -> None:
        self._objs.append(float(obj))
        if len(self._objs) == len(self._gen):     # generation complete
            self.es.tell(self._gen, self._objs)