"""Stage E: the swappable search interface.

A Searcher proposes feasible scenarios and observes their robustness:
    ask()        -> u in [0,1]^DIM (feasible)
    tell(u, rho) -> record the outcome (no-op for non-adaptive strategies)

Every strategy -- random, LHS, CMA-ES, BO, curiosity -- implements THIS and
nothing else. The engine and oracle are fixed, so comparing strategies is fair.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Searcher(ABC):
    @abstractmethod
    def ask(self) -> np.ndarray:
        """Return a feasible scenario vector u in [0,1]^DIM."""

    def tell(self, u: np.ndarray, rho: float) -> None:
        """Observe the robustness of a proposed scenario. No-op by default."""
        return None