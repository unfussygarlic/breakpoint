"""Stage C: STL oracle with quantitative (robustness) semantics.

Spec:  [](clearance >= d_safe)  /\\  <>_[0,T](dist_to_goal <= eps)

From a recorded trajectory of positions we compute two signals and reduce them:

    clearance(t) = min_obs ( ||pos_t - centre|| - radius )     # signed; <0 inside
    dist(t)      = ||pos_t - goal||

    rho_safety = min_t ( clearance(t) - d_safe )   # [] -> min over time
    rho_goal   = eps - min_t dist(t)               # <> -> max_t (eps - dist)
    rho_spec   = min(rho_safety, rho_goal)         # /\\ -> min

Sign = pass/fail, magnitude = margin. Failure iff rho_spec < 0. The oracle is
pure geometry: trajectory + obstacles + goal in, robustness out. It is the
AUTHORITY on failure -- nothing else in the system gets to declare a failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STLSpec:
    d_safe: float = 0.05   # required clearance buffer from obstacle surfaces
    eps: float = 0.05      # goal tolerance


@dataclass
class OracleResult:
    rho_safety: float
    rho_goal: float
    rho_spec: float
    failed: bool
    # per-step signals, for plotting / debugging
    clearance: np.ndarray
    dist_to_goal: np.ndarray

    @property
    def vector(self) -> dict:
        """The per-requirement rho vector we log (not just the min)."""
        return {"rho_safety": self.rho_safety, "rho_goal": self.rho_goal}


def _clearance_series(traj: np.ndarray, obstacles) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    margins = np.full(len(traj), np.inf)
    for (cx, cy, r) in obstacles:
        d = np.hypot(traj[:, 0] - cx, traj[:, 1] - cy) - r
        margins = np.minimum(margins, d)
    return margins


def evaluate(traj: np.ndarray, obstacles, goal, spec: STLSpec | None = None) -> OracleResult:
    spec = spec or STLSpec()
    traj = np.asarray(traj, dtype=float)

    clearance = _clearance_series(traj, obstacles)
    dist = np.hypot(traj[:, 0] - goal[0], traj[:, 1] - goal[1])

    rho_safety = float(np.min(clearance - spec.d_safe))   # [] (always)
    rho_goal = float(spec.eps - np.min(dist))             # <> (eventually)
    rho_spec = min(rho_safety, rho_goal)                  # /\\ (and)

    return OracleResult(
        rho_safety=rho_safety,
        rho_goal=rho_goal,
        rho_spec=rho_spec,
        failed=rho_spec < 0.0,
        clearance=clearance,
        dist_to_goal=dist,
    )