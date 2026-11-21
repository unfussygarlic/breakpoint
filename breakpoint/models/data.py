"""Stage G/G.5: transition logging + dataset for the dynamics world model.

Collect (s, a, s') by rolling the fixed controller in scenarios, varying the
obstacles/goal for state coverage. Physics (mass, friction) is either pinned
(fixed-physics, Stage G) or sampled per scenario (varying-physics, Stage G.5).
The learned target is the delta:

    s = [x, y, vx, vy]   a = [Fx, Fy] (clamped force actually applied)
    Delta s = s' - s     model regresses  f(s, a [, mass, friction]) -> Delta s

Train/val split is BY SCENARIO (no transition leakage). Wall-clip transitions
(velocity zeroed at the boundary) are flagged and excluded by default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breakpoint.envs.nav2d import Nav2DEnv
from breakpoint.controllers.potential_field import PotentialFieldController, APFConfig
from breakpoint.scenarios.space import rejection_sample, Scenario, to_env_config


def _clamp(u: np.ndarray, f_max: float) -> np.ndarray:
    n = np.linalg.norm(u)
    return u * (f_max / n) if n > f_max else u


def collect_transitions(n_scenarios: int, seed: int = 0, mass: float = 1.0,
                        friction: float = 0.5, mass_range=None, friction_range=None,
                        apf: APFConfig | None = None,
                        eps: float = 0.05, max_steps: int = 400) -> list[dict]:
    """Return per-scenario dicts with arrays S, A, Snext, W and scalars mass,
    friction. If *_range is given, physics is sampled per scenario (varying
    physics); otherwise pinned (fixed-physics)."""
    rng = np.random.default_rng(seed)
    apf = apf or APFConfig()
    trajs = []
    for _ in range(n_scenarios):
        _, sc0, _ = rejection_sample(rng)
        m = float(rng.uniform(*mass_range)) if mass_range else mass
        b = float(rng.uniform(*friction_range)) if friction_range else friction
        sc = Scenario(mass=m, friction=b, obstacles=sc0.obstacles,
                      goal=sc0.goal, sensor_noise=0.0)        # noise off; physics per sample
        cfg = to_env_config(sc, max_steps=max_steps)
        env = Nav2DEnv(cfg, seed=0)
        ctrl = PotentialFieldController(cfg.goal, cfg.obstacles, apf)

        S, A, Snext, W = [], [], [], []
        s = env.reset()
        for _ in range(max_steps):
            a = _clamp(ctrl(s), cfg.f_max)
            # flag wall-clip: would the UNCLIPPED next position leave the box?
            v_un = s[2:] + (a - cfg.friction * s[2:]) / cfg.mass * cfg.dt
            pos_un = s[:2] + v_un * cfg.dt
            wall = bool(np.any(pos_un < 0.0) or np.any(pos_un > cfg.world_size))
            s2 = env.step(a)
            S.append(s.copy()); A.append(a.copy()); Snext.append(s2.copy()); W.append(wall)
            s = s2
            if env.dist_to_goal(s[:2]) <= eps:
                break
        trajs.append({"S": np.array(S), "A": np.array(A),
                      "Snext": np.array(Snext), "W": np.array(W),
                      "mass": m, "friction": b})
    return trajs


def split_by_scenario(trajs: list[dict], val_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    n_val = max(1, int(round(val_frac * len(trajs))))
    val_idx = set(idx[:n_val].tolist())
    train = [t for i, t in enumerate(trajs) if i not in val_idx]
    val = [t for i, t in enumerate(trajs) if i in val_idx]
    return train, val


def to_xy(trajs: list[dict], exclude_wall: bool = True, with_physics: bool = False):
    """Flatten trajectories into X and D=Delta s (N,4).
    X = [s,a] (N,6), or [s,a,mass,friction] (N,8) if with_physics.
    By default drops wall-clip transitions (discontinuous, out of scope)."""
    S = np.concatenate([t["S"] for t in trajs])
    A = np.concatenate([t["A"] for t in trajs])
    Snext = np.concatenate([t["Snext"] for t in trajs])
    X = np.concatenate([S, A], axis=1)
    if with_physics:
        M = np.concatenate([np.full(len(t["S"]), t["mass"]) for t in trajs])
        B = np.concatenate([np.full(len(t["S"]), t["friction"]) for t in trajs])
        X = np.concatenate([X, M[:, None], B[:, None]], axis=1)
    D = Snext - S
    if exclude_wall:
        keep = ~np.concatenate([t["W"] for t in trajs])
        X, D = X[keep], D[keep]
    return X, D


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(A: np.ndarray) -> "Standardizer":
        return Standardizer(A.mean(0), A.std(0) + 1e-8)

    def transform(self, A):
        return (A - self.mean) / self.std

    def inverse(self, A):
        return A * self.std + self.mean

    def inverse_std(self, S):
        return S * self.std