"""Stage H: curiosity search -- the model-based active tester.

For each candidate scenario theta we IMAGINE its rollout through the learned
physics-conditioned ensemble (controller + oracle are known), giving:
    rho_hat(theta) = predicted clearance robustness  (lead on a collision)
    u(theta)       = mean epistemic uncertainty over the imagined rollout
Acquisition (minimized), with both terms z-scored over the pool so kappa is
scale-free:
    a(theta) = z(rho_hat) - kappa * z(u)
kappa=0 is pure exploitation of rho_hat; kappa>0 adds curiosity (seek novelty).

Discipline: rho_hat is a LEAD, never a verdict. Every proposal is confirmed in
the REAL sim + oracle; the real transitions grow the dataset and the ensemble is
retrained periodically (active experimentation -> uncertainty collapses where we
have looked).
"""
from __future__ import annotations

import numpy as np

from breakpoint.controllers.potential_field import APFConfig
from breakpoint.oracle.stl import STLSpec
from breakpoint.scenarios.space import DIM, START, rejection_sample, unnormalize
from breakpoint.models.data import Standardizer
from breakpoint.models.ensemble import DynamicsEnsemble
from breakpoint.engine.runner import run, RunConfig

DT, WORLD = 0.05, 1.0


def _apf_batch(pos, goal, obst, cfg: APFConfig, f_max: float) -> np.ndarray:
    f = cfg.k_att * (goal - pos)                       # attractive (M,2)
    for j in range(obst.shape[1]):
        c = obst[:, j, :2]; r = obst[:, j, 2]
        dvec = pos - c
        dc = np.linalg.norm(dvec, axis=1)
        dirn = dvec / (dc[:, None] + 1e-9)
        d = dc - r
        inside = d <= 0.0
        within = (~inside) & (d < cfg.influence)
        dd = np.where(within, d, 1.0)
        mag = cfg.k_rep * (1.0 / dd - 1.0 / cfg.influence) / (dd * dd)
        contrib = np.where(inside[:, None], cfg.inside_push * dirn,
                           np.where(within[:, None], mag[:, None] * dirn, 0.0))
        f = f + contrib
    nrm = np.linalg.norm(f, axis=1)
    scale = np.where(nrm > f_max, f_max / (nrm + 1e-12), 1.0)
    return f * scale[:, None]


def _clearance_batch(pos, obst) -> np.ndarray:
    c = obst[:, :, :2]; r = obst[:, :, 2]
    dist = np.linalg.norm(pos[:, None, :] - c, axis=2) - r
    return dist.min(axis=1)


def imagine(ens, sx, sd, U, apf: APFConfig, spec: STLSpec, f_max=10.0, H=50):
    """Batched imagined rollout over M candidates -> (rho_safety_hat, u)."""
    scen = [unnormalize(u) for u in U]
    M = len(U)
    pos = np.tile(np.array(START, float), (M, 1))
    vel = np.zeros((M, 2))
    goal = np.array([s.goal for s in scen], float)
    obst = np.array([list(s.obstacles) for s in scen], float)
    mass = np.array([s.mass for s in scen], float)
    fric = np.array([s.friction for s in scen], float)

    clear_min = np.full(M, np.inf)
    epi_sum = np.zeros(M); nstep = np.zeros(M)
    active = np.ones(M, bool)
    for _ in range(H):
        s = np.concatenate([pos, vel], 1)
        a = _apf_batch(pos, goal, obst, apf, f_max)
        X = np.concatenate([s, a, mass[:, None], fric[:, None]], 1)
        out = ens.predict(sx.transform(X))
        d = sd.inverse(out["mean"])
        epi = out["epistemic"].mean(1)
        s2 = s + d
        pos, vel = s2[:, :2], s2[:, 2:]
        clr = _clearance_batch(pos, obst)
        clear_min = np.minimum(clear_min, np.where(active, clr, np.inf))
        epi_sum += epi * active; nstep += active
        active &= (np.linalg.norm(pos - goal, axis=1) > spec.eps)
        if not active.any():
            break
    rho_safety_hat = clear_min - spec.d_safe
    u = epi_sum / np.maximum(nstep, 1)
    return rho_safety_hat, u


class CuriositySearch:
    def __init__(self, seed: int = 0, kappa: float = 1.0, n_seed: int = 40,
                 retrain_every: int = 15, pool: int = 250, H_imagine: int = 45,
                 n_members: int = 5, ens_seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.kappa = kappa
        self.n_seed = n_seed; self.K = retrain_every; self.pool = pool; self.H = H_imagine
        self.apf = APFConfig(); self.spec = STLSpec()
        self.X: list = []; self.D: list = []
        self.ens = None; self.sx = None; self.sd = None
        self.sims = 0; self.since = 0
        self.n_members = n_members; self.ens_seed = ens_seed

    def ask(self) -> np.ndarray:
        if self.sims < self.n_seed or self.ens is None:
            u, _, _ = rejection_sample(self.rng)        # seed phase: random coverage
            return u
        U = np.array([rejection_sample(self.rng)[0] for _ in range(self.pool)])
        rho_hat, u_unc = imagine(self.ens, self.sx, self.sd, U, self.apf, self.spec, H=self.H)
        rz = (rho_hat - rho_hat.mean()) / (rho_hat.std() + 1e-9)
        uz = (u_unc - u_unc.mean()) / (u_unc.std() + 1e-9)
        acq = rz - self.kappa * uz                      # minimize
        return U[int(np.argmin(acq))]

    def observe(self, u: np.ndarray, rec) -> None:
        sc = unnormalize(np.asarray(u))
        for (s, a, s2) in (rec.transitions or []):
            # drop wall-clip transitions (discontinuous, out of the smooth model)
            v_un = s[2:] + (a - sc.friction * s[2:]) / sc.mass * DT
            pos_un = s[:2] + v_un * DT
            if np.any(pos_un < 0.0) or np.any(pos_un > WORLD):
                continue
            self.X.append(np.concatenate([s, a, [sc.mass], [sc.friction]]))
            self.D.append(s2 - s)
        self.sims += 1; self.since += 1
        if self.sims >= self.n_seed and (self.ens is None or self.since >= self.K):
            self._retrain(); self.since = 0

    def _retrain(self) -> None:
        X = np.array(self.X); D = np.array(self.D)
        self.sx = Standardizer.fit(X); self.sd = Standardizer.fit(D)
        self.ens = DynamicsEnsemble(in_dim=8, n_members=self.n_members, seed=self.ens_seed)
        self.ens.train(self.sx.transform(X), self.sd.transform(D),
                       epochs=60, warmup=25, seed=self.ens_seed)


def run_curiosity_campaign(cur: CuriositySearch, sim_budget: int,
                           run_seed: int = 0, spec: STLSpec | None = None) -> list:
    cfg = RunConfig(seed=run_seed, spec=spec or STLSpec(),
                    store_trajectory=False, store_transitions=True)
    records = []
    for _ in range(sim_budget):
        u = cur.ask()
        rec = run(u, cfg)
        cur.observe(u, rec)
        records.append(rec)
    return records