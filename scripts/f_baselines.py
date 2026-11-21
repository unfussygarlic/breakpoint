"""Stage F checkpoint (full): all four baselines on the collision-hunt objective
(rho_safety) -- random, LHS, CMA-ES, Bayesian optimization -- over several seeds.

Currency = feasible simulator calls. Headline = calls to first collision.

Run from the repo root:
    PYTHONPATH=. python scripts/f_baselines.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.search.samplers import RandomSearch, LatinHypercube
from breakpoint.search.cmaes import CMASearch
from breakpoint.search.bo import BOSearch
from breakpoint.search.driver import run_campaign, summarize

SIM_BUDGET = 300
SEEDS = range(5)
OBJ = "safety"
COLORS = {"random": "C0", "lhs": "C2", "cma": "C3", "bo": "C4"}


def make(method: str, seed: int):
    if method == "random":
        return RandomSearch(seed=seed)
    if method == "lhs":
        return LatinHypercube(budget=SIM_BUDGET, seed=seed)
    if method == "cma":
        return CMASearch(seed=seed, sigma0=0.25)
    if method == "bo":
        return BOSearch(seed=seed)
    raise ValueError(method)


def cumulative_collisions(records) -> np.ndarray:
    hits = np.array([r.rho_vector["rho_safety"] < 0 for r in records], dtype=int)
    return np.cumsum(hits)


def main():
    methods = ["random", "lhs", "cma", "bo"]
    curves = {m: [] for m in methods}

    for m in methods:
        first, total = [], []
        for seed in SEEDS:
            recs = run_campaign(make(m, seed), SIM_BUDGET, objective=OBJ)
            s = summarize(recs)
            curves[m].append(cumulative_collisions(recs))
            first.append(s.first_collision_at if s.first_collision_at is not None else np.nan)
            total.append(s.n_safety_fail)
        fh = np.array(first, float)
        print(f"{m:7s}  first-collision@ mean={np.nanmean(fh):5.1f}  "
              f"(per seed {[None if np.isnan(x) else int(x) for x in fh]})   "
              f"total/{SIM_BUDGET} mean={np.mean(total):.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(1, SIM_BUDGET + 1)
    for m in methods:
        arr = np.array(curves[m])
        mean = arr.mean(0); sd = arr.std(0)
        ax.plot(x, mean, color=COLORS[m], label=m)
        ax.fill_between(x, mean - sd, mean + sd, color=COLORS[m], alpha=0.12)
    ax.set_xlabel("feasible simulator calls")
    ax.set_ylabel("cumulative collisions found")
    ax.set_title(f"Stage F: all baselines, collision discovery ({len(list(SEEDS))} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("stage_f_baselines.png", dpi=110)
    print("saved plot -> stage_f_baselines.png")


if __name__ == "__main__":
    main()