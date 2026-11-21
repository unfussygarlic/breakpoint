"""Stage F checkpoint (part 1): CMA-ES vs the random/LHS baselines on the
collision-hunting objective (rho_safety), over several search seeds.

Currency = feasible simulator calls. Headline metric = calls to first collision.

Run from the repo root:
    PYTHONPATH=. python scripts/f_cma.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.search.samplers import RandomSearch, LatinHypercube
from breakpoint.search.cmaes import CMASearch
from breakpoint.search.driver import run_campaign, summarize

SIM_BUDGET = 300
SEEDS = range(5)
OBJ = "safety"


def make(method: str, seed: int):
    if method == "random":
        return RandomSearch(seed=seed)
    if method == "lhs":
        return LatinHypercube(budget=SIM_BUDGET, seed=seed)
    if method == "cma":
        return CMASearch(seed=seed, sigma0=0.25)
    raise ValueError(method)


def cumulative_collisions(records) -> np.ndarray:
    hits = np.array([r.rho_vector["rho_safety"] < 0 for r in records], dtype=int)
    return np.cumsum(hits)


def main():
    methods = ["random", "lhs", "cma"]
    curves = {m: [] for m in methods}
    first_hit = {m: [] for m in methods}
    total = {m: [] for m in methods}

    for m in methods:
        for seed in SEEDS:
            recs = run_campaign(make(m, seed), SIM_BUDGET, objective=OBJ)
            s = summarize(recs)
            curves[m].append(cumulative_collisions(recs))
            first_hit[m].append(s.first_collision_at if s.first_collision_at is not None else np.nan)
            total[m].append(s.n_safety_fail)
        fh = np.array(first_hit[m], float)
        print(f"{m:7s}  first-collision@ (per seed): "
              f"{[None if np.isnan(x) else int(x) for x in fh]}  "
              f"mean={np.nanmean(fh):.1f}   total collisions/{SIM_BUDGET}: "
              f"mean={np.mean(total[m]):.1f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c in [("random", "C0"), ("lhs", "C2"), ("cma", "C3")]:
        arr = np.array(curves[m])
        mean = arr.mean(0); sd = arr.std(0)
        x = np.arange(1, SIM_BUDGET + 1)
        ax.plot(x, mean, color=c, label=m)
        ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.15)
    ax.set_xlabel("feasible simulator calls")
    ax.set_ylabel("cumulative collisions found")
    ax.set_title(f"Stage F: collision discovery (objective=rho_safety, {len(list(SEEDS))} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("stage_f_cma.png", dpi=110)
    print("saved plot -> stage_f_cma.png")


if __name__ == "__main__":
    main()