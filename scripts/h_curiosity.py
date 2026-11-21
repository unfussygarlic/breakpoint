"""Stage H checkpoint: the curiosity tester vs baselines on collision discovery.

  random              - the honest floor
  bo                  - best black-box baseline
  curiosity (kappa=0) - model-based EXPLOIT of rho_hat, no curiosity term
  curiosity (kappa>0) - rho_hat - kappa*u  (the full method)

The kappa=0 vs kappa>0 pair IS the ablation: does the uncertainty term help?
Currency = feasible simulator calls. Headline = calls to first collision.

Run from the repo root:
    PYTHONPATH=. python scripts/h_curiosity.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.search.samplers import RandomSearch
from breakpoint.search.bo import BOSearch
from breakpoint.search.curiosity import CuriositySearch, run_curiosity_campaign
from breakpoint.search.driver import run_campaign

SIM_BUDGET = 120
SEEDS = range(3)
N_SEED = 25


def cum_collisions(records):
    return np.cumsum([r.rho_vector["rho_safety"] < 0 for r in records]).astype(int)


def run_method(name, seed):
    if name == "random":
        return run_campaign(RandomSearch(seed=seed), SIM_BUDGET, objective="safety")
    if name == "bo":
        return run_campaign(BOSearch(seed=seed), SIM_BUDGET, objective="safety")
    if name.startswith("cur"):
        kappa = float(name.split("=")[1])
        cur = CuriositySearch(seed=seed, kappa=kappa, n_seed=N_SEED, retrain_every=20)
        return run_curiosity_campaign(cur, SIM_BUDGET)
    raise ValueError(name)


def main():
    methods = ["random", "bo", "cur_k=0.0", "cur_k=1.5"]
    colors = {"random": "C0", "bo": "C4", "cur_k=0.0": "C1", "cur_k=1.5": "C3"}
    curves = {m: [] for m in methods}

    for m in methods:
        first, total = [], []
        for seed in SEEDS:
            recs = run_method(m, seed)
            c = cum_collisions(recs)
            curves[m].append(c)
            hit = np.argmax(c > 0) if c[-1] > 0 else np.nan
            first.append(hit); total.append(int(c[-1]))
        fh = np.array(first, float)
        print(f"{m:10s}  first-collision@ mean={np.nanmean(fh):5.1f} {[None if np.isnan(x) else int(x) for x in fh]}"
              f"   total/{SIM_BUDGET} mean={np.mean(total):.1f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(1, SIM_BUDGET + 1)
    for m in methods:
        arr = np.array(curves[m]); mean = arr.mean(0); sd = arr.std(0)
        ax.plot(x, mean, color=colors[m], label=m)
        ax.fill_between(x, mean - sd, mean + sd, color=colors[m], alpha=0.12)
    ax.axvline(N_SEED, color="gray", ls=":", lw=1)
    ax.text(N_SEED + 1, 0.5, "curiosity seed phase ends", fontsize=7, color="gray")
    ax.set_xlabel("feasible simulator calls"); ax.set_ylabel("cumulative collisions found")
    ax.set_title(f"Stage H: curiosity vs baselines ({len(list(SEEDS))} seeds)")
    ax.legend()
    fig.tight_layout(); fig.savefig("stage_h_curiosity.png", dpi=110)
    print("saved plot -> stage_h_curiosity.png")


if __name__ == "__main__":
    main()