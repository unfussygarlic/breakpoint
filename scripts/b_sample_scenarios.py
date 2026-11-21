"""Stage B checkpoint: sample feasible scenarios from Theta, report the rejection
acceptance rate, and plot a few so we can eyeball that they look sane.

Run from the repo root:
    PYTHONPATH=. python scripts/b_sample_scenarios.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.scenarios.space import (
    DIM, START, rejection_sample, unnormalize, is_feasible,
)

SEED = 0
N_PLOT = 6          # scenarios to draw
N_STATS = 5000      # raw unit-box samples to estimate feasibility fraction


def feasibility_fraction(rng, n=N_STATS) -> float:
    ok = sum(is_feasible(unnormalize(rng.random(DIM))) for _ in range(n))
    return ok / n


def main():
    rng = np.random.default_rng(SEED)

    frac = feasibility_fraction(rng)
    print(f"Theta dimension d        : {DIM}")
    print(f"feasible fraction (~p)   : {frac:.3f}   -> ~{1/frac:.1f} samples per feasible draw")

    # draw N_PLOT feasible scenarios and report avg tries
    scens, tries = [], []
    for _ in range(N_PLOT):
        _, sc, n = rejection_sample(rng)
        scens.append(sc)
        tries.append(n)
    print(f"rejection tries (N={N_PLOT}) : {tries}  (mean {np.mean(tries):.1f})")

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, sc in zip(axes.ravel(), scens):
        for (cx, cy, r) in sc.obstacles:
            ax.add_patch(plt.Circle((cx, cy), r, color="0.6", alpha=0.6))
        ax.plot(*START, "o", color="green", ms=8)
        ax.plot(*sc.goal, "*", color="red", ms=14)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"m={sc.mass:.2f} b={sc.friction:.2f}\n"
                     f"noise={sc.sensor_noise:.3f}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("Stage B: sampled feasible scenarios (green=start, red=goal)")
    fig.tight_layout()
    fig.savefig("stage_b_scenarios.png", dpi=110)
    print("saved plot -> stage_b_scenarios.png")


if __name__ == "__main__":
    main()