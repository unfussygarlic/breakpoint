"""Stage E checkpoint: calibrate Theta by measuring the failure rate p.

Sweep N random scenarios through the fixed engine, read off p and the failure-type
split (goal vs safety), compare against LHS, and plot the rho_spec distribution.

Run from the repo root:
    PYTHONPATH=. python scripts/e_calibrate.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.search.samplers import RandomSearch, LatinHypercube
from breakpoint.search.driver import run_campaign, summarize

N = 1000
SEED = 0


def main():
    rnd = RandomSearch(seed=SEED)
    rec_rnd = run_campaign(rnd, N)
    s_rnd = summarize(rec_rnd)
    print("RANDOM :", s_rnd)

    lhs = LatinHypercube(budget=N, seed=SEED)
    rec_lhs = run_campaign(lhs, N)
    s_lhs = summarize(rec_lhs)
    print("LHS    :", s_lhs, f"(replaced {lhs.n_replaced} infeasible)")

    rho = s_rnd.rho_spec
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.hist(np.clip(rho, -1.0, 0.3), bins=60, color="C0", alpha=0.85)
    ax1.axvline(0, color="k", lw=1)
    ax1.set_title(f"rho_spec distribution (random, N={N})\np_fail = {s_rnd.p_fail:.3f}")
    ax1.set_xlabel("rho_spec  (<0 = failure)")
    ax1.set_ylabel("count")

    labels = ["pass", "goal-fail", "safety-fail"]
    vals = [N - s_rnd.n_goal_fail - s_rnd.n_safety_fail, s_rnd.n_goal_fail, s_rnd.n_safety_fail]
    ax2.bar(labels, vals, color=["C2", "C3", "C1"])
    ax2.set_title("failure-type split (random)")
    for i, v in enumerate(vals):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig("stage_e_calibration.png", dpi=110)
    print("saved plot -> stage_e_calibration.png")


if __name__ == "__main__":
    main()