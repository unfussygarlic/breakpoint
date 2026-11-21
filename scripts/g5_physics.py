"""Stage G.5 checkpoint: physics-conditioned dynamics ensemble f(s,a,mass,friction),
trained on VARYING physics with a HELD-OUT high-mass region. The real question:
does epistemic uncertainty RISE where the model hasn't trained?

  train/val : scenarios with mass < 2.25   (in-distribution)
  OOD       : scenarios with mass >= 2.25   (never seen)

Run from the repo root:
    PYTHONPATH=. python scripts/g5_physics.py
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.models.data import collect_transitions, split_by_scenario, to_xy, Standardizer
from breakpoint.models.ensemble import DynamicsEnsemble

SEED = 0
N = 300
MASS_CUT = 2.25
np.random.seed(SEED); torch.manual_seed(SEED)


def eval_group(ens, sx, sd, trajs, tag):
    X, D = to_xy(trajs, with_physics=True)
    out = ens.predict(sx.transform(X))
    err = np.abs(sd.inverse(out["mean"]) - D).mean(1)
    rmse = np.sqrt(((sd.inverse(out["mean"]) - D) ** 2).mean())
    epi = out["epistemic"].mean(1)
    print(f"  {tag:10s} n={len(X):5d}  RMSE={rmse:.5f}  mean-epistemic={epi.mean():.4e}")
    return err, epi


def main():
    trajs = collect_transitions(N, seed=SEED, mass_range=(0.5, 3.0),
                                friction_range=(0.1, 1.5), max_steps=200)
    in_pool = [t for t in trajs if t["mass"] < MASS_CUT]
    ood = [t for t in trajs if t["mass"] >= MASS_CUT]
    train, val = split_by_scenario(in_pool, val_frac=0.2, seed=SEED)
    print(f"scenarios: train={len(train)} val={len(val)} OOD(mass>={MASS_CUT})={len(ood)}")

    Xtr, Dtr = to_xy(train, with_physics=True)
    sx = Standardizer.fit(Xtr)
    sd = Standardizer.fit(Dtr)
    ens = DynamicsEnsemble(in_dim=8, seed=SEED)
    ens.train(sx.transform(Xtr), sd.transform(Dtr), epochs=150, warmup=60, seed=SEED)

    print("evaluation:")
    err_id, epi_id = eval_group(ens, sx, sd, val, "in-dist")
    err_ood, epi_ood = eval_group(ens, sx, sd, ood, "OOD")
    print(f"  --> OOD/in-dist   epistemic x{epi_ood.mean()/epi_id.mean():.1f}   "
          f"error x{err_ood.mean()/err_id.mean():.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.boxplot([np.log10(epi_id + 1e-12), np.log10(epi_ood + 1e-12)], showfliers=False)
    ax1.set_xticks([1, 2]); ax1.set_xticklabels(["in-dist", "OOD (mass>=2.25)"])
    ax1.set_ylabel("log10 epistemic uncertainty")
    ax1.set_title(f"epistemic rises OOD (x{epi_ood.mean()/epi_id.mean():.1f})")

    e = np.concatenate([epi_id, epi_ood]); er = np.concatenate([err_id, err_ood])
    col = ["C0"] * len(epi_id) + ["C3"] * len(epi_ood)
    ax2.scatter(e, er, s=4, c=col, alpha=0.3)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("epistemic uncertainty"); ax2.set_ylabel("actual |error|")
    corr = np.corrcoef(np.log(e + 1e-12), np.log(er + 1e-12))[0, 1]
    ax2.set_title(f"uncertainty tracks error (log-corr={corr:.2f})\nblue=in-dist  red=OOD")

    fig.tight_layout(); fig.savefig("stage_g5_physics.png", dpi=110)
    print(f"saved plot -> stage_g5_physics.png   (log-corr = {corr:.2f})")


if __name__ == "__main__":
    main()