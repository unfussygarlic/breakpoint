"""Stage G checkpoint: train the dynamics ensemble on FIXED-physics transitions,
then evaluate one-step accuracy, uncertainty calibration, and multi-step rollout
error (where errors compound).

Run from the repo root:
    PYTHONPATH=. python scripts/g_worldmodel.py
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr, erfinv

from breakpoint.models.data import (
    collect_transitions, split_by_scenario, to_xy, Standardizer,
)
from breakpoint.models.ensemble import DynamicsEnsemble

SEED = 0
N_SCENARIOS = 200
MAX_STEPS = 200
np.random.seed(SEED); torch.manual_seed(SEED)


def multistep(ens, sx: Standardizer, sd: Standardizer, traj, horizon=60):
    S, A, Wf = traj["S"], traj["A"], traj["W"]
    H = min(horizon, len(A))
    wall_hits = np.where(Wf[:H])[0]
    if len(wall_hits):
        H = int(wall_hits[0])
    s = S[0].copy()
    pred = [s[:2].copy()]
    pvar = np.zeros(2)
    pos_var = [pvar.copy()]
    for t in range(H):
        x = np.concatenate([s, A[t]])[None, :]
        out = ens.predict(sx.transform(x))
        d = sd.inverse(out["mean"][0])
        var_phys = (sd.std ** 2) * out["total"][0]
        s = s + d
        pred.append(s[:2].copy())
        pvar = pvar + var_phys[:2]
        pos_var.append(pvar.copy())
    true_xy = np.vstack([S[:1, :2], traj["Snext"][:H, :2]])
    err = np.linalg.norm(np.array(pred) - true_xy, axis=1)
    sigma = np.sqrt(np.array(pos_var).sum(1))
    return err, sigma


def main():
    trajs = collect_transitions(N_SCENARIOS, seed=SEED, max_steps=MAX_STEPS)
    train, val = split_by_scenario(trajs, val_frac=0.2, seed=SEED)
    Xtr, Dtr = to_xy(train)
    Xva, Dva = to_xy(val)
    print(f"transitions: train={len(Xtr)}  val={len(Xva)}  "
          f"(scenarios {len(train)}/{len(val)})")

    sx = Standardizer.fit(Xtr)
    sd = Standardizer.fit(Dtr)
    ens = DynamicsEnsemble(seed=SEED)
    ens.train(sx.transform(Xtr), sd.transform(Dtr), epochs=150, warmup=60, seed=SEED)

    out = ens.predict(sx.transform(Xva))
    pred_d = sd.inverse(out["mean"])
    rmse = np.sqrt(((pred_d - Dva) ** 2).mean(0))
    base = np.sqrt(((Dtr.mean(0) - Dva) ** 2).mean(0))
    names = ["dx", "dy", "dvx", "dvy"]
    print("one-step RMSE (physical):")
    for i, nm in enumerate(names):
        print(f"  {nm}: model {rmse[i]:.5f}   const-baseline {base[i]:.5f}   "
              f"({base[i]/rmse[i]:.1f}x better)")

    z = (sd.transform(Dva) - out["mean"]) / np.sqrt(out["total"])
    for k in (1, 2):
        cov = np.mean(np.abs(z) <= k)
        print(f"  coverage within {k}sigma: observed {cov:.3f}  expected {2*ndtr(k)-1:.3f}")

    H = 60
    errs, sigs = [], []
    for t in val:
        if len(t["A"]) >= 10:
            e, s = multistep(ens, sx, sd, t, horizon=H)
            L = min(len(e), H + 1)
            errs.append(np.pad(e[:L], (0, H + 1 - L), constant_values=np.nan))
            sigs.append(np.pad(s[:L], (0, H + 1 - L), constant_values=np.nan))
    errs = np.array(errs); sigs = np.array(sigs)
    me = np.nanmean(errs, 0); ms = np.nanmean(sigs, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    h = np.arange(len(me))
    ax1.plot(h, me, color="C0", label="mean position error")
    ax1.plot(h, ms, color="C3", ls="--", label="predicted sigma (uncertainty)")
    ax1.set_xlabel("rollout horizon (steps)"); ax1.set_ylabel("position error")
    ax1.set_title("multi-step: error & uncertainty compound"); ax1.legend(fontsize=8)

    qs = np.linspace(0.05, 0.95, 19)
    ks = np.sqrt(2) * erfinv(qs)
    obs = [np.mean(np.abs(z) <= k) for k in ks]
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax2.plot(qs, obs, "o-", color="C2", ms=3, label="observed")
    ax2.set_xlabel("expected coverage"); ax2.set_ylabel("observed coverage")
    ax2.set_title("calibration"); ax2.legend(fontsize=8)

    fig.tight_layout(); fig.savefig("stage_g_worldmodel.png", dpi=110)
    print("saved plot -> stage_g_worldmodel.png")


if __name__ == "__main__":
    main()