"""Stage C checkpoint: score two trajectories with the STL oracle and plot the
robustness signals over time.

  A = nominal scenario        -> expect PASS  (both rho > 0)
  B = obstacle on the path    -> expect FAIL on GOAL (stall), safety still OK

Run from the repo root:
    PYTHONPATH=. python scripts/c_oracle.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breakpoint.envs.nav2d import Nav2DEnv
from breakpoint.controllers.potential_field import PotentialFieldController, APFConfig
from breakpoint.scenarios.space import Scenario, to_env_config
from breakpoint.oracle.stl import STLSpec, evaluate

SEED = 0
SPEC = STLSpec(d_safe=0.05, eps=0.05)


def run_trajectory(sc: Scenario, spec: STLSpec, seed=SEED) -> np.ndarray:
    """Roll the fixed controller in the scenario; stop early once within eps of goal.
    (This loop moves into the Stage D execution engine; local here for the checkpoint.)"""
    cfg = to_env_config(sc)
    env = Nav2DEnv(cfg, seed=seed)
    ctrl = PotentialFieldController(cfg.goal, cfg.obstacles, APFConfig())
    s = env.reset()
    traj = [s[:2].copy()]
    for _ in range(cfg.max_steps):
        s = env.step(ctrl(s))
        traj.append(s[:2].copy())
        if env.dist_to_goal(s[:2]) <= spec.eps:
            break
    return np.array(traj)


def report(name: str, sc: Scenario):
    traj = run_trajectory(sc, SPEC)
    res = evaluate(traj, sc.obstacles, sc.goal, SPEC)
    verdict = "FAIL" if res.failed else "pass"
    print(f"[{name}] {verdict}  steps={len(traj)-1}")
    print(f"      rho_safety = {res.rho_safety:+.4f}   rho_goal = {res.rho_goal:+.4f}"
          f"   rho_spec = {res.rho_spec:+.4f}")
    return traj, res


def main():
    # A: nominal -- obstacles off the diagonal
    scA = Scenario(mass=1.0, friction=0.5,
                   obstacles=((0.1, 0.8, 0.12), (0.7, 0.3, 0.08)),
                   goal=(0.9, 0.9), sensor_noise=0.0)
    # B: a blocker sitting on the start->goal line
    scB = Scenario(mass=1.0, friction=0.5,
                   obstacles=((0.5, 0.5, 0.12), (0.7, 0.3, 0.08)),
                   goal=(0.9, 0.9), sensor_noise=0.0)

    trajA, resA = report("A nominal", scA)
    trajB, resB = report("B blocker", scB)

    # plot the two robustness signals over time for each scenario
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name, res in [(axes[0], "A nominal", resA), (axes[1], "B blocker", resB)]:
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(res.clearance - SPEC.d_safe, label="clearance - d_safe  (safety)", color="C0")
        ax.plot(SPEC.eps - res.dist_to_goal, label="eps - dist  (goal)", color="C3")
        ax.set_title(f"{name}:  rho_spec = {res.rho_spec:+.3f}  "
                     f"({'FAIL' if res.failed else 'pass'})", fontsize=10)
        ax.set_xlabel("timestep")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("per-step robustness")
    fig.suptitle("Stage C: STL robustness signals  (dips below 0 = requirement violated)")
    fig.tight_layout()
    fig.savefig("stage_c_oracle.png", dpi=110)
    print("saved plot -> stage_c_oracle.png")


if __name__ == "__main__":
    main()