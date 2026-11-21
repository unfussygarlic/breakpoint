"""Stage A checkpoint: run ONE rollout of the fixed controller in the 2D env and
plot the trajectory so we can eyeball it.

Run from the repo root:
    PYTHONPATH=. python scripts/a_run_rollout.py
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # file output, no display needed
import matplotlib.pyplot as plt

from breakpoint.envs.nav2d import Nav2DEnv, Nav2DConfig
from breakpoint.controllers.potential_field import PotentialFieldController, APFConfig

SEED = 0


def rollout(env: Nav2DEnv, controller) -> dict:
    s = env.reset()
    traj = [s[:2].copy()]
    min_clear = env.signed_clearance(s[:2])
    reached_at = None
    for t in range(env.cfg.max_steps):
        u = controller(s)
        s = env.step(u)
        traj.append(s[:2].copy())
        min_clear = min(min_clear, env.signed_clearance(s[:2]))
        if reached_at is None and env.at_goal(s[:2]):
            reached_at = t + 1
            break
    return {
        "traj": np.array(traj),
        "steps": len(traj) - 1,
        "reached_at": reached_at,
        "min_clearance": min_clear,
        "final_dist": env.dist_to_goal(s[:2]),
    }


def main():
    np.random.seed(SEED)
    cfg = Nav2DConfig()
    env = Nav2DEnv(cfg, seed=SEED)
    controller = PotentialFieldController(cfg.goal, cfg.obstacles, APFConfig())

    r = rollout(env, controller)

    print(f"steps taken     : {r['steps']} / {cfg.max_steps}")
    print(f"reached goal at : {r['reached_at']}")
    print(f"final dist2goal : {r['final_dist']:.4f}  (eps = {cfg.goal_radius})")
    print(f"min clearance   : {r['min_clearance']:.4f}  (<0 => entered an obstacle)")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    for (cx, cy, rr) in cfg.obstacles:
        ax.add_patch(plt.Circle((cx, cy), rr, color="0.6", alpha=0.6, zorder=1))
    ax.plot(r["traj"][:, 0], r["traj"][:, 1], "-", lw=1.5, color="C0", zorder=2, label="trajectory")
    ax.plot(*cfg.start, "o", color="green", ms=9, zorder=3, label="start")
    ax.plot(*cfg.goal, "*", color="red", ms=16, zorder=3, label="goal")
    ax.add_patch(plt.Circle(cfg.goal, cfg.goal_radius, color="red", fill=False, ls="--", alpha=0.5))
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_title("Stage A: one rollout (APF controller)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("stage_a_rollout.png", dpi=120)
    print("saved plot -> stage_a_rollout.png")


if __name__ == "__main__":
    main()