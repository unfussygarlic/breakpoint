"""Stage A: fixed, deliberately imperfect controller -- artificial potential field (APF).

Force = attractive pull to the goal + repulsive push from nearby obstacles.
    F_att = k_att * (goal - pos)
    F_rep = sum over obstacles within influence range d0 of
                k_rep * (1/d - 1/d0) * (1/d^2) * (unit vector away from obstacle)
            where d = ||pos - centre|| - radius  (surface distance)

This is imperfect ON PURPOSE:
  * NO velocity damping  -> momentum causes overshoot / oscillation.
  * Local minima         -> repulsion can cancel attraction and stall the robot.
  * Narrow-gap chatter    -> opposing repulsors induce oscillation.
We are testing this controller, not fixing it. The env clamps force to the
actuator limit f_max, so the controller does not clamp its own output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class APFConfig:
    k_att: float = 3.0
    k_rep: float = 0.02          # weakened (Stage E, Setup 2): momentum can breach obstacles
    influence: float = 0.22      # d0: repulsion acts only within this surface distance
    inside_push: float = 4.0     # weak ejection -> collisions are a rare-but-present critical failure


class PotentialFieldController:
    def __init__(self, goal, obstacles, config: APFConfig | None = None):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = [tuple(o) for o in obstacles]
        self.cfg = config or APFConfig()

    def __call__(self, state) -> np.ndarray:
        pos = np.asarray(state[:2], dtype=float)

        # attractive term
        f_att = self.cfg.k_att * (self.goal - pos)

        # repulsive term
        f_rep = np.zeros(2)
        for (cx, cy, r) in self.obstacles:
            d_vec = pos - np.array([cx, cy])
            dist_centre = np.linalg.norm(d_vec)
            direction = d_vec / (dist_centre + 1e-9)
            d = dist_centre - r  # surface distance
            if d <= 0.0:
                # inside obstacle (ghost): push straight out, hard
                f_rep += self.cfg.inside_push * direction
            elif d < self.cfg.influence:
                mag = self.cfg.k_rep * (1.0 / d - 1.0 / self.cfg.influence) / (d * d)
                f_rep += mag * direction

        return f_att + f_rep