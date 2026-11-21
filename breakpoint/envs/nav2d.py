from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Nav2DConfig:
    dt: float = 0.05
    mass: float = 3.0
    friction: float = 0.5
    f_max: float = 10.0
    world_size: float = 1.0
    start: Tuple[float, float] = (0.1, 0.1)
    goal: Tuple[float, float] = (0.9, 0.9)
    goal_radius: float = 0.05

    obstacles: Tuple[Tuple[float, float, float], ...] = (
        (0.1, 0.8, 0.12),
        (0.7, 0.3, 0.08),
    )
    max_steps: int = 400


class Nav2DEnv:

    def __init__(self, config: Nav2DConfig | None = None, seed: int | None = None):
        self.cfg = config or Nav2DConfig()
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self) -> np.ndarray:
        x, y = self.cfg.start
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        return self.state.copy()

    def step(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        norm = np.linalg.norm(u)
        if norm > self.cfg.f_max:
            u = u * (self.cfg.f_max / norm)

        pos = self.state[:2].copy()
        v = self.state[2:].copy()

        a = (u - self.cfg.friction * v) / self.cfg.mass
        v = v + a * self.cfg.dt
        pos = pos + v * self.cfg.dt

        for i in (0, 1):
            if pos[i] < 0.0:
                pos[i] = 0.0
                v[i] = 0.0
            elif pos[i] > self.cfg.world_size:
                pos[i] = self.cfg.world_size
                v[i] = 0.0

        self.state = np.concatenate([pos, v])
        return self.state.copy()


    def dist_to_goal(self, pos) -> float:
        return float(np.linalg.norm(np.asarray(pos) - np.asarray(self.cfg.goal)))


    def signed_clearance(self, pos) -> float:
        pos = np.asarray(pos)
        margins = [
            np.linalg.norm(pos - np.array([cx, cy])) - r
            for (cx, cy, r)  in self.cfg.obstacles
        ]

        return float(min(margins)) if margins else float('inf')

    def at_goal(self, pos) -> bool:
        return self.dist_to_goal(pos) <= self.cfg.goal_radius
    