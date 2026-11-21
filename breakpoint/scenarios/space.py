"""Stage B: the scenario space Theta.

Theta is a normalized box [0,1]^d. Search strategies ALWAYS operate on the raw
unit vector u in [0,1]^d; a flat BOUNDS table maps u -> physical units, and the
Scenario dataclass is the structured, human-readable view of one point.

v0 knobs (d = 11):
    mass, friction, obs1(cx,cy,r), obs2(cx,cy,r), goal(gx,gy), sensor_noise
Start is fixed at (0.1, 0.1). sensor_noise is DEFINED here but inert until the
Stage D execution engine wires it into the control loop.

Feasibility is enforced by rejection sampling: no obstacle covering start or goal,
goal inside the workspace with a border, and goal far enough from start to be a
non-trivial task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from breakpoint.envs.nav2d import Nav2DConfig

START = (0.1, 0.1)

# (name, low, high) -- order defines the layout of the unit vector u
BOUNDS = [
    ("mass",         0.5, 3.0),
    ("friction",     0.1, 1.5),
    ("obs1_cx",      0.0, 1.0),
    ("obs1_cy",      0.0, 1.0),
    ("obs1_r",       0.05, 0.15),
    ("obs2_cx",      0.0, 1.0),
    ("obs2_cy",      0.0, 1.0),
    ("obs2_r",       0.05, 0.15),
    ("goal_x",       0.0, 1.0),
    ("goal_y",       0.0, 1.0),
    ("sensor_noise", 0.0, 0.05),
]
DIM = len(BOUNDS)
_LOW = np.array([b[1] for b in BOUNDS])
_HIGH = np.array([b[2] for b in BOUNDS])


@dataclass
class Scenario:
    mass: float
    friction: float
    obstacles: Tuple[Tuple[float, float, float], ...]
    goal: Tuple[float, float]
    sensor_noise: float

    @property
    def physical(self) -> np.ndarray:
        """Flat physical vector in BOUNDS order (inverse of unnormalize)."""
        (o1x, o1y, o1r), (o2x, o2y, o2r) = self.obstacles
        return np.array([
            self.mass, self.friction,
            o1x, o1y, o1r, o2x, o2y, o2r,
            self.goal[0], self.goal[1], self.sensor_noise,
        ])


# --- normalization map ---------------------------------------------------
def unnormalize(u: np.ndarray) -> Scenario:
    """[0,1]^DIM  ->  Scenario (physical units)."""
    u = np.asarray(u, dtype=float)
    p = _LOW + u * (_HIGH - _LOW)
    return Scenario(
        mass=float(p[0]),
        friction=float(p[1]),
        obstacles=((float(p[2]), float(p[3]), float(p[4])),
                   (float(p[5]), float(p[6]), float(p[7]))),
        goal=(float(p[8]), float(p[9])),
        sensor_noise=float(p[10]),
    )


def normalize(sc: Scenario) -> np.ndarray:
    """Scenario  ->  u in [0,1]^DIM (for search methods living in the unit box)."""
    return (sc.physical - _LOW) / (_HIGH - _LOW)


# --- feasibility ---------------------------------------------------------
def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def is_feasible(sc: Scenario, start=START, margin: float = 0.05,
                min_task_dist: float = 0.4, border: float = 0.03) -> bool:
    gx, gy = sc.goal
    # goal inside workspace with a border
    if not (border <= gx <= 1 - border and border <= gy <= 1 - border):
        return False
    # non-trivial task: goal far enough from start
    if _dist(start, sc.goal) < min_task_dist:
        return False
    for (cx, cy, r) in sc.obstacles:
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            return False
        if _dist(start, (cx, cy)) < r + margin:   # not covering start
            return False
        if _dist(sc.goal, (cx, cy)) < r + margin:  # not covering goal
            return False
    return True


# --- rejection sampler ---------------------------------------------------
def rejection_sample(rng: np.random.Generator, max_tries: int = 1000, **feas):
    """Return (u, Scenario, n_tries). Raises if no feasible point in max_tries."""
    for n in range(1, max_tries + 1):
        u = rng.random(DIM)
        sc = unnormalize(u)
        if is_feasible(sc, **feas):
            return u, sc, n
    raise RuntimeError(f"no feasible scenario in {max_tries} tries")


# --- bridge to the env ---------------------------------------------------
def to_env_config(sc: Scenario, max_steps: int = 400) -> Nav2DConfig:
    """Build a runnable env config from a scenario. start is fixed; sensor_noise
    is carried on the Scenario, not here -- the engine consumes it in Stage D."""
    return Nav2DConfig(
        mass=sc.mass,
        friction=sc.friction,
        f_max=10.0,
        world_size=1.0,
        start=START,
        goal=sc.goal,
        obstacles=sc.obstacles,
        max_steps=max_steps,
    )