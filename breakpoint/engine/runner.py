"""Stage D: the execution engine -- fixed shared infrastructure.

run(u, cfg) is the ONE place a scenario is executed. Every search strategy calls
it, so method comparisons are fair by construction. It:
  * unnormalizes the search vector u -> Scenario -> env config
  * rolls the fixed controller, feeding it a NOISY observation (sensor_noise),
    while the TRUE state evolves by real dynamics
  * scores the TRUE trajectory with the STL oracle (the only authority on failure)
  * returns a RunRecord with everything needed to replay the run bit-for-bit

RNG discipline: SeedSequence(seed).spawn(2) gives independent, seed-determined
streams for the env and for observation noise.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

from breakpoint.envs.nav2d import Nav2DEnv
from breakpoint.controllers.potential_field import PotentialFieldController, APFConfig
from breakpoint.scenarios.space import unnormalize, normalize, to_env_config, Scenario, DIM
from breakpoint.oracle.stl import STLSpec, evaluate


@dataclass
class RunConfig:
    seed: int = 0
    spec: STLSpec = field(default_factory=STLSpec)
    apf: APFConfig = field(default_factory=APFConfig)
    max_steps: int = 400
    store_trajectory: bool = True   # keep full traj in the record (for replay/plots)


@dataclass
class RunRecord:
    # --- inputs (reproducibility) ---
    u: list
    scenario: dict
    seed: int
    spec: dict
    apf: dict
    env_meta: dict
    versions: dict
    timestamp: str
    # --- outputs ---
    steps: int
    reached: bool
    rho_vector: dict
    rho_spec: float
    failed: bool
    trajectory: list | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "RunRecord":
        return RunRecord(**json.loads(s))


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git": _git_hash(),
    }


def run(u, cfg: RunConfig | None = None) -> RunRecord:
    cfg = cfg or RunConfig()
    u = np.asarray(u, dtype=float)
    assert u.shape == (DIM,), f"expected u of shape ({DIM},), got {u.shape}"

    sc = unnormalize(u)
    env_cfg = to_env_config(sc, max_steps=cfg.max_steps)

    # independent, seed-determined streams for env and observation noise
    env_seed, noise_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    env = Nav2DEnv(env_cfg, seed=env_seed)
    noise_rng = np.random.default_rng(noise_seed)
    ctrl = PotentialFieldController(env_cfg.goal, env_cfg.obstacles, cfg.apf)

    s = env.reset()
    traj = [s[:2].copy()]
    reached = False
    for _ in range(cfg.max_steps):
        obs = s.copy()
        if sc.sensor_noise > 0.0:
            obs[:2] = obs[:2] + noise_rng.normal(0.0, sc.sensor_noise, size=2)
        u_ctrl = ctrl(obs)
        s = env.step(u_ctrl)            # TRUE dynamics, noise-free
        traj.append(s[:2].copy())
        if env.dist_to_goal(s[:2]) <= cfg.spec.eps:
            reached = True
            break

    traj = np.array(traj)
    res = evaluate(traj, sc.obstacles, sc.goal, cfg.spec)

    return RunRecord(
        u=[float(x) for x in u],
        scenario={
            "mass": sc.mass, "friction": sc.friction,
            "obstacles": [list(o) for o in sc.obstacles],
            "goal": list(sc.goal), "sensor_noise": sc.sensor_noise,
        },
        seed=cfg.seed,
        spec=asdict(cfg.spec),
        apf=asdict(cfg.apf),
        env_meta={
            "dt": env_cfg.dt, "f_max": env_cfg.f_max,
            "world_size": env_cfg.world_size, "start": list(env_cfg.start),
            "max_steps": cfg.max_steps,
        },
        versions=_versions(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        steps=len(traj) - 1,
        reached=reached,
        rho_vector=res.vector,
        rho_spec=res.rho_spec,
        failed=res.failed,
        trajectory=[[float(x), float(y)] for x, y in traj] if cfg.store_trajectory else None,
    )


def run_scenario(sc: Scenario, cfg: RunConfig | None = None) -> RunRecord:
    """Convenience: run a structured Scenario (normalizes to u first)."""
    return run(normalize(sc), cfg)