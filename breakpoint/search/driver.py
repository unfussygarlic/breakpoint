"""Stage E/F: the ask-tell campaign driver + a summary over records.

The currency is FEASIBLE simulator calls. Continuous optimizers (CMA, BO) may
propose infeasible scenarios; those get a penalty objective with NO simulation,
so they steer the optimizer without spending budget. The loop runs until
`sim_budget` feasible scenarios have been executed.

objective selects what the searcher minimizes:
  "safety" -> rho_safety  (Setup 2: hunt collisions)
  "spec"   -> rho_spec    (either-mode failure)
The full rho vector is logged regardless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from breakpoint.engine.runner import run, RunConfig, RunRecord
from breakpoint.oracle.stl import STLSpec
from breakpoint.scenarios.space import unnormalize, is_feasible
from breakpoint.search.base import Searcher

INFEASIBLE_PENALTY = 1.0   # worse than any feasible rho; minimizer avoids it


def _objective(rec: RunRecord, which: str) -> float:
    if which == "safety":
        return rec.rho_vector["rho_safety"]
    return rec.rho_spec


def run_campaign(searcher: Searcher, sim_budget: int, objective: str = "spec",
                 run_seed: int = 0, spec: STLSpec | None = None,
                 max_asks: int | None = None) -> list[RunRecord]:
    spec = spec or STLSpec()
    cfg = RunConfig(seed=run_seed, spec=spec, store_trajectory=False)
    max_asks = max_asks or sim_budget * 50
    records: list[RunRecord] = []
    sims = asks = 0
    while sims < sim_budget and asks < max_asks:
        u = np.asarray(searcher.ask()); asks += 1
        if is_feasible(unnormalize(u)):
            rec = run(u, cfg)
            sims += 1
            searcher.tell(u, _objective(rec, objective))
            records.append(rec)
        else:
            searcher.tell(u, INFEASIBLE_PENALTY)   # no sim call spent
    return records


@dataclass
class CampaignSummary:
    n: int
    p_fail: float
    n_goal_fail: int
    n_safety_fail: int          # = collisions
    first_failure_at: int | None
    first_collision_at: int | None
    rho_spec: np.ndarray

    def __str__(self) -> str:
        return (f"n={self.n}  p_fail={self.p_fail:.3f}  "
                f"(goal={self.n_goal_fail}, collision={self.n_safety_fail})  "
                f"first-fail@={self.first_failure_at}  "
                f"first-collision@={self.first_collision_at}")


def summarize(records: list[RunRecord]) -> CampaignSummary:
    rho = np.array([r.rho_spec for r in records])
    failed = np.array([r.failed for r in records])
    collided = np.array([r.rho_vector["rho_safety"] < 0 for r in records])
    goal_fail = int(sum(r.rho_vector["rho_goal"] < 0 for r in records))
    first_fail = int(np.argmax(failed)) if failed.any() else None
    first_coll = int(np.argmax(collided)) if collided.any() else None
    return CampaignSummary(
        n=len(records),
        p_fail=float(failed.mean()),
        n_goal_fail=goal_fail,
        n_safety_fail=int(collided.sum()),
        first_failure_at=first_fail,
        first_collision_at=first_coll,
        rho_spec=rho,
    )