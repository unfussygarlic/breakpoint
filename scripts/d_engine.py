"""Stage D checkpoint: exercise the execution engine.

  1. Run nominal A with NO sensor noise          -> baseline rho
  2. Run A with sensor_noise = 0.04              -> noise is now LIVE; traj differs
  3. Replay the noisy run with the same seed     -> trajectory identical (reproducible)
  4. Serialize the record to JSON and reload it  -> round-trips

Run from the repo root:
    PYTHONPATH=. python scripts/d_engine.py
"""
from __future__ import annotations

import numpy as np

from breakpoint.scenarios.space import Scenario, normalize
from breakpoint.engine.runner import run, RunConfig, RunRecord

SEED = 0


def show(tag, rec: RunRecord):
    v = rec.rho_vector
    print(f"[{tag}] {'FAIL' if rec.failed else 'pass'}  steps={rec.steps}  reached={rec.reached}")
    print(f"      rho_safety={v['rho_safety']:+.4f}  rho_goal={v['rho_goal']:+.4f}"
          f"  rho_spec={rec.rho_spec:+.4f}")


def main():
    cfg = RunConfig(seed=SEED)

    quiet = Scenario(1.0, 0.5, ((0.1, 0.8, 0.12), (0.7, 0.3, 0.08)), (0.9, 0.9), sensor_noise=0.0)
    noisy = Scenario(1.0, 0.5, ((0.1, 0.8, 0.12), (0.7, 0.3, 0.08)), (0.9, 0.9), sensor_noise=0.04)

    rec_quiet = run(normalize(quiet), cfg)
    rec_noisy = run(normalize(noisy), cfg)
    show("no noise ", rec_quiet)
    show("noise .04", rec_noisy)

    # --- reproducible replay: same u, same seed -> identical trajectory ---
    rec_replay = run(normalize(noisy), cfg)
    t1 = np.array(rec_noisy.trajectory)
    t2 = np.array(rec_replay.trajectory)
    identical = np.array_equal(t1, t2)
    print(f"\nreplay identical (bitwise): {identical}  "
          f"(max abs diff = {np.abs(t1 - t2).max():.2e})")

    # --- different seed -> different noise draw -> different trajectory ---
    rec_seed7 = run(normalize(noisy), RunConfig(seed=7))
    t3 = np.array(rec_seed7.trajectory)
    diff_seed = not np.array_equal(t1, t3)
    print(f"seed 0 vs seed 7 differ   : {diff_seed}  "
          f"(max abs diff = {np.abs(t1[:min(len(t1),len(t3))] - t3[:min(len(t1),len(t3))]).max():.3f})")

    # --- JSON round-trip ---
    s = rec_noisy.to_json()
    back = RunRecord.from_json(s)
    roundtrip = np.array_equal(np.array(back.trajectory), t1) and back.rho_spec == rec_noisy.rho_spec
    print(f"JSON round-trip ok        : {roundtrip}  (record size {len(s)} chars)")
    print(f"versions logged           : {rec_noisy.versions}")


if __name__ == "__main__":
    main()