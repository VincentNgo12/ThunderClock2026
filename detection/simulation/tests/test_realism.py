# File: test_realism.py
# Purpose: Realism regression gates. Each pins a gap that the realism scorecard
#          (detection/simulation/realism, REPORT.md) measured against real recordings and that a
#          simulator change closed, so a later edit cannot silently reopen it.
# Contributors: Unassigned
"""Sim-only checks run everywhere; the scorecard check runs only when the real-data cache exists
(it lives outside the repo, see realism/paths.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realism import emulate  # noqa: E402
from realism.features import event_features, find_event, trace_features, event_trace  # noqa: E402
from realism.mapping import EVENTS  # noqa: E402
from realism.paths import SPECS  # noqa: E402


def emulated_event(category: str, seed: int):
    cat = next(c for c in EVENTS if c.name == category)
    run = emulate.event_run(cat, seed)
    if run is None:
        return None
    spec = SPECS["erciyes"]
    acc, gyro = emulate.body_signals(run["track"], run["worker"], run["soft_fn"], run["soft_zeta"], "erciyes")
    acc, gyro = emulate.sample_like(acc, gyro, run["track"].fs, spec, np.random.default_rng(seed))
    i0 = find_event(acc, spec.fs, *run["window"])
    f = event_features(acc, gyro, spec.fs, i0)
    f.update(trace_features(event_trace(acc, gyro, spec.fs, i0)))
    return f


class RealismRegression(unittest.TestCase):
    SEEDS = [emulate.dev_seed(90, i) for i in range(6)]

    def features(self, category):
        rows = [emulated_event(category, s) for s in self.SEEDS]
        return [r for r in rows if r is not None]

    def test_trunk_lies_flat_after_a_fall(self):
        # Iteration 1: real chest-minus-waist inclination after lying falls is ~0 +/- 10 deg; the
        # sim used to keep the descent curl (median +14..+26 deg).
        rows = self.features("fall_forward") + self.features("fall_backward")
        diff = np.array([r["post_incl_c"] - r["post_incl_w"] for r in rows])
        self.assertLess(abs(np.median(diff)), 8.0, diff)

    def test_no_rebound_flight_after_body_impact(self):
        # Iteration 2: real waist |a| stays ~0.6 g after a body-first impact (median 0.57-0.67 g);
        # the sim bounced off the floor (0.15-0.3 g) and the trunk rang (chest 4-10 rad/s).
        # Round 2 (slower, resisted descent): per-run minima spread widely (0.03-0.82 g), so this
        # uses 12 seeds per category; the threshold sits between the old bounce (scorecard medians
        # 0.15-0.31 g) and the current sim (0.40 here, 0.52-0.58 in the 40-run scorecard).
        seeds = [emulate.dev_seed(90, i) for i in range(12)]
        rows = [r for c in ("fall_forward", "fall_backward") for r in (emulated_event(c, s) for s in seeds) if r]
        self.assertGreater(np.median([r["postmin_w"] for r in rows]), 0.33)
        self.assertLess(np.median([r["gyro_post_c"] for r in rows]), 4.0)

    def test_trip_descent_is_not_faster_than_real(self):
        # Round 2: UNIVRFall's video-labelled onset->impact for walking trips is 0.62 s (IQR
        # 0.54-0.71); the sim tripped in ~0.37-0.44 s with hip speeds of 3.6-4.0 m/s (real falls
        # 2.1-3.0 m/s). Protective resistance fixed both.
        from thundersim.scenarios import build_run

        durations, speeds = [], []
        for seed in self.SEEDS:
            run = build_run("L-TRIP", seed, obs_s=5.0)
            onset = next(t for t, n, _ in run.events if n == "fall_onset")
            impact = next(t for t, n, _ in run.events if n == "impact" and t >= onset)
            durations.append(impact - onset)
            speeds.append(run.params["hip_speed_ms"])
        self.assertGreater(np.median(durations), 0.5, durations)
        self.assertLess(np.median(speeds), 3.2, speeds)

    def test_syncope_lands_with_an_impact(self):
        # Iteration 3: acted syncope in Erciyes lands at ~3.5 g at the waist and fires Stage 1 in
        # 95 % of trials; the guided sag landed at ~1.4 g and never fired it.
        rows = self.features("syncope")
        self.assertGreater(np.median([r["peak_w"] for r in rows]), 2.2)


class ScorecardSmoke(unittest.TestCase):
    def test_scorecard_runs_if_real_cache_present(self):
        from realism.score import FALLDET_CACHE_VERSION, REAL_CACHE_VERSION, real_side, score, sim_side

        cache = Path(__file__).resolve().parents[4] / "research" / "artifacts" / "realism_runs" / "cache"
        if not ((cache / f"real_side_{REAL_CACHE_VERSION}.pkl").exists()
                and (cache / f"real_falldet_{FALLDET_CACHE_VERSION}.pkl").exists()):
            self.skipTest("real-data cache not present (outside the repo)")
        real = real_side(cache, True)
        sim = sim_side(6, 2, 2, False, only={"fall_side"})
        card = score(real, sim, only={"fall_side"})
        self.assertIn("fall_side", card["events"])
        self.assertTrue(np.isfinite(card["events"]["fall_side"]["post_incl_c"]["nW"]))


if __name__ == "__main__":
    unittest.main()
