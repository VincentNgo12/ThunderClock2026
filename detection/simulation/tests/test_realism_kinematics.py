# File: test_realism_kinematics.py
# Purpose: The kinematics provider (realism/kinematics.py) returns motionval metrics in the realism
#          scorecard's row format, and the scorecard can score and rank them.
# Contributors: Unassigned
"""Unit checks on synthetic motionval rows, plus one small end-to-end run (one A-CARRY seed and the
mocap clips in motionlib/canonical; no real IMU data needed)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realism import kinematics, score  # noqa: E402
from thundersim import motion_validation as mv  # noqa: E402

ROW_KEYS = {"level", "category", "feature", "real", "real_groups", "sim", "sim_groups", "relevance", "confidence",
            "description"}


def metric_row(source: str, clip: str, activity: str, **values) -> dict:
    """One motionval metric row, as window_metrics returns it."""
    return {"_source": source, "_clip": clip, "_activity": activity, "_seconds": 5.0, "_technique": None,
            "_load_kg": None, "_kind": None, **values}


def synthetic_rows() -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    mocap_clips = ["cmu_07_01", "cmu_07_12", "cmu_08_02", "cmu_09_01", "accad_male1_b3_walk", "accad_female1_b03_walk1",
                   "100style_neutral_fw", "100style_heavyset_fw"]
    for clip in mocap_clips:
        for _ in range(2):   # two windows per clip: the provider keeps one value per clip
            rows.append(metric_row("mocap", clip, "walk", trunk_rate_p95_rads=1.2 + 0.2 * rng.standard_normal()))
        rows.append(metric_row("mocap", clip, "run", pelvis_pitch_mean_deg=2.0 + rng.standard_normal()))
    for seed in range(8):
        rows.append(metric_row("sim", f"A-CARRY/{seed}", "walk", trunk_rate_p95_rads=0.8 + 0.02 * rng.standard_normal()))
        rows.append(metric_row("sim", f"A-RUN/{seed}", "run", pelvis_pitch_mean_deg=7.0 + 0.2 * rng.standard_normal()))
    return rows


class Helpers(unittest.TestCase):
    def test_performer_groups_clips_of_one_person(self):
        self.assertEqual(kinematics.performer("cmu_07_01"), kinematics.performer("cmu_07_12"))
        self.assertNotEqual(kinematics.performer("cmu_07_01"), kinematics.performer("cmu_08_02"))
        self.assertEqual(kinematics.performer("niosh_s3_t0"), "niosh_s3")
        self.assertEqual(kinematics.performer("accad_female1_a05_pickupbox"), "accad_female1")
        self.assertEqual(kinematics.performer("100style_neutral_fw"), kinematics.performer("100style_heavyset_fw"))

    def test_range_distance_matches_the_motionval_verdict(self):
        self.assertEqual(kinematics.range_distance(5.0, 4.0, 6.0), 0.0)
        self.assertEqual(kinematics.range_distance(6.2, 4.0, 6.0), 0.0)            # inside the 5 % pad
        self.assertAlmostEqual(kinematics.range_distance(8.0, 4.0, 6.0), (8.0 - 6.3) / 2.0)
        self.assertAlmostEqual(kinematics.range_distance(2.0, 4.0, 6.0), (3.7 - 2.0) / 2.0)
        self.assertLess(kinematics.range_distance(0.2, 0.3, 0.3), 10.0)           # one-clip range stays bounded


class Rows(unittest.TestCase):
    def setUp(self):
        self.rows = kinematics.build_rows(synthetic_rows())
        self.by_key = {(r["category"], r["feature"]): r for r in self.rows}

    def test_row_format(self):
        self.assertIn(("walk", "trunk_rate_p95_rads"), self.by_key)
        for r in self.rows:
            self.assertTrue(ROW_KEYS <= set(r), r.keys())
            self.assertEqual(r["level"], kinematics.LEVEL)
            self.assertEqual(len(r["real"]), len(r["real_groups"]))
            self.assertEqual(len(r["sim"]), len(r["sim_groups"]))
            self.assertIn(r["confidence"], score.CONFIDENCE_WEIGHT)
            self.assertTrue(0.0 <= r["relevance"] <= 1.0)

    def test_one_value_per_clip_grouped_by_performer(self):
        r = self.by_key[("walk", "trunk_rate_p95_rads")]
        self.assertEqual(len(r["real"]), 8)
        self.assertEqual(len(set(r["real_groups"])), 6)    # cmu_07 twice, 100style twice
        self.assertEqual(len(r["sim"]), 8)
        self.assertEqual(r["verdict"], "low")
        self.assertIn("motionval low", r["description"])

    def test_not_judged_rows_get_no_ranking_weight(self):
        r = self.by_key[("run", "pelvis_pitch_mean_deg")]
        self.assertIn(("run", "pelvis_pitch_mean_deg"), mv.REFERENCE_CAVEATS)
        self.assertEqual(r["relevance"], 0.0)
        self.assertIn("not judged", r["description"])

    def test_scorecard_scores_and_ranks_the_rows(self):
        gaps = score.score_provider_rows(self.rows)
        g = gaps[kinematics.LEVEL]["walk"]["trunk_rate_p95_rads"]
        self.assertTrue(np.isfinite(g["nW"]))
        self.assertGreater(g["nW"], 1.0)                    # sim 0.8 vs mocap 1.2 +- 0.2
        card = {"events": {}, "activities": {}, "providers": gaps}
        ranked = score.rank(card)
        self.assertTrue(any(r["level"] == kinematics.LEVEL and r["feature"] == "trunk_rate_p95_rads" for r in ranked))
        not_judged = [r for r in ranked if (r["category"], r["feature"]) == ("run", "pelvis_pitch_mean_deg")]
        self.assertTrue(all(r["impact"] == 0.0 for r in not_judged))

    def test_tables_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = kinematics.write_tables(self.rows, score.score_provider_rows(self.rows), Path(folder))
            text = path.read_text()
            self.assertIn("trunk_rate_p95_rads", text)
            self.assertTrue((Path(folder) / "kinematics.json").exists())


class Registration(unittest.TestCase):
    def test_provider_is_registered_with_the_scorecard(self):
        self.assertIn("realism.kinematics", score.PROVIDER_MODULES)
        self.assertIn(kinematics.kinematics_provider, score.PROVIDERS)


@unittest.skipUnless(mv.MOCAP_DIR.exists(), "motionlib/canonical not present")
class EndToEnd(unittest.TestCase):
    def test_real_mocap_and_one_sim_run(self):
        rows = kinematics.build_rows(mv.collect(("A-CARRY",), seeds=1, seed_base=0, jobs=1))
        by_key = {(r["category"], r["feature"]): r for r in rows}
        r = by_key[("walk", "trunk_rate_p95_rads")]
        self.assertGreaterEqual(len(r["real"]), kinematics.MIN_REAL)
        self.assertGreaterEqual(len(r["sim"]), 1)
        self.assertTrue(np.all(np.isfinite(r["real"])))


if __name__ == "__main__":
    unittest.main()
