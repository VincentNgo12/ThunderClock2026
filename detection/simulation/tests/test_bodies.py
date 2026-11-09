# File: test_bodies.py
# Purpose: The Anny body presets load into a valid Worker.
import dataclasses
import unittest

from thundersim.bodies import available_bodies, worker_from_anny
from thundersim.spine import Worker


class AnnyBodyTests(unittest.TestCase):
    def test_presets_are_shipped(self):
        self.assertIn("avg_male", available_bodies())
        self.assertIn("female_160", available_bodies())

    def test_avg_male_builds_a_worker(self):
        worker = Worker(**worker_from_anny("avg_male"))
        self.assertAlmostEqual(worker.height_m, 1.765, delta=0.002)
        self.assertAlmostEqual(worker.mass_kg, 69.6, delta=0.2)
        self.assertEqual(len(worker.spacings()), 3)

    def test_overrides_and_unknown_fields(self):
        self.assertEqual(worker_from_anny("avg_male", load_kg=12)["load_kg"], 12.0)
        with self.assertRaises(KeyError):
            worker_from_anny("avg_male", shoe_size=42)

    def test_fall_geometry_is_opt_in(self):
        self.assertNotIn("pelvis_height_m", worker_from_anny("avg_male"))
        kwargs = worker_from_anny("avg_male", fall_geometry=True)
        self.assertAlmostEqual(kwargs["pelvis_height_m"], 1.014, delta=0.002)
        self.assertEqual(len(kwargs["com_to_pelvis_m"]), 3)
        if "pelvis_height_m" in {f.name for f in dataclasses.fields(Worker)}:
            self.assertAlmostEqual(Worker(**kwargs).pelvis_height, kwargs["pelvis_height_m"])


if __name__ == "__main__":
    unittest.main()
