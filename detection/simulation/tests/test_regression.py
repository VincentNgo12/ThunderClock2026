# File: test_regression.py
# Purpose: End-to-end regression gates on fixed seeds: the full chain (protobuf + backend
#          decoder) runs, the baseline keeps its known behaviour, the firmware gap stays
#          visible, and external models plug in through the FallModel API.
# Contributors: Unassigned
"""These pin behaviour on SIMULATED scenarios so refactors cannot silently change it.
If a deliberate change moves a result, update the expectation here and say why."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thundersim.model_api import Event, load_model  # noqa: E402
from thundersim.runner import StreamOptions, run_scenario  # noqa: E402

OBS = 60.0


class BaselineRegression(unittest.TestCase):
    def check(self, scenario, seed, expected_output, profile="target", fast=True):
        result = run_scenario(scenario, seed, "baseline", StreamOptions(profile=profile, fast=fast), obs_s=OBS)
        self.assertEqual(result.output, expected_output, f"{scenario} seed {seed} on {profile}: {result.events}")
        return result

    def test_full_chain_through_backend_decoder(self):
        result = self.check("L-TRIP", 3, "fall_confirmed", fast=False)
        self.assertLessEqual(result.link["max_packet_bytes"], 150)
        self.assertLess(result.latency_s, 30.0)

    def test_falls_are_confirmed_on_target_firmware(self):
        for scenario, seed in (("L-SLIP", 3), ("H-SCAF", 3), ("H-LAD", 3)):
            self.check(scenario, seed, "fall_confirmed")

    def test_hard_negatives_stay_quiet(self):
        for scenario, seed in (("A-LIEWORK", 3), ("DEVICE-OFF", 3), ("A-KNEEL", 5)):
            self.check(scenario, seed, "normal")

    def test_stumble_is_a_near_fall(self):
        self.check("NF-STUMBLE", 3, "near_fall")

    def test_current_firmware_gap_is_visible(self):
        self.check("L-TRIP", 3, "normal", profile="current")


class Echo:
    """Minimal external model: proves the FallModel contract is enough to plug in."""

    name = "echo"

    def reset(self, context):
        self.fired = False

    def update(self, batch):
        if not self.fired and batch.t[-1] - batch.t[0] > 0.2 and (abs(batch.acc[:, 0, 1]) < 3).any():
            self.fired = True
            return [Event(float(batch.t[-1]), "near_fall")]
        return []


class PluginTests(unittest.TestCase):
    def test_external_model_object_runs(self):
        result = run_scenario("L-TRIP", 3, Echo(), StreamOptions(fast=True), obs_s=OBS)
        self.assertIn(result.output, ("near_fall", "normal"))

    def test_falldet_adapter_if_available(self):
        try:
            load_model("falldet")
        except ImportError as error:
            self.skipTest(str(error))
        result = run_scenario("L-TRIP", 5, "falldet", StreamOptions(fast=True), obs_s=OBS)
        self.assertEqual(result.output, "fall_confirmed")


if __name__ == "__main__":
    unittest.main()
