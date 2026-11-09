# File: test_mounting.py
# Purpose: Mounting calibration over HTTP (capture -> fit -> save -> reload -> auto re-zero) with a fake strip
#          whose boards are mounted arbitrarily and whose fused heading drifts, plus the trunk posture rules.
# Contributors: Unassigned
import asyncio
from datetime import UTC, datetime
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from app.main import create_app
from app.models import Diagnostics, Imu, Sample
from app.segments import quat_unit
from app.trunk_posture import TrunkPosture
from test_segments import Sensor, axis_angle

MOUNTS = [quat_unit(q) for q in ((0.3, -0.7, 0.2, 0.6), (0.9, 0.1, -0.4, 0.2), (0.1, 0.2, 0.9, -0.3), (0.5, 0.5, -0.5, 0.5))]


class FakeStrip:
    """Streams the four sensors' fused quaternions for whatever pose the test sets, at 50 Hz."""

    packet_format = "quaternion"
    instances = []

    def __init__(self, device, packet_format):
        self.stats = Diagnostics()
        self.latest = None
        self.received_monotonic = None
        self.received_at = None
        self.sensors = [Sensor(m, h) for m, h in zip(MOUNTS, (10, -75, 140, 33))]
        self.pose = (1.0, 0.0, 0.0, 0.0)                 # body -> world for every segment
        self.drift_deg_per_s = 0.0
        FakeStrip.instances.append(self)

    async def updates(self):
        self.stats.state = "subscribed"
        try:
            while True:
                for sensor in self.sensors:
                    sensor.heading += self.drift_deg_per_s * 0.02
                imus = tuple(Imu(i + 1, quaternion=s.quaternion(self.pose)) for i, s in enumerate(self.sensors))
                self.latest = Sample("quaternion", int(time.monotonic() * 1000), imus, 0)
                self.received_monotonic = time.monotonic()
                self.received_at = datetime.now(UTC)
                yield self.latest
                await asyncio.sleep(0.02)
        finally:
            self.stats.state = "disconnected"


class MountingFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        FakeStrip.instances.clear()

        async def scan():
            return [("device", None)]

        env = {"TELEMETRY_SOURCE": "manual", "MOUNTINGS_DIR": self.folder.name, "DEVICE_ID": "strip-A"}
        for patcher in (patch.dict(os.environ, env), patch("app.main.scan", scan),
                        patch("app.main.DeviceStream", FakeStrip),
                        patch("app.main.select_device", lambda devices, _id: devices[0][0])):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("MOUNTING_FILE", None)
        self.app, self.http = await self.start_app()

    async def start_app(self):
        app = create_app()
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        self.addAsyncCleanup(lifespan.__aexit__, None, None, None)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(http.aclose)
        return app, http

    async def connect(self, http):
        self.assertEqual((await http.post("/device/connect")).status_code, 200)
        await asyncio.sleep(0.1)
        return FakeStrip.instances[-1]

    def trunk(self, app):
        return app.state.segments["upper_thoracic"]

    async def test_calibrate_save_reload_and_rezero(self):
        strip = await self.connect(self.http)
        self.assertFalse((await self.http.get("/mounting")).json()["calibrated"])
        self.assertEqual((await self.http.post("/mounting/fit")).status_code, 409)       # nothing captured

        stand = (await self.http.post("/mounting/capture", params={"pose": "stand", "seconds": 0.5})).json()
        self.assertTrue(all(info["still"] for info in stand["sensors"].values()))
        self.assertEqual((await self.http.post("/mounting/fit")).status_code, 409)       # needs a bend
        strip.pose = axis_angle((1, 0, 0), -45)                                          # bend forward
        forward = (await self.http.post("/mounting/capture", params={"pose": "forward", "seconds": 0.5})).json()
        self.assertAlmostEqual(forward["sensors"]["pelvis"]["tilt_from_stand_deg"], 45, delta=1)
        fit = (await self.http.post("/mounting/fit")).json()
        self.assertEqual(len(fit["regions"]), 4)
        self.assertTrue(os.path.exists(fit["saved_path"]))

        # Bend left 30 deg while every sensor's heading drifts by tens of degrees: view still says "left 30".
        strip.pose = axis_angle((0, 0, 1), 30)
        strip.drift_deg_per_s = 40
        await asyncio.sleep(0.6)
        self.assertAlmostEqual(self.trunk(self.app)["lateral_deg"], -30, delta=1)
        self.assertAlmostEqual(self.trunk(self.app)["flexion_deg"], 0, delta=1)

        # Hold the bend: a trunk posture warning appears (2 s sustained).
        await asyncio.sleep(2.2)
        message = __import__("json").loads(self.app.state.latest)
        self.assertEqual(message["posture_source"], "gravity")
        self.assertEqual(message["trunk_posture"]["zone"], "moderate")
        self.assertIn("side bend to the left", message["alerts"][0]["summary"])

        # Restart: the saved mounting is loaded; the strip now sits 8 deg further forward when "upright".
        app2, http2 = await self.start_app()
        self.assertTrue((await http2.get("/mounting")).json()["calibrated"])
        strip2 = await self.connect(http2)
        strip2.pose = axis_angle((1, 0, 0), -8)
        await asyncio.sleep(0.3)
        self.assertAlmostEqual(app2.state.segments["upper_thoracic"]["flexion_deg"], 8, delta=1)
        self.assertTrue(app2.state.calibrator.auto_zero_pending)
        await asyncio.sleep(2.4)                                                          # stands still 2 s
        self.assertFalse(app2.state.calibrator.auto_zero_pending)
        self.assertAlmostEqual(app2.state.segments["upper_thoracic"]["flexion_deg"], 0, delta=1)
        strip2.pose = axis_angle((1, 0, 0), -38)                                          # 30 deg past the new upright
        await asyncio.sleep(0.3)
        self.assertAlmostEqual(app2.state.segments["upper_thoracic"]["flexion_deg"], 30, delta=1.5)

    async def test_manual_zero(self):
        strip = await self.connect(self.http)
        self.assertEqual((await self.http.post("/mounting/zero")).status_code, 409)       # no mounting yet
        await self.http.post("/mounting/capture", params={"pose": "stand", "seconds": 0.5})
        strip.pose = axis_angle((0, 0, 1), -30)                                           # bend right
        await self.http.post("/mounting/capture", params={"pose": "right", "seconds": 0.5})
        await self.http.post("/mounting/fit")
        strip.pose = axis_angle((0, 0, 1), -5)
        zero = await self.http.post("/mounting/zero", params={"seconds": 0.5})
        self.assertEqual(zero.status_code, 200)
        await asyncio.sleep(0.2)
        self.assertAlmostEqual(self.trunk(self.app)["lateral_deg"], 0, delta=1)

    async def test_capture_needs_a_stream_and_a_known_pose(self):
        self.assertEqual((await self.http.post("/mounting/capture", params={"pose": "stand"})).status_code, 409)
        await self.connect(self.http)
        self.assertEqual((await self.http.post("/mounting/capture", params={"pose": "sideways"})).status_code, 409)


def seg(flexion=0.0, lateral=0.0, inclination=None):
    inclination = abs(flexion) if inclination is None else inclination
    return {"flexion_deg": flexion, "lateral_deg": lateral, "inclination_deg": inclination}


class TrunkPostureTests(unittest.TestCase):
    def run_for(self, posture, segments, seconds):
        state = None
        for k in range(int(seconds * 20) + 1):
            state = posture.update(segments, k / 20)
        return state

    def test_neutral_has_no_alerts(self):
        posture = TrunkPosture()
        state = self.run_for(posture, {"pelvis": seg(), "upper_thoracic": seg(10, 5)}, 5)
        self.assertEqual(state["zone"], "neutral")
        self.assertEqual(posture.alerts, [])

    def test_forward_zones_need_to_be_sustained(self):
        posture = TrunkPosture(warning_seconds=2)
        self.run_for(posture, {"upper_thoracic": seg(35)}, 1.5)
        self.assertEqual(posture.alerts, [])
        self.run_for(posture, {"upper_thoracic": seg(35)}, 2.5)
        self.assertEqual(posture.alerts[0]["severity"], "warning")
        posture.reset()
        state = self.run_for(posture, {"upper_thoracic": seg(70)}, 2.5)
        self.assertEqual(state["zone"], "high")
        self.assertEqual(posture.alerts[0]["severity"], "high")

    def test_extension_and_side_bend(self):
        posture = TrunkPosture()
        self.run_for(posture, {"upper_thoracic": seg(-25)}, 2.5)
        self.assertIn("backward", posture.alerts[0]["summary"])
        posture.reset()
        self.run_for(posture, {"upper_thoracic": seg(0, 20, 20)}, 2.5)
        self.assertIn("right", posture.alerts[0]["summary"])

    def test_lying_is_a_state_not_a_posture_alert(self):
        posture = TrunkPosture()
        state = self.run_for(posture, {"pelvis": seg(85), "upper_thoracic": seg(88)}, 5)
        self.assertTrue(state["lying"])
        self.assertEqual(state["zone"], "lying")
        self.assertEqual(posture.alerts, [])

    def test_joint_bend_between_segments(self):
        state = TrunkPosture().update({"pelvis": seg(10), "lumbar": seg(30), "thoracic": seg(45), "upper_thoracic": seg(50)}, 0)
        self.assertEqual(state["joints"]["lumbar"]["flexion_deg"], 20)
        self.assertEqual(state["joints"]["upper_thoracic"]["flexion_deg"], 5)


if __name__ == "__main__":
    unittest.main()
