# File: test_replay.py
# Purpose: Replay panel endpoints: list recordings, start (mounting fitted from the session), pause, seek,
#          stop and restore the previous source, with a small synthetic recording.
# Contributors: Unassigned
import asyncio
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.main import create_app
from test_segments import Sensor, axis_angle
from test_mounting import MOUNTS

POSES = [("stand#1:hold", (1.0, 0.0, 0.0, 0.0)), ("forward#1:hold", axis_angle((1, 0, 0), -40)),
         ("right#1:hold", axis_angle((0, 0, 1), -25)), ("forward#2:hold", axis_angle((1, 0, 0), -50))]


def write_recording(folder, name="20261109T000000Z_test.jsonl", step_s=1.5, rate_hz=40):
    sensors = [Sensor(m, h) for m, h in zip(MOUNTS, (5, 80, -100, 150))]
    lines = [{"type": "meta", "session": name[:-6], "started_at": "2026-11-09T00:00:00+00:00", "source": "hardware",
              "device_format": "quaternion", "client": {"wearer": "T", "notes": "synthetic"}}]
    t = 0.0
    for label, pose in POSES:
        lines.append({"type": "label", "t": t, "name": label})
        for _ in range(int(step_s * rate_hz)):
            for s in sensors:
                s.heading += 0.5                              # drifting heading, as on the real strip
            imus = [{"id": i + 1, "valid": True, "q": list(s.quaternion(pose)), "a": None, "g": None}
                    for i, s in enumerate(sensors)]
            lines.append({"type": "sample", "t": round(t, 4), "origin": "device", "format": "quaternion",
                          "sample_time_ms": int(t * 1000), "sequence": 0, "imus": imus})
            t += 1 / rate_hz
    lines.append({"type": "end", "t": t, "samples": 0, "labels": len(POSES), "aborted": False, "truncated": False})
    path = pathlib.Path(folder) / name
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


class ReplayPanelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = write_recording(self.folder.name)
        env = {"TELEMETRY_SOURCE": "manual", "RECORDINGS_DIR": self.folder.name, "MOUNTINGS_DIR": self.folder.name}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("MOUNTING_FILE", None)
        self.app = create_app()
        lifespan = self.app.router.lifespan_context(self.app)
        await lifespan.__aenter__()
        self.addAsyncCleanup(lifespan.__aexit__, None, None, None)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.http.aclose)

    async def test_list_start_pause_seek_stop(self):
        listing = (await self.http.get("/replay/recordings")).json()
        self.assertEqual([r["file"] for r in listing["recordings"]], [self.path.name])
        self.assertEqual(listing["recordings"][0]["wearer"], "T")
        self.assertIsNone(listing["active"])

        self.assertEqual((await self.http.post("/replay/start", params={"file": "../etc/passwd"})).status_code, 404)
        self.assertEqual((await self.http.post("/replay/start", params={"file": "missing.jsonl"})).status_code, 404)

        started = await self.http.post("/replay/start", params={"file": self.path.name, "speed": 1, "start_t": 3.1})
        self.assertEqual(started.status_code, 200)
        self.assertEqual(self.app.state.source, "replay")
        await asyncio.sleep(0.4)                              # inside right#1 (3.0-4.5 s; started at 3.1)
        message = json.loads(self.app.state.latest)
        self.assertEqual(message["replay"]["label"], "right#1:hold")
        self.assertIn("fitted from", message["segment_model"]["source"])
        self.assertAlmostEqual(message["segments"]["upper_thoracic"]["lateral_deg"], 25, delta=1.5)

        paused = (await self.http.post("/replay/pause", params={"paused": True})).json()
        await asyncio.sleep(0.4)
        self.assertAlmostEqual(self.app.state.stream.t, paused["t"], delta=0.11)
        await self.http.post("/replay/pause", params={"paused": False})
        await asyncio.sleep(1.3)                              # on into forward#2 (from 4.5 s)
        self.assertEqual(json.loads(self.app.state.latest)["replay"]["label"], "forward#2:hold")
        self.assertAlmostEqual(self.app.state.segments["upper_thoracic"]["flexion_deg"], 50, delta=1.5)

        stopped = (await self.http.post("/replay/stop")).json()
        self.assertEqual(stopped["source"], "manual")
        self.assertIsNone(self.app.state.segment_model)      # restored: the strip was never calibrated
        await asyncio.sleep(0.05)                             # next published snapshot
        self.assertNotIn("replay", json.loads(self.app.state.latest))
        self.assertEqual((await self.http.post("/replay/stop")).status_code, 409)

    async def test_session_without_bends_needs_a_current_mounting(self):
        bare = pathlib.Path(self.folder.name) / "20261109T000001Z_bare.jsonl"
        lines = self.path.read_text().splitlines()
        bare.write_text("\n".join(l for l in lines if '"forward' not in l and '"right' not in l) + "\n")
        response = await self.http.post("/replay/start", params={"file": bare.name})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.app.state.source, "manual")    # nothing changed


if __name__ == "__main__":
    unittest.main()
