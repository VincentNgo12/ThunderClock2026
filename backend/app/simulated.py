# File: simulated.py
# Purpose: Produce an explicit synthetic quaternion stream for a repeatable posture demo.
# Contributors: Unassigned

import asyncio
from datetime import UTC, datetime
import math
import time

from .models import Diagnostics, Imu, Sample
from .site_scenario import scenario_at


class SimulatedStream:
    def __init__(self):
        self.stats = Diagnostics()
        self.latest = None
        self.received_monotonic = None
        self.received_at = None

    async def updates(self):
        self.stats.state = "simulated"
        self.stats.started_at = time.monotonic()
        sequence = 0
        try:
            while True:
                now = time.monotonic()
                elapsed = now - self.stats.started_at
                phase = elapsed % 20
                # Upright for 10 seconds, then ramp into and out of a sustained bend.
                bend = max(0, min(1, phase - 10, 19 - phase))
                imus = []
                for sensor_id, degrees in enumerate((0, 32, 54, 65), 1):
                    # Forward bend is a negative rotation about wearer-right (+X).
                    half_angle = math.radians(-degrees * bend) / 2
                    rotation = (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0)
                    imus.append(Imu(sensor_id, quaternion=rotation))
                environment, heart_rate_bpm, spo2_percent = scenario_at(elapsed)
                self.latest = Sample("quaternion", int(elapsed * 1000), tuple(imus), sequence,
                                     environment, heart_rate_bpm, spo2_percent)
                self.received_monotonic = self.stats.last_notification = self.stats.last_valid = now
                self.received_at = datetime.now(UTC)
                self.stats.received += 1
                self.stats.decoded += 1
                self.stats.usable += 1
                yield self.latest
                sequence += 1
                await asyncio.sleep(0.05)
        finally:
            self.stats.state = "disconnected"
            self.stats.finished_at = time.monotonic()
