# File: demo_app.py
# Purpose: Spine-demo server. Runs the backend app unchanged with a ThunderSim replay as its telemetry
#          source: quaternions drive the spine view and the same run's raw samples feed the detection
#          hub, whose state is published as `fall_state`/`detector`/`demo` in the existing snapshot.
# Contributors: Unassigned
"""Run (from the repository root; needs the simulation venv with the backend API packages):

    cd detection && HUB_DEMO=1 PYTHONPATH=.:simulation:../backend simulation/.venv/bin/python \\
        -m uvicorn hub.demo_app:app --host 127.0.0.1 --port 8000
    cd frontend && npm run dev          # http://localhost:5173

No backend, firmware or simulator file is edited. The wrapper
  * replaces `app.main.SimulatedStream` with ReplayStream (TELEMETRY_SOURCE=simulated),
  * wraps `app.main.snapshot` to add the hub's state to the 20 Hz websocket message,
  * adds /demo/scenarios, /demo/scenario (POST) and /incident/dismiss (POST).

Real time: the replay is paced by the wall clock (raw samples at the recorded 100 Hz device
timestamps, quaternions at 20 Hz), the hub runs on its own thread, and nothing waits on ML.
Live strip: TELEMETRY_SOURCE=hardware DEVICE_ID="<id>" DEVICE_FORMAT=both (or raw_fusion) runs the
same hub on the real BLE stream with no replay; the snapshot carries the same `fall_state`/`detector`
fields (`demo.phase` is "live"). The simulated-source path is the only one tested without hardware.
POST routes take query parameters, not a body, so the browser sends no CORS preflight.

Prototype indicator on simulated data. Not a medical or safety-certified device.
"""
import asyncio
import os
import time
from datetime import UTC, datetime

import numpy as np

os.environ.setdefault("TELEMETRY_SOURCE", "simulated")
LIVE = os.environ["TELEMETRY_SOURCE"] in ("hardware", "manual")   # real strip over BLE (manual: Connect button)
if not LIVE:
    os.environ.setdefault("RAW_IMU_AXES", ";".join(["x,y,z"] * 4))   # replay is already in the segment frame

from .backend_hook import DetectorService  # noqa: E402

START_DELAY_S = 0.5
END_HOLD_S = 8.0
TICK_S = 0.005
QUAT_PERIOD_S = 0.01   # replay quaternions at the 100 Hz IMU rate
EVENTS_SHOWN = 8
DEFAULT_SCENARIO = "SYNC"
DEFAULT_SEED = 1
# Test menu for the spine demo: (group, scenario id, label). Any catalogue id also works.
TEST_MENU = (
    ("Collapse", "SYNC", "Collapse from standing (syncope)"),
    ("Collapse", "SYNC-KNEEL", "Collapse while kneeling"),
    ("Fall", "L-TRIP", "Trip, forward fall"),
    ("Fall", "L-SLIP", "Slip, backward fall"),
    ("Fall", "STAIR", "Fall down stairs"),
    ("Fall", "H-LAD", "Fall from a ladder"),
    ("Fall", "FALL-GETUP", "Fall, then gets up"),
    ("Near fall", "NF-STUMBLE", "Stumble and catch"),
    ("Near fall", "NF-RAIL", "Slip and grab a rail"),
    ("Bad posture", "A-REBAR", "Sustained deep stoop (rebar tying)"),
    ("Bad posture", "A-CARRY", "Carrying and bending to lift"),
    ("Bad posture", "A-KNEEL", "Kneeling floor work, trunk bent"),
    ("Bad posture", "A-DRILL", "Overhead drilling, trunk extended"),
    ("Normal work", "A-SIT", "Sitting down hard"),
    ("Normal work", "A-LIE", "Lying down on purpose"),
    ("Normal work", "A-HAMMER", "Hammering"),
    ("Normal work", "A-RUN", "Running"),
)
PREVIEW_SCENARIOS = tuple(item[1] for item in TEST_MENU)

STATE_OF_KIND = {"suspected_fall": "suspected", "near_fall": "near_fall", "recovered": "normal",
                 "fall_confirmed": "fall_confirmed", "man_down_warning": "man_down_warning",
                 "man_down_critical": "man_down_critical", "cleared": "normal"}
ALERT_STATES = ("fall_confirmed", "man_down_warning", "man_down_critical")


class Demo:
    """Playback control plus the observed detection timeline (touched only from the event loop)."""

    def __init__(self, service, live=False):
        self.service = service
        self.live = live
        self.replay = None
        self.phase = "live" if live else "building"   # building | playing | holding | error | live
        self.error = None
        self.playback_started = time.monotonic() if live else None   # playback t = 0; live: server start
        self.generation = 0               # bumps on every (re)start so the stream notices
        self.pending = None               # Replay waiting to start
        self.timeline = []                # events seen this playback, with playback seconds
        self.seen_id = 0
        self.expect_reset = 0
        self.need_neutral = False
        self.dismissed = set()
        self.building = None
        self.catalogue = []
        self.app_state = None                 # the backend app's state (recorder), set by make_demo_app

    # ------------------------------------------------------------------ control
    async def start(self, scenario, seed):
        from .demo_replay import build_replay

        if self.live:
            return False
        self.phase, self.error, self.building = "building", None, scenario
        try:
            # A separate process: the simulator is CPU-bound and would starve the 100 Hz stream of the GIL.
            replay = await asyncio.get_running_loop().run_in_executor(_builder(), build_replay, scenario, seed)
        except Exception as error:   # unknown scenario etc.: keep the previous replay playing
            self.error = f"{type(error).__name__}: {error}"
            self.phase = "playing" if self.replay is not None else "error"
            self.building = None
            return False
        self.building = None
        self.pending = replay
        self.generation += 1
        return True

    def begin_playback(self, replay, now):
        """Called by the stream when it picks up a pending replay."""
        self.replay, self.pending = replay, None
        self.playback_started = now
        self.phase = "playing"
        self.timeline, self.dismissed = [], set()
        self.expect_reset = self.service.reset_count + 1
        self.service.reset()                  # new stream: fresh model, intake and calibration
        self.need_neutral = True              # replay starts upright: capture the posture neutral

    def playback_s(self, now=None):
        if self.playback_started is None:
            return None
        return round((time.monotonic() if now is None else now) - self.playback_started, 2)

    def dismiss(self):
        episode = self.current_episode()
        if episode is None:
            return None
        self.dismissed.add(episode)
        self.service.dismiss(episode)
        return episode

    # ------------------------------------------------------------------ observation
    def observe(self):
        """Copy new hub events into the timeline (called at 20 Hz from the snapshot wrapper)."""
        if self.service.reset_count < self.expect_reset:
            return                            # the hub has not yet applied this playback's reset
        self.seen_id = max(self.seen_id, self.service.floor_id)
        now = self.playback_s()
        for event in list(self.service.events):
            if event["id"] > self.seen_id:
                detail = event["detail"]
                self.timeline.append({"id": event["id"], "kind": event["kind"], "confidence": event["confidence"],
                                      "episode_id": detail.get("episode_id"), "path": detail.get("path"),
                                      "rule": detail.get("rule"), "playback_s": now})
                self.seen_id = event["id"]

    def current_episode(self):
        for event in reversed(self.timeline):
            if event["kind"] in ALERT_STATES and event["episode_id"] is not None:
                return event["episode_id"]
        return None

    def fall_state(self):
        self.observe()
        last = self.timeline[-1] if self.timeline else None
        state = "normal" if last is None else STATE_OF_KIND[last["kind"]]
        episode = self.current_episode()
        # A dismissal covers fall_confirmed only; man_down_* keep escalating (docs 3.7).
        dismissed = state == "fall_confirmed" and episode in self.dismissed
        return {
            "state": "dismissed" if dismissed else state,
            "confidence": None if last is None else last["confidence"],
            "episode_id": episode,
            "dismissable": state == "fall_confirmed" and not dismissed,
            "since_s": None if last is None else last["playback_s"],
            "events": self.timeline[-EVENTS_SHOWN:],
            "label": "prototype indicator",
        }

    def demo_block(self):
        replay = self.replay
        return {
            "phase": self.phase, "error": self.error, "building": self.building,
            "live": self.live,
            "scenario": None if replay is None else replay.scenario,
            "seed": None if replay is None else replay.seed,
            "description": None if replay is None else replay.description,
            "expected": None if replay is None else replay.expected,
            "duration_s": None if replay is None else round(replay.duration_s, 1),
            "playback_s": self.playback_s(),
            "truth": [] if replay is None else replay.truth,
            "link": None if replay is None else replay.link,
        }


_POOL = None


def _builder():
    global _POOL
    if _POOL is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _POOL = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    return _POOL


class ReplayStream:
    """Stands in for backend SimulatedStream: same attributes, same `updates()` contract."""

    packet_format = "quaternion"

    def __init__(self, demo):
        from app.models import Diagnostics

        self.demo = demo
        self.stats = Diagnostics()
        self.latest = None
        self.received_monotonic = None
        self.received_at = None

    async def updates(self):
        from app.models import Imu, Sample
        from app.site_scenario import scenario_at

        demo, service = self.demo, self.demo.service
        self.stats.state = "simulated"
        self.stats.started_at = time.monotonic()
        sequence, replay, cursor, last_quat, ended = 0, None, 0, 0.0, None
        try:
            while True:
                now = time.monotonic()
                if demo.pending is not None:
                    replay, cursor, ended = demo.pending, 0, None
                    demo.begin_playback(replay, now + START_DELAY_S)
                    self.stats.resets += 1
                if replay is None:
                    await asyncio.sleep(TICK_S)
                    yield None
                    continue
                elapsed = now - demo.playback_started
                if elapsed >= 0:
                    upto = int(np.searchsorted(replay.t, elapsed, side="right"))
                    recorder = getattr(demo.app_state, "recorder", None)
                    for k in range(cursor, upto):
                        imus = tuple(Imu(i + 1, accel=tuple(replay.acc[k, i]), gyro=tuple(replay.gyro[k, i]))
                                     for i in range(4))
                        raw = Sample("raw", int(replay.time_ms[k]), imus)
                        service.on_raw_sample(raw, now)
                        if recorder is not None:      # "Collect data" also gets the replay's raw stream
                            recorder.add(raw)
                    cursor = upto
                if cursor >= len(replay.t):
                    ended = now if ended is None else ended
                    demo.phase = "holding"
                    if now - ended > END_HOLD_S:           # loop the same scenario from its start
                        demo.pending = replay
                        demo.generation += 1
                if now - last_quat >= QUAT_PERIOD_S:
                    last_quat = now
                    k = max(0, min(cursor, len(replay.t)) - 1)
                    imus = tuple(Imu(i + 1, quaternion=tuple(float(v) for v in replay.quat[k, i])) for i in range(4))
                    environment, heart_rate_bpm, spo2_percent = scenario_at(max(0.0, elapsed))
                    self.latest = Sample("quaternion", int(replay.time_ms[k]), imus, sequence,
                                         environment, heart_rate_bpm, spo2_percent)
                    self.received_monotonic = self.stats.last_notification = self.stats.last_valid = now
                    self.received_at = datetime.now(UTC)
                    self.stats.received += 1
                    self.stats.decoded += 1
                    self.stats.usable += 1
                    sequence += 1
                    yield self.latest
                await asyncio.sleep(TICK_S)
        finally:
            self.stats.state = "disconnected"
            self.stats.finished_at = time.monotonic()


def make_demo_app(service=None, scenario=DEFAULT_SCENARIO, seed=DEFAULT_SEED):
    from contextlib import asynccontextmanager

    from fastapi import HTTPException

    import app.main as backend_main
    from app.dashboard import snapshot as base_snapshot

    service = service or DetectorService()
    if not LIVE:
        warm_up()
    demo = Demo(service, live=LIVE)

    if LIVE:
        def accept(sample):   # same receipt-time rule as backend_hook.make_app
            stream = getattr(app_ref.get("app"), "state", None)
            stream = getattr(stream, "stream", None)
            receipt = getattr(stream, "received_monotonic", None) if getattr(stream, "packet_format", None) == "raw" else None
            service.on_raw_sample(sample, receipt)
        on_raw = accept
    else:
        backend_main.SimulatedStream = lambda: ReplayStream(demo)
        on_raw = None
    app_ref = {}

    def snapshot(stream, sequence, *args, **kwargs):
        posture = kwargs.get("posture")
        if demo.need_neutral and posture is not None and stream.latest is not None and demo.playback_s() is not None \
                and demo.playback_s() >= 0:
            try:   # same call as POST /posture/calibrate, on the first upright frame of the playback
                posture.capture(stream.latest, stream.received_monotonic, stream.received_at, time.monotonic())
                demo.need_neutral = False
            except ValueError:
                pass
        state = base_snapshot(stream, sequence, *args, **kwargs)
        state["fall_state"] = demo.fall_state()
        state["detector"] = service.snapshot()["detector"]
        state["demo"] = demo.demo_block()
        state["freshness"]["fall_state"] = {"status": "valid" if (demo.live or demo.replay is not None) else "unavailable",
                                            "source": "hub", "received_at": None, "age_ms": 0}
        return state

    backend_main.snapshot = snapshot
    inner = backend_main.create_app(on_raw_sample=on_raw)
    app_ref["app"] = inner
    inner_lifespan = inner.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with inner_lifespan(application):
            if not LIVE:
                demo.catalogue = await asyncio.to_thread(_catalogue)
            task = asyncio.create_task(demo.start(scenario, seed))
            try:
                yield
            finally:
                task.cancel()
                service.stop()

    inner.router.lifespan_context = lifespan
    inner.state.demo = demo
    demo.app_state = inner.state
    inner.state.detector = service

    @inner.get("/demo/scenarios")
    async def scenarios():
        known = {s["id"] for s in demo.catalogue}
        menu = [{"group": g, "id": i, "label": label} for g, i, label in TEST_MENU if i in known]
        return {"preview": list(PREVIEW_SCENARIOS), "menu": menu, "scenarios": demo.catalogue}

    @inner.post("/demo/scenario")
    async def pick(scenario: str, seed: int = DEFAULT_SEED):
        if demo.live:
            raise HTTPException(status_code=409, detail="Live strip: there is no scenario to pick")
        if demo.phase == "building":
            raise HTTPException(status_code=409, detail="A scenario is still being built")
        if scenario not in {s["id"] for s in demo.catalogue}:
            raise HTTPException(status_code=404, detail=f"Unknown scenario {scenario}")
        asyncio.create_task(demo.start(scenario, seed))
        return {"building": scenario, "seed": seed}

    @inner.post("/incident/dismiss")
    async def dismiss():
        episode = demo.dismiss()
        if episode is None:
            raise HTTPException(status_code=409, detail="No fall alert to dismiss")
        return {"dismissed_episode": episode}

    return inner


def warm_up():
    """Run one synthetic collapse through a throwaway hub so the classifier's first call (about
    0.2 s) happens before the demo starts, not on the first real fall."""
    try:
        import importlib.util
        import pathlib

        path = pathlib.Path(__file__).parent / "tests" / "synth.py"
        spec = importlib.util.spec_from_file_location("hub_warm_synth", path)
        synth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(synth)
        from types import SimpleNamespace

        t, acc, gyro, _ = synth.stream(synth.topple_script(pre_s=8, after_s=12), noise=0.002)
        warm = DetectorService(start_thread=False)
        for k in range(len(t)):
            imus = [SimpleNamespace(sensor_id=i + 1, accel=tuple(acc[k, i]), gyro=tuple(gyro[k, i]), valid=True)
                    for i in range(4)]
            warm.on_raw_sample(SimpleNamespace(sample_time_ms=int(round(t[k] * 1000)) + 5000, imus=imus), 0.0)
            if k % 10 == 9:
                warm.step()
    except Exception:   # warm-up is an optimisation only
        pass


def _catalogue():
    from .demo_replay import ensure_profile, scenario_catalogue

    ensure_profile()
    return scenario_catalogue()


if os.environ.get("HUB_DEMO") == "1":   # opt-in so importing this module stays side-effect free
    app = make_demo_app()
