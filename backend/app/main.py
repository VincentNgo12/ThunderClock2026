# File: main.py
# Purpose: Run one telemetry source, own neutral capture and publish shared dashboard state.
# Contributors: Unassigned

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .ble import DEVICE_FORMATS, DeviceStream, scan, select_device
from .dashboard import snapshot
from .fusion import RawFusion
from .models import Diagnostics, REGIONS, Sample, STREAMING_STATES
from .mounting import Calibrator, identity_model, load_model, mounting_path, rezero, save_model
from .posture import Posture
from .recording import Recorder
from .segments import SegmentModel
from .trunk_posture import TrunkPosture
from .simulated import SimulatedStream


# The team's strip. DEVICE_ID overrides it; POST /device/connect?device_id= overrides both.
DEFAULT_DEVICE_ID = "4496845D-954E-138F-09F8-D841CF3F5B37"
ACTIVE_STATES = ("connecting", "connected", "subscribed")
# Dashboard snapshot and WebSocket rate. 100 Hz matches the IMU sampling; the view only gets new
# orientations as fast as the source makes them (strip quaternions 20 Hz, raw_fusion 100 Hz).
TELEMETRY_RATE_HZ = float(os.getenv("TELEMETRY_RATE_HZ", "100"))
if not 1 <= TELEMETRY_RATE_HZ <= 200:
    raise ValueError("TELEMETRY_RATE_HZ must be between 1 and 200")
PUBLISH_PERIOD_S = 1 / TELEMETRY_RATE_HZ


class IdleStream:
    """Placeholder source before the dashboard asks to connect (TELEMETRY_SOURCE=manual)."""

    packet_format = None

    def __init__(self):
        self.stats = Diagnostics()
        self.latest = None
        self.received_monotonic = None
        self.received_at = None


def create_app(*, on_raw_sample: Callable[[Sample], None] | None = None):
    """Optionally hand each decoded raw frame to a quick, synchronous consumer."""

    def start_stream(app, stream, fusion):
        """Make `stream` the live source and start reading it."""
        app.state.stream = stream
        app.state.pose_stream = stream if fusion is None else fusion
        app.state.posture.reset()
        app.state.trunk.reset()
        app.state.segments = None
        # A (re)connected strip may sit slightly differently: re-zero upright on the first still stand.
        app.state.calibrator.auto_zero_pending = app.state.segment_model is not None and app.state.source in ("hardware", "manual")
        app.state.acquire_task = asyncio.create_task(acquire(app, stream, fusion))

    async def open_device(device_id, packet_format):
        if packet_format not in (*DEVICE_FORMATS, "raw_fusion"):
            raise ValueError("DEVICE_FORMAT must be raw, quaternion, both or raw_fusion")
        stream = DeviceStream(select_device(await scan(), device_id),
                              "raw" if packet_format == "raw_fusion" else packet_format)
        fusion = RawFusion(stream.stats, os.getenv("RAW_IMU_AXES")) if packet_format == "raw_fusion" else None
        return stream, fusion

    async def acquire(app, stream, fusion):
        posture = app.state.posture
        pose_stream = stream if fusion is None else fusion
        resets = stream.stats.resets
        raw_consumer = on_raw_sample
        try:
            async for sample in stream.updates():
                if sample is None:
                    continue
                recorder = app.state.recorder
                if recorder is not None:
                    recorder.add(sample)                      # as the device sent it (raw or quaternion)
                if sample.format == "raw":
                    if raw_consumer is not None:
                        try:
                            raw_consumer(sample)
                        except Exception:
                            logging.getLogger(__name__).exception("Raw IMU consumer disabled for this session")
                            raw_consumer = None
                    if fusion is None or not fusion.update(sample, stream.received_monotonic, stream.received_at):
                        continue
                    sample = fusion.latest
                    if recorder is not None:
                        recorder.add(sample, origin="laptop_fusion")
                if stream.stats.resets != resets:
                    # A restarted wearable or estimator may establish a new orientation frame.
                    posture.reset()
                    resets = stream.stats.resets
                posture.update(sample, pose_stream.received_monotonic)
                update_segments(app, sample)
        except Exception as error:
            # Keep HTTP diagnostics available after a device/setup failure; never substitute a source.
            stream.stats.state = "error"
            stream.stats.last_error = f"{type(error).__name__}: {error}"
            logging.getLogger(__name__).error("Device stream stopped: %s", stream.stats.last_error)

    def update_segments(app, sample):
        """Per quaternion sample: calibration capture / auto re-zero, segment poses, trunk posture."""
        if sample.format != "quaternion":
            return
        now = time.monotonic()
        rezeroed = app.state.calibrator.watch(sample, app.state.segment_model, now)
        if rezeroed is not None:
            app.state.segment_model = rezeroed
            logging.getLogger(__name__).info("Mounting re-zeroed upright after (re)connect")
        model = app.state.segment_model
        if model is None:
            return
        app.state.segments = model.compute(sample, REGIONS)
        app.state.segments_at = now
        app.state.trunk.update(app.state.segments, now)

    def add_segments(app, state):
        """Heading-free segment poses, trunk posture and replay position for the dashboard."""
        model = app.state.segment_model
        calibrator = app.state.calibrator
        state["mounting"] = {"calibrated": model is not None, "source": None if model is None else model.source,
                             "regions": [] if model is None else sorted(model.mountings),
                             "auto_zero_pending": calibrator.auto_zero_pending,
                             "capturing": calibrator.active is not None, "captured_poses": sorted(calibrator.captures)}
        if model is not None:
            fresh = app.state.segments_at is not None and time.monotonic() - app.state.segments_at <= 0.5
            state["segments"] = app.state.segments if fresh else None
            state["segment_model"] = {"source": model.source, "regions": sorted(model.mountings)}
            state["trunk_posture"] = app.state.trunk.state if fresh else None
            # The joint posture compares drifting sensor headings; with a mounting, alerts come from the trunk.
            state["alerts"] = app.state.trunk.alerts if fresh else []
            state["posture_source"] = "gravity"
        if hasattr(app.state.stream, "recording"):
            state["replay"] = app.state.stream.status()

    @asynccontextmanager
    async def lifespan(app):
        app.state.posture = Posture(
            warning_deg=float(os.getenv("POSTURE_WARNING_DEG", "20")),
            warning_seconds=float(os.getenv("POSTURE_WARNING_SECONDS", "2")),
        )
        app.state.acquire_task = None
        app.state.connecting = False
        app.state.recorder = None
        app.state.calibrator = Calibrator()
        app.state.before_replay = None
        app.state.trunk = TrunkPosture(float(os.getenv("POSTURE_WARNING_SECONDS", "2")))
        app.state.segments = None
        app.state.segments_at = None
        source = os.getenv("TELEMETRY_SOURCE", "hardware")
        app.state.source = source
        # Heading-free segment model (segments.py) needs each sensor's mounting. In order: MOUNTING_FILE, the
        # mounting saved for this strip by Calibrate... (mountings/<device>.json), body-aligned for simulated
        # data, or (replay) fitted from the recording itself.
        app.state.device_key = os.getenv("DEVICE_ID", DEFAULT_DEVICE_ID) if source in ("hardware", "manual") else source
        app.state.segment_model = None
        if os.getenv("MOUNTING_FILE"):
            with open(os.environ["MOUNTING_FILE"], encoding="utf-8") as handle:
                app.state.segment_model = SegmentModel.from_json(json.load(handle))
        elif source in ("hardware", "manual"):
            app.state.segment_model = load_model(app.state.device_key)
        elif source == "simulated":
            app.state.segment_model = identity_model("simulated: body-aligned sensors")
        if source == "simulated":
            start_stream(app, SimulatedStream(), None)
        elif source == "hardware":
            device_id = os.getenv("DEVICE_ID", DEFAULT_DEVICE_ID)
            packet_format = os.getenv("DEVICE_FORMAT")
            if packet_format not in (*DEVICE_FORMATS, "raw_fusion"):
                raise ValueError("Set DEVICE_FORMAT=raw, quaternion, both or raw_fusion for hardware mode")
            start_stream(app, *await open_device(device_id, packet_format))
        elif source == "manual":
            # Wait for POST /device/connect from the dashboard.
            app.state.stream = app.state.pose_stream = IdleStream()
        elif source == "replay":
            # Play a "Collect data" recording back; its own stand + bend holds give the sensor mountings.
            from .recording_io import fit_model, hold_up_vectors
            from .replay import ReplayStream

            path = os.getenv("REPLAY_FILE")
            if not path:
                raise ValueError("Set REPLAY_FILE=<recording.jsonl> for TELEMETRY_SOURCE=replay")
            replay = ReplayStream(path, float(os.getenv("REPLAY_SPEED", "1")), os.getenv("REPLAY_LOOP", "1") == "1",
                                  float(os.getenv("REPLAY_START_DELAY_S", "0")))
            if os.getenv("REPLAY_FIT", "1") == "1":
                app.state.segment_model = fit_model(hold_up_vectors(replay.recording), source=f"fitted from {path}")
            start_stream(app, replay, None)
        else:
            raise ValueError("Set TELEMETRY_SOURCE=hardware, manual, simulated or replay")
        app.state.latest = json.dumps(snapshot(app.state.pose_stream, 0, source=source, posture=app.state.posture))

        async def publish():
            sequence = 1  # The initial snapshot already uses sequence zero.
            while True:
                state = snapshot(app.state.pose_stream, sequence, source=app.state.source, posture=app.state.posture)
                add_segments(app, state)
                app.state.latest = json.dumps(state, allow_nan=False)
                sequence += 1
                await asyncio.sleep(PUBLISH_PERIOD_S)

        publisher = asyncio.create_task(publish())
        try:
            yield
        finally:
            tasks = [publisher] + ([app.state.acquire_task] if app.state.acquire_task else [])
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="ThunderClock telemetry diagnostics", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                       *filter(None, os.getenv("CORS_ORIGINS", "").split(","))],
        allow_methods=["GET", "POST"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "source": app.state.source}

    @app.get("/diagnostics")
    async def diagnostics():
        stream = app.state.stream
        result = stream.stats.snapshot()
        if app.state.source == "hardware" and stream.packet_format == "both":
            result["raw"] = stream.stats_by_format["raw"].snapshot()
            result["raw"]["state"] = result["state"]
        return result

    @app.post("/device/connect")
    async def connect(device_format: str | None = None, device_id: str | None = None):
        """Scan for the strip and start streaming (query parameters only: no CORS preflight)."""
        if app.state.source == "simulated":
            raise HTTPException(status_code=409, detail="Simulated source: restart with TELEMETRY_SOURCE=manual")
        if app.state.connecting or app.state.stream.stats.state in ACTIVE_STATES:
            raise HTTPException(status_code=409, detail="Already connecting or connected")
        device_id = device_id or os.getenv("DEVICE_ID", DEFAULT_DEVICE_ID)
        device_format = device_format or os.getenv("DEVICE_FORMAT", "quaternion")
        app.state.connecting = True
        try:
            await stop_acquire()
            stream, fusion = await open_device(device_id, device_format)
        except ValueError as error:   # device not found in the scan, or a bad format
            raise HTTPException(status_code=404, detail=f"{error} (looked for {device_id})") from error
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail="BLE scan timed out: is Bluetooth on, and does this "
                                "terminal have Bluetooth permission (System Settings > Privacy)?") from error
        except Exception as error:   # Bluetooth off or unavailable (BleakError, OSError, ...)
            raise HTTPException(status_code=503, detail=f"BLE unavailable: {type(error).__name__}: {error}") from error
        finally:
            app.state.connecting = False
        app.state.source = "hardware"
        app.state.before_replay = None
        start_stream(app, stream, fusion)
        return {"device_id": device_id, "format": device_format}

    @app.post("/device/disconnect")
    async def disconnect():
        if app.state.source == "simulated":
            raise HTTPException(status_code=409, detail="Simulated source")
        await stop_acquire()
        return {"state": app.state.stream.stats.state}

    async def stop_acquire():
        task = app.state.acquire_task
        app.state.acquire_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)   # DeviceStream disconnects in its finally

    # ---- replay a recorded session from the dashboard (Replay panel), see replay.py
    @app.get("/replay/recordings")
    async def replay_recordings():
        from .replay import list_recordings, recordings_dir

        return {"folder": str(recordings_dir()), "recordings": await asyncio.to_thread(list_recordings),
                "active": app.state.stream.status() if hasattr(app.state.stream, "recording") else None}

    @app.post("/replay/start")
    async def replay_start(file: str, speed: float = 1.0, loop: bool = False, start_t: float = 0.0,
                           model: str = "recording"):
        """Replace the live source with a recording. model=recording fits the mounting from the session's own
        stand + bend steps (faithful to how the vest sat then); model=current keeps the strip's mounting."""
        from .recording_io import fit_model, hold_up_vectors
        from .replay import ReplayStream, resolve_recording

        if app.state.recorder is not None:
            raise HTTPException(status_code=409, detail="Stop the data collection first")
        if not 0.1 <= speed <= 10:
            raise HTTPException(status_code=400, detail="speed must be between 0.1 and 10")
        try:
            path = resolve_recording(file)
            replay = await asyncio.to_thread(ReplayStream, path, speed, loop, 0.0, start_t)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        segment_model = app.state.segment_model
        if model == "recording":
            try:
                segment_model = await asyncio.to_thread(
                    lambda: fit_model(hold_up_vectors(replay.recording), source=f"fitted from {path.name}"))
            except ValueError as error:          # no stand / bend steps: fall back to the current mounting
                if segment_model is None:
                    raise HTTPException(status_code=409, detail=f"Cannot fit a mounting from {path.name}: {error}") from error
        await stop_acquire()
        if app.state.before_replay is None:      # remember the live setup once, to restore on stop
            app.state.before_replay = {"source": app.state.source, "model": app.state.segment_model}
        app.state.source = "replay"
        app.state.segment_model = segment_model
        start_stream(app, replay, None)
        return replay.status()

    @app.post("/replay/pause")
    async def replay_pause(paused: bool = True):
        stream = app.state.stream
        if not hasattr(stream, "recording"):
            raise HTTPException(status_code=409, detail="No replay is running")
        stream.paused = paused
        return stream.status()

    @app.post("/replay/stop")
    async def replay_stop():
        if not hasattr(app.state.stream, "recording"):
            raise HTTPException(status_code=409, detail="No replay is running")
        await stop_acquire()
        before = app.state.before_replay or {"source": "manual", "model": None}
        app.state.before_replay = None
        app.state.segment_model = before["model"]
        app.state.segments = None
        app.state.trunk.reset()
        if before["source"] == "simulated":
            app.state.source = "simulated"
            start_stream(app, SimulatedStream(), None)
        else:                                    # a live strip was released by the replay: reconnect from the UI
            app.state.source = "manual" if before["source"] in ("hardware", "manual") else before["source"]
            app.state.stream = app.state.pose_stream = IdleStream()
        return {"source": app.state.source}

    # ---- labelled data collection ("Collect data" in the dashboard). Query parameters and a text/plain
    # body only, so the browser sends no CORS preflight.
    @app.post("/recording/start")
    async def recording_start(request: Request, name: str = "session"):
        if app.state.recorder is not None:
            raise HTTPException(status_code=409, detail="A recording is already running")
        if app.state.stream.stats.state not in STREAMING_STATES:
            raise HTTPException(status_code=409, detail="Connect the strip (or a simulated source) first")
        client = {}
        body = (await request.body()).decode("utf-8", "replace").strip()
        if body:
            try:
                client = json.loads(body)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Client metadata must be JSON") from error
        stream = app.state.stream
        meta = {
            "source": app.state.source,
            "device_format": getattr(stream, "packet_format", None),
            "device_id": os.getenv("DEVICE_ID", DEFAULT_DEVICE_ID) if app.state.source == "hardware" else None,
            "raw_imu_axes": os.getenv("RAW_IMU_AXES"),
            "telemetry_rate_hz": TELEMETRY_RATE_HZ,
            "regions": {"1": "pelvis", "2": "lumbar", "3": "thoracic", "4": "upper_thoracic"},
            "frame": "quaternion w,x,y,z segment-local -> sensor world (+Y up); raw accel m/s^2, gyro rad/s, board axes",
            "backend_posture_calibration": app.state.posture.calibration,
            "client": client,
        }
        app.state.recorder = Recorder(name, meta)
        return app.state.recorder.status()

    @app.post("/recording/label")
    async def recording_label(name: str):
        recorder = app.state.recorder
        if recorder is None:
            raise HTTPException(status_code=409, detail="No recording is running")
        recorder.set_label(name[:80])
        return recorder.status()

    @app.post("/recording/stop")
    async def recording_stop(aborted: bool = False):
        recorder = app.state.recorder
        if recorder is None:
            raise HTTPException(status_code=409, detail="No recording is running")
        app.state.recorder = None
        return await asyncio.to_thread(recorder.save, aborted)

    @app.get("/recording/status")
    async def recording_status():
        recorder = app.state.recorder
        return recorder.status() if recorder is not None else {"active": False}

    # ---- sensor mounting calibration (Calibrate... in the dashboard), see mounting.py
    @app.get("/mounting")
    async def mounting_status():
        model = app.state.segment_model
        return {"calibrated": model is not None, "device": app.state.device_key,
                "saved_path": str(mounting_path(app.state.device_key)),
                "model": None if model is None else model.to_json(),
                "captured_poses": sorted(app.state.calibrator.captures)}

    @app.post("/mounting/capture")
    async def mounting_capture(pose: str, seconds: float = 2.0):
        if app.state.stream.stats.state not in STREAMING_STATES:
            raise HTTPException(status_code=409, detail="Connect the strip first")
        if pose == "stand":
            app.state.calibrator.reset()            # a new calibration starts with the stand
        try:
            return await app.state.calibrator.capture(pose, min(max(seconds, 0.5), 10.0))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/mounting/fit")
    async def mounting_fit(save: bool = True):
        try:
            model = app.state.calibrator.fit(f"calibrated {time.strftime('%Y-%m-%d %H:%M')} ({app.state.device_key})")
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        app.state.segment_model = model
        app.state.calibrator.auto_zero_pending = False
        app.state.trunk.reset()
        saved = save_model(model, app.state.device_key) if save and app.state.source in ("hardware", "manual") else None
        return {"report": model.report, "regions": sorted(model.mountings), "saved_path": saved}

    @app.post("/mounting/zero")
    async def mounting_zero(seconds: float = 2.0):
        """Stand still: keep each sensor's forward direction, take a new upright."""
        model = app.state.segment_model
        if model is None:
            raise HTTPException(status_code=409, detail="No mounting yet: run Calibrate first")
        if app.state.stream.stats.state not in STREAMING_STATES:
            raise HTTPException(status_code=409, detail="Connect the strip first")
        calibrator = app.state.calibrator
        try:
            result = await calibrator.capture("stand", min(max(seconds, 0.5), 10.0))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        stand = {region: info["up"] for region, info in calibrator.captures.pop("stand").items()}
        app.state.segment_model = rezero(model, stand, model.source)
        calibrator.auto_zero_pending = False
        app.state.trunk.reset()
        saved = save_model(app.state.segment_model, app.state.device_key) if app.state.source in ("hardware", "manual") else None
        return {"sensors": result["sensors"], "saved_path": saved}

    @app.post("/posture/calibrate")
    async def calibrate():
        stream = app.state.pose_stream
        if stream.stats.state not in STREAMING_STATES:
            raise HTTPException(status_code=409, detail="Neutral capture requires an active telemetry source")
        posture = app.state.posture
        try:
            posture.capture(stream.latest, stream.received_monotonic, stream.received_at, time.monotonic())
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return posture.calibration

    @app.websocket("/ws/telemetry")
    async def telemetry(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # No per-client history: one in-flight latest snapshot, with a bounded send.
                await asyncio.wait_for(websocket.send_text(app.state.latest), 1)
                await asyncio.sleep(PUBLISH_PERIOD_S)
        except TimeoutError:
            with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
                await asyncio.wait_for(websocket.close(code=1013), 1)
        except (WebSocketDisconnect, RuntimeError, OSError):
            return

    return app


app = create_app()
