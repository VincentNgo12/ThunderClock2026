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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .ble import DEVICE_FORMATS, DeviceStream, scan, select_device
from .dashboard import snapshot
from .fusion import RawFusion
from .models import Diagnostics, Sample, STREAMING_STATES
from .posture import Posture
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
                if stream.stats.resets != resets:
                    # A restarted wearable or estimator may establish a new orientation frame.
                    posture.reset()
                    resets = stream.stats.resets
                posture.update(sample, pose_stream.received_monotonic)
        except Exception as error:
            # Keep HTTP diagnostics available after a device/setup failure; never substitute a source.
            stream.stats.state = "error"
            stream.stats.last_error = f"{type(error).__name__}: {error}"
            logging.getLogger(__name__).error("Device stream stopped: %s", stream.stats.last_error)

    @asynccontextmanager
    async def lifespan(app):
        app.state.posture = Posture(
            warning_deg=float(os.getenv("POSTURE_WARNING_DEG", "20")),
            warning_seconds=float(os.getenv("POSTURE_WARNING_SECONDS", "2")),
        )
        app.state.acquire_task = None
        app.state.connecting = False
        source = os.getenv("TELEMETRY_SOURCE", "hardware")
        app.state.source = source
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
        else:
            raise ValueError("Set TELEMETRY_SOURCE=hardware, manual or simulated")
        app.state.latest = json.dumps(snapshot(app.state.pose_stream, 0, source=source, posture=app.state.posture))

        async def publish():
            sequence = 1  # The initial snapshot already uses sequence zero.
            while True:
                state = snapshot(app.state.pose_stream, sequence, source=app.state.source, posture=app.state.posture)
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
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
