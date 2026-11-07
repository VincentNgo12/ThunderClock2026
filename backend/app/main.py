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
from .models import Sample, STREAMING_STATES
from .posture import Posture
from .simulated import SimulatedStream


def create_app(*, on_raw_sample: Callable[[Sample], None] | None = None):
    """Optionally hand each decoded raw frame to a quick, synchronous consumer."""
    @asynccontextmanager
    async def lifespan(app):
        posture = Posture(
            warning_deg=float(os.getenv("POSTURE_WARNING_DEG", "20")),
            warning_seconds=float(os.getenv("POSTURE_WARNING_SECONDS", "2")),
        )
        source = os.getenv("TELEMETRY_SOURCE", "hardware")
        if source == "simulated":
            stream = SimulatedStream()
        elif source == "hardware":
            device_id = os.getenv("DEVICE_ID")
            packet_format = os.getenv("DEVICE_FORMAT")
            if not device_id or packet_format not in DEVICE_FORMATS:
                raise ValueError("Set DEVICE_ID and DEVICE_FORMAT=raw, quaternion or both for hardware mode")
            stream = DeviceStream(select_device(await scan(), device_id), packet_format)
        else:
            raise ValueError("Set TELEMETRY_SOURCE=hardware or simulated")
        app.state.source = source
        app.state.stream = stream
        app.state.posture = posture
        app.state.latest = json.dumps(snapshot(stream, 0, source=source, posture=posture))

        async def acquire():
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
                        continue
                    if stream.stats.resets != resets:
                        # A restarted wearable may have established a new orientation frame.
                        posture.reset()
                        resets = stream.stats.resets
                    posture.update(sample, stream.received_monotonic)
            except Exception as error:
                # Keep HTTP diagnostics available after a device/setup failure; never substitute a source.
                stream.stats.state = "error"
                stream.stats.last_error = f"{type(error).__name__}: {error}"
                logging.getLogger(__name__).error("Device stream stopped: %s", stream.stats.last_error)

        async def publish():
            sequence = 0
            while True:
                state = snapshot(stream, sequence, source=source, posture=posture)
                app.state.latest = json.dumps(state, allow_nan=False)
                sequence += 1
                await asyncio.sleep(0.05)

        tasks = [asyncio.create_task(acquire()), asyncio.create_task(publish())]
        try:
            yield
        finally:
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

    @app.post("/posture/calibrate")
    async def calibrate():
        stream = app.state.stream
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
                await asyncio.sleep(0.05)
        except TimeoutError:
            with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
                await asyncio.wait_for(websocket.close(code=1013), 1)
        except (WebSocketDisconnect, RuntimeError, OSError):
            return

    return app


app = create_app()
