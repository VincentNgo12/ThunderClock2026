# File: main.py
# Purpose: Optional HTTP/WebSocket adapter for the same diagnostic core; never imported by the CLI.
# Contributors: Unassigned

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .ble import DeviceStream, scan, select_device
from .dashboard import snapshot


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        device_id = os.getenv("DEVICE_ID")
        packet_format = os.getenv("DEVICE_FORMAT")
        if not device_id or packet_format not in ("raw", "quaternion"):
            raise ValueError("Set DEVICE_ID and explicit DEVICE_FORMAT=raw or quaternion; do not run CLI streaming simultaneously")
        device = select_device(await scan(), device_id)
        stream = DeviceStream(device, packet_format)
        app.state.stream = stream
        app.state.latest = json.dumps(snapshot(stream, 0))

        async def acquire():
            try:
                async for _sample in stream.updates():
                    pass  # Core retains the latest sample; this adapter performs no sensor interpretation.
            except Exception as error:
                # Keep HTTP diagnostics available after a device/setup failure; never substitute a source.
                stream.stats.state = "error"
                stream.stats.last_error = f"{type(error).__name__}: {error}"
                logging.getLogger(__name__).error("Device stream stopped: %s", stream.stats.last_error)

        async def publish():
            sequence = 0
            while True:
                app.state.latest = json.dumps(snapshot(stream, sequence), allow_nan=False)
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
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "source": "hardware"}

    @app.get("/diagnostics")
    async def diagnostics():
        return app.state.stream.stats.snapshot()

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
