# File: server.py
# Purpose: Local web server for the 3D viewer: serves the static page and runs scenarios on
#          request (/api/run), with an in-memory and on-disk cache.
# Contributors: Unassigned
"""Standard library only (http.server). Binds to 127.0.0.1 by default.

GET /api/catalogue                       scenarios, profiles, models
GET /api/default_seed?scenario=H-LAD     the scenario's test-scene seed
GET /api/run?scenario=H-LAD&seed=0&model=baseline&profile=target&obs=60&dead=&physics=1&mocap=0&contact=0
    physics=1 (default) adds the render-only limb ragdoll and rest lift (physics.py);
    physics=0 gives the purely procedural pose. mocap=1 drives the limbs from the clips in
    motionlib/manifest.json where one applies (mocap.py); without the parameter the server's
    default is used: off, or on when started with THUNDERSIM_MOCAP=1. contact=1 runs the contact and
    balance pass (contact.py); without the parameter: contact.DEFAULT_ON, or THUNDERSIM_CONTACT. The
    sensor data is identical either way.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import traceback
import webbrowser
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..device import PROFILES
from ..scenarios import CATALOGUE
from . import contact as contact_lib
from . import mocap as mocap_lib
from .export import available_models, run_payload
from .scenes_index import default_seed

STATIC = Path(__file__).resolve().parent / "static"
CACHE_DIR = Path(__file__).resolve().parents[2] / "output" / "viz_cache"
GROUP_NAMES = {"height": "Falls from height", "level": "Same-level falls", "arrest": "Arrested falls",
               "collapse": "Collapse", "near_fall": "Near-falls", "negative": "Construction hard negatives"}


def code_version() -> str:
    """Hash of the simulator and viewer sources, the calibration and the mocap manifest (which
    names every clip with its sha256), so cached runs refresh when any of them changes."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha1()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.read_bytes())
    calib = root.parent / "calibration.json"
    if calib.exists():
        digest.update(calib.read_bytes())
    digest.update(b"mocap-manifest:" + mocap_lib.manifest_bytes())
    return digest.hexdigest()[:12]


class RunCache:
    def __init__(self, size: int = 16):
        self.size = size
        self.mem: OrderedDict[str, bytes] = OrderedDict()
        self.lock = threading.Lock()
        self.building: dict[str, threading.Event] = {}
        self.version = code_version()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, build) -> bytes:
        key = f"{self.version}-{key}"
        with self.lock:
            if key in self.mem:
                self.mem.move_to_end(key)
                return self.mem[key]
            waiter = self.building.get(key)
            if waiter is None:
                self.building[key] = threading.Event()
        if waiter is not None:  # someone else is building the same run
            waiter.wait()
            return self.get(key.split("-", 1)[1], build)
        try:
            path = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.json.gz"
            if path.exists():
                blob = path.read_bytes()
            else:
                blob = gzip.compress(json.dumps(build(), separators=(",", ":")).encode(), compresslevel=5)
                path.write_bytes(blob)
            with self.lock:
                self.mem[key] = blob
                while len(self.mem) > self.size:
                    self.mem.popitem(last=False)
            return blob
        finally:
            with self.lock:
                self.building.pop(key).set()


class Handler(SimpleHTTPRequestHandler):
    cache: RunCache = None
    models: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):  # quieter console: only API calls and errors
        if "/api/" in (args[0] if args else "") or (len(args) > 1 and str(args[1]) >= "400"):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/catalogue":
                return self.send_json(self.catalogue())
            if url.path == "/api/default_seed":
                return self.send_json({"seed": default_seed(query["scenario"])})
            if url.path == "/api/run":
                return self.send_run(query)
            self.send_error(404, "Unknown API path")
        except KeyError as error:
            self.send_json({"error": f"missing or unknown parameter {error}"}, status=400)
        except Exception as error:  # report simulator errors to the page instead of dropping the connection
            traceback.print_exc()
            self.send_json({"error": repr(error)}, status=500)

    def catalogue(self) -> dict:
        return {
            "groups": GROUP_NAMES,
            "scenarios": [{"id": s.id, "group": s.group, "expected": s.expected, "description": s.description,
                           "report_only": s.report_only} for s in CATALOGUE.values()],
            "profiles": [{"name": p.name, "description": p.description, "odr_hz": p.odr_hz} for p in PROFILES.values()],
            "models": self.models,
        }

    def send_run(self, q: dict) -> None:
        scenario = q["scenario"]
        if scenario not in CATALOGUE:
            raise KeyError("scenario")
        seed = default_seed(scenario) if q.get("seed", "auto") in ("", "auto") else int(q["seed"])
        model = q.get("model", "baseline")
        if model not in self.models:
            raise KeyError("model")
        profile = q.get("profile", "target")
        if profile not in PROFILES:
            raise KeyError("profile")
        obs = float(q.get("obs", 60))
        dead = None if q.get("dead", "") in ("", "none") else int(q["dead"])
        if q.get("physics", "1") not in ("0", "1"):
            raise KeyError("physics")
        physics = q.get("physics", "1") == "1"
        if q.get("mocap", "") not in ("", "0", "1"):
            raise KeyError("mocap")
        mocap = mocap_lib.requested(None if q.get("mocap", "") == "" else q["mocap"] == "1")
        if q.get("contact", "") not in ("", "0", "1"):
            raise KeyError("contact")
        contact = contact_lib.requested(None if q.get("contact", "") == "" else q["contact"] == "1")
        key = (f"{scenario}|{seed}|{model}|{profile}|{obs}|{dead}|physics={int(physics)}|mocap={int(mocap)}"
               f"|contact={int(contact)}")
        blob = self.cache.get(key, lambda: run_payload(scenario, seed, model, profile, obs, dead, physics=physics,
                                                       mocap=mocap, contact=contact))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    Handler.cache = RunCache()
    Handler.models = available_models()
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"ThunderSim 3D viewer at {url}  (models: {', '.join(Handler.models)}; Ctrl+C to stop)", flush=True)
    print("Simulated data only: limbs and props are illustrative; spine, sensors and events come from the simulator.", flush=True)
    if mocap_lib.requested():
        print("Mocap limbs on by default (THUNDERSIM_MOCAP=1); ?mocap=0 turns them off per run.", flush=True)
    if contact_lib.requested():
        print("Contact pass on by default (THUNDERSIM_CONTACT=1 or contact.DEFAULT_ON); ?contact=0 turns it off per run.",
              flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
