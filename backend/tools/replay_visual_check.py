# File: replay_visual_check.py
# Purpose: End-to-end check of what the spine view DRAWS for a real recording: replays a "Collect data"
#          session through the backend (TELEMETRY_SOURCE=replay) into the real frontend in headless Chrome,
#          and at every labelled hold takes a screenshot and reads the drawn segment directions from the page
#          (window.__spineDebug). Fails if any segment is drawn bending the wrong way.
# Contributors: Unassigned
"""Run from backend/ (needs the API packages + websockets, Node for the frontend, and Google Chrome):

    python tools/replay_visual_check.py recordings/<session>.jsonl --out /tmp/spine_check [--speed 2]

Outputs in --out: one PNG per hold, contact_sheet.png (needs matplotlib) and report.json.
By default it starts its own backend (port 8795) and frontend (port 5198); pass --frontend URL to use a
running dev server (its origin is allowed through CORS_ORIGINS).
"""
import argparse
import asyncio
import base64
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import websockets

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
CHROME = os.getenv("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
EXPECTED = {"forward": ("flex", 1), "backward": ("flex", -1), "left": ("lat", -1), "right": ("lat", 1),
            "lie_back": ("flex", -1), "lie_front": ("flex", 1), "lie_left": ("lat", -1), "lie_right": ("lat", 1)}
WRONG_IF_BEYOND_DEG = 10.0
SETTLE_S = 1.3            # wall-clock seconds into a hold before capturing (smoothing + re-centring)


def wait_http(url, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def angles(up):
    x, y, z = up
    return math.degrees(math.atan2(-z, y)), math.degrees(math.atan2(x, y))


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expression):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result.get("result", {}).get("value")

    async def click(self, label):
        await self.js(f"[...document.querySelectorAll('button')].find(b => b.textContent.trim() === {json.dumps(label)})?.click()")

    async def screenshot(self, path):
        data = await self.call("Page.captureScreenshot", format="png")
        pathlib.Path(path).write_bytes(base64.b64decode(data["data"]))


POSTURE_ZONES = {"forward": {"moderate", "high"}, "backward": {"moderate"}, "left": {"moderate"}, "right": {"moderate"},
                 "lie_back": {"lying"}, "lie_front": {"lying"}, "lie_left": {"lying"}, "lie_right": {"lying"},
                 "stand": {"neutral"}, "stand_end": {"neutral"}}
CALIBRATION_HOLDS = {"stand#1:hold": "stand", "forward#1:hold": "forward", "backward#1:hold": "backward",
                     "right#1:hold": "right", "left#1:hold": "left"}


def api_post(base, path):
    request = urllib.request.Request(base + path, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path}: {error.code} {error.read().decode()}") from error


async def start_from_panel(page, recording, speed):
    """Start the replay the way a user does: Replay -> pick the session -> speed -> Play."""
    await page.click("Replay")
    await asyncio.sleep(1.5)
    session = json.dumps(pathlib.Path(recording).stem)
    picked = await page.js(f"""(() => {{
        const item = [...document.querySelectorAll('.replay-item')].find(i => i.textContent.includes({session}));
        if (!item) return false;
        item.querySelector('input').click();
        const select = document.querySelector('.replay-options select');
        Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(select, '{speed:g}');
        select.dispatchEvent(new Event('change', {{ bubbles: true }}));
        return true; }})()""")
    if not picked:
        raise RuntimeError(f"recording {session} is not listed in the Replay panel")
    await asyncio.sleep(0.3)
    await page.click("Play")
    await asyncio.sleep(1.0)
    await page.js("document.querySelector('.replay-panel .calib-close')?.click()")


async def run(args, frontend_url, backend_port, out):
    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.debug_port}/json"))
    page_ws = next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
    api = f"http://127.0.0.1:{backend_port}"
    results, calibration = [], {}
    async with websockets.connect(page_ws, max_size=None) as ws:
        page = Page(ws)
        await page.call("Page.navigate", url=f"{frontend_url}/?ws=ws://127.0.0.1:{backend_port}/ws/telemetry")
        await asyncio.sleep(4)
        if args.mode == "panel":
            await start_from_panel(page, args.recording, args.speed)
        done, seen_end = set(), False
        deadline = time.time() + args.timeout
        hold_started = {}
        while time.time() < deadline and not seen_end:
            replay = await page.js("document.querySelector('.pill.replay')?.textContent ?? null")
            label = replay.split("·")[-1].strip() if replay else None
            if label and label.startswith("stand_end") and label.endswith(":hold"):
                seen_end = label in done
            if label and label.endswith(":hold") and label not in done:
                hold_started.setdefault(label, time.time())
                if args.calibrate and label in CALIBRATION_HOLDS:
                    # Calibrate through the API during the recording's own stand + round-1 bends.
                    if time.time() - hold_started[label] >= 0.5:
                        pose = CALIBRATION_HOLDS[label]
                        result = await asyncio.to_thread(api_post, api, f"/mounting/capture?pose={pose}&seconds=1.2")
                        calibration[pose] = result["sensors"]
                        if pose == "left":
                            calibration["fit"] = await asyncio.to_thread(api_post, api, "/mounting/fit")
                        print(f"{label:20s} captured for calibration", flush=True)
                        done.add(label)
                elif time.time() - hold_started[label] >= SETTLE_S:
                    movement = label.split("#")[0]
                    lateral = movement in ("left", "right", "lie_left", "lie_right")   # these tilt toward a side camera
                    await page.click("Back" if lateral else "Side")
                    await asyncio.sleep(0.4)
                    debug = await page.js("JSON.stringify(window.__spineDebug ?? null)")
                    zone = await page.js("(document.querySelector('.posture-panel')?.className.match(/zone-(\\w+)/) || [])[1] ?? null")
                    posture_text = await page.js("document.querySelector('.posture-panel')?.innerText ?? null")
                    shot = out / f"{len(done):02d}_{label.replace(':', '_').replace('#', '')}.png"
                    await page.screenshot(shot)
                    anchors = json.loads(debug)["anchors"] if debug and debug != "null" else []
                    segs = {a["region"]: angles(a["up"]) for a in anchors}
                    wrong = None
                    if movement in EXPECTED:
                        kind, sign = EXPECTED[movement]
                        wrong = [r for r, (f, l) in segs.items()
                                 if (f if kind == "flex" else l) * sign < 0 and abs(f if kind == "flex" else l) > WRONG_IF_BEYOND_DEG]
                    posture_ok = None if movement not in POSTURE_ZONES else zone in POSTURE_ZONES[movement]
                    results.append({"hold": label, "view": "back" if lateral else "side", "screenshot": shot.name,
                                    "segments": {r: {"flexion_deg": round(f, 1), "lateral_deg": round(l, 1)} for r, (f, l) in segs.items()},
                                    "wrong": wrong, "posture_zone": zone, "posture_ok": posture_ok,
                                    "posture_text": posture_text})
                    flag = "-" if wrong is None else ("OK" if not wrong else f"WRONG {wrong}")
                    pflag = "" if posture_ok is None else ("" if posture_ok else "  POSTURE?")
                    cells = "  ".join(f"{r[:5]} {f:+4.0f}/{l:+4.0f}" for r, (f, l) in segs.items())
                    print(f"{label:20s} {cells}   {flag}  posture {zone}{pflag}", flush=True)
                    done.add(label)
            await asyncio.sleep(0.15)
        model = await page.js("document.querySelector('.pill.model')?.title ?? null")
    return results, model, calibration


def contact_sheet(out, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping contact_sheet.png")
        return
    cols = 5
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, results):
        img = mpimg.imread(out / r["screenshot"])
        h, w = img.shape[:2]
        ax.imshow(img[int(h * 0.12):int(h * 0.92), int(w * 0.22):int(w * 0.78)])
        status = "" if r["wrong"] is None else ("  ✓" if not r["wrong"] else "  ✗ " + ",".join(r["wrong"]))
        ax.set_title(f"{r['hold'].replace(':hold', '')} ({r['view']}){status}", fontsize=8,
                     color="#b91c1c" if r["wrong"] else "#111")
    fig.tight_layout()
    fig.savefig(out / "contact_sheet.png", dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording")
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--frontend", help="URL of a running frontend dev server (default: start one on :5198)")
    parser.add_argument("--backend-port", type=int, default=8795)
    parser.add_argument("--debug-port", type=int, default=9335)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--mode", choices=("env", "panel"), default="env",
                        help="env: backend starts the replay (TELEMETRY_SOURCE=replay); panel: click Replay -> Play in the UI")
    parser.add_argument("--calibrate", action="store_true",
                        help="env mode: calibrate through /mounting/capture + /mounting/fit during the recording's "
                             "stand and round-1 bends instead of fitting from the file; checks the rest")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recording = str(pathlib.Path(args.recording).resolve())
    procs = []
    try:
        frontend_url = args.frontend
        if not frontend_url:
            frontend_url = "http://localhost:5198"
            procs.append(subprocess.Popen(["npx", "vite", "--port", "5198", "--strictPort"], cwd=FRONTEND_DIR,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        origin = frontend_url.rstrip("/")
        if args.calibrate and args.mode != "env":
            raise SystemExit("--calibrate needs --mode env")
        cors = f"{origin},{origin.replace('localhost', '127.0.0.1')}"
        scratch = out / "mountings"
        if args.mode == "env":
            env = {**os.environ, "TELEMETRY_SOURCE": "replay", "REPLAY_FILE": recording, "REPLAY_SPEED": str(args.speed),
                   "REPLAY_LOOP": "0", "REPLAY_START_DELAY_S": "12", "REPLAY_FIT": "0" if args.calibrate else "1",
                   "CORS_ORIGINS": cors, "MOUNTINGS_DIR": str(scratch)}
        else:
            env = {**os.environ, "TELEMETRY_SOURCE": "manual", "RECORDINGS_DIR": str(pathlib.Path(recording).parent),
                   "CORS_ORIGINS": cors, "MOUNTINGS_DIR": str(scratch), "DEVICE_ID": "replay-visual-check"}
        env.pop("MOUNTING_FILE", None)
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                                       "--port", str(args.backend_port)], cwd=BACKEND_DIR, env=env,
                                      stdout=subprocess.DEVNULL, stderr=open(out / "backend.log", "w")))
        procs.append(subprocess.Popen([CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                       "--window-size=1400,900", f"--remote-debugging-port={args.debug_port}",
                                       f"--user-data-dir={out / 'chrome-profile'}", "about:blank"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        wait_http(f"http://127.0.0.1:{args.backend_port}/health")
        wait_http(frontend_url)
        wait_http(f"http://127.0.0.1:{args.debug_port}/json")
        results, model, calibration = asyncio.run(run(args, frontend_url, args.backend_port, out))
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    checked = [r for r in results if r["wrong"] is not None]
    wrong = sum(len(r["wrong"]) for r in checked)
    posture_checked = [r for r in results if r["posture_ok"] is not None]
    posture_bad = [r["hold"] for r in posture_checked if not r["posture_ok"]]
    report = {"recording": recording, "mode": args.mode, "calibrate": args.calibrate, "speed": args.speed, "model": model,
              "holds": len(results), "checked_segments": sum(len(r["segments"]) for r in checked), "wrong_segments": wrong,
              "posture_checked": len(posture_checked), "posture_wrong": posture_bad, "calibration": calibration,
              "results": results}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    contact_sheet(out, results)
    print(f"\nmode {args.mode}{' + calibrate' if args.calibrate else ''}; model: {model}; holds captured: {len(results)}")
    print(f"segments drawn the wrong way: {wrong} of {report['checked_segments']}; "
          f"posture zone wrong: {len(posture_bad)} of {len(posture_checked)} {posture_bad or ''}  ->  {out}")
    return 1 if wrong or posture_bad or not results else 0


if __name__ == "__main__":
    sys.exit(main())
