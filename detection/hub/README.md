# Detection hub (collapse and man-down)

Prototype indicator, not a medical or safety-certified device. Implements the detection side
(migration steps D1-D3, D4 report) of [`docs/DETECTION_ARCHITECTURE.md`](../../docs/DETECTION_ARCHITECTURE.md):
one `MainModel` consumes the raw sensor queue, cheap per-sample triggers run on every sample, a
window is classified when one fires, and an arbiter turns it into section-5 events. The same object
runs in ThunderSim and behind the backend's raw handoff. Runtime needs numpy only (plus falldet's
own scikit-learn GBT backend).

## Modules (doc section -> code)

| Doc | File | What it does |
|---|---|---|
| 2, 3 | `main_model.py` | `MainModel`: reset/update/finish/dismiss/status; one internal clock across restarts; interleaves falldet events with per-sample paths |
| 3.1, 3.2, 3.9 | `intake.py` | `RawFrames`: backend `Sample` -> arrays; uint32 ms unwrap, duplicates, `RAW_IMU_AXES` remap, all-zero slot = dead, neutral gravity alignment (`uncalibrated` until done) |
| 3.1 | `backend_hook.py` | `DetectorService` + `make_app()`: wraps `create_app(on_raw_sample=...)`, bounded Q2 (1000 = 10 s), one detector thread every 100 ms, drop counter, scoped dismissal, `snapshot()` |
| 3.2 | `main_model.py` | restart = device time back > 1 s (flush + recreate falldet, keep hub state, arm T4 if a trigger <= 60 s before or "down" after); gaps > 0.1 s reset only the tilt filter, >= 2 s also T3 timing |
| 3.3 | `falldet_path.py` | `choose_profile`: explicit (`FALLDET_PROFILE`/config) wins, else `current` for DLPF <= 20 Hz or range <= 4 g; backend forced `gbt`; hidden `error:` rules -> `status = error` |
| 3.4 | `rate_watch.py`, `main_model._rate_check` | rolling 5 s rate from device time; latched or < 40 Hz -> `insufficient_rate`; >= 5 s at >= 90 Hz while latched -> flush + recreate falldet, `rate_recoveries += 1` |
| 3.5 | `posture_track.py`, `topple.py`, `collapse.py` | heading-free tilt per anchor with real dt; T3 topple trigger (O(1)); C3 fast/slow/recovered rules on 1 s bins, "not worn" only tags `no_motion` |
| 3.6 | `man_down.py` | T4 gated timer (armed by T1/T3/episode/restart, disarmed after 5 s upright), warning 30 s, critical 60 s, `no_motion_after_trigger` escalates; `lying_work` 150/180 s timer on breathing-only stillness, mode expires after 30 min |
| 3.7 | `arbiter.py` | one `suspected_fall` per 10 s; first `fall_confirmed` opens an episode, other paths merge; `man_down_*` once per episode and never covered by a dismissal; `cleared` = upright 5 s + gait, >= 8 s after the trigger |
| 5 | `events.py` | `Event(t, kind, confidence, detail)` with the section-5 detail block; diagnostics records |
| - | `config.json`, `config.py` | every threshold (seconds, degrees, rad/s, g) and its hash |
| 7 | `sim_models.py`, `eval/` | ThunderSim arms A-E, harness (fw_main profile, robustness arms, shifts), summaries, lab evaluation |

Man-down fixes (on by default since collapse_v1, approved by the team after dev shifts showed
false man-downs; see `reports/collapse_v1.md` section 5):
`man_down.disarm_on_upright` (5 s upright cancels T4 arming at once, instead of keeping it for
120 s) and `man_down.lying_work_quiet` (the lying_work timer counts only strict, breathing-only
stillness: gyro deviation < `quiet_gyro_rad_s`, accel deviation < `quiet_acc_g`). Setting either to
false restores the earlier reading of doc 3.6; the tests keep both opt-out cases.

Stillness is "mostly still": per sample, every live anchor's gyro and accel deviate little from a
0.5 s low-pass (bias-insensitive, so an uncalibrated MPU-6050 with a 20 dps zero-rate offset still
counts as still), and windows use fractions, never zero motion.

## Run

```sh
cd detection
# unit tests (unittest; the simulation venv has numpy, scikit-learn 1.9.1 and the simulator)
PYTHONDONTWRITEBYTECODE=1 simulation/.venv/bin/python -m unittest discover -s hub/tests -t hub/tests

# hub as a ThunderSim model (fw_main is injected by the harness; the stock CLI has target_8g)
PYTHONPATH=.:simulation simulation/.venv/bin/python -m thundersim run SYNC --model hub.sim_models:HubFull --profile target_8g
PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.harness scenarios \
    --scenarios SYNC,SYNC-KNEEL --runs 20 --variant base --out "$TMPDIR/hub/run"
PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.summarize "$TMPDIR/hub/run"
```

Backend (needs the API dependencies and the proposals in `proposals/` for the full B1):

```sh
cd backend
HUB_BACKEND_APP=1 PYTHONPATH=../detection TELEMETRY_SOURCE=hardware DEVICE_ID="<id>" DEVICE_FORMAT=both \
    python -m uvicorn hub.backend_hook:app --host 127.0.0.1 --port 8000
```

## Spine demo (hub + spine frontend, real time)

`demo_app.py` runs the backend app unchanged, with a ThunderSim scenario replayed as its telemetry
source, and publishes the hub's state in the existing 20 Hz websocket snapshot (`fall_state`,
`detector`, `demo`). The spine view draws the run's segment quaternions; the same run's raw 100 Hz
samples (profile `fw_main`, link losses included) go through `DetectorService` on its own thread.
Everything is paced by the wall clock; nothing waits on ML.

```sh
cd detection
HUB_DEMO=1 PYTHONPATH=.:simulation:../backend simulation/.venv/bin/python \
    -m uvicorn hub.demo_app:app --host 127.0.0.1 --port 8000      # needs: uv pip install --python simulation/.venv/bin/python -r ../backend/requirements-api.txt
cd frontend && npm run dev                                        # http://localhost:5173
```

The card at the top left shows the state (Normal, Checking, Fall detected, Man down), a scenario
picker (SYNC, SYNC-KNEEL, L-TRIP, ... all 35) and a Dismiss button for `fall_confirmed`. Dismissal
never covers man-down, which keeps escalating (doc 3.7). The first scenario takes about 5 s to build.
Measured on a laptop, SYNC: trigger at the impact, `fall_confirmed` 8 s later, `man_down_warning` at
about 30 s and `man_down_critical` at about 60 s after the fall; raw rate 99 Hz, 0 dropped, update
p-max 28 ms.

Live strip: `TELEMETRY_SOURCE=hardware DEVICE_ID="<id>" DEVICE_FORMAT=both` (or `raw_fusion`) with the same
command runs the hub on real BLE frames (default `RAW_IMU_AXES`) and the card shows "Live strip". Not
tested without hardware, and it needs the firmware 100 Hz raw path (blocker B1).

Routes added: `GET /demo/scenarios`, `POST /demo/scenario?scenario=ID&seed=N`, `POST /incident/dismiss`.
`backend/`, `firmware/`, `detection/simulation` and `detection/models` are not edited; the wrapper
replaces `app.main.SimulatedStream` and `app.main.snapshot` in memory.

## Rules this code keeps

- No edits to `detection/models` (falldet), `detection/simulation`, `backend/`, `firmware/`.
  Proposed diffs are in `proposals/`.
- Tune on dev seeds only; held-out blocks go in `SEED_LEDGER.md` first (the harness refuses others).
- Report: `reports/collapse_v1.md`.
