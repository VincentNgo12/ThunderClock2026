# Proposed changes outside detection/hub

Nothing here is applied. Each diff is small, has a reason, evidence and a test, and needs its
owner's review. The hub already works without them (it wraps falldet and uses the documented
backend hook), so they can land in any order.

## 1. `falldet_R1_latch_profile_openmp.diff` (owner: research-95, step R1)

Base: the working tree of `detection/models/falldet` (unchanged from b123c40). Apply from the repo
root with `patch -p1`.

| Change | Why | Evidence | Test |
|---|---|---|---|
| `resample.Preprocessor`: the `insufficient_rate` latch clears after 5 s at a median rate >= 90 Hz (hysteresis); the grid restarts as after a gap. `FallDetectorStream.push_batch` then decides pending triggers, re-arms the warning and emits a `rate_recovered` diagnostic | Blocker B2: 1.5 s at 20 Hz at stream start blinds falldet for the whole session | Dev `slowstart` arm (SYNC + SYNC-KNEEL, n = 40): falldet B detects 0/40, the hub (which recreates the stream itself) detects as in `base` | Scratch copy with the patch applied: `tests/test_falldet.py` and `tests/test_sim_adapter.py` pass; 2 s at 20 Hz then 7 s at 100 Hz gives `insufficient_rate`, then `rate_recovered`, and samples flow again |
| `sim_adapter.SpineFallModel.reset`: an explicit `FALLDET_PROFILE` wins; else `current` when the context reports `dlpf_hz <= 20` (optional field), a `current*` profile or range <= 4 g | The ±8 g, 10 Hz firmware gets the 44 Hz `target` model today | Dev base, fw_main: A (shipped, picks `target`) SYNC 4/20, SYNC-KNEEL 6/20; B (forced `current`) 16/20, 15/20 | Patched copy imports and passes the same scripts; `StreamContext` has no `dlpf_hz` yet, so ThunderSim still needs `FALLDET_PROFILE=current` or the hub |
| `falldet/__init__.py`: `OMP_NUM_THREADS` defaults to 1 | One detector thread next to the backend loop; GBT inference must not fan out | Doc 3.1 | Import check |

## 2. `backend_B1_raw_queue_lag_diagnostics.diff` (owner: backend, step B1)

Base: `origin/main` **f3337e3** (`backend/app/ble.py`, `backend/app/main.py`). Apply with `patch -p1`.

| Change | Why | Evidence | Test |
|---|---|---|---|
| `DeviceStream` default queue 256 when raw frames are subscribed (`raw`, `both`; `raw_fusion` uses `raw`), limit raised to 512 | f3337e3 uses 8 entries for `raw`/`raw_fusion` and 32 for `both`; at 100 Hz that is 80-320 ms of loop stall before raw packets drop | Doc 3.1 review probe: queue 64 + 1 s stall dropped 48 raw packets | Applies cleanly to f3337e3 and compiles; the backend's unittest suite could not run here (no fastapi/bleak in the available venv) |
| `publish()` measures how late the 20 Hz publisher wakes (`loop_lag_ms_max` in `/diagnostics`) | Loop-lag monitor (3.1) | - | same |
| `/diagnostics` gains the hub's `detector` block when `app.state.detector` exists | `detector.status` must be visible (3.3-3.4) | - | same |

Not proposed yet (B2, backend owner): `IncidentManager`, `fall_state`, `POST /incident/dismiss`,
`/worker/mode`, `simulated_raw`. The hub side of B1 is `detection/hub/backend_hook.py`: it wraps
`create_app(on_raw_sample=...)`, so the documented hook needs **no** backend edit to get frames or
receipt time (it reads `app.state.stream.received_monotonic`). A separate raw queue (not just a
larger shared one) is the full fix and needs a reviewed `ble.py` change.

## 3. Requests to the simulator owner (step S1), as specs only

1. `fw_main` profile in `device.PROFILES`: 100 Hz, ±8 g, ±1000 dps, DLPF 10 Hz (ImuManager at
   f3337e3). The hub harness injects it in memory today. Also add `dlpf_hz` to `StreamContext`.
2. Scenarios: **SYNC-WALL** (slide down a wall to seated), **SYNC-SEATED** (slump in a seat or cab),
   **SYNC-BRIEF** (collapse, moves again after 5-20 s), **FLOP** (fast deliberate lie-down, trunk rate
   3-5 rad/s; today voluntary rotation is capped at 3.14 rad/s, so simulated lie-downs pass T3's 3.0
   rad/s gate partly by construction).
3. Post-collapse realism: breathing on/off and amplitude per run, brief jerks/fidgets while
   unconscious (today `sway = 0`, `fidget = 0`), curled endings. The harness emulates these with a
   Builder subclass (`hub/eval/harness.py`, variants `jerks`, `apnea`, `damped`, `curl`).
4. A scoped check-in in `runner.apply_checkin`: one cancel should cover only the alert it answered,
   not every kind for 600 s (dismiss-then-collapse cannot be scored with it; the harness scores
   `X-DISMISS` itself).
5. An `--errors` flag on `suite` (the `uncalibrated` error model exists in `StreamOptions` only).
