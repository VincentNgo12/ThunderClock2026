# ThunderClock detection architecture

**Status:** proposal, 2026-11-07, pinned to `origin/main` **656b209** (symbols cited, not line numbers). Nothing is implemented. Every number is **prototype evidence** from ThunderSim or acted lab data (Erciyes, VTT), not a result of this design and not from our strip.

**Wording:** prototype indicator. Say "fall", "near-fall", "collapse (no impact)", "man down". No medical or compliance claims; DGUV and similar rules are reference points only.

**Shape:** one main model consumes the raw sensor queue. Cheap triggers run on every sample. When one fires, its window is classified. An arbiter and a backend incident layer escalate.

## 1. Status and blockers (656b209)

| # | Blocker | Evidence | Owner |
|---|---|---|---|
| B1 | **Critical path: no 100 Hz raw path.** Raw from today's hand-off would arrive at ≤ 20 Hz; below 40 Hz falldet detects nothing | `IMUTask` writes depth-1 `xQueueOverwrite` mailboxes (`imuToTelemetryQueue`, `imuToSnapshotQueue`) at debug periods `pdMS_TO_TICKS(1000)`/`(2000)`; `telemetryTask` TODO; `blePublish` NotImplemented; `kInitImuStrip`, `kInitBLE`, `kStartTelemetryTask`, `kStartIMUTask` false | firmware (team to assign) |
| B2 | falldet blind for the session after 1.5 s at 20 Hz, or 0.4 s keeping 1 frame in 3 | `resample.Preprocessor.insufficient_rate` never clears (review probe) | detection; research-95 |
| B3 | Strip read time unmeasured at `kI2cClockHz = 100000` (estimate 7-10 ms per 10 ms); MPU path makes 3 unchecked transactions (read it as one burst like `readIcm`) | `ImuManager::read` | firmware |
| B4 | Boot hangs without USB (`while (!Serial)`); IMU/audio init failure loops forever | `main.cpp` `setup()` | firmware |
| B5 | Dead slot: contract suppresses the strip, backend rejects it, firmware sends zeros (= 0 g free fall) | `protocol/ORIENTATION.md`, `protocol.decode`, `ImuManager::read` | team decision |
| B6 | `docs/TELEMETRY_PROTOCOL.md` has conflict markers; `fall_state` is `null` | file at 656b209 | backend owner |

## 2. Data flow

```
ESP32  IMUTask every 10 ms --> raw FIFO >= 32 --BLE 8888, 1 strip/notification----------+
                           --> fusion mailbox --BLE 8889, 20 Hz quaternions---------+   |
Laptop DeviceStream._notify (asyncio loop):  Q1 quaternion -> Posture (unchanged) <-+   |
                                             Q1 raw (>= 256) <---------------------------+
       acquire() -> on_raw_sample(sample, received_mono) -> Q2 deque(1000) = 10 s  <== SENSOR QUEUE
       detector thread, every 100 ms: Q2 -> RawFrames (unwrap, restart, remap, align) -> MainModel
       loop: IncidentManager -> snapshot() fall_state/detector --WS 20 Hz--> dashboard --POST--> ack, dismiss
MainModel (detection/hub; same object in ThunderSim)
  falldet (profile forced, 'gbt'): T1 impact -> C1 [-2 s, +8 s]; T2 perturbation -> C2 [-4 s, +3 s]
  PostureTrack (100 Hz grid):     T3 topple -> C3 [T3+2 s, T3+30 s]; T4 man-down (armed only)
  RateWatch (5 s rate, latch recovery, errors) -> Arbiter (one episode per event) -> Events
```

## 3. MVP

### 3.1 Intake and threading (backend)
- **Use the documented hook** (`backend/DIAGNOSTICS.md`, "Raw IMU handoff"): a separate module wraps `create_app(on_raw_sample=...)` and owns intake and worker; `main.py` only gains fall fields and endpoints. Agree with the backend owner.
- The hook appends `(sample, received_monotonic)` (2-line change in `acquire()`) and counts drops. One exception disables it.
- **Q1, not Q2, is the weak point:** notifications enter the shared `DeviceStream.queue` on the loop (default 32, max 64); with 64, a 1 s loop stall dropped 48 raw packets (review probe). Give raw its own queue (≥ 256), add a loop-lag monitor, never load or reset models on the loop.
- One detector thread runs RawFrames, `update` and every `reset` (no locks), with OpenMP limited to 1 thread.
- falldet hides errors (rule `error: ...`, `internal_error`); map both to `detector.status = error`.
- **Detection-only inclination is a feature computation, not posture fusion;** posture stays on firmware quaternions (AGENTS.md). Needs team agreement.

### 3.2 Time, gaps, resets
- Unwrap `ImuStripData.time` (uint32 ms); drop duplicates. A µs field wraps every 71.6 min.
- **Keep MainModel across reconnects** and feed the gap through; falldet decides pending triggers itself after gaps > 5 s.
- **Reset only on a proven device restart** (time back > 1 s; boot id in raw v2). `flush()` falldet first. Keep posture ring, last trigger, man-down arming. A restart within 60 s of a trigger, or followed by "down", arms T4.
- Gaps < 2 s reset only the tilt filter, never T3's rotation history (clearing it: 8/12 → 2/12 with a 0.7 s gap at the event, review probe).
- PostureTrack runs on falldet's resampled 100 Hz grid or real `dt` (`TiltFilter` assumes 0.01 s). Windows are in seconds.

### 3.3 falldet with an explicit profile
- **Pick by filter bandwidth:** explicit `FALLDET_PROFILE` wins; else `current` if DLPF ≤ 20 Hz or range ≤ 4 g, else `target`. 656b209 firmware (10-11.6 Hz, ±8 g, ±1000 dps) → `current`.
- `SpineFallModel.reset` picks `target` (44 Hz model, 2.25 g gate) for ranges > 4 g; `StreamContext` has no filter field. New profile, dev, prototype: collapses caught by falldet as shipped **25 %**, forced `current` **78 %**, rule baseline **75 %**; all misses were Stage-1 trigger misses.
- `current` is a stopgap (trained on ±2 g / ±250 dps); retraining needs an ML-scope decision.
- Force `backend='gbt'` (`logreg_json` differs: A-EQUIP 5/20 vs 1/20, dev). Show profile and backend in `/diagnostics` and events.

### 3.4 Rate status and latch recovery
- RateWatch: rolling 5 s rate from device time (`valid_frame_hz` is a session average).
- Latched or < 40 Hz → `detector.status = insufficient_rate`; dashboard says "fall detection unavailable", never "normal".
- **Recovery:** ≥ 5 s at ≥ 90 Hz while latched → flush and recreate the falldet stream, keep hub state, count `rate_recoveries`. Diff to research-95: a recoverable latch with hysteresis.

### 3.5 T3 topple trigger + C3 collapse confirmer
- **T3:** "not down" (3+ anchors < 45°) seen ≤ 3 s ago; now "down" (3+ anchors > 45° and pelvis > 40°); `max(|w_pelvis|, |w_lumbar|)` ≥ 3.0 rad/s since 0.5 s before the last not-down sample. Once per descent. O(1) state, JSON thresholds, ESP32-portable.
- **C3 tolerates brief movement** (the simulated unconscious body has `sway = 0`; with five 0.2 s jerks the prototype C3 rejected 8/8):
  - fast: still for 6 s from T3 + 2 s, median inclination "down";
  - slow, until T3 + 30 s: still ≥ 70 % of a 10 s window, "down", no return to upright (3+ anchors < 30° for 5 s);
  - else `recovered` (`moving_or_up`), timeline only. "Not worn" never rejects; it tags `no_motion`.
- **Thin margin:** SYNC's T3-window peak was 3.04-6.3 rad/s; T3 fired on 17/20 dev SYNC-KNEEL. ThunderSim caps voluntary rotation at 3.14 rad/s, so simulated lie-downs pass partly by construction; 16 % of lab lying-into-bed (815) exceed 3.0. Report a ±10/20 % sweep.

### 3.6 T4 man-down timer (gated)
- **Armed** by T1/T3 or an open episode, for up to 120 s, and disarmed at once when the worker is upright for 5 s (`man_down.disarm_on_upright`, on by default since collapse v1). Before this change, a Stage-1 trigger kept T4 armed for the full 120 s, so stooped rebar work or crawling right after a false impact trigger escalated on dev shifts.
- **Once armed, "down" = C3's definition.** The old 60°/55° rule missed curled endings (SYNC-KNEEL escalated 18/20). "Not upright" (< 3 anchors below 30°) is experiment E4: covers seated endings, but fires after a hard sit.
- Down and mostly still (≥ 70 % of each 10 s) → `man_down_warning` at 30 s, `man_down_critical` at 60 s. Brief movement does not reset it; upright does.
- **"Not worn" after a trigger escalates** (`no_motion_after_trigger`): apnea or a supine body damping the strip looks like "strip off" (man-down 0/8 without breathing, probe). Suppress only with no trigger for 5 min plus a removal pattern.
- **`lying_work`:** man-down after 150 s of full stillness (120-180 s); the mode expires after 30 min and shows on the fall card. "Full stillness" means breathing-only: gyro deviation < 0.1 rad/s and accel deviation < 0.03 g from a 0.5 s low-pass (`man_down.lying_work_quiet`, on by default since collapse v1). Slow lying-work movements of about 5° otherwise passed the looser stillness test and raised false man-downs on dev shifts.

### 3.7 Arbiter
- Triggers within 10 s → at most one `suspected_fall`. The first `fall_confirmed` (C1 or C3) opens an episode; other paths merge (`merged_from`). Each `man_down_*` once per episode.
- A new T1/T3 trigger after a dismissal opens a new episode.
- `cleared` needs upright for 5 s **and** gait, from 8 s after the event (the prototype cleared a suspended ARR worker on upright alone).

### 3.8 Incident layer (backend, loop thread)
```
NORMAL -suspected_fall-> SUSPECTED -recovered, or 20 s undecided with raw fresh-> NORMAL
SUSPECTED or open incident + raw stale >= 3 s -> DATA_LOST (warning) -> critical at 15 s (never NORMAL while stale)
fall_confirmed -> CHECK_IN (15 s) -not dismissed-> ALERT;  man_down_* -> MAN_DOWN_WARNING / _CRITICAL
every non-NORMAL state is latched until the supervisor acknowledges; a reconnect clears nothing
```
- **No worker input exists** (PRODUCT.md): the button reads **"Supervisor dismissed (demo)"**, and the **headline false-alarm metric is alerts before check-in**.
- A dismissal covers **only the alert it answered**, never `man_down_*`; the episode closes 120 s later if nothing new happens. Each episode gets its own incident.
- ThunderSim's `apply_checkin` snoozes every kind for 600 s after one cancel; simulated supervisor numbers need a scoped harness variant.

### 3.9 Calibration
Raw is in board axes; detection reuses the backend's mounting remap. Gravity alignment runs on the first still 2-5 s of standing, as `canonical.py` does (0.15 g tolerance); `POST /detection/calibrate` redoes it. Never block `/posture/calibrate` on raw. Until aligned: `uncalibrated`.

## 4. STRETCH (not needed for the demo)
- **T5 gap/outage trigger:** gap ≥ 0.3 s, upright before, down after → arms an episode; infer rotation from the inclination change.
- **Raw packet v2:** `seq`, boot id, `valid_mask`, `sat_mask`; `RAW_MAX_BYTES`; log the negotiated MTU (quaternion v2 needs ≥ 204).
- **BLE reconnect loop:** `DeviceStream` is single-use; separate reviewed change with the backend owner. MVP: `stale`, restart the service.
- **Full escalation table:** no incident + stale > 10 s → "worker unmonitored", > 60 s → warning; `strip_off` advisory.
- **Device-side event buffer:** a Stage-1 counter in each strip reports impacts during BLE gaps.
- Experiments: E1 landing gate; E4 armed stillness in any posture; E5 upright hold for hard sits; E6 20 Hz quaternion mode; retraining; ungated pre-alarm.

## 5. Event schema
`Event(t, kind, confidence, detail)`; `kind` from ThunderSim `EVENT_KINDS`; only `fall_confirmed` and `man_down_*` alert.
```
detail = {"episode_id": 3, "trigger_t": 812.4, "path": "impact|perturbation|topple|man_down",
  "rule": "model|topple_still_down|topple_mostly_still|moving_or_up|no_motion_after_trigger",
  "severity": "info|warning|critical", "merged_from": ["impact", "topple"],
  "tags": ["ended_upright", "gap_in_window", "decided_early", "no_motion"],
  "evidence": {"rot_peak_rad_s": 7.4, "incl_median_deg": [86, 92, 101, 110], "still_frac": 0.9},
  "data": {"model_profile": "current", "falldet_backend": "gbt", "rate_hz_5s": 99.8}}
diagnostic = {"type": "device", "code": "insufficient_rate|rate_recovered|link_gap|restart|sensor_dead|internal_error"}
"fall_state": {"state": "normal|suspected|check_in|alert|man_down_warning|man_down_critical|data_lost|unavailable",
  "severity": "normal|info|warning|critical", "incident_id": 3, "checkin_deadline_s": 9.5,
  "acknowledged": false, "label": "prototype indicator"}
"detector": {"status": "ok|uncalibrated|insufficient_rate|stale|error", "model_profile": "current",
  "raw_rate_hz_5s": 99.8, "intake_dropped": 0, "loop_lag_ms_max": 12, "rate_recoveries": 0, "work_mode": "normal"}
```
- `freshness.fall_state` uses raw receipt; fall alerts are latched, not gated on quaternion freshness. `detection_events`: last 20, monotonic ids.
- `POST /incident/dismiss` only in CHECK_IN, never for `man_down_*`; `/worker/mode` expires; `simulated_raw` is labelled `simulated`.

## 6. Failure modes
| Failure | Handling |
|---|---|
| Raw < 40 Hz; slow burst latches falldet; wrong profile; silent `logreg_json` | `insufficient_rate` ("unavailable"), RateWatch recovery, explicit config (3.3-3.4); pre-demo 5 s rate ≥ 90 Hz |
| Loop stall; link lost at the impact; restart; 0.1-1 s gap at the event | raw Q1 ≥ 256 + lag monitor; DATA_LOST; keep MainModel; keep T3 history |
| Dead IMU | B5 decision; mask the slot as `canonical.py` does; explicit 3-live-anchor rules |
| Apnea, supine damping, movement after collapse, curled ending, collapse in `lying_work` | "not worn" escalates; mostly-still rules; C3 "down" for T4; 150 s timer |
| **Collapse ending seated or against a wall** | **Not covered**: T3 fires on 0 % of lab code 920 and the worker is never "down"; the impact path confirms 64-77 % (acted) without escalation. Say so in the demo; request SYNC-WALL, SYNC-SEATED; E4 |
| Hard sit, jackhammer, ladder dismount, fast flop | supervisor-visible false confirmations (A-SIT 3/8; about 1.5-2 per 8 h before check-in; 16 % of lab 815 pass T3's gate); reported; E1/E5 |

## 7. Testing and acceptance
- **Profile** `fw_main` = 100 Hz, ±8 g, ±1000 dps, 10 Hz DLPF, injected from a scratch harness until the simulator owner adds it; always `--out <scratch>`. Rate caveat: also 50 Hz, and 20 Hz (falldet off).
- **Arms, identical streams:** A falldet as shipped, B `current`, C hub without T4, D full hub, E rule baseline. State each arm's error model; add `errors=uncalibrated` (dev: A-SIT 10/20 → 0/20).
- **Seeds:** tune on dev only (`--seed-base 0`; shifts 500000+). **Block 1000000 is burned** (three prototypes used it). Use 2000000+ with a seed-use ledger in `detection/hub/reports/`; `thundersim export` defaults to `--seed-base 2000000`, so skip exported blocks (e.g. start at 2100000). One held-out run per frozen config hash.
- **Metrics:** by event kind (`runner.classify` counts `man_down_*` as `fall_confirmed`). Headline = alerts **before** check-in (the shift runner's forced 5 s check-in makes supervisor alerts 0). Wilson 95 % CIs; exact Poisson for rates. Hard negatives n ≥ 20 each. "≤ 0.25 alerts per 8 h" with zero events needs about 120 h of shifts.
- **Robustness arms (dev):** jerks after collapse, apnea, curled endings, gap at the event (0.3/0.7/1.5 s), restart mid-event, slow start, 50 Hz with real `dt`, real dead-slot contract, protobuf path (not `--fast`), dismiss-then-collapse.

| Bar (fresh held-out block) | Value |
|---|---|
| SIM_SPEC `report.PASS_BARS` | unchanged; "≤ 1 false alarm per 8 h before check-in" is **known unmet** by the impact path; report it |
| Collapse | SYNC, SYNC-KNEEL each ≥ 18/20 confirmed; with no check-in, `man_down_critical` ≤ 90 s on ≥ 19/20 |
| Falls, near-falls | none more than 1/20 below arm B |
| Hard negatives | ≤ 1/20 alerts before check-in: A-LIE, A-NAP, A-LIEWORK, A-KNEEL, A-REBAR, DEVICE-OFF, HAT-OFF, STRAP-SLIP; A-SIT, A-EQUIP, A-JUMP no worse than B; add A-RUN, A-CARRY, A-LADDER, A-JACK |
| Escalation safety | ARR never `cleared` while suspended; apnea escalates ≥ 19/20; latch always recovers |
| Runtime (demo laptop) | `update` p99 ≤ 50 ms per 100 ms batch; 0 raw drops in 10 min; loop lag < 200 ms |

- **Lab LOSO** (Erciyes + VTT): T3 fires/h on ADL (≤ 3/h); codes 919, 920, 815, 816. Only about 8 s of aftermath; 919 is acted like an impact fall.
- **Tests:** every trigger/confirmer branch, T1 + T3 = one alert, event t ≤ batch end, latch recovery, restart, RawFrames vs `canonical.py` parity; backend: fake BLE → suspected → check_in → alert, scoped dismiss, DATA_LOST, Q1 stall.
- **Our strip:** supervised volunteers on crash mats; replay through MainModel.

## 8. Migration steps (each ends runnable)
| Step | Owner | Done when |
|---|---|---|
| F1 raw FIFO (≥ 32, drop counter) filled at `kAcquisitionPeriodMs`; fusion gets every strip (or its real rate); flags on | firmware | Serial ≥ 99 strips/s, read < 10 ms |
| F2 BLE sender drains the FIFO in order, 1 strip per notification; `ORIENTATION.md`: latest-value for quaternions only | firmware | raw ≥ 90 Hz, 0 drops in 10 min |
| F3 boot without USB, no init halt, B5 encoding | firmware | battery cold boot streams |
| B1 hook module, raw Q1, lag monitor, `simulated_raw` | backend + detection | SYNC replay gives events |
| B2 IncidentManager, `fall_state`, endpoints, protocol doc | backend | unittest passes |
| D1 MainModel pass-through; D2 PostureTrack, T3, C3; D3 T4, arbiter, robustness arms | detection | D1 equals arm B; dev bars met |
| D4 freeze, ledger, one held-out run, `reports/collapse_v1.md` | detection | report with CIs |
| R1 diffs: recoverable latch, bandwidth pick, README, OpenMP guard | research-95 | reviewed |
| S1 `fw_main`; SYNC-WALL/-SEATED/-BRIEF, FLOP; scoped check-in; jerk/apnea | simulator owner | available |
| U1 fall card, timeline, two buttons | frontend | works on `simulated_raw` |

## 9. Open questions for the team
1. Who builds F1-F2, by when? Is a tested 50 Hz fallback acceptable?
2. Who owns `detection/hub` (and adds `detection/` to AGENTS.md's structure)? Does research-95 accept a wrapper plus diffs?
3. Does the backend owner accept the hook route, a raw queue and `simulated_raw`?
4. Dead slot: require raw v2 `valid_mask`, or an all-zero record means "invalid"?
5. Accept "detection-only inclination is a feature computation" under AGENTS.md?
6. Keep `current` as a stopgap, or approve retraining?
7. Accept about 1.5-2 supervisor-visible false alarms per 8 h, plus hard sits?
8. Agree that "not worn"/"no data" after a trigger escalates and no button cancels man-down?
9. `lying_work` timer and expiry? Does harness suspension bypass a dismissal?
10. Request seated/wall-slide scenarios and state the gap in the demo?
11. Acted collapses on crash mats: OK under our safety and consent rules?
12. Who keeps the seed ledger? Which laptop is the demo laptop?

## Appendix: evidence
- Sources are scratch prototypes and review probes; regenerate before quoting. falldet-as-shipped recall was also reported as 16/40 and 15/40; the event-hub prototype's 40/40 used the burned block.
- References only: Bourke & Lyons 2008 (trunk gyro 3.1 rad/s); Bagalà 2012 (impact methods miss low-impact falls); DGUV Regel 112-139 (alarm ≤ 90 s, pre-alarm cancellable ≤ 15 s; unverified).
