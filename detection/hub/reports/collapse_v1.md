# Collapse (no impact) and man down: hub v1 report

**Prototype indicator. Simulation and acted lab data only. No medical or safety claim.**
Date 2026-11-07/08. Design: `docs/DETECTION_ARCHITECTURE.md` (pinned to 656b209; section 3.6 is being
updated by the team lead for the two man-down fixes). Code read at GitHub `origin/main` **f3337e3**
(backend, docs) and **979afef** (firmware, re-checked before freezing), and the local checkout b123c40
(`detection/models`, `detection/simulation`, unchanged by both).

**Frozen config `hub/config.json`, hash `cf8c7456b67e`** (snapshot `reports/collapse_v1_config.json`;
ledger row 2100000 in `SEED_LEDGER.md`). History: 60323b2f32be failed the dev shift guard and was never
run held-out; cf8c7456b67e = 60323b2f32be + `man_down.disarm_on_upright` + `man_down.lying_work_quiet`
(team-approved, section 5).

**Current result (new simulator, section 6b).** The frozen config was re-run unchanged on the
simulator pinned at `f05639c99ccacc4635258cbc48aa5fecaf817fc790beb995fb3b071a3b8fd68a` (checkout
86faf2e): dev bars all pass, then held-out block **2200000** (run once) passes every acceptance bar.
D confirms 20/20 SYNC and 20/20 SYNC-KNEEL [0.84-1.00] and escalates all 40 to `man_down_critical`
within 90 s. With syncope now landing with an impact, falldet alone also confirms 20/20 and 20/20
(A and B), so the hub's added value on these scenarios is the man-down escalation (A/B: 0/40) and
robustness (slow start, restart, 20 Hz, apnea), not first detection. Sections 2-6 below are the
earlier run on the old simulator, kept for the record.

**Earlier result (old simulator, sections 2-6).** Held-out block 2100000: D 20/20 and 20/20,
man-down 40/40; falldet as shipped 3/20 and 6/20, forced `current` 15/20 and 13/20.

## 0a. firmware 979afef (after f3337e3)

`origin/main` moved during the work (979afef "working snapshot + working imu"): the IMU task now runs
at 10 ms and notifies the raw characteristic every cycle (`pCharIMU->notify()`), snapshot at 50 ms,
and `kInitImuStrip`, `kInitBLE`, `kStartIMUTask` are on. `ImuManager` is unchanged, so **fw_main
(±8 g, ±1000 dps, ~10 Hz DLPF, 100 Hz) still holds**. Whether those notifications reach the backend
at 100 Hz, in order and in the decoded raw format, was not checked (no hardware).

## Validity note (added 2026-11-08; resolved by the re-run in section 6b)

Every simulator number in this report (dev and held-out) was produced with the `detection/simulation`
working tree as it was up to 2026-11-08 ~00:05. After that, the realism work changed
`thundersim/falls.py` and `site.py` (uncommitted, modified 02:59): `slow_collapse` is now a
0.4-0.8 s buckle and a partially supported drop (waist peak ~3.6 g, fitted to Erciyes 919 acted
syncope; falldet Stage 1 reportedly fires on all simulated syncopes), plus Hunt-Crossley contact
damping, a damped trunk landing, trunk curl relaxing on landing and a softer crash_mat for all
body-first falls. The SYNC / SYNC-KNEEL and fall results here therefore describe the **old** collapse
model (slow guided sag, no Stage-1 impact) and are stale against the current simulator. The old path
is reproducible with `falls.SYNC_SUPPORT = None` and `falls.SYNC_BUCKLE_S = (1.0, 2.0)`.
A later change (also uncommitted) slows trips and knock-downs (`PIVOT_MOMENTUM_KEEP=0.5`,
`PIVOT_RESIST=2.0` in `Faller.pivot`; restore with 1.0 / 0.0): this touches L-TRIP, STRUCK-KD,
FALL-GETUP, secondary topples and the harness's X-DISMISS (its first fall is a trip), so those
regression-guard numbers are stale too.
Block 2100000 stays used; a re-evaluation of the frozen config on the new simulator needs dev runs
first and, for a held-out claim, the next block (2200000), reported as a new run.

## 0. What f3337e3 changes for this design (656b209 -> f3337e3)

| Design assumption | f3337e3 | Effect on the hub |
|---|---|---|
| 3.1 the raw hook needs a 2-line `acquire()` change to pass receipt time | `create_app(on_raw_sample=...)` still passes only the `Sample` | **Not needed** for `raw`/`raw_fusion`: `backend_hook.make_app` reads `app.state.stream.received_monotonic` (set before the sample is yielded). In `both` mode that field tracks quaternion packets only, so the hook uses the dequeue time (late by at most the queue latency). Zero backend edits for B1 intake; an exact raw receipt time in `both` mode needs the 2-line change |
| 3.1 Q1 = shared `DeviceStream.queue`, default 32, max 64 | unchanged for `both`; the new `raw_fusion` mode opens the stream as `raw`, whose default queue is **8** | Q1 is weaker than the doc says in `raw`/`raw_fusion` (80 ms of loop stall). Proposal B1 raises it to 256 |
| raw frames reach detection only with `DEVICE_FORMAT=both` | `raw_fusion` also calls the raw hook (it is `raw` on the wire) | the hub works in `both`, `raw` and `raw_fusion` |
| posture comes from firmware quaternions; detection-only inclination is separate (3.1, AGENTS.md) | new opt-in laptop fusion `RawFusion` (`fusion.py`) for the dashboard posture | confirmed separate: `RawFusion` resets every filter on any gap > 100 ms and on a lost sensor, which would erase T3's descent (3.2 says gaps < 2 s must keep T3 history). The hub keeps its own heading-free tilt |
| a restart can be inferred from backend counters | `RawFusion` now increments `stream.stats.resets` on estimator resets (gaps > 100 ms, sensor loss) | **invalidated**: `stats.resets` is not a device-restart signal. The hub detects a restart only from device time going back > 1 s (3.2) |
| mounting remap = backend's | `RAW_IMU_AXES` (board -> right, up, back; default `-y,z,-x`) | `RawFrames` parses the same format and default |
| dead slot = all-zero record (B5) | `RawFusion` also treats an all-zero accel+gyro record as a failed read | consistent: `RawFrames` marks it dead (`slot_mask` False, NaN) |
| B6 `TELEMETRY_PROTOCOL.md` conflict markers; `fall_state` null | markers resolved; `dashboard.snapshot` still sets `fall_state: None` | B6 half fixed; `fall_state` remains B2 work |
| B1 firmware critical path, fw_main settings | no firmware change (±8 g, ±1000 dps, ~10-11.6 Hz DLPF, mailboxes at debug periods, init flags false) | unchanged: fw_main profile and the 20/50 Hz caveat still apply |

## 1. Setup

- Device profile **fw_main** (injected in memory by `hub/eval/harness.py`): 100 Hz, ±8 g,
  ±1000 dps, DLPF 10 Hz, as `ImuManager::begin` sets at f3337e3. ThunderSim `fast` transport
  unless marked "protobuf". Error model `calibrated` unless marked `uncalibrated`.
- Arms on identical streams (one simulated stream feeds every arm):
  A falldet as shipped (`SpineFallModel()`, picks `target` for ±8 g, backend auto = gbt);
  B falldet forced `current`, gbt; C hub without T4; D hub as frozen; D1 hub with T3 and T4 off;
  E ThunderSim rule baseline; L hub with falldet `logreg_json`.
- Metrics by event kind: **fall_confirmed** counts only that kind (a man-down does not pass for a
  collapse confirmation). "any alert" = any of fall_confirmed / man_down_*: with no worker input
  every alert is an **alert before check-in**. Latency = first alert on the laptop clock minus
  the ground-truth reference (impact; arrest end for ARR). Wilson 95 % CIs in brackets.
- Dev seeds: scenarios 0-19, shifts 500000-500014. Held-out: see section 6.

## 2. Reproduction (dev, n = 20, fw_main, calibrated)

| Scenario | A shipped | B current | E baseline | Doc 656b209 (prototype) |
|---|---|---|---|---|
| SYNC fall_confirmed | 4/20 [0.08-0.42] | 16/20 [0.58-0.92] | 17/20 [0.64-0.95] | A 25 %, B 78 %, E 75 % (collapses) |
| SYNC-KNEEL | 6/20 [0.15-0.52] | 15/20 [0.53-0.89] | 13/20 [0.43-0.82] | T3 fired 17/20 |
| A-SIT any alert | 13/20 | 10/20 | 0/20 | 3/8 |
| A-EQUIP, L arm | B 1/20; L 6/20 | | | logreg 5/20 vs gbt 1/20 |

All within the CIs of the doc's numbers; T3 fired on 17/20 dev SYNC-KNEEL runs in the first
dev run, as in the doc. **D1 equals B**: identical fall_confirmed outcome on all 700 dev runs
(31 runs differ by <= 0.04 s in event time because the hub hands falldet decisions on at the next
frame time rather than falldet's grid time).

## 3. Dev results, base suite (n = 20 per scenario, fw_main)

These tables are for the **frozen config cf8c7456b67e** (re-run 2026-11-07 after the man-down
fixes, arms A, B, C, D, D1, E). Every cell is identical to the earlier run with 60323b2f32be: the
two fixes change no outcome on any of the 700 scenario runs (the L arm was not re-run).

### 3.1 Collapse set

| Scenario | Arm | fall_confirmed | man_down_critical <= 90 s | latency median / p95 s | first path |
|---|---|---|---|---|---|
| SYNC | B | 16/20 [0.58-0.92] | 0/20 | 7.6 / 7.9 | falldet 16 |
| SYNC | C | 20/20 [0.84-1.00] | 0/20 | 7.6 / 7.9 | impact 14, topple 6 |
| SYNC | **D** | **20/20 [0.84-1.00]** | **20/20 [0.84-1.00]** | 7.6 / 7.9 | impact 14, topple 6 |
| SYNC | E | 17/20 [0.64-0.95] | 17/20 | 9.4 / 9.6 | baseline 17 |
| SYNC-KNEEL | B | 15/20 [0.53-0.89] | 0/20 | 7.5 / 7.9 | falldet 15 |
| SYNC-KNEEL | C | 20/20 [0.84-1.00] | 0/20 | 7.5 / 8.0 | impact 12, topple 8 |
| SYNC-KNEEL | **D** | **20/20 [0.84-1.00]** | **20/20 [0.84-1.00]** | 7.5 / 8.0 | impact 12, topple 8 |
| SYNC-KNEEL | E | 13/20 [0.43-0.82] | 13/20 | 9.4 / 9.5 | baseline 13 |

`man_down_critical` arrives about 60 s after the impact (T4 counts from the first down-and-still bin).

### 3.2 Hard negatives: alerts before check-in (runs with >= 1 alert)

| Scenario | A | B | **D** | E | D alert paths |
|---|---|---|---|---|---|
| A-SIT | 13/20 | 10/20 [0.30-0.70] | **10/20** [0.30-0.70] | 0/20 | impact (falldet model) only |
| A-EQUIP | 2/20 | 1/20 | **1/20** | 0/20 | impact |
| A-JUMP | 3/20 | 3/20 | **3/20** | 1/20 | impact (`free_fall_no_gait`) |
| A-KNEEL | 0 | 0 | **0/20** | 4/20 | - |
| A-LIE | 0 | 0 | **0/20** | 17/20 | - |
| A-NAP | 0 | 0 | **0/20** | 20/20 | - |
| A-LIEWORK, A-REBAR, A-JACK, A-RUN, A-CARRY, A-LADDER, A-HAMMER, A-DRILL, DEVICE-OFF, HAT-OFF, STRAP-SLIP | 0 | 0 | **0/20 each** [0.00-0.16] | 0 | - |

**T3 fired 0 times on all 340 dev hard-negative runs** (also 0 under `uncalibrated`); every D alert
on a negative is the unchanged impact path. A-SIT, A-EQUIP and A-JUMP are equal to B (bar: no worse).

### 3.3 Regression guard (falls, near-falls)

D is never below B: H-LAD 18 = 18, H-SCAF/H-ROOF/H-HOLE/H-EXC 20 = 20, L-SLIP/L-TRIP/STRUCK-KD
20 = 20, **STAIR 20 vs 17**, **NET 20 vs 19**, SLIP-FRICTION 11 = 11, ARR 20 = 20, FALL-GETUP
20 = 20. NF-STUMBLE, NF-RAIL, STRUCK-NF: 0/20 fall_confirmed in every arm. **ARR: 0 `cleared`
in every hub arm** (gait required). SIM_SPEC bars (`report.PASS_BARS`) on the D arm: height
98/100 >= 90 %, same-level/stairs/struck 80/80 >= 80 %, ARR 20/20, NF false alarms 0 %, latency
p95 <= 30 s. The "<= 1 false alarm per 8 h before check-in" bar: section 5.

### 3.4 Error model `uncalibrated` (dev, n = 20, config 60323b2f32be)

Frozen-config re-run (reduced, n = 10, arms B and D, machine load ~240): SYNC D 10/10 confirmed and
10/10 critical (B 6/10, 0); SYNC-KNEEL D 10/10, 10/10 (B 9/10, 0); ARR D 7/10 = B, 0 cleared;
A-SIT, A-JUMP, A-LIE, A-NAP, A-LIEWORK, DEVICE-OFF 0/10 alerts in both arms.

| Scenario | A | B | **D** | E |
|---|---|---|---|---|
| SYNC fall_confirmed / crit <= 90 s | 4 / 0 | 15 / 0 | **20 / 20** | 17 / 0 |
| SYNC-KNEEL | 6 / 0 | 16 / 0 | **20 / 20** | 13 / 0 |
| ARR | 16 | 13 | 13 (= B) | 14 |
| A-SIT any alert | 2 | 0 | **0** | 0 |
| A-JUMP | 3 | 3 | **3** | 1 |
| other 15 hard negatives | 0 | 0 | **0** | 0 |

The first hub build used absolute stillness (|w| < 0.35 rad/s, ||a| - g| < 0.1 g). Under
`uncalibrated` (MPU-6050 zero-rate offset up to 20 dps, 3 % accel scale) it never saw stillness:
D collapsed to B (15/20, 16/20) and man-down fired 0/40. Stillness now uses the deviation from a
0.5 s low-pass per anchor; this was the only change after the first dev pass, and every dev table
in this report was regenerated with it.

## 4. Robustness arms (dev, SYNC + SYNC-KNEEL)

### 4.1 Frozen config cf8c7456b67e (reduced re-run: n = 10 each, arms B and D, seeds 0-9)

The machine was shared with other jobs (load average 200-300), so the re-run was cut to the arms
that test the changed man-down logic. Not re-run with the frozen config: `damped`, `gap0.3`,
apnea + uncalibrated, dead slots 0, 2 and 3, the protobuf path (the table in 4.2, n = 20, is from
60323b2f32be; the fixes touch only the man-down timer and lying-work timer).

| Arm | B conf | **D conf** | **D crit <= 90 s** |
|---|---|---|---|
| jerks | 6, 8 | **10, 10** | **10, 10** |
| apnea | 6, 8 | **10, 10** | **10, 10** |
| curl | 6, 8 | **10, 10** | **10, 10** |
| gap 0.7 s at the impact | 5, 5 | **10, 10** | **10, 10** |
| gap 1.5 s at the impact | 0, 1 | **2, 3** | **2, 3** (known miss) |
| restart | 6, 8 (latency 45-54 s) | **10, 10** (3.5 s) | **10, 10** |
| slow start | 0, 0 | **10, 10**, latch recovered in 20/20 | **10, 10** |
| 50 Hz | 6, 8 | **10, 10** | **10, 10** |
| 20 Hz | 0, 0 | **10, 10** | **10, 10** |
| dismiss, then collapse (X-DISMISS) | 7 | **10** | **10** |
| dead slot 1 lumbar | 0, 0 | **10, 10** | **10, 10** |

### 4.2 Config 60323b2f32be (n = 20 each, arms B, D, E)

| Arm | B conf | **D conf** | **D crit <= 90 s** | E conf / crit | Note |
|---|---|---|---|---|---|
| jerks (five 0.2 s 10-20 deg jerks in the first 30 s) | 16, 15 | **20, 20** | **20, 20** | 19/14, 18/13 | |
| apnea (breathing removed) | 16, 15 | **20, 20** | **20, 20** | 0/0, 0/0 | baseline treats it as "not worn" |
| apnea + uncalibrated | 15, 12 | **20, 18** | **20, 20** | - | |
| damped (breathing x 0.2) | 16, 15 | **20, 20** | **20, 20** | 7/7, 6/6 | |
| curl (trunk flexes 25-45 deg after landing) | 16, 15 | **20, 20** | **20, 20** | 0/0, 0/0 | |
| gap 0.3 s at the impact | 16, 14 | **20, 20** | **20, 20** | 17, 13 | |
| gap 0.7 s at the impact | 11, 9 | **20, 19** | **20, 19** | 17, 13 | |
| gap 1.5 s at the impact | 0, 3 | **6, 7** | **6, 7** | 17, 8 | **known miss**: the gap hides the descent; T5 (stretch) |
| restart 3 s after the event (0.5 s reboot) | 16, 15 (latency ~50-60 s) | **20, 20** (3.5 s) | **20, 20** | 18/0, 13/0 | A/B/E are blind until device time passes the old clock |
| slow start (1.5 s at 20 Hz) | **0, 0** | **20, 20** | **20, 20** | 17, 13 | falldet latch; hub recovers it |
| 50 Hz (real dt) | 16, 15 | **20, 20** | **20, 20** | 15, 12 | |
| 20 Hz | 0, 0 | **20, 17** | **20, 17** | 19, 19 | falldet off (`insufficient_rate`); T3 + C3 + T4 only |
| dismiss, then collapse (X-DISMISS: trip + get up, alert dismissed after 10 s, then a collapse) | 17 | **20** | **20** | 15/15 | scored by the harness, not `apply_checkin` |
| dead slot 0 pelvis | 19, 20 | **20, 20** | **20, 20** | 18, 13 | |
| dead slot 1 lumbar | **0, 0** | **20, 17** | **20, 17** | 16, 13 | falldet loses Stage 1 with a dead lumbar |
| dead slot 2 thoracic | 16, 15 | **20, 20** | **20, 20** | 16, 13 | |
| dead slot 3 upper thoracic | 16, 15 | **20, 20** | **20, 20** | 9, 9 | |
| protobuf path (n = 10) | 6, 8 | **10, 10** | **10, 10** | - | A-SIT 8/10 = B; A-LIE, A-NAP, DEVICE-OFF 0/10; ARR 10/10, 0 cleared |

20 Hz works for T3/C3/T4 in this test (falldet does not); 50 Hz works for every path.

## 5. Shifts (dev, fw_main) and what changed

**Reduced:** 9 fall-free shifts of 2 h (seeds 500000-500008, 18.1 h, SIM_SPEC 5.2 mix incl.
jackhammer, ladder, rebar, kneel/crawl, lying work, jump-downs, runs), fast transport, calibrated.
A 15 x 8 h (120 h) run could not finish on the shared machine (load 70-300 on 10 cores), so only
exact Poisson CIs are given; 18 h cannot support a "<= 0.25 per 8 h" claim.

| Arm | alert events | per 8 h [95 % CI] | episodes (check-ins) | per 8 h [95 % CI] | hub-path episodes | by path:kind |
|---|---|---|---|---|---|---|
| A | 9 | 3.97 [1.82-7.54] | 9 | 3.97 [1.82-7.54] | - | falldet fall_confirmed 9 |
| B | 5 | 2.21 [0.72-5.15] | 5 | 2.21 [0.72-5.15] | - | falldet fall_confirmed 5 |
| C | 5 | 2.21 [0.72-5.15] | 5 | 2.21 [0.72-5.15] | 0 | impact 5 (= B) |
| D, 60323b2f32be (before) | 21 | 9.26 [5.73-14.16] | 11 | 4.85 [2.42-8.68] | **6** | impact 5; man_down warning 8, critical 8 |
| **D, frozen cf8c7456b67e** | 9 | 3.97 [1.82-7.54] | **5** | **2.21 [0.72-5.15] (= B)** | **0** (upper 95 % bound 1.63 per 8 h) | impact 5; man_down warning 2, critical 2 |
| E | 3 | 1.32 [0.27-3.87] | 3 | 1.32 [0.27-3.87] | - | baseline |

T3 fired 0 times in 18.1 h. The frozen D still has 4 man-down alert events; all four escalate
episodes that the unchanged impact path opened (falsely) before rebar and lying work, so they add
no check-in. The impact path itself is the known SIM_SPEC "<= 1 false alarm per 8 h before
check-in" miss (2.2 per 8 h here).

**What changed and why (60323b2f32be -> cf8c7456b67e, approved by the team).** The first config
added 6 hub-path episodes in 18.1 h from two mechanisms:

1. The `lying_work` timer counted slow 5 deg lying-work motions as "full stillness" (4 episodes);
   the resulting warning opened an episode that armed T4, which escalated to critical.
2. T4 stayed armed for the 120 s minimum after a Stage-1 trigger whose Stage 2 said normal, and a
   following rebar stoop, kneel/crawl or lying work was "down and mostly still" by C3's definition
   (2 episodes; 2 more escalations of impact-path false alarms).

Fixes, both now default (`config.json`, `man_down`), each with an opt-out test:
- `disarm_on_upright: true`: 5 s upright (>= 3 anchors < 30 deg) cancels T4 arming at once.
  A worker who never stands up stays armed, so collapses are unaffected.
- `lying_work_quiet: true`: the lying_work timer counts only breathing-only stillness (per-sample
  gyro deviation < 0.10 rad/s and accel deviation < 0.03 g from the 0.5 s low-pass; new
  `posture.quiet_gyro_rad_s`, `posture.quiet_acc_g`).

No other threshold changed. The dev base suite is cell-for-cell identical (section 3).

## 6. Held-out block 2100000 (frozen cf8c7456b67e, run once)

Registered in `SEED_LEDGER.md` and `reports/seed_ledger.csv` before running. Seeds 2100000-2100019,
fw_main, calibrated, fast transport. Runs: all 35 catalogue scenarios (arms A, B, C, D, E);
SYNC and SYNC-KNEEL with `apnea` and `slowstart` (B, D); X-DISMISS with `dismiss` (B, D). 0 run
errors. No threshold was changed after it. Not run held-out: shifts (dev seeds only), the other
robustness arms, `uncalibrated`, the protobuf path.

| Scenario | A | B | C | **D** | E |
|---|---|---|---|---|---|
| SYNC fall_confirmed | 3/20 [0.05-0.36] | 15/20 [0.53-0.89] | 20/20 | **20/20 [0.84-1.00]** | 13/20 |
| SYNC man_down_critical <= 90 s | 0 | 0 | 0 | **20/20 [0.84-1.00]** (median 61.5 s, max 63.1 s) | 13/20 |
| SYNC-KNEEL fall_confirmed | 6/20 [0.15-0.52] | 13/20 [0.43-0.82] | 20/20 | **20/20 [0.84-1.00]** | 10/20 |
| SYNC-KNEEL man_down_critical <= 90 s | 0 | 0 | 0 | **20/20 [0.84-1.00]** (median 61.8 s, max 62.8 s) | 10/20 |
| apnea: SYNC / SYNC-KNEEL critical <= 90 s | - | 0 / 0 | - | **20/20, 20/20** | - |
| slow start: SYNC / SYNC-KNEEL confirmed | - | 0 / 0 | - | **20/20, 20/20** (latch recovered 40/40) | - |
| X-DISMISS (dismiss, then collapse) confirmed / critical | - | 17 / 0 | - | **20/20, 20/20** | - |

First-alert latency for D: SYNC median 7.6 s, p95 8.2 s; SYNC-KNEEL 7.7 / 8.1 s. First path:
SYNC impact 14, topple 6; SYNC-KNEEL impact 10, topple 10.

| Hard negatives (any alert before check-in) | A | B | **D** | E |
|---|---|---|---|---|
| A-SIT | 13/20 | 12/20 [0.39-0.78] | **12/20** [0.39-0.78] (impact path) | 0/20 |
| A-EQUIP | 3/20 | 2/20 | **2/20** (impact) | 0/20 |
| A-JUMP | 3/20 | 3/20 | **3/20** (impact) | 4/20 |
| A-LIE, A-NAP, A-LIEWORK, A-KNEEL, A-REBAR, A-JACK, A-RUN, A-CARRY, A-LADDER, A-HAMMER, A-DRILL, DEVICE-OFF, HAT-OFF, STRAP-SLIP | | | **0/20 each** [0.00-0.16] | |

T3 fired 0 times on the 340 held-out negative runs; no D alert on a negative came from the topple
or man-down path.

Falls and near-falls, D vs B: H-LAD 19 = 19, H-SCAF/H-ROOF/H-HOLE/H-EXC 20 = 20, L-SLIP/L-TRIP/
STRUCK-KD 20 = 20, STAIR 20 vs 19, NET 20 = 20, ARR 20 = 20 (0 cleared), FALL-GETUP 20 = 20,
SLIP-FRICTION 9 = 9; NF-STUMBLE, NF-RAIL, STRUCK-NF 0/20. SIM_SPEC bars on D: height 99/100, level
80/80, ARR 20/20, latency p95 8.2 s. Man-down on falls (bonus, no bar): e.g. NET 20/20, STAIR 20/20,
H-LAD 15/20, H-HOLE 14/20 critical within 90 s (some fallers move or get up).

Acceptance bars (`hub/eval/bars.py`): **all pass** (SYNC, SYNC-KNEEL >= 18/20 confirmed; critical
<= 90 s >= 19/20; ARR never cleared; apnea >= 19/20; latch always recovers; no fall more than 1/20
below B; A-SIT/A-EQUIP/A-JUMP no worse than B; other hard negatives <= 1/20; SIM_SPEC bars). The
"<= 1 false alarm per 8 h before check-in" SIM_SPEC bar is still unmet by the impact path (dev
shifts, section 5), as the doc expects.

## 6b. Re-run on the new simulator (frozen cf8c7456b67e, unchanged)

Simulator: working tree pinned by sha256 `f05639c99ccacc4635258cbc48aa5fecaf817fc790beb995fb3b071a3b8fd68a`
(LC_ALL=C-sorted `shasum -a 256` manifest of `thundersim/**/*.py` + `calibration.json`, 38 files; same as
realism's acceptance pin), checkout HEAD 86faf2e. Collapse: 0.4-0.8 s buckle + supported drop
(onset to impact median 1.26 s for SYNC, 1.05 s for SYNC-KNEEL; old model 2.64 s and 2.08 s);
slower trips; damped contacts. falldet is as it is now: its near-fall classifier gets 79/100 on
NF-STUMBLE on this simulator (was 89/100; every stumble still triggers Stage 1), and research-95 may
retrain it; every result below depends on that falldet.

### 6b.1 Dev (seeds 0-19, shifts 500000-500008), full plan as specified

Base suite, 35 scenarios x 20, arms A, B, C, D, D1, E; `uncalibrated` on 20 scenarios; all robustness
arms n = 20; dead slots 0-3; protobuf n = 10; 9 x 2 h shifts. 0 run errors. **All bars pass**.

| Scenario | A | B | E | **D** |
|---|---|---|---|---|
| SYNC confirmed / critical <= 90 s | 20 / 0 | 20 / 0 | 12 / 12 | **20/20 / 20/20** |
| SYNC-KNEEL | 18 / 0 | 19 / 0 | 12 / 12 | **20/20 / 20/20** |
| A-SIT / A-EQUIP / A-JUMP alerts | 12 / 2 / 1 | 10 / 1 / 1 | 0 / 0 / 1 | **10 / 1 / 1** (impact path) |
| 14 other hard negatives | 0 | 0 | A-LIE 17, A-NAP 20, A-KNEEL 4 | **0/20 each** |

Falls: D >= B everywhere (STAIR 20 vs 19, NET 20 vs 19, others equal); near-falls 0/20 confirmed;
ARR 0 cleared. Shifts (18.2 h): check-ins per 8 h A 5.73 [3.05-9.80], B 3.53 [1.52-6.95],
C 2.64 [0.97-5.76], **D 2.64 [0.97-5.76], 0 hub-path episodes** (2 man-down events escalate one
impact-path false alarm), E 3.08; T3 fired 0 times. (C and D have fewer check-ins than B because the
arbiter merges falldet confirmations that fall inside one open episode.)

Robustness, D (SYNC, SYNC-KNEEL; n = 20): jerks, apnea, damped, curl, restart (3.5 s), slow start,
50 Hz, X-DISMISS: 20/20 confirmed and critical; 20 Hz 20, 18; dead slots 0/2/3 20, 20; dead lumbar 19, 20;
uncalibrated 20, 20; apnea + uncalibrated 19-20 confirmed, 20/20 critical; protobuf 10/10.
**Regression vs the old simulator: data gaps placed at the impact.** 0.3 s gap: D 19/20 and
**10/20** (B 4, 4); 0.7 s: **9/20 and 9/20** (B 2, 2); 1.5 s: **0/20 and 0/20** (B 0, 0; E 12, 3, because
its collapse trigger needs no rotation). The faster descent now fits inside the gap, so neither T1
(impact) nor T3 (rotation) sees it, and T4 is never armed (misses had 0 T3 fires). This was a known
gap (T5 gap trigger, stretch) and is now much more likely; it is not an acceptance bar.

### 6b.2 Held-out block 2200000 (run once; seeds 2200000-2200019)

Ledger row written before the run (`SEED_LEDGER.md`). Runs: 35 scenarios (A-E), SYNC and SYNC-KNEEL
apnea and slow start (B, D), X-DISMISS (B, D). 0 errors. Nothing tuned after. **All bars pass.**

| Scenario | A | B | C | **D** | E |
|---|---|---|---|---|---|
| SYNC fall_confirmed | 20/20 | 20/20 | 20/20 | **20/20 [0.84-1.00]** | 13/20 [0.43-0.82] |
| SYNC critical <= 90 s | 0/20 | 0/20 | 0/20 | **20/20 [0.84-1.00]** (median 62.2 s, max 64.3) | 13/20 |
| SYNC-KNEEL fall_confirmed | 20/20 | 20/20 | 20/20 | **20/20 [0.84-1.00]** | 13/20 |
| SYNC-KNEEL critical <= 90 s | 0/20 | 0/20 | 0/20 | **20/20 [0.84-1.00]** (median 62.2 s, max 63.8) | 13/20 |
| apnea critical (SYNC, SYNC-KNEEL) | - | 0, 0 | - | **20/20, 20/20** | - |
| slow start confirmed | - | 0, 0 | - | **20/20, 20/20** (latch recovered 40/40) | - |
| X-DISMISS confirmed / critical | - | 20 / 0 | - | **20/20 / 20/20** | - |
| A-SIT any alert | 15/20 | 11/20 [0.34-0.74] | 11/20 | **11/20** [0.34-0.74] | 0/20 |
| A-EQUIP / A-JUMP | 1 / 0 | 0 / 0 | 0 / 0 | **0 / 0** | 0 / 0 |
| other 14 hard negatives | 0 | 0 | 0 | **0/20 each** [0.00-0.16] | A-LIE 16, A-NAP 20, A-KNEEL 6 |

First-alert latency D: SYNC 8.2 / 8.3 s (median / p95), SYNC-KNEEL 8.3 / 8.4 s. T3 fired 0 times
on 340 negative runs. Falls D vs B: H-LAD 18 = 18, STAIR 20 vs 18, all others equal; SLIP-FRICTION
7 = 7; near-falls 0/20 confirmed (NF-STUMBLE near_fall 15/20 in A, B, D; 20/20 in E); ARR 20/20,
0 cleared. SIM_SPEC on D: height 98/100, level 80/80, ARR 20/20, latency p95 8.2 s.

**Which path catches the collapse** (D; T1 read from falldet alone, since the arbiter merges
suspected events):

| | T1 Stage 1 fired | T3 topple fired | both | first confirmation impact / topple | falldet alone confirms |
|---|---|---|---|---|---|
| held-out SYNC | 20/20 | 20/20 | 20/20 | 14 / 6 | 20/20 |
| held-out SYNC-KNEEL | 20/20 | 20/20 | 20/20 | 20 / 0 | 20/20 |
| held-out X-DISMISS (2nd collapse) | 20/20 | 5/20 | 5/20 | 19 / 1 | 20/20 |
| dev SYNC | 20/20 | 20/20 | 20/20 | 14 / 6 | 20/20 |
| dev SYNC-KNEEL | 19/20 | 18/20 | 17/20 | 18 / 2 | 19/20 |

So on this simulator both paths see almost every collapse; the impact path usually confirms first
(C1 decides 8 s after the trigger; C3's fast rule needs 8 s from T3 + 2 s). The topple path is the
backup when the impact is missed (dead lumbar, 20 Hz, gap at the impact, slow start before recovery),
and T4 provides the escalation that falldet does not.

## 7. Thresholds: margins and +-10/20 % sweep (dev)

- **T3 rotation peak at the fire** (D, base + uncalibrated + jerks, threshold 3.0 rad/s):
  SYNC n = 60 fires, min 3.05, p10 3.30, median 4.32 (5 below 3.3, 11 below 3.6);
  SYNC-KNEEL n = 52, min 3.06, p10 3.50, median 4.41. The margin on positives is thin, as the doc says.
- **Negatives in simulation show no margin at all**: 0 T3 fires on 340 dev negative runs even at
  2.4 rad/s, because ThunderSim caps voluntary trunk rotation at 3.14 rad/s and its lie-downs are
  slow. This is a simulator artifact; the lab estimate is below.
- Sweep on dev (n = 10 per scenario, 13 scenarios, identical streams): `topple.rot_rad_s` 2.4 / 2.7 /
  3.3 / 3.6, `posture.down_deg` 36 / 40.5 / 49.5 / 54, `collapse.slow_still_frac` 0.56 / 0.84,
  `man_down.still_frac` 0.56 / 0.84, `posture.still_gyro_rad_s` 0.28 / 0.42: SYNC and SYNC-KNEEL
  stay 10/10 confirmed and 10/10 critical <= 90 s in every arm except `down_deg` 36
  (SYNC-KNEEL 9/10 T3-fired, 9/9 critical); the 11 negatives do not change (A-SIT 8/10 and
  A-EQUIP 1/10 are the impact path in all arms).
- (Margins and the sweep in this section were measured with 60323b2f32be; T3, C3 and stillness
  are unchanged in cf8c7456b67e, so they apply to the frozen config except for the lying-work timer.)
- Still fraction at the decision: C3 fast `topple_still_down` min 0.903-0.914 (threshold 0.90),
  C3 slow min 0.72-0.78 (0.70), T4 min 0.73 on NET (0.70), 1.00 on SYNC/SYNC-KNEEL.

- **Lab sweep of `topple.rot_rad_s`** (same rules, Erciyes/VTT, section 8):

  | rot_rad_s | 2.4 (-20 %) | 2.7 (-10 %) | **3.0** | 3.3 (+10 %) | 3.6 (+20 %) |
  |---|---|---|---|---|---|
  | 919 T3 fired (n = 92) | 92 | 92 | **92** | 92 | 90 |
  | 815 lying into bed: T3 fired / C3 confirmed (n = 93) | 47 / 27 | 34 / 16 | **15 / 7** | 11 / 5 | 3 / 1 |
  | 816 rising from bed: T3 fired (n = 94) | 15 | 12 | **7** | 5 | 3 |
  | T3 fires per hour, all ADL (10.9 h) | 7.3 | 4.5 | **2.1** | 1.6 | 0.6 |

  Lowering the gate is clearly worse on lab lie-downs; raising it to 3.3-3.6 rad/s cuts lab
  lie-down triggers 1.4-5x while the acted syncope keeps firing, but 5/60 (3.3) and 11/60 (3.6)
  of the simulated SYNC fires sit below those values. Not changed (no tuning on lab data); listed
  as a team decision.

## 8. Erciyes/VTT (read-only, `research/fall_model/data/processed/canonical_current.npz`)

No threshold was fitted to lab data (all from `config.json`), so leave-one-subject-out reduces
to scoring each held-out subject with the same rules; pooled over 17 Erciyes subjects (13 VTT
for ADL):

| Code | Activity | n | T3 fired | C3 confirmed | undecided at recording end |
|---|---|---|---|---|---|
| 919 | syncope (acted, like an impact fall) | 92 | 92 [0.96-1.00] | 60 [0.55-0.74] | 32 |
| 920 | syncope against a wall (ends seated) | 90 | **0 [0.00-0.04]** | 0 | 0 |
| 815 | lying down into bed (ADL) | 93 | **15 [0.10-0.25]** | 7 [0.04-0.15] | 8 |
| 816 | rising from bed (ADL) | 94 | 7 [0.04-0.15] | 0 [0.00-0.04] | 0 |
| 917 | rolling out of bed (fall, starts lying) | 95 | 5 [0.02-0.12] | 1 [0.00-0.06] | 4 |

T3 fires on all ADL: **23 in 10.9 h = 2.1 per hour** (bar <= 3/h). Limits: recordings end about
8 s after the event (C3's fast rule needs T3 + 8 s, hence "undecided"), acts are performed by
volunteers, data harmonised to ±2 g / 100 Hz, and the lumbar and thoracic anchors are synthetic
(only pelvis and upper thoracic are real), so the 3-anchor rules run on 2 real anchors.

## 9. Runtime

`MainModel.update_arrays` on 10-frame (100 ms) batches of a synthetic 527 s stream with eight
collapses: median 0.27 ms, **p99 11.2 ms**, max 28 ms, on this laptop while other jobs loaded it
(load average > 50); bar p99 <= 50 ms. CPU per hour of data on dev scenarios: falldet B 2.6 s,
hub D 9.7 s, baseline E 26.7 s. Not measured on the demo laptop; the 10-min "0 raw drops" and
loop-lag bars need the backend proposals and hardware.

## 10. Not covered or not verified

- New simulator (6b): a data gap at the impact now defeats both triggers far more often (0.3 s gap:
  SYNC-KNEEL 10/20; 0.7 s: 9/20; 1.5 s: 0/40). A T5 gap trigger (doc section 4) is the fix to discuss.
- Sections 7-9 (threshold margins and sweeps, lab evaluation, runtime) were measured before the
  simulator change; the lab evaluation does not use the simulator, the margins and sweep do.
- Results depend on falldet as it is now (near-fall 79/100 on NF-STUMBLE on the new simulator); a
  retrained falldet needs a new dev run and a new held-out block.
- Held-out shifts were never run; dev shifts are 18 h, not 120 h.

- Held-out covers only the runs listed in section 6; shifts are dev-only and 18 h, not 120 h.
- The frozen-config dev re-run of robustness arms was reduced (section 4.1); `damped`, `gap0.3`,
  apnea + uncalibrated, dead slots 0/2/3 and the protobuf path were last run with 60323b2f32be.

- Seated and wall-slide endings (lab 920: T3 0/90); collapse into a seat or cab; a brief faint
  with recovery within ~8 s; fast deliberate flops (lab 815: 16 % pass T3's gate, 7.5 % confirmed
  by C3 within 8 s); hard sits (impact path, unchanged).
- A 1.5 s data gap that covers the whole descent (6-7/20); the T5 gap trigger is a stretch item.
- Robustness arms are my harness's emulations (Builder subclass, batch perturbations), not
  simulator scenarios; the simulated unconscious body still has `sway = 0`.
- Backend wiring is a module plus proposals; the backend test suite was not run (no fastapi/bleak
  in the available venv). Nothing ran on hardware.
