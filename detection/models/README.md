# ThunderClock 4-IMU spine-strip fall detector (model card)

A two-stage man-down detector for the T'Work It / ThunderClock spine strip (NatHacks 2026).
**Stage 1** is a threshold trigger on the pelvis and lumbar IMUs, written so it ports directly to
ESP32 C++. **Stage 2** is a small classical-ML confirmer over an 8 s post-trigger window that uses
all four anchors. Outputs: `normal`, `near_fall` (fell or stumbled, then recovered) and
`fall_confirmed`, each with a confidence.

> **Read this first.** Every fall in the training and test data was **simulated by young
> volunteers onto mats in a lab**. Real-world performance is expected to be much lower. On 143
> real falls of older adults, the best published feature model reached **F 64.6 % at
> 0.56 false alarms/h**, and a threshold method fell to 30 % sensitivity (Palmerini et al. 2020).
> Two of the four anchors in our data are **synthetic** (see below). This is a prototype safety
> aid with **no medical or diagnostic claim**.

---

## 1. Quick start

```bash
cd detection/models
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # or pip
.venv/bin/python run_demo.py                      # replay all fixtures (target firmware)
.venv/bin/python run_demo.py --profile current    # same, with today's firmware settings
.venv/bin/python tests/test_falldet.py            # self-checks (stream == batch, frames, parsing)
```

```python
from falldet import detect, FallDetectorStream

# one recorded window: acc, gyro are (N, 4, 3) numpy arrays at 100 Hz
result = detect((acc, gyro))            # or detect({"acc": acc, "gyro": gyro})
result["state"], result["confidence"], result["stage1_triggered"], result["features"]

# live: one 4-IMU sample at a time
stream = FallDetectorStream()           # profile="current" for today's firmware settings
for sample in samples:                  # telemetry-style dicts, see section 2
    for event in stream.push(sample):   # [] most of the time
        print(event["type"], event["state"])
stream.state                            # 'normal' | 'suspected' | 'near_fall' | 'fall_confirmed'
stream.flush()                          # end of data: decide any pending trigger
```

For **real-world-shaped streams** (irregular device timestamps, lost packets, NaN, missing slots,
low sample rates), use `push_batch`. It takes ThunderSim `FrameBatch` shapes:

```python
stream = FallDetectorStream(profile="target", backend="auto")   # backend "logreg_json" = numpy only
events = stream.push_batch(t_s, acc, gyro, slot_mask=None, sat_mask=None, work_mode="normal")
```
* Frames are interpolated onto a 100 Hz grid. Gaps over 0.1 s are not filled and reset the Stage-1 timing.
* NaN or missing slots are filled from the nearest anchor.
* Below 40 Hz you get one `insufficient_rate` diagnostic and no detections.
* `falldet.sim_adapter:SpineFallModel` wraps the same thing in ThunderSim's `FallModel` protocol.
* See HANDOFF.md ("ThunderSim integration"). `tests/test_sim_adapter.py` exercises it at 100 Hz and at 0.5 Hz.

**Dependencies:** the default GBT model needs `scikit-learn==1.9.1` + `joblib`. `backend="logreg_json"`
needs numpy only (LOSO F1 0.976 vs 0.981), and `"auto"` falls back to it if the GBT cannot be loaded.

## 2. Inputs

| Item | Value |
|---|---|
| Rate | **100 Hz** uniform for `detect()` / `FallDetectorStream` (firmware `kAcquisitionPeriodMs = 10`). `push_batch()` / `SpineFallModel` accept any timestamps and resample. Below 40 Hz they return `normal` (0.5 Hz cannot see a fall). |
| Anchor order | `pelvis, lumbar, thoracic, upper_thoracic` = `sensor_id` 1, 2, 3, 4 (ORIENTATION.md) |
| Accel | **m/s²**, specific force (reads ≈ +9.81 on +Y when upright and still) |
| Gyro | **rad/s** |
| Frame | segment frame, **+X wearer-right, +Y up, +Z posterior** (ORIENTATION.md, "proposed"). Sensor-to-segment mounting is already corrected, so in the neutral upright pose each anchor's +Y points up. |
| Device setting the model expects | target: ±16 g, ±2000 °/s, DLPF 44 Hz. A separate model ships for today's ±2 g, ±250 °/s, 10 Hz DLPF. |

The stream accepts any of these per sample:

```json
{"time": 1234, "data": [{"sensor_id": 1, "accel": {"x": 0.1, "y": 9.8, "z": 0.2}, "gyro": {"x": 0, "y": 0, "z": 0}}, "... x4"]}
```
This is `ImuStripData` from `firmware/proto/telemetry.proto` as JSON. `time` is device ms. If
`sensor_id` is missing, list order is used. The stream also accepts `{"t_ms": 1234, "acc": [[x,y,z] x4], "gyro": [[x,y,z] x4]}`
or a tuple `(acc(4,3), gyro(4,3))`. Missing `time` means samples are assumed 10 ms apart.

Heading (yaw) is never used. Stage 2 only needs each segment's tilt from vertical, which the
detector estimates itself from accel + gyro (`falldet/tilt.py`). It does not consume quaternions,
because the planned quaternion-only packet carries no linear acceleration and cannot express impact.

## 3. Outputs

`detect()` returns

```python
{"state": "normal" | "near_fall" | "fall_confirmed",
 "confidence": float,                 # Stage-2 probability of `state` (1.0 if Stage 1 never fired)
 "probabilities": {"normal": p, "near_fall": p, "fall_confirmed": p},
 "stage1_triggered": bool, "trigger_index": int | None,
 "rule": "model" | "free_fall_no_gait" | "long_free_fall" | "lying_work_suppressed",
 "features": {28 named features} | None, "aux": {"free_fall_s", "walking_after"} | None,
 "note": "... (optional)"}
```

`FallDetectorStream.push()` emits:
* `{"type": "stage1_trigger", "state": "suspected", "t_ms", "sample", "svm_g", "rot_rad_s"}` immediately
  on a trigger. This maps onto today's telemetry `fall_state = "suspected"`.
* `{"type": "stage2_result", "state", "confidence", "probabilities", "t_ms" (trigger time), "decided_at_ms", "features"}`
  **8 s after the trigger**, once the aftermath has been observed. Telemetry has only `normal` and
  `suspected` today. `near_fall` and `fall_confirmed` would need to be added there (TELEMETRY_PROTOCOL.md TODO).

**Confidence caveat:** on this simulated data the classes separate almost perfectly, so the
gradient-boosted model outputs probabilities close to 0 or 1. Treat confidence as "how typical this
event looks of the training data". It is not a calibrated real-world probability.

## 4. How it works

### Stage 1: trigger (`falldet/stage1.py`, constants in `models/stage1_thresholds.json`)
Per sample: `svm_g = mean(|a_pelvis|, |a_lumbar|) / g` and `rot = max(|ω_pelvis|, |ω_lumbar|)`.

| Condition | Shipped value (target) | Current-firmware value |
|---|---|---|
| FREEFALL: `svm_g` below | 0.7 g for ≥ 100 ms | 0.6 g for ≥ 100 ms |
| IMPACT: `svm_g` above | **2.25 g** | 1.9 g (±2 g clips each axis) |
| ROTATION: `rot` above | 3.0 rad/s | 3.0 rad/s |
| Fire when | IMPACT plus ≥ 1 other condition, all within 1000 ms | same |
| Refractory | 5000 ms | same |

The report proposed requiring all three conditions. On this data a strict 3-of-3 rule could not
reach the 95 % recall target at an acceptable trigger rate, so the grid search picked 2-of-3 with
impact mandatory. The thresholds were chosen by grid search, inside each LOSO training fold, as the
lowest-trigger-rate setting that catches ≥ 95 % of fall and near-fall recordings. All 30 folds chose
the same target setting. The rotation threshold is a population constant for now. The report
recommends each worker's own 99.9th percentile once personal data exist.

### Stage 2: confirmer (`falldet/features.py`, `models/stage2_model.joblib`)
Window: 2 s before to 8 s after the trigger. 28 features:
* **Multiphase** (Palmerini-style, pelvis+lumbar): free-fall dip, impact peak, jerk, peak trunk
  rotation, activity before the event, acceleration SD after impact (early and late), periodicity
  after impact (did walking resume?), mean angular rate after impact.
* **Four-anchor**: per-anchor median inclination 2–8 s after the trigger and its change from before,
  number of anchors lying (> 60°), cross-anchor spread, upper-thoracic minus pelvis inclination,
  fraction of time all anchors are still, and the longest still run.

The classifier was chosen on LOSO results: **gradient-boosted trees** (`HistGradientBoostingClassifier`,
depth 3, 200 iterations, balanced class weights). Logistic regression and an RBF SVM came within
the confidence intervals (table below). `models/stage2_logreg_coefficients.json` is the portable
logistic-regression version, plain arithmetic `softmax(W·((x−mean)/scale)+b)`, for a C++ or JS port.
`falldet.features.rule_based_stage2` is the report's hand-written rule ("≥ 3 anchors lying and
≥ 4 s still"), kept as a baseline and fallback.

### Post-rules (`falldet.detector.apply_rules`, constants in `models/post_rules.json`)
The training data contains only falls that end on the floor. Two situations it never showed are
handled by hand-written rules applied after the model, in `detect()` and the stream alike:
* **Harness arrest / fall from height.** After an arrested fall the worker hangs near-upright,
  still or struggling, not lying. A free fall (svm < 0.5 g) of **≥ 0.35 s** (about a 0.6 m drop)
  followed by **no gait** in the next 2–8 s gives `fall_confirmed`; **≥ 0.6 s** (about 1.8 m) always
  confirms. Public falls from standing never exceed 0.33 s, so the rule changes no fall or near-fall
  result. It does turn 2 VTT jump-downs into false alarms (0.73 → **0.92 false alarms/h** overall).
  How it behaves on a synthetic arrest (0.5 s free fall, 6 g arrest, hanging 25° forward):
  * **still:** the model alone already says `fall_confirmed`. That is extrapolation; do not rely on it.
  * **struggling:** the model says `near_fall` and the rule confirms.
  * **drops under about 0.6 m:** expect `near_fall` (a SIM_SPEC partial pass) or `normal`.
* **Lying work.** With `work_mode="lying_work"`, a `fall_confirmed` is downgraded to `normal` if
  ≥ 3 anchors were already lying (> 60°) before the trigger.

The LOSO table below is for the model alone. With the post-rules, only the false-alarm column moves.

## 5. Data

| Dataset | What we used | Licence | Access |
|---|---|---|---|
| **Erciyes "Simulated Falls and Daily Living Activities"** (UCI 455) | 17 volunteers, 20 fall + 16 ADL types, ~5 repeats; waist + chest Xsens MTw, **25 Hz**; 3,303 recordings | **CC BY 4.0** | direct download, no login ([UCI](https://archive.ics.uci.edu/dataset/455/simulated+falls+and+daily+living+activities+data+set)) |
| **VTT-ConIot** | 13 users × 16 construction activities × 1 min; hip ("trousers") + back IMUs, 100 Hz; 207 recordings, 3.3 h | **CC BY 4.0** | Zenodo, no login ([record 4683703](https://zenodo.org/records/4683703)) |

Citations: Özdemir A.T., Barshan B. *Detecting Falls with Wearable Sensors Using Machine Learning
Techniques.* Sensors 2014, 14, 10691–10708. Mäkela S.-M. et al. *Introducing VTT-ConIot: A Realistic
Dataset for Activity Recognition of Construction Workers Using IMU Devices.* Sustainability 2022,
14(1), 220. The fixture files are derived from these datasets and carry the same CC BY 4.0
attribution in their headers.

We did not use KFall, FallAllD or SisFall because they need registration or a login, or their
mirror terms are unclear. The two extra no-login Zenodo sets we checked (TST-modified, ASSIST-IoT)
have no trunk IMU.

**Labels.** `fall_confirmed` = the 16 Erciyes falls that end on the floor. `near_fall` = the falls
with recovery (905, 906, 914, 916) plus *stumble with recovery* (808). `normal` = all other ADLs and
all VTT activities. Recording counts: 1,464 fall_confirmed, 454 near_fall, 1,385 Erciyes ADL
(7.6 h) and 207 VTT (3.3 h).

### What is real and what is synthetic in the four anchors

| Anchor | Source | Status |
|---|---|---|
| pelvis | Erciyes **waist** / VTT **hip ("trousers")** | **REAL** |
| lumbar | slerp pelvis→upper thoracic at f = 0.30, rigid-trunk accel | **SYNTHETIC** |
| thoracic | slerp pelvis→upper thoracic at f = 0.65, rigid-trunk accel | **SYNTHETIC** |
| upper thoracic | Erciyes **chest** (sternum) + lever arm to the spine / VTT **back** | **REAL** |

Harmonisation, all source → device (`scripts/harmonize.py`):
1. **Units**: SI (VTT accel is stored in half-g units; gyro °/s → rad/s).
2. **Sensor-to-segment calibration, from the data.** +Y comes from the standing gravity direction;
   +Z (posterior) is the direction gravity tilts towards during forward bends; +X = Y×Z. The VTT
   hip sensor's bend direction is ambiguous because squats tilt the pelvis either way, so its heading
   comes from the magnetometers. Check on held-out end postures: prone → +Z up, supine → −Z,
   right side → −X, left side → +X at both real anchors (|component| ≥ 0.90).
3. **Resample** to 100 Hz. Erciyes is 25 Hz upsampled ×4, so its **true bandwidth is only 12.5 Hz**.
4. **Lever arm**: sternum → upper-thoracic spine, `a + α×r + ω×(ω×r)`, r = (0, 0.08, 0.20) m (an estimate).
5. **Fusion**: one 6-axis Mahony filter for every source gives a shared-heading orientation per real anchor.
6. **Synthesis**: orientation `q_p ⊗ slerp(I, q_p⁻¹q_ut, f)`. World-frame acceleration is interpolated
   linearly, which is exact for a rigid trunk because rigid-body acceleration is affine in position.
   World-frame gyro is interpolated linearly too (approximate). The results are then rotated into
   the synthetic segment frame.
7. **MPU-6050 emulation** per profile: residual bias, datasheet white noise, clip to full scale,
   causal 2nd-order low-pass at the DLPF corner, clip, 16-bit quantisation.

**Dataset problem found:** the Erciyes `Information.pdf` lists sensor 340527 as the chest and
340539 as the right thigh. In all 17 subjects the data behave the other way round. In sit-chair,
340527 ends about 70° from vertical (a horizontal thigh) while 340539 stays upright. In a 90° bend,
340539 tilts about 90° (the trunk). We use **340539 as the chest**. Head and ankle are swapped the
same way; we do not use them.

## 6. Evaluation

Protocol: **leave-one-subject-out** over 30 folds (17 Erciyes volunteers and 13 VTT workers).
Stage-1 thresholds are re-tuned inside every training fold (nested). One prediction is made per
recording: `fall_confirmed` if any trigger in it is confirmed. Sensitivity, specificity and F1 use
the 3,303 Erciyes recordings, with positive = `fall_confirmed`. **False alarms/h** = Stage-2
`fall_confirmed` decisions inside `normal` recordings ÷ their actual duration (10.9 h: 7.6 h
Erciyes ADL + 3.3 h VTT construction). Brackets are 95 % Wilson intervals.

| Firmware config | Model | Sensitivity | Specificity | Precision | F1 | 3-class macro-F1 | False alarms/h (all / lab ADL / construction) | Stage-1 recall (falls) | Stage-1 triggers/h on ADL |
|---|---|---|---|---|---|---|---|---|---|
| **Target** (±16 g, ±2000 °/s, 44 Hz) | **GBT (shipped)** | **0.969** [0.959–0.977] | **0.995** [0.991–0.997] | 0.994 | **0.981** | 0.951 | **0.73** / 1.05 / 0.00 | 0.972 | 22.6 |
| Target | SVM | 0.966 | 0.995 | 0.994 | 0.980 | 0.949 | 0.64 / 0.92 / 0.00 | 0.972 | 22.6 |
| Target | Logistic reg. | 0.959 | 0.995 | 0.993 | 0.976 | 0.945 | 0.73 / 1.05 / 0.00 | 0.972 | 22.6 |
| Target | Rule (report) | 0.764 | 0.997 | 0.996 | 0.864 | 0.816 | 0.46 / 0.65 / 0.00 | 0.972 | 22.6 |
| **Current** (±2 g, ±250 °/s, 10 Hz), **retrained** | **GBT (shipped `_current`)** | 0.966 [0.955–0.974] | 0.995 | 0.994 | 0.980 | 0.952 | 0.64 / 0.92 / 0.00 | 0.968 | 26.8 |
| **Current, but target-trained model and thresholds** | GBT | **0.738** | 0.999 | 0.998 | 0.849 | 0.798 | 0.09 | **0.745** | 2.8 |

**What the firmware comparison does and does not show.**
* Using a model built for one configuration on data from the other **costs about 23 points of
  sensitivity**: Stage 1 misses a quarter of the falls because ±2 g clips the impact. Median
  pelvis/lumbar impact falls from 3.85 g to 2.48 g, and 98 % of fall recordings clip.
* After **retraining** on ±2 g data, the scores barely change. This is **not** evidence that ±2 g is
  fine. (a) Erciyes is already band-limited to 12.5 Hz, so this data cannot show what a 10 Hz DLPF
  removes from genuine 100 Hz impact transients. (b) Stage 2 leans on posture and stillness after
  the fall, which survive clipping; the report notes the same for Palmerini's ±2 g real falls.
  (c) Our emulation keeps 100 Hz sampling. **Today's firmware samples at 0.5 Hz**
  (`kAcquisitionPeriodMs = 2000`), which would miss falls entirely. Data recorded at ±2 g can never
  be "upgraded" later, so the reconfiguration still matters.

**Per fall type** (target, shipped GBT, LOSO; full table in `reports/per_type_target.csv`):
* Falls ending on the floor are confirmed in 93–100 % of recordings. Exceptions: rolling out of bed
  93 %, syncope 95 %, and **syncope sliding down a wall 74 %** (slow, little impact, so Stage 1 often
  stays silent).
* Falls with recovery (905, 906, 914, 916) are labelled `near_fall` in 98–100 %.
  **Stumble with recovery (808) is the weak spot**: Stage 1 fires on only 53 % and 39 % are flagged.
  Stumbles without a hard impact need a jerk- or gyro-based trigger.
* All 8 lab false alarms came from lying down on a bed (5 of 93; lying + still looks like a fall)
  and sitting hard onto a sofa (3 of 94). **Construction (VTT): 0 false `fall_confirmed`** in 3.3 h,
  although Stage 1 fires on every jumping-down and on 54–62 % of stairs recordings. Stage 2 rejects
  them because the worker stays upright and keeps moving.

**Ablation: do four anchors help?** (`reports/ablation_target.json`, GBT, LOSO)

| Features | F1 | FA/h |
|---|---|---|
| all 28 (four anchors) | 0.981 | 0.73 |
| real anchors only (no lumbar, no thoracic, no cross-anchor) | 0.981 | 0.73 |
| **pelvis only** (multiphase + pelvis inclination) | **0.982** | **0.64** |
| multiphase only (no posture) | 0.968 | 0.73 |
| posture only (no multiphase) | 0.923 | 0.82 |

On this data the four-anchor features add **nothing measurable**. That is expected: the synthetic
anchors are functions of the two real ones, and lab falls rotate the whole trunk. The strip's
proposed advantages remain **untested hypotheses**: "picking up a tool bends the thorax over an
upright pelvis" and "a hard hat knocked off produces impact without trunk motion". Only real 4-IMU
recordings can test them. The shipped model keeps all 28 features so it is ready for that data.

## 7. Limitations (please repeat these in any pitch)

* Simulated falls by young, healthy volunteers onto mats. Expect much lower performance on real
  falls (FARSEEING: best feature model F 64.6 % at 0.56 FA/h). No falls from height, no ladders, no
  older or impaired workers.
* Harness arrests and lying work are handled by **untested hand-written rules** (section 4), not
  learned. Short arrested drops (< 0.35 s free fall) will be missed, and a jump-down followed by
  standing still will be confirmed.
* Lab ADL plus only 3.3 h of construction activity. 0.73 FA/h (0.92 with the post-rules) is about **6–7 false confirmations per
  8 h shift**, which is too many for a live alarm without a worker check-in (hat double-tap or button).
* Erciyes is 25 Hz: impact peaks and jerk are under-sampled. On real 100 Hz data the Stage-1 impact
  threshold will need re-tuning upwards. Re-tune on team recordings before trusting it.
* Lumbar and thoracic anchors are synthetic; the lever arm and anchor fractions are estimates. The
  VTT hip heading from the magnetometers was poorly determined for users V05 and V06 (agreement 0.54
  and 0.20). VTT gyros saturate at ±250 °/s in the source data, even in the "target" profile.
* Model selection used the same LOSO results it reports (mild optimism). LR, SVM and GBT are
  statistically indistinguishable here.
* The shipped model was trained on all subjects, so the fixtures were part of its training data.
  The LOSO numbers above are the honest estimate.
* Stage 1 fires about 23 times per hour of (lab and construction) activity. That is fine for an
  on-device trigger plus an event buffer, but each trigger costs a BLE event.
* Not a medical device; no claim about stroke, syncope, cardiac events or any diagnosis.

## 8. Files

```
falldet/            the detector package (numpy; + scikit-learn 1.9.1 + joblib for the GBT backend)
  constants.py      anchors, units, frame, device profiles
  tilt.py           heading-free tilt estimator (complementary filter)
  stage1.py         ESP32-portable trigger state machine
  features.py       Stage-2 features + rule baseline
  detector.py       detect(), FallDetectorStream, post-rules, fixture helpers
  resample.py       irregular timestamps / gaps / NaN / missing slots -> clean 100 Hz (push_batch)
  sim_adapter.py    SpineFallModel: push_batch behind ThunderSim's FallModel protocol
models/             stage1_thresholds[_current].json, stage2_model[_current].joblib, post_rules.json,
                    stage2_logreg_coefficients[_current].json
fixtures/{target,current}/   13 recordings each (.jsonl + .csv) + index.json with expected outputs
run_demo.py         replay fixtures through the stream
tests/test_falldet.py, tests/test_sim_adapter.py (FrameBatch smoke test, 100 Hz and 0.5 Hz)
scripts/            load_raw.py -> harmonize.py -> train_eval.py -> make_fixtures.py; ablation.py
reports/            metrics_*.json, per_type_*.csv, stage1_grid_*.csv, loso_predictions_*.csv, ablation
data/raw/           downloaded datasets (untrusted; not part of the hand-off)
```

Rebuild everything (about 15 minutes on a laptop; the downloaded and extracted datasets take about 9 GB):

```bash
.venv/bin/python -I scripts/load_raw.py && .venv/bin/python -I scripts/harmonize.py
.venv/bin/python -I scripts/train_eval.py && .venv/bin/python -I scripts/make_fixtures.py
```
