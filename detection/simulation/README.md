# ThunderSim: fall and man-down simulator

ThunderSim tests fall and man-down models before real sensors are worn. It generates
physically consistent data for the 4-IMU spine strip on a simulated construction worker and
sends it through the same path real packets will take. It then scores any model against
falls, near-falls, construction hard negatives and full fall-free shifts, under both the
current and the target firmware.

> **Simulated data only.** Results show how a model behaves on these scenarios. They are a
> necessary, not sufficient, gate before real-world pilot data, and they are not validated
> performance. Do not make medical claims from them.

## Pipeline

```
scenario script ──> spine body model (4 rigid links, 1 kHz) ──> soft tissue / strap
     │  falls physics + site physics                                      │
     ▼                                                                    ▼
ground truth                          MPU-6050 emulator (DLPF, noise, bias, temperature,
(events, phases)                      int16, clipping, sequential reads, millis())
                                                                          │
     raw ImuStripData protobuf ──> BLE link (loss, outages, throughput) ──> backend
     app.protocol.decode() ──> canonicalize (neutral calibration) ──> MODEL ──> events
     ──> worker check-in ──> supervisor alerts ──> SIM_SPEC 5.3 metrics + 5.4 pass bars
```

The simulator imports the backend decoder (`backend/app/protocol.py`) unchanged. `--fast` skips
the protobuf step for long runs.

## Setup

```sh
cd detection/simulation
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python numpy scipy "protobuf==7.36.2" matplotlib
# Only for research-95's trained model (falldet), which needs this exact version:
uv pip install --python .venv/bin/python "scikit-learn==1.9.1" "joblib==1.6.0"
.venv/bin/python -m unittest discover -s tests    # 21 tests, about 15 s
```

## Commands

Run every command from `detection/simulation/` as `.venv/bin/python -m thundersim <command>`.

| Command | What it does |
|---|---|
| `list` | Scenario catalogue and device profiles. |
| `run H-LAD --seed 3 --model falldet --profile target --plot out.png` | One run: prints ground truth and model events, and draws spine snapshots plus signals. |
| `suite --models baseline,falldet --profiles target,current --runs 20` | Randomised suite. Writes `output/.../runs.jsonl`, `report.md` and `summary.json`. |
| `suite ... --dead-slot 0,3` | The same suite with one anchor dead (per-anchor degradation). |
| `suite ... --seed-base 1000000` | Held-out **test** seeds. Tune on the default dev seeds only. |
| `shifts --models falldet --count 10 --hours 8` | Fall-free day-in-the-life shifts: false alarms per 8 h and Stage-1 triggers per hour. |
| `tune --trials 40` | Random search over the baseline's thresholds and timers on cached dev frames. Writes `tuned_baseline.json`. |
| `export --out data/ --runs 5` | Labelled synthetic dataset: one `.npz` per run with frames and per-frame ground-truth phase. |
| `calibrate` | Fits the soft-tissue coupling against the Erciyes waist impact peaks and writes `calibration.json`. |
| `posture --runs 10` | Bad spine posture test: 13 posture scenarios, true against estimated segment angles, ergonomic bad-posture rules. Writes `posture_report.md` and figures. |
| `report output/a output/b` | Rebuilds one report from several output folders. |

To plug in a model, see [MODEL_API.md](MODEL_API.md). Built-in models:

- `baseline`: the report's two-stage rules plus the fixes the simulator suggested. `thundersim/baseline.py`.
- `falldet`: research-95's trained detector, loaded from `detection/models`.

## What is modelled, and where the numbers come from

Tags follow `thundersim/site.py`: **[src]** sourced, **[design]** starting value randomised per run,
**[verify]** sourced but unconfirmed. Values merged from `fall_model_sim_spec/SIM_SPEC.md` keep
that file's tags.

**Body and spine** (`spine.py`; from the spine-model research note)

- **Chain:** four rigid links (pelvis, lumbar, thoracic, upper thoracic), built with the note's recipe
  `p(i+1) = p(i) + L(i) · axis(i)`.
- **Frame and joint angles:** frame +X right, +Y up, +Z posterior. Joints use the ISB spine order
  mapped to intrinsic X-Z′-Y″.
- **Spine curve:** a cubic Hermite curve through the anchors, for display only.
- **Sensors:** each IMU sits 4 cm behind the spine line, so rotation produces lever-arm terms
  (ω×(ω×r) and α×r).
- **Joint limits:** soft physiological range-of-motion limits.
- **Movement:** voluntary trunk rate is capped at 180°/s. Every join between motions is C1-continuous,
  so no velocity step can appear as a fake impact.
- **Small background motion:**
  - breathing at 0.2–0.33 Hz, about ±0.01 g;
  - postural sway;
  - fidget.

  An unconscious worker still breathes; a strip on a table does not.
- **Soft tissue and strap:** a 2nd-order mount filter per anchor at 8–25 Hz with ζ 0.4–0.9 [design],
  lowered in winter clothing.

**Falls** (`falls.py`)

- **Loss of balance:** an inverted pendulum over the stance foot with knee collapse. The trunk lags
  the leg line.
- **Slips:** the foot slides out from under the body, then slows by kinetic friction.
- **Flight:** ballistic about the centre of mass, so the free-fall reading is ω²r, not 0 g.
  Free-fall time matches √(2h/g) (tested).
- **Body-first impact (Mode A):** effective mass 0.35 × body mass (20–40 kg) and pelvis stiffness
  39–55 kN/m [src: hip-protector consensus], in series with the surface. Same-level hip contact
  speeds come out around 3.3–4.0 m/s, against the consensus 3.4 ± 0.5 (tested).
- **Feet-first landing (Mode B):** peak ≈ (h/d + 1) g, often followed by a topple.
- **Harness arrest:** OSHA limits [src: 1926.502(d)(16)] of at most 1.8 m free fall, a stop within
  1.07 m and at most 8 kN (tested). The worker then hangs 15–35° forward of upright, with pendulum
  swing.
- **Safety nets:** at most 9.1 m below the work level [src: 1926.502(c)(1)].
- **Other falls:** stairs (several impacts), roof and excavation slides, fainting collapse with no
  free fall, near-falls, struck-by. After a fall the worker is unconscious, still then moving,
  struggling, or gets up.

**Construction site** (`site.py`)

- **Surfaces:** concrete, steel deck, plywood deck, scaffold plank, rebar mat, gravel, packed soil,
  mud, crash mat and safety net, each with its own stiffness, damping, boot friction, heel-strike
  transient and unevenness [design].
- **Conditions:** dry, dusty, wet, muddy, oily or icy.
- **Slips:** happen when the slide distance v²/(2μg) exceeds 0.15 m [design].
- **Fall-protection triggers:** OSHA 1.8 m [src] and Alberta 3 m [src via SIM_SPEC].
- **Ladders:** set at 1:4 [src: 1926.1053(b)(5)(i)].
- **Clothing, temperature and radio:**
  - winter clothing;
  - temperature-driven gyro drift;
  - BLE links for open site, steel frame, heavy equipment, and a throughput-constrained case that
    shows 100 Hz raw streaming overflowing a 6 kB/s link.

**Device** (`device.py`): MPU-6050 values from the register map and product spec (LSB per g,
DLPF table, noise densities, offsets).

| Profile | Rate | Accel range | Gyro range | DLPF | Notes |
|---|---|---|---|---|---|
| `current` | 0.5 Hz | ±2 g | ±250 °/s | 10 Hz | Firmware today, from `ImuManager.cpp` and `SystemConfig.h` |
| `current_100hz` | 100 Hz | ±2 g | ±250 °/s | 10 Hz | Firmware today, with only the sampling period fixed |
| `target` | 100 Hz | ±16 g | ±2000 °/s | 44 Hz | The report's recommendation |
| `target_8g` | 100 Hz | ±8 g | ±1000 °/s | 44 Hz | Alternative from the preprocessing notes |

Other device effects:
- sequential mux reads 0.8–1.5 ms apart, sharing one `millis()` timestamp;
- calibrated bias residuals;
- int16 quantisation and clipping.

## Scenarios

There are 35 scenarios in six groups; `list` prints all of them.

| Group | Scenarios |
|---|---|
| Falls from height | H-LAD, H-SCAF, H-ROOF, H-HOLE, H-EXC |
| Same-level falls | L-SLIP, L-TRIP, STAIR, STRUCK-KD |
| Arrested falls | ARR (harness), NET |
| Collapse | SYNC, SYNC-KNEEL (both report-only) |
| Near-falls | NF-STUMBLE, NF-RAIL, STRUCK-NF, SLIP-FRICTION (the outcome follows site friction) |
| Construction hard negatives | hammering, overhead drilling, jackhammer, equipment cab, kneeling and crawling, rebar tying, ladder, jump-down, hard sit, lying down, declared lying work, lunch nap, carrying, running, hard hat knocked off, strap slip, strip left on a table |

Shifts mix the activities in SIM_SPEC 5.2's proportions, plus about 10 jump-downs and 5 runs per
8 h.

## Calibration finding

The `calibrate` command compares simulated standing falls onto mats with Erciyes waist falls, like
for like (12 Hz band-limited, sampled at 25 Hz).

| Source | Median peak | Interquartile range |
|---|---|---|
| Erciyes waist falls (25 Hz) | 3.95 g | 3.3–4.6 g |
| Simulated falls at the consensus hip speed | 6.5–7 g | |
| Simulated falls, all hands-first | about 5.5 g | |

- **Soft tissue can't close the gap**, because the mount filter is transparent below 12.5 Hz.
- **Interpretation:** the volunteers made controlled falls onto thick mats.
- **Decision:** site falls are not scaled down to match them. Models trained on Erciyes will see
  larger impacts on site. Details are in `calibration.json`.

## Limits

- **Body model:** kinematic, with physics only in the fall phases. There are no arms, legs or head
  segments. Impacts use single-degree-of-freedom contact.
- **Design values:** many are untuned against real recordings. Measure the team's own MPU-6050s
  (bias, noise, read skew) and strap mounting, then update `DeviceErrors` and `Worker`.
- **Real falls:** real falls of workers are not simulated in any validated sense. The point is to
  exercise the pipeline, thresholds, timers and failure modes before people are involved, and to
  find where a model breaks.
- **Sources:** the research report, `research_notes/...`, and
  `fall_model_sim_spec/SIM_SPEC.md` (research-95), whose values were merged where they were better
  sourced.
