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
.venv/bin/python -m unittest discover -s tests    # 118 tests, about 2 min
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
| `motionval --out output/motionval` | Motion validation: the simulated trunk against open mocap clips, the team's own vest recordings and literature ranges, per activity. Writes `MOTION_VALIDATION.md`, `metrics.json` and figures. See below. |
| `report output/a output/b` | Rebuilds one report from several output folders. |
| `viz` | 3D viewer at http://127.0.0.1:8765: replays any scenario with the worker, the strip and the site. See below. |

To plug in a model, see [MODEL_API.md](MODEL_API.md). Built-in models:

- `baseline`: the report's two-stage rules plus the fixes the simulator suggested. `thundersim/baseline.py`.
- `falldet`: research-95's trained detector, loaded from `detection/models`.

## 3D viewer

```sh
.venv/bin/python -m thundersim viz              # opens http://127.0.0.1:8765 (add --no-browser, --port N)
```

It runs locally. It needs no extra Python packages, no internet and no build step: three.js 0.160 (MIT) is
vendored in `thundersim/viz/static/vendor/`. Pick a scenario on the left. The server runs it through the same
chain as `run`: body, MPU-6050 emulator, BLE link, then the selected model and the score. The page then
replays the run.

- **Scene per scenario.** Each scenario gets the physical setup it implies, placed where the simulated worker
  actually was:
  - **Falls from height:** a ladder against a wall; a scaffold with an open edge just past the worker's feet;
    a roof slope at the simulated slide angle; a floor opening; a trench.
  - **Arrested falls:** a harness anchor and lanyard (a simulated rope); a safety net (a simulated cloth).
  - **Same-level falls:** stairs (0.18 m rise); a trip hazard or slick patch matching the site condition;
    a beam that a mobile crane lifts off a timber crib and swings into the worker at the struck-by event.
  - **Hard negatives:** a wheel loader that drives the simulated path, with a suspension seat; a carried
    stepladder; a step-up platform with a short ladder; buckets, a rebar mat, a formwork ceiling, a
    jackhammer, sacks, a creeper board; a hard hat knocked off on its cable; a table for the removed strip.
  - **Fitted props:** where the simulated path fixes the geometry (stairs, roof slope, platform, bucket, deck
    edge, crane strike point), the prop is fitted to the drawn body, so the body neither floats above it nor
    sinks into it.
  - **Default seed:** the scenario's *test scene*, the first dev seed (0–19) whose run fits one consistent
    scene. Any seed can be typed in, including test seeds; a seed that does not fit says so in its notes.
- **Human model.** The vest-wearing torso follows the simulated 4-link spine exactly; it is swept along the
  same Hermite curve as `spine.spine_curve`. Legs, arms and head are **illustrative**: posed from what the
  scenario script did, then simulated in falls (see Physics):
  - feet land on the simulated heel strikes, roll over heel and toe, climb ladder rungs and never jump
    between poses;
  - kneeling, sitting, slipping, collapsing, lying, flight and hanging poses follow the scenario;
  - in a lift or put-down the feet stay planted with the heels down while the knees bend forward, and the
    hands sit on the load at the simulated grip point (events `lift_grasp`, `lift_off`, `put_down_touch`);
  - hammer strikes are synced to the simulated impulse times.
- **Mocap limbs** (`thundersim/viz/mocap.py`, **off by default**). Real motion-capture clips from
  `motionlib/` replace some procedural limb targets. The spine stays the simulated one, and the signals,
  spine, segment and chip orientations, phases and the worn flag are byte-identical with mocap on or off
  (`motionlib/tools/check_invariants.py`).
  - **Walking and carrying (legs):** each clip is time-warped so that its heel strikes land on the simulated
    ones. The rig still plans and plants the feet, so they do not skate. The clip sets the toe-off time, the
    swing path, the heel-strike and heel-rise roll and the knee direction. A step lifts early if the
    simulated hip height would overstretch the leg or extend the thigh past 26°.
  - **Walking (arms):** the clip's arm swing in the simulated upper-thoracic frame, scaled with walking
    speed. While carrying, the arms hold the load as before.
  - **Hammering at waist height (arms):** the clip's strokes are warped onto the simulated impulse times.
    The hand reaches the impact point at each impulse. The open hammer clips (CMU subject 62) are light taps
    of about 5 cm, much smaller than the procedural swing.
  - **Kneeling floor work (arms):** NIOSH roofing trials, played near real speed with their wrist events on
    the work impulses. The hands reach toward the work surface only as far as the arm allows: the simulated
    kneeling shoulder is about 0.95 m above the floor, about 0.2 m higher than the roofers'. The hands
    therefore stay 0.35–0.65 m above the floor.
  - **Procedural:** lifts and put-downs (hands on the simulated grip point), ladders, running, falls and
    everything else, and any activity without a listed clip.
  - **Turning it on:** the Mocap limbs button, M, `mocap=1` in the page URL, or
    `THUNDERSIM_MOCAP=1 .venv/bin/python -m thundersim viz` for a server-wide default. The notes panel
    lists the clips used and their credits.
  - **Clips and gates:** clips come from `motionlib/manifest.json` with a sha256 each. They are gated on the
    real payload by `motionlib/tools/payload_gates.py`: bones, penetration, skate, pops, joint limits,
    sync and reach. A clip becomes on by default only after a person reviews it (`reviewed_by` on its
    card). See `motionlib/README.md`.
- **Contact pass** (`thundersim/viz/contact.py`, **off by default**). A whole-body contact and balance pass
  between the rig (procedural or mocap limbs) and the physics ragdoll. It moves only limbs and props, never the
  simulated spine: signals, spine, segment and chip orientations, phases and the worn flag are byte-identical with
  it on or off (`motionlib/tools/check_invariants.py --switch contact`), and with it off the whole payload is
  unchanged.
  - **Feet:** each planted foot keeps the sole point it landed on. Where the rig would drag a foot, or the hip
    pulls it out of reach, the foot lifts earlier or takes a short step. Before each lift and put-down the feet
    step to where the simulator's lift geometry puts the ankles (`motions.lift_root`). A trailing leg rolls
    onto the ball, so the thigh stays within the hip limit. Feet step one after the other instead of sliding
    when the worker kneels down or steps onto a ladder. Kneeling feet lie with the toes on the floor. A swing
    the pass redraws keeps its lowest point 1.5 cm above the floor in mid-swing (as the mocap swings do), so a
    foot rolled far onto the ball does not drag its toe.
  - **Hands on held objects:** fixed handholds on the sack's sides, on the ladder rails (hand over hand, one
    hand at a time) and on the stepladder's upper rail. The object moves rigidly with the hands. The sack's
    handholds are at the simulated grip height of each end: the lift's through the rise, then one hand after
    the other slides to the put-down's grip once upright. A held sack clears the drawn thighs, shanks and
    torso by the arms swinging forward about the shoulders, with the elbows unchanged, so the load is never
    curled up. A handhold out of the arm's reach is reported, not faked.
  - **Turning it on:** the Contact pass button, C, `contact=1` in the page URL, or
    `THUNDERSIM_CONTACT=1 .venv/bin/python -m thundersim viz`. The notes panel lists what the pass changed.
  - **Gates:** `.venv/bin/python -m thundersim.viz.contact_gates` (regression set, seeds 0-2):
    - hand to handhold while held <= 1 cm, with no handhold out of reach;
    - a held sack or stepladder into the legs and torso <= 5 mm;
    - foot skate <= 1 cm per stance, and heels down while squatting in a lift;
    - no heel or toe within 5 mm of the floor moving faster than 0.5 m/s (a swing skimming the floor; the
      rig's ladder climbing steps are reported apart);
    - static centre of mass outside the support polygon in <= 2 % of quasi-static frames;
    - no joint-limit violations, and no frame pops (feet < 0.12 m, hands < 0.20 m per frame).

    With the pass on, handholds, skate, floor drag, joint limits and pops pass. Two gates still fail, so the pass stays off by
    default (`contact.DEFAULT_ON`):
    - **Penetration (A-LADDER seeds 1 and 2):** 4 frames each, while the stepladder turns during a pick-up.
    - **Balance (A-KNEEL):**
      - steady kneeling work: the simulated trunk (30-55 deg) puts the centre of mass ahead of the knee
        contact in 16-73 % of frames, by a median of 1.4-3.2 cm and at most 5.6 cm. The hands cannot reach the
        floor, because the simulated kneeling shoulder is about 0.95 m up;
      - kneeling down and getting up: the support changes, and the centre of mass goes up to 0.6-0.67 m
        outside it.
- **Physics** (`thundersim/viz/physics.py`). A small deterministic position-based dynamics (XPBD) engine in
  numpy, with 250 Hz substeps. It is render only and never moves the simulated spine.
  - **Limbs and head:** a ragdoll around the kinematic simulated trunk. Hips, shoulders and neck stay on the
    spine. Knees, feet, elbows, hands and head are particles with exact bone lengths and joint limits,
    pulled toward the procedural pose by a muscle-tone schedule built from the ground-truth events, phases
    and consciousness:
    - full tone in normal activity, so the procedural pose is kept exactly;
    - limp in a fall, and while unconscious or still, so limbs swing, hit the floor and settle;
    - tone back after a 1.5 s stun for struggling, getting up and recovery;
    - in free flight the limbs keep the drawn body's centre of mass on the simulator's ballistic path.

    Only the time windows that need it are simulated (low tone, or lying in the net).
  - **Collisions:** the floor, boxes and slabs, rails and scaffold tubes, ladder rails and rungs, roof and
    bank slopes and the net, with friction and no bounce.
  - **Props:** these three collide with the drawn body and the scene but never move the body. They are
    simulated whether or not physics limbs are on.
    - the harness lanyard is a rope from the D-ring to the anchor: slack in free fall, taut at the arrest,
      with its energy absorber paying out up to 1.07 m;
    - the safety net is a cloth pinned along its border, pushed down by the drawn body;
    - a knocked-off hard hat is a rigid body on a 0.55 m cable from the upper sensor. Its cable goes taut at
      the simulated tug, then the hat swings, hits the worker's back and hangs.
  - **Rest lift:** the simulated trunk sometimes lies a few cm below the support (ground truth). While the
    body lies at rest, the whole drawn body and the worn strip are lifted by up to 15 cm, eased over 1 s.
  - **Ground truth unchanged:** IMU signals, segment and chip orientations, phases, events and the worn flag
    are identical with physics on or off. Spine and strip positions differ only by the rest lift.
  - **Toggle:** the Physics limbs button, P, or `physics=0` in the page URL (on by default). The page reloads
    the run with `/api/run?...&physics=0|1`. The notes panel shows the engine, the simulated windows, the
    largest rest lift and the ragdoll frame count. Physics adds roughly 5–40 % to a run's build time.
- **Playback.** Real time by default, so falls play at true gravity; [ and ] change the speed.
  - **Slow-mo** is opt-in: the Slow-mo button, S, or `slowmo=1`. It applies from 0.5 s before to 1.3 s after
    each fall event and eases to 0.25× and back (0.28 s time constant) instead of switching.
  - **Badge:** while playback is slower than real time, a badge shows the effective rate and the gravity it
    implies on screen. The clock shows any rate other than 1×.
- **Camera.** Follow (F, on by default) tracks the worker 1:1 horizontally but follows height through a
  critically damped spring (0.8 s, in simulated time). A falling worker therefore drops through the frame
  against the site, stays in view, and the camera settles within about 2 s of the impact. A seek snaps the
  camera to the worker. Dragging orbits around the worker.
- **Sensors.** The four pods use the true chip orientation, including mounting error and strap slip. Arrows
  show the specific force and angular rate each IMU *reported*: the emulated device output after clipping,
  sample-and-hold at the profile's rate, and lost packets. The charts show |a| and |ω| per anchor as
  delivered, the clip level, ground-truth events and model events.
- **Controls.**
  - **Playback:** Space plays and pauses. ←/→ steps by 0.1 s (Shift: 1 s), and Home goes back to the start.
    J/K jumps to the previous or next event. Click the timeline to seek.
  - **Camera:** 1–6 pick camera presets (chase, side, front, strip, top, site).
  - **Views:** X, V and G toggle the x-ray spine, force vectors and gyro; L and T the sensor labels and
    pelvis path.
  - **Modes:** S toggles slow-mo, P physics limbs, M mocap limbs and C the contact pass.
- **Deep links.** For example `/?s=H-SCAF&seed=3&model=falldet&profile=current_100hz&at=impact&dt=-1&cam=side`.
  `at` is a ground-truth event name or a model event kind. Also `t`, `play`, `speed`, `follow`, `xray`,
  `gyro`, `slowmo=1`, `physics=0`, `mocap=1` and `contact=1`.
- Runs are cached in `output/viz_cache/`, separately with physics, mocap and the contact pass on and off. They refresh when the
  simulator or viewer code, or `motionlib/manifest.json`, changes.

Rendering-only adjustments, also listed in each run's notes:
- **Jackhammer bouts:** `Builder.vibrate` lets the pelvis drift slowly up or down, by up to about 1 m in
  long bouts. The viewer removes that slow part from the body; it does not change the signals.
- **Rest lift:** the vertical lift of a body lying at rest partly below the support (see Physics).
- **After DEVICE-OFF:** the worker steps back from the table.

Known viewer limits:
- **Simulator motion is shown as simulated.** The trunk path is ground truth, so the viewer does not hide:
  - the rebound after hard body-first impacts, and the safety net throwing the worker back up;
  - trips and knock-downs that topple faster than free fall: the simulated centre of mass reaches about
    1.3–1.7 g downward;
  - the trunk dipping into the floor or steps during impacts, since the rest lift acts only at rest.
- **Flights:** the centre-of-mass hold ends early when a drawn leg still touches the roof, platform or
  block edge after takeoff, so a few flights read slightly off g.
- **Inconsistent seeds:** in some seeds the worker walks over the drop edge before falling backward along the
  same line, so they cross the opening for a few seconds. The notes say so.
- **Net impact:** the 10 m/s fall into the net has a short burst of fast limb motion at impact.

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

**Lifting** (`motions.py`, `Builder.lift` / `Builder.put_down`, used by every carry)

- **Technique per worker:** stoop, semi-squat or squat, drawn once per worker at 42 / 51 / 7 %
  [src: Geissinger 2020, 666 bends observed at work, the symmetric classes renormalised]. A habitual
  stooper switches to a semi-squat more often as the load grows past 10 kg [design, after
  Burgess-Limerick & Abernethy 1997].
- **Lowest posture:** the legs take a technique-specific share of the hands' drop to the grip and the
  trunk inclines until the hands reach it, with the feet planted. A semi-squat to a 0.15 m grip gives
  knee 87°, thorax 97°, pelvis 56° (Saraceni 2021 self-selected floor lifts: 86, 103, 51); a squat drops
  the pelvis 0.33–0.39 m. Lumbar share of the bend: von Arx 2021 [src]. Where the trunk reaches its 120°
  cap first (a stoop to a grip below about 0.25 m), the knees bend further, so a planned stoop can end as a
  semi-squat by knee angle. The events carry both the planned `technique` and the `knee_class` reached
  (Straker 2003: stoop below 45°; squat from 112.5° [design midpoint]), plus `knee_stand_deg`, the
  standing knee flexion (about 14°) included in `knee_deg`.
- **Timing:** lower, a short set and grip (0.1–0.4 s), then a leg-drive rise in which the knees extend
  first, then the hips, then the lumbar spine, the spine lagging more under heavier loads [src:
  Burgess-Limerick 1995 ordering; design windows]. The rest of a carry's time budget is spent standing
  upright with the load (real bends stay near their deepest point for only about 0.4 s, motionval). Rise times follow von Arx 2021 (about 1.4–1.8 s) [src]. The thorax turns at most
  150°/s (Saraceni 2021: peak 138, CI 126–150) [src] and every segment stays under the 180°/s cap.
- **Signals:** a carry keeps the main random stream and its total duration, so every other motion of a
  run is unchanged; lift details come from a separate sub-generator.

**Motion validation** (`motion_validation.py`, `motionval`)

The simulated trunk is compared with real motion using the same metrics for every source: walking
(cadence, step length, pelvis bob and sway, pelvis rotation, obliquity and tilt per stride, thorax
counter-rotation, lean, pelvis IMU step peaks and spectrum), carrying, running, lifting (peak inclination per
segment, pelvis share, knee, pelvis drop, thorax rate, lower, hold and rise times), kneeling, ladders and
quiet standing.

- **References:**
  - the class A mocap clips in `motionlib/canonical` (CMU, ACCAD, 100STYLE, NIOSH roofing);
  - the team's own vest recordings, if any, which are judged first (`validation/RECORDING_PROTOCOL.md`);
  - cited literature ranges (Orendurff 2004, Saraceni 2021, von Arx 2021, Latorre 2019).
- **Tuned from the comparison (step 1):**
  - walking: pelvis bob, rotation and obliquity, and thorax counter-rotation;
  - the extra lean when carrying;
  - running: pelvis rotation, tilt and sway, and counter-rotation;
  - ladders: the pelvis rolls and shifts to alternate sides on successive rungs;
  - kneeling: working trunk depth, and the trunk keeps moving;
  - quiet-standing sway;
  - lifts: a short hold at the lowest posture.

  Each value is tagged `[src: motionval ...]` in `motions.py`. Values the data cannot judge stay
  `[design]`. Rebuild the report with `motionval`. Remaining gaps: a lift still holds its deepest posture
  longer than real bends do (min-jerk approach), and carrying and running have only 2-3 reference clips.

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

## Realism scorecard

`realism/` measures how close the simulated trunk signals are to **real recordings**, like for like,
and ranks the gaps. It emulates each real sensor on the simulated body (placement and lever arm,
soft tissue, bandwidth, rate, range), compares about 22 fall-event and 35 activity features per category
(Wasserstein distance with subject/seed bootstrap CIs against a real split-half noise floor, plus a
classifier two-sample test), and runs falldet on both sides to show the downstream effect.

```sh
.venv/bin/python -m realism score --out <scratch dir>       # about 7 min once the real-data cache exists
.venv/bin/python -m realism compare <before dir> <after dir>
```

- **Report:** `realism/REPORT.md`: data and licences, the scenario-to-real mapping, the ranked gaps,
  and every simulator change with before/after numbers.
- **Real data:** Erciyes (UCI 455), VTT-ConIot and UNIVRFall, all CC BY 4.0, read from
  `research/fall_model/data` outside the repo (override with `THUNDERSIM_REALDATA`). They are never
  committed. The first run builds a cache (about 20 min).
- **Plug-ins:** other feature modules register with `realism.score.register_provider`, for example the
  voluntary-motion kinematics checks.
- **Caveats:**
  - Erciyes falls are acted, from standing, onto mats, at 25 Hz. Hard-floor impacts are not measured.
  - VTT sensors are clothing-mounted.
  - A classifier AUC near 1.0 means the sim is still distinguishable from real data. Judge progress by
    the per-feature gaps against the real noise floor.
  - Only dev seeds are used.

## Calibration finding

The `calibrate` command compares simulated standing falls onto mats with Erciyes waist falls, like
for like (12 Hz band-limited, sampled at 25 Hz).

| Source | Median peak | Interquartile range |
|---|---|---|
| Erciyes waist falls (25 Hz) | 3.95 g | 3.3–4.6 g |
| Simulator before the realism fixes (`calibration.json`) | 6.8 g | 5.7–7.5 g |
| Current simulator (re-run 2026-11-08, any soft-tissue scale) | 3.9 g | 3.4–4.6 g |

- **The gap is closed by fall physics, not by soft tissue.** The realism fixes did it (`realism/REPORT.md`):
  - no bounce at impact;
  - fitted mat stiffness;
  - protective resistance in trips, which brings hip speeds to 2.7–2.8 m/s.
- **Soft tissue is still unconstrained.** Every soft-tissue scale from 0.4 to 1.5 now fits within a log error of
  0.03–0.08, because the mount filter is transparent below 12.5 Hz.
- **`calibration.json` was not rewritten:**
  - the re-run's rule would pick the grid edge (0.4) on the 90th percentile alone, which Erciyes cannot
    support;
  - the realism scorecard uses scale 1.0;
  - the runner still reads 1.5 from the file.
  - Decide this with the team's own strip recordings.
- **Mats only.** Hard-floor peaks are not measured by any open dataset on disk.

## Limits

- **Body model:** kinematic, with physics only in the fall phases. There are no arms, legs or head
  segments. Impacts use single-degree-of-freedom contact.
- **Design values:** voluntary trunk motion is checked against open mocap (`motionval`), but not yet
  against the team's own vest. Record sessions as in `validation/RECORDING_PROTOCOL.md`. Device and
  strap values are still untuned: measure the team's own MPU-6050s (bias, noise, read skew) and strap
  mounting, then update `DeviceErrors` and `Worker`. Heel-strike transients (`site.py`) have no
  verified literature value; the vest recordings would set them.
- **Real falls:** real falls of workers are not simulated in any validated sense. The point is to
  exercise the pipeline, thresholds, timers and failure modes before people are involved, and to
  find where a model breaks.
- **Sources:** the research report, `research_notes/...`, and
  `fall_model_sim_spec/SIM_SPEC.md` (research-95), whose values were merged where they were better
  sourced.
