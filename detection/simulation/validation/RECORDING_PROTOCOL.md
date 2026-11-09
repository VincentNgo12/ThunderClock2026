# Recording your own vest sessions for motion validation

`thundersim motionval` compares the simulator's trunk motion with real motion. Today it uses open
motion-capture clips and literature ranges. Your own vest recordings are the best reference, because
they come from the real strip, straps and firmware. This page explains how to record them safely and
how to turn them into the report.

Recordings are optional. The tool works without them. When they exist, the report judges the
simulator against them first.

> **Not a clinical study.** Record only normal, comfortable everyday movements. The results show how
> close the simulator is to your movements. They say nothing medical about anyone.

## 1. Before you record

- **Consent.** Every person signs the team's consent form before recording. Give each person a
  consent id (for example `TC01`). The form itself never goes in git; keep it with the team lead.
  A person may withdraw at any time: delete their session folders.
- **People.** Two people: the person wearing the vest and an operator who runs the laptop and keeps
  time. A third person foots the ladder.
- **Hardware.** Use firmware that streams at 100 Hz (the `target` or `current_100hz` settings). The
  0.5 Hz firmware cannot measure gait. With the ±2 g range (`current_100hz`), running and heel strikes
  can clip: leave out running in that case.
- **Sensor check.** Do the backend's physical acceptance first (`backend/DIAGNOSTICS.md`). Confirm that
  sensor IDs 1-4 are on the pelvis, lumbar, thoracic and upper-thoracic positions.
- **Video (optional, for labelling).** Film from a tripod at 3-4 m, one camera from the side and one
  at 45 degrees (two cameras, or repeat the take). Frame the whole body. Raw video and faces never go
  in git. Keep the video on the team drive only until the labels are written, or as long as the
  consent form allows.

## 2. Safety rules

- **No falls of any kind:** no real falls, stumbles, slips, trips, jumps down, hard sits or "acted"
  falls. The simulator covers those.
- **Loads:** at most 15 kg, and only what the person lifts comfortably. Use a box or a sack with
  handles. Light loads (5 kg) are enough for the stoop and squat trials.
- **Ladder:** use a stepladder or a short fixed ladder, at most 4 rungs (about 1.2 m). The ladder is
  footed by a spotter, and the climber keeps three points of contact.
- **Kneeling:** wear knee pads or use a mat.
- **Running:** optional. Jog only on a dry, flat, clear surface.
- Stop at once if anything hurts or feels unsafe.

## 3. What to record

Each session starts with a **5 s neutral stand**: stand upright and still, arms relaxed, looking
ahead. The tool uses it to align the sensors, like the product's neutral calibration. Then record the
blocks below in any order, with a short pause standing between blocks.

| Activity (label) | What to do | Repetitions |
|---|---|---|
| `neutral` | Stand still, upright (first 5 s of every session) | 1 |
| `idle` | Quiet standing, 20 s | 3 |
| `walk` | Walk 20 m straight at a comfortable pace; turn outside the labelled part | 3 normal, 1 slow, 1 brisk |
| `carry_walk` | Walk 20 m carrying a box or sack in front with both hands; set `load_kg` | 2 at 10 kg, 1 at 15 kg if comfortable |
| `lift` | Lift a box from the floor to standing, then put it down; label each bend separately | 5 lifts + 5 put-downs your own way; then 3 stoop and 3 squat with 5 kg if comfortable; set `technique` |
| `kneel` | Kneel and work at floor level for 60 s (sort screws, place tiles) | 2 |
| `ladder` | Climb up and down the ladder; label only the climbing | 3 up and 3 down |
| `run` | Optional: jog 15 m | 2 |

Labels need to be only roughly right (about ±0.5 s). Start each label after the movement has begun
and end it before it stops. Walking windows shorter than 1.5 s are ignored.

## 4. Record

The backend CLI records one stream format per connection. Record each session twice if you can: once
with `quaternion` (trunk angles: lifts, kneeling, ladder, lean) and once with `raw` (accelerations:
step rate, heel strikes). Each recording is its own session folder.

From `backend/`, with the backend's environment active:

```sh
SESSION=../detection/simulation/validation/recordings/20261110_TC01_01_quat
mkdir -p "$SESSION"
python -m app.cli scan --timeout 5
python -m app.cli stream --device "<exact OS id>" --format quaternion --duration 900 --record "$SESSION/quat.jsonl"
```

For the raw pass, use `--format raw` and `--record "$SESSION/raw.jsonl"` in a folder ending in
`_raw`. Check the drop counter at the end. If many frames were dropped, record again.

## 5. Name the files

One folder per recording, in `detection/simulation/validation/recordings/`:

```
<YYYYMMDD>_<consent id>_<take number>_<quat|raw>/
    quat.jsonl or raw.jsonl     the backend recording, unchanged
    labels.csv                  start_s,end_s,activity,load_kg,technique,notes
    meta.json                   see below
```

Example: `20261110_TC01_01_quat/`.

- **`labels.csv`.** Times are seconds from the first sample of the recording. Start with
  `0,5,neutral,,,`. Use the activity names from the table. `technique` is `stoop`, `semi_squat` or
  `squat` if known.

  ```
  start_s,end_s,activity,load_kg,technique,notes
  0,5,neutral,,,
  12.5,26.0,walk,,,
  40.2,44.8,lift,10,semi_squat,from floor
  ```

- **`meta.json`.** Only what the analysis needs. No names.

  ```json
  {"consent_id": "TC01", "recorded_on": "2026-11-10", "height_m": 1.78,
   "firmware": "<git hash>", "profile": "target", "format": "quaternion",
   "sensor_placement_checked": true, "notes": "winter jacket over the vest"}
  ```

The `recordings/` folder is ignored by git. Commit derived results only, and only if the consent form
allows it and the team agrees.

## 6. Make the report

From `detection/simulation/`:

```sh
.venv/bin/python -m thundersim motionval --out output/motionval
```

The report is `output/motionval/MOTION_VALIDATION.md`, with figures in `output/motionval/figures/` and
every number in `metrics.json`. Use `--recordings <folder>` to read sessions from somewhere else.

To try the whole path without hardware, write a synthetic session from the simulator. It is marked
`synthetic` and is never used as a reference unless you ask for it:

```sh
.venv/bin/python -m thundersim motionval --write-example validation/recordings/SYNTHETIC_example
.venv/bin/python -m thundersim motionval --include-synthetic --out output/motionval_selftest
```

## 7. What the tool does with your files

- It reads the JSONL rows, drops repeated or backwards device times, and resamples to 100 Hz.
- **Raw recordings.** It rotates the pelvis accelerometer so +Y is gravity during the neutral stand,
  as the product does.
- **Quaternion recordings.** It re-zeroes each sensor on the neutral stand and converts the four
  orientations into the simulator's trunk angles.
- It cuts the recording at your labels and computes the same metrics as for the simulator and the
  mocap clips.
- Pelvis height (bob, lift drop) cannot be measured from the vest. Those rows stay empty for vest
  sessions.
