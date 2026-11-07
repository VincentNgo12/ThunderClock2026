# ThunderSim model contract

How a fall / man-down model plugs into the simulator. The code version is
`thundersim/model_api.py`. Anything that implements this contract can be tested against every
scenario, both firmware configurations and long fall-free shifts, without touching real sensors.

## Load a model

```
python -m thundersim suite --models baseline,falldet,mypkg.mymodule:MyModel
```

`load_model(spec)` accepts:

- `baseline`: the rule baseline.
- `baseline:<config.json>`: the baseline with a tuned config.
- `falldet`: research-95's `falldet.sim_adapter:SpineFallModel`. It loads from
  `detection/models`, or from the path in `THUNDERSIM_FALLDET`.
- `package.module:ClassName`: any class whose constructor takes no arguments.

The module must be importable; add its folder to `PYTHONPATH`.

## Interface

```python
class MyModel:
    name = "my-model"
    def reset(self, context: StreamContext) -> None: ...        # once per stream
    def update(self, batch: FrameBatch) -> list[Event]: ...     # every batch, in time order
    def finish(self) -> list[Event]: ...                        # optional, at the end of a stream
```

### `StreamContext`

| Field | Meaning |
|---|---|
| `fs_hz` | Nominal frame rate. 100 for the `target` profile, 0.5 for `current`. |
| `profile` | `target`, `target_8g`, `current_100hz` or `current`. See `device.PROFILES`. |
| `accel_range_g`, `gyro_range_dps` | Full scale. Values at full scale are clipped. |

### `FrameBatch`

Each batch is a chunk of about 0.5 s. Frames arrive through the real backend decoder
(`backend/app/protocol.py`), so lost packets leave gaps in `t`.

| Field | Shape / type | Meaning |
|---|---|---|
| `t` | (N,) float64 | Device time in seconds, from the firmware's `millis()` at the end of each 4-sensor scan. Strictly increasing. |
| `acc` | (N,4,3) float32 | Specific force in m/s², gravity included. Standing still reads about (0, +9.81, 0). |
| `gyro` | (N,4,3) float32 | Angular rate in rad/s. |
| `slot_mask` | (N,4) bool | False where that sensor is dead (an all-zero record). `acc` and `gyro` are NaN there. |
| `sat_mask` | (N,4) bool | True where any axis is at full scale, meaning the value is clipped. |
| `fs_hz` | float | Same as the context. |
| `work_mode` | str | `normal` or `lying_work`, as declared by the worker for the whole batch. |

**Slot order:** pelvis, lumbar, thoracic, upper_thoracic (sensor IDs 1–4).

**Frame:** each slot is rotated so that +Y points along gravity during the 5 s neutral stand at
stream start. That is the product's neutral calibration. Heading about +Y is unobservable, so the
horizontal axes are only as good as the strap mounting. That mounting is random per run, about
±6° (±9° in winter clothing). Use features that don't depend on heading where you can.

### `Event`

`Event(t, kind, confidence=1.0, detail={})`. `t` must not be later than the last frame of the
batch it is returned from. Models must be causal; the runner raises an error otherwise.

| `kind` | Meaning |
|---|---|
| `suspected_fall` | Stage-1 candidate. Not an alert; counted as Stage-1 load. |
| `near_fall` | A disturbance that ended without lying down. |
| `recovered` | A candidate was dismissed. |
| `fall_confirmed` | **Alert.** |
| `man_down_warning`, `man_down_critical` | **Alert** and escalation. |
| `cleared` | A previous alert no longer applies. |

## What happens to your events

1. **Run output.** Each run's output is `fall_confirmed` if any alert fired after calibration,
   otherwise `near_fall` if a near-fall fired, otherwise `normal`. This is SIM_SPEC's 3-state scheme.
2. **Worker check-in.** Every alert opens a 15 s worker check-in. A worker who is fine (hard
   negatives, a worker who gets up) cancels it. An unconscious, suspended or injured worker, or a
   strip left on a table, does not. Uncancelled alerts count as **supervisor alerts**.
3. **Latency.** Latency is measured from the ground-truth reference (the first impact, or the end
   of the arrest for ARR) to the alert's arrival on the laptop, including BLE latency.

## Rules for honest results

- Tune only on dev seeds (`--seed-base 0`, the default). Report on held-out test seeds
  (`--seed-base 1000000`) once.
- Use `export` data to test and augment a model, never as its only training data. The scenarios are
  simulated, and their impacts and motions are idealised.
- A simulator pass is necessary, not sufficient. It does not show real-world fall performance.
