# File: export.py
# Purpose: Run one scenario through the full chain (body -> device -> link -> model -> score)
#          and package everything the 3D viewer needs: body pose, strip pose, sensor readings,
#          events, the scene and the score.
# Contributors: Unassigned
"""Payload builder for the 3D viewer.

The run is the same one `thundersim run` produces for the same scenario, seed, profile and
model: build_run() with a recording Builder (it draws no extra random numbers), the same
mounts and soft tissue, and the same Pipeline. RecordingPipeline only listens to the samples
it delivers, so the charts and sensor arrows show exactly what the model received.
"""

from __future__ import annotations

import hashlib
from functools import partial

import numpy as np

from ..calibrate import load_fn_scale
from ..model_api import ALERT_KINDS, GRAVITY, SLOTS, load_model
from ..runner import CALIBRATION_S, Pipeline, StreamOptions, score
from ..scenarios import CATALOGUE, build_run
from ..spine import PHASES, random_mounts, strip_kinematics
from . import contact as contact_lib
from . import mocap as mocap_lib
from .geom import fwd, smoothstep
from .physics import ENGINE, SUBSTEP_HZ, VERSION, apply_limb_physics, rest_lift
from .recorder import RecordingBuilder
from .rig import JOINTS, build_rig
from .scene import Ctx, build_scene, dynamics

OUT_HZ = 50.0
LABELS = {
    "hold": None, "stand": "Standing", "walk": "Walking", "run": "Running", "stoop": "Bending to lift",
    "squat": "Squatting", "kneel": "Kneeling", "quadruped": "On hands and knees", "crawl": "Crawling",
    "sit_ground": "Sitting on the ground", "lie": "Lying down", "get_up": "Getting up", "stand_up": "Standing up",
    "ladder": "Ladder", "hammer": "Hammering", "drill_overhead": "Drilling overhead", "jackhammer": "Jackhammer",
    "carry": "Carrying a load", "lift": "Lifting a load", "put_down": "Putting a load down",
    "rebar_tie": "Tying rebar", "kneel_work": "Kneeling floor work",
    "sit": "Sitting", "cab_ride": "Riding in the equipment cab", "jump_down": "Jumping down",
    "lying_activity": "Working while lying", "remove_strip": "Taking the strip off", "hat_off": "Hard hat knocked off",
}


class NullModel:
    """Model slot filler when no model is selected: the device and link still run."""

    name = "none"

    def reset(self, context) -> None:
        pass

    def update(self, batch) -> list:
        return []


class RecordingPipeline(Pipeline):
    """Pipeline that also keeps every device sample and whether the radio delivered it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_t, self.samples_acc, self.samples_gyro, self.samples_ok = [], [], [], []

    def _deliver(self, t_true, time_ms, acc, gyro, modes_t, modes_v) -> None:
        before = len(self.arrivals_at)
        dead_acc, dead_gyro = acc.copy(), gyro.copy()
        for slot, t0, t1 in self.slot_failures:
            if t0 <= t_true < t1:
                dead_acc[slot] = 0.0
                dead_gyro[slot] = 0.0
        super()._deliver(t_true, time_ms, acc, gyro, modes_t, modes_v)
        self.samples_t.append(float(t_true))
        self.samples_acc.append(dead_acc)
        self.samples_gyro.append(dead_gyro)
        self.samples_ok.append(len(self.arrivals_at) > before)


def body_shift(ctx, frames: np.ndarray, track) -> tuple[np.ndarray, list]:
    """Rendering-only offsets of the body (N,3).

    1. Builder.vibrate() double-integrates band-limited noise in the frequency domain with
       freqs[0] = 1.0 and removes only a linear trend, so leftover content below ~0.1 Hz grows
       into a slow vertical pelvis drift (up to ~1 m in long jackhammer bouts). Its acceleration
       is < 0.01 g, so the IMU signals barely see it, but the body would sink into the floor.
       The slow part (< 0.2 Hz) is removed inside each trunk-vibration bout.
    2. After DEVICE-OFF the simulator tracks only the strip. The worker steps back 0.6 m from
       the table so the strip can lie where the simulator put it.
    """
    from scipy import signal

    fs = track.fs
    shift = np.zeros((len(frames), 3))
    notes = []
    worst = 0.0
    for call in ctx.rec.find("vibrate"):
        if call.args.get("slots", "trunk") != "trunk":
            continue
        a, b = int(call.t0 * fs), min(int(call.t1 * fs), len(track) - 1)
        if b - a < 2000:
            continue
        y = track.root_pos[a:b + 1, 1]
        slow = signal.sosfiltfilt(signal.butter(2, 0.2, fs=fs, output="sos"), y)
        base = slow[0] + (slow[-1] - slow[0]) * np.linspace(0, 1, len(slow))
        drift = slow - base
        worst = max(worst, float(np.abs(drift).max()))
        inside = (frames >= a) & (frames <= b)
        shift[inside, 1] -= drift[frames[inside] - a]
    if worst > 0.05:
        notes.append(f"Rendering removes a slow vertical drift of up to {worst:.2f} m that the simulator's vibration model "
                     "(Builder.vibrate) adds to the pelvis in this bout. IMU signals are unchanged (the drift's acceleration "
                     "is below 0.01 g).")
    t_off = ctx.t_ev("strip_removed")
    if t_off is not None:
        t = frames / fs
        back = -fwd(ctx.heading(t_off)) * 0.6
        shift += back[None] * smoothstep((t - t_off - 0.2) / 1.6)[:, None]
        notes.append("After the strip is taken off the simulator tracks only the strip; the worker's pose is held and "
                     "stepped back from the table for display.")
    return shift, notes


def available_models() -> list[str]:
    models = ["none", "baseline"]
    try:
        load_model("falldet")
        models.append("falldet")
    except Exception:  # falldet needs scikit-learn==1.9.1 and joblib; the viewer works without it
        pass
    return models


def rnd(arr, digits: int = 4) -> list:
    return np.round(np.asarray(arr, float), digits).ravel().tolist()


def run_payload(scenario: str, seed: int, model: str = "baseline", profile: str = "target", obs_s: float = 60.0,
                dead_slot: int | None = None, fast: bool = True, physics: bool = True, mocap: bool | None = None,
                mocap_only: dict | None = None, mocap_manifest: dict | None = None, keep: dict | None = None,
                contact: bool | None = None) -> dict:
    """physics=True (default) adds the render-only limb ragdoll and rest lift (physics.py);
    physics=False gives the purely procedural pose. mocap=True drives the limbs from the clips in
    motionlib/manifest.json where one applies (mocap.py); None reads THUNDERSIM_MOCAP (default off).
    mocap_only={role: clip} restricts a role to one clip and mocap_manifest replaces
    motionlib/manifest.json (both for the S6 gate tool). contact=True runs the contact and balance
    pass (contact.py) on the rig before the ragdoll; None reads THUNDERSIM_CONTACT (default
    contact.DEFAULT_ON). Signals, spine, segq, anchors, chipq, phase and worn are identical
    whatever physics, mocap and contact are.
    keep: optional dict that receives the scene, rig and ctx objects (for gate tools)."""
    spec = CATALOGUE[scenario]
    sink: list = []
    run = build_run(scenario, seed, obs_s=obs_s, fn_scale=load_fn_scale(), slot_failure=dead_slot,
                    builder=partial(RecordingBuilder, sink=sink))
    rec = sink[0]
    track = run.track
    fs = track.fs
    mounts = random_mounts(run.worker, np.random.default_rng(seed + 3))
    kin = strip_kinematics(track, run.worker, mounts)

    # --- device, radio and model: the same chain as `thundersim run` ----------------------
    model_obj = NullModel() if model == "none" else load_model(model)
    options = StreamOptions(profile=profile, fast=fast)
    pipe = RecordingPipeline(model_obj, run.worker, run.site, mounts, run.soft_fn, run.soft_zeta, options, seed,
                             run.slot_failures)
    total = len(track) / fs
    pipe.feed(track, run.work_modes, run.site.ambient_c, total)
    out = pipe.finish()
    result = None
    if model != "none":
        res = score(run, out, profile, getattr(model_obj, "name", model), dead_slot)
        result = {k: getattr(res, k) for k in ("output", "passed", "partial", "latency_s", "supervisor_latency_s",
                                               "supervisor_alerts", "cancelled_alerts", "stage1_triggers", "frames",
                                               "saturated_frames", "ref_time")}
        result["link"] = res.link

    # --- output frames ---------------------------------------------------------------------
    step = int(round(fs / OUT_HZ))
    frames = np.arange(0, len(track), step)
    worn = track.worn[frames]
    last_worn = np.maximum.accumulate(np.where(track.worn, np.arange(len(track)), 0))
    body_idx = last_worn[frames]
    ctx = Ctx(run, rec, kin)
    scene = build_scene(ctx, scenario)
    shift, drift_notes = body_shift(ctx, frames, track)
    physics_meta = {"engine": ENGINE, "version": VERSION, "substep_hz": SUBSTEP_HZ, "enabled": False, "windows": []}
    if physics:
        # Render-only lift of a body resting partly below the support (ground truth), shared by
        # the drawn body and the worn strip through the existing shift.
        lift, lift_notes = rest_lift(ctx, scene, frames, body_idx, shift)
        shift[:, 1] += lift
        drift_notes += lift_notes
        physics_meta["rest_lift_max_m"] = round(float(lift.max()), 3)
    strip_anchors = kin.anchors[frames] + np.where(worn[:, None, None], shift[:, None, :], 0.0)
    session = mocap_lib.Session.create(mocap_lib.requested(mocap), f"{scenario}:{seed}", mocap_only, mocap_manifest)
    if session is None:
        rig = build_rig(ctx, scene, frames, body_idx, strip_anchors, shift)
    else:
        rig = build_rig(ctx, scene, frames, body_idx, strip_anchors, shift, mocap=session)
    contact_meta = {"enabled": False, "version": contact_lib.VERSION}
    if contact_lib.requested(contact):
        # Feet on their plants, hands on held objects, load clear of the legs, balance (render only).
        contact_meta = contact_lib.apply_contact(ctx, scene, frames, rig)
    if physics:
        # Limbs and head as a ragdoll around the simulated spine; before dynamics() so props
        # held in the hands or hung on the head follow the simulated limbs.
        physics_meta.update(apply_limb_physics(ctx, scene, frames, rig, body_idx))
    dyn = dynamics(scene, ctx, frames, rig)
    segq = np.stack([kin.seg_rot[i][body_idx].as_quat() for i in range(4)], axis=1)
    chipq = np.stack([kin.chip_rot[i][frames].as_quat() for i in range(4)], axis=1)

    # --- sensor samples as the model received them ------------------------------------------
    st = np.asarray(pipe.samples_t)
    sacc = np.asarray(pipe.samples_acc).reshape(-1, 4, 3)
    sgyro = np.asarray(pipe.samples_gyro).reshape(-1, 4, 3)
    ok = np.asarray(pipe.samples_ok, bool)
    if len(st):
        k = np.clip(np.round(st * fs).astype(int), 0, len(track) - 1)
        world_f = np.stack([kin.chip_rot[i][k].apply(sacc[:, i]) for i in range(4)], axis=1) / GRAVITY
        world_w = np.stack([kin.chip_rot[i][k].apply(sgyro[:, i]) for i in range(4)], axis=1)
    else:
        world_f = world_w = np.zeros((0, 4, 3))
    acc_lim = pipe.profile.accel_range_g * GRAVITY * (32767 / 32768) - 1e-3
    gyro_lim = np.deg2rad(pipe.profile.gyro_range_dps) * (32767 / 32768) - 1e-5
    sat = (np.abs(sacc) >= acc_lim).any(axis=2) | (np.abs(sgyro) >= gyro_lim).any(axis=2)

    # --- events, phases, activities ----------------------------------------------------------
    truth = [{"t": round(float(t), 3), "name": n,
              "info": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in info.items()
                       if isinstance(v, (int, float, str, bool, np.generic))}} for t, n, info in run.events]
    model_events = [{"t": round(float(e["t"]), 3), "t_laptop": round(float(e["t_laptop"]), 3), "kind": e["kind"],
                     "confidence": round(float(e["confidence"]), 3), "alert": e["kind"] in ALERT_KINDS,
                     "detail": {k: v for k, v in e["detail"].items() if isinstance(v, (int, float, str, bool))}}
                    for e in out.events]
    change = np.flatnonzero(np.diff(track.phase)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(track)]])
    phases = [[round(s / fs, 3), round(e / fs, 3), int(track.phase[s])] for s, e in zip(starts, ends, strict=True)]
    activities = []
    for c in rec.calls:
        if c.depth == 0 and LABELS.get(c.name) and c.t1 - c.t0 > 0.05:
            label = LABELS[c.name]
            if c.name == "ladder":
                label = "Climbing up" if c.args.get("up", True) else "Climbing down"
                if c.args.get("height", 1.0) < 0.9:
                    label = "Stepping up"
            if c.name == "hammer" and c.args.get("overhead"):
                label = "Hammering overhead"
            activities.append([round(c.t0, 3), round(c.t1, 3), label])
    activities.insert(0, [0.0, CALIBRATION_S, "Calibration stand"])

    notes = list(scene.notes) + drift_notes
    if contact_meta["enabled"]:
        notes.append("Contact pass on (contact.py): feet stay on their plants, hands hold the sack, ladder rails and "
                     "stepladder at fixed handholds, a held sack clears the legs, and the static balance is checked; "
                     "only limbs and props move, never the simulated spine.")
    notes.append("Spine, sensors and events are simulated; legs, arms, head and props are illustrative.")
    mocap_meta = mocap_info(session, rig)
    if mocap_meta["uses"]:
        notes.append("Some limb motion comes from motion capture, time-warped onto the simulated events (mocap.py); "
                     "the spine is still the simulated one. " + " ".join(mocap_meta["credits"]))
    if physics_meta["windows"]:
        notes.append(f"During falls, lying and suspension the limbs and head are a ragdoll ({ENGINE} {VERSION}, "
                     f"{SUBSTEP_HZ:.0f} Hz) under gravity around the simulated spine, with muscle tone set by the "
                     "ground-truth events and consciousness; in free flight they keep the drawn body's centre of "
                     "mass on the simulated ballistic path.")
    payload = {
        "meta": {
            "scenario": scenario, "seed": seed, "group": spec.group, "description": spec.description,
            "expected": run.expected, "report_only": spec.report_only, "model": model, "profile": profile,
            "obs_s": obs_s, "dead_slot": dead_slot, "duration": round(total, 3), "fps": OUT_HZ,
            "frames": int(len(frames)), "slots": list(SLOTS), "joint_names": list(JOINTS),
            "params": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in run.params.items()},
            "worker": {"height_m": round(run.worker.height_m, 3), "mass_kg": round(run.worker.mass_kg, 1),
                       "load_kg": round(run.worker.load_kg, 1), "clothing": run.site.clothing},
            "site": {"surface": scene.surface, "condition": scene.condition, "radio": options.radio or run.site.radio,
                     "ambient_c": [round(x, 1) for x in run.site.ambient_c]},
            "profile_info": {"odr_hz": pipe.profile.odr_hz, "accel_range_g": pipe.profile.accel_range_g,
                             "gyro_range_dps": pipe.profile.gyro_range_dps, "dlpf_hz": pipe.profile.dlpf_hz},
            "ref_time": run.ref_time, "notes": notes, "phase_names": PHASES, "body": rig.dims.export(),
            "checkin_s": run.checkin_s, "physics": physics_meta, "mocap": mocap_meta, "contact": contact_meta,
        },
        "result": result,
        "truth": truth,
        "model_events": model_events,
        "phases": phases,
        "activities": activities,
        "modes": {"legs": [[round(a, 3), round(b, 3), m] for a, b, m in rig.leg_modes],
                  "arms": [[round(a, 3), round(b, 3), m] for a, b, m in rig.arm_modes]},
        "frames": {
            "spine": rnd(rig.spine), "segq": rnd(segq), "anchors": rnd(strip_anchors), "chipq": rnd(chipq),
            "joints": rnd(rig.joints), "headq": rnd(rig.head_q), "phase": track.phase[frames].astype(int).tolist(),
            "worn": worn.astype(int).tolist(), "tool": rig.tool.astype(int).tolist(),
        },
        "dyn": {k: (rnd(v) if isinstance(v, np.ndarray) else v) for k, v in dyn.items()},
        "signals": {
            "t": rnd(st, 3), "ok": ok.astype(int).tolist(), "acc_g": rnd(np.linalg.norm(sacc, axis=2) / GRAVITY, 3),
            "gyro_dps": rnd(np.rad2deg(np.linalg.norm(sgyro, axis=2)), 1), "sat": sat.astype(int).ravel().tolist(),
            "fw": rnd(world_f, 3), "ww": rnd(world_w, 3),
        },
        "scene": {"ground_y": scene.ground_y, "surface": scene.surface, "condition": scene.condition,
                  "props": scene.props, "focus": rnd(scene.focus, 3),
                  "solids": [{"c": [round(float(w.centre[0]), 3), round(float(w.centre[2]), 3)], "y": round(w.y, 3),
                              "h": [round(float(w.half[0]), 3), round(float(w.half[1]), 3)], "yaw": round(w.yaw, 4)}
                             for w in scene.walkables]},
    }
    if keep is not None:
        keep.update({"scene": scene, "rig": rig, "ctx": ctx, "frames": frames, "session": session})
    return payload


def mocap_info(session, rig) -> dict:
    """meta.mocap: whether mocap was on, the manifest it used, and every clip use with its credit."""
    blob = mocap_lib.manifest_bytes()
    info = {"enabled": session is not None, "manifest_sha256": hashlib.sha256(blob).hexdigest() if blob else None,
            "uses": [], "credits": [], "problems": []}
    if session is not None:
        meta = getattr(rig, "mocap", None) or {}
        info["uses"] = [{k: v for k, v in use.items() if k != "credit"} for use in meta.get("uses", [])]
        credits = []
        for use in meta.get("uses", []):
            line = f"{use['clip']} (class {use['licence_class']}): {use['credit']}"
            if line not in credits:
                credits.append(line)
        info["credits"] = credits
        info["problems"] = meta.get("problems", [])
    return info
