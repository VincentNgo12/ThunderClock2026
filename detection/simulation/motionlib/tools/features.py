# File: features.py
# Purpose: Stage S4 of the motion library: turn each annotated canonical clip into the limb
#          targets the viewer rig uses (rig.py leg_target / arm_target -> two_bone): ankle,
#          foot orientation and knee pole in the heading frame, hand and elbow pole in the
#          upper-thoracic frame and in a task frame, normalised by limb length.
#          Writes motionlib/clips/<clip>.npz (float32). The trunk is never stored as a target:
#          the simulated spine stays the IMU ground truth.
# Contributors: Unassigned
"""Run from detection/simulation after annotate.py:

    .venv/bin/python motionlib/tools/features.py             # every clip
    .venv/bin/python motionlib/tools/features.py cmu_62      # clips whose id starts with this

Frames: 100 Hz, as the canonical clip. Frames and units of motionlib/clips/<clip>.npz:
  heading (N,)            smoothed pelvis heading, simulator convention (fwd(h) = (-sin h, 0, -cos h))
  Heading frame           axes (right, up, forward) of that heading; "hf" below
  Upper-thoracic frame    the clip's segment 3 (seg_q[:, 3]) in the simulator convention
                          (+X wearer-right, +Y up the spine, +Z posterior): the rig's M[:, 3]
  leg_len (2,), arm_len (2,)  source thigh + shank and upper arm + elbow-to-hand (m), per side
  Legs (has_legs = 1 only with a floor and heel/toe joints):
    ankle_hip (N,2,3)     hip -> ankle in hf / leg_len
    knee_pole (N,2,3)     unit direction of the knee from the hip-ankle line, in hf; blends to the
                          foot's forward direction where the knee is almost straight (< 15 deg)
    ankle_lift (N,2)      ankle height above its flat-foot height / leg_len
    foot_u (N,2,3)        sole direction heel -> toe in hf (unit): the ankle -> toe line turned down by
                          its flat-stance pitch, so a flat foot reads horizontal (rig _foot_axes u)
    foot_n (N,2,3)        sole normal in hf (unit, perpendicular to foot_u; rig _foot_axes n)
    foot_yaw (N,2)        sole direction minus heading (rad, unwrapped over time)
    foot_pitch (N,2)      sole pitch, toes up + (rad), zero when the foot stands flat
    contact (N,2,2), phase (N,2)  from annotate.py (heel, toe)
    pelvis_h (N,)         pelvis height above the floor / mean leg_len
    pelvis_vel (N,3)      pelvis velocity in hf / mean leg_len (1/s)
  Arms:
    hand_shoulder (N,2,3) shoulder -> hand in the upper-thoracic frame / arm_len
    elbow_pole (N,2,3)    unit direction of the elbow from the shoulder-hand line, upper-thoracic frame;
                          blends to posterior and slightly outward where the arm is almost straight
    hand_shoulder_hf (N,2,3)  shoulder -> hand in the heading frame (gravity-aligned) / arm_len: where
                          the hands are relative to the shoulders whatever the trunk angle (kneeling work)
  hand_deck_h (N,2)       kneeling clips without a floor (NIOSH): hand height above the kneeling surface /
                          arm_len, the surface taken KNEE_ABOVE_DECK_M below the knees' 5th-percentile
                          height (the roof mock-up's slope is unknown, so this is an estimate)
    hand_task (N,2,3)     hand in the task frame / mean leg_len; task_origin (3,) world (m), task_heading ()
    task                  "floor": origin on the floor under the pelvis, axes hf, both per frame;
                          "hammer_post": origin = median striking-hand position at the wrist
                          deceleration peaks, axes = hf at the median strike (fixed);
                          "ladder": origin = mean rung-contact foot position on the floor, axes = hf
                          at the median rung contact (fixed); "none" without a floor (NIOSH)
  meta (json str)         clip, source, licence credit, segments, events, frame and unit notes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # our own tools only, never raw/
from annotate import above_floor, flat_frames  # noqa: E402
from cliplib import (ANNOTATIONS, CLIPS, SIDES, bend_pole, clip_names, heading_frame, inner_angle_deg,  # noqa: E402
                     load_card, load_clip, save_card, segment_matrix, smoothed_heading, to_local)

FORMAT_VERSION = 1
POLE_TRUST_DEG = (5.0, 15.0)   # below 5 deg of knee/elbow flexion the clip's bend direction is noise; the
                               # pole blends to a default up to 15 deg [design]
ELBOW_DEFAULT_OUT = 0.3        # default elbow pole: posterior, 0.3 outward [design]
KNEE_ABOVE_DECK_M = 0.05       # a kneeling knee joint centre sits about this far above the surface [design]


def unit_vector(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def blend_pole(pole: np.ndarray, default: np.ndarray, flex_deg: np.ndarray) -> np.ndarray:
    """The clip's pole where the joint is clearly bent, the default where it is almost straight."""
    w = np.clip((flex_deg - POLE_TRUST_DEG[0]) / (POLE_TRUST_DEG[1] - POLE_TRUST_DEG[0]), 0.0, 1.0)
    w = (w * w * (3 - 2 * w))[:, None]
    out = w * np.nan_to_num(pole) + (1 - w) * default
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def limb_length(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - b, axis=1)))


def raw_pitch(ankle: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """Pitch (toes up +) of the ankle -> toe line."""
    d = toe - ankle
    return np.arctan2(d[:, 1], np.maximum(np.hypot(d[:, 0], d[:, 2]), 1e-9))


def sole_direction(ankle: np.ndarray, toe: np.ndarray, knee: np.ndarray, pitch_flat: float, R: np.ndarray):
    """Sole direction u and sole normal n (N,3) in the heading frame, the yaw of u relative to
    the heading (unwrapped) and its pitch: the ankle -> toe line turned about the foot's own
    lateral axis by -pitch_flat. The lateral axis is the normal of the shank-foot plane, which
    stays well defined when the foot points steeply down (toe-off, kneeling)."""
    d = toe - ankle
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    lateral = np.cross(d, knee - ankle)
    lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-12)
    # Rodrigues rotation of d about `lateral` by -pitch_flat (positive angle about d x up lifts the toes).
    a = -pitch_flat
    u = d * np.cos(a) + np.cross(lateral, d) * np.sin(a) + lateral * np.sum(lateral * d, axis=1, keepdims=True) * (1 - np.cos(a))
    normal = np.cross(lateral, u)
    local = to_local(R, u)
    yaw = np.unwrap(np.arctan2(-local[:, 0], local[:, 2]))    # 0 = along the heading, + = toward the left
    pitch = np.arcsin(np.clip(local[:, 1], -1, 1))
    return local, to_local(R, normal), yaw, pitch


def task_frame(clip, card: dict, floor: np.ndarray, heading: np.ndarray):
    """(name, origin (N,3), axes (N,3,3)) of the task frame for this clip's hands."""
    s3 = card["s3"]
    tags = set(card["tags"])
    if np.isnan(floor).any():
        return "none", None, None
    peaks = [e for e in s3.get("wrist_decel_peaks", [])]
    rungs = s3.get("rung_contacts", [])
    if "hammer" in tags and len(peaks) >= 3:
        # The post: where the busier hand stops at its strikes.
        side = max(SIDES, key=lambda s: sum(e["side"] == s for e in peaks))
        frames = np.array([e["frame"] for e in peaks if e["side"] == side])
        origin = np.median(clip.j(f"hand_{side}")[frames], axis=0)
        k = int(np.median(frames))
        return "hammer_post", np.repeat(origin[None], clip.n, 0), np.repeat(heading_frame(heading[k:k + 1]), clip.n, 0)
    if "ladder" in tags and rungs:
        points = []
        for e in rungs:
            a, b = int(round(e["t0"] * clip.fs)), int(round(e["t1"] * clip.fs))
            points.append(clip.j(f"ankle_{e['side']}")[a:b].mean(axis=0))
        origin = np.mean(points, axis=0)
        origin[1] = floor[0] + floor[1] * origin[0] + floor[2] * origin[2]
        k = int(np.median([round(0.5 * (e["t0"] + e["t1"]) * clip.fs) for e in rungs]))
        return "ladder", np.repeat(origin[None], clip.n, 0), np.repeat(heading_frame(heading[k:k + 1]), clip.n, 0)
    pelvis = clip.j("pelvis")
    origin = pelvis.copy()
    origin[:, 1] = pelvis[:, 1] - above_floor(floor, pelvis)
    return "floor", origin, heading_frame(heading)


def build(name: str) -> dict:
    clip = load_clip(name)
    card = load_card(name)
    ann = np.load(ANNOTATIONS / f"{name}.npz")
    floor = ann["floor"].astype(float)
    offsets = ann["flat_offsets"].astype(float)
    heading = smoothed_heading(clip)
    R = heading_frame(heading)
    upper = segment_matrix(clip, 3)
    n = clip.n
    out: dict = {"fs": np.float32(clip.fs), "heading": heading.astype(np.float32), "format": np.int32(FORMAT_VERSION)}

    leg_len = np.array([limb_length(clip.j(f"hip_{s}"), clip.j(f"knee_{s}"), clip.j(f"ankle_{s}")) for s in SIDES])
    arm_len = np.array([limb_length(clip.j(f"shoulder_{s}"), clip.j(f"elbow_{s}"), clip.j(f"hand_{s}")) for s in SIDES])
    out["leg_len"], out["arm_len"] = leg_len.astype(np.float32), arm_len.astype(np.float32)
    scale = float(leg_len.mean())

    has_legs = not np.isnan(floor).any() and clip.has("heel_l", "heel_r", "toe_l", "toe_r")
    out["has_legs"] = np.int32(has_legs)
    if has_legs:
        ankle_hip = np.zeros((n, 2, 3))
        knee_pole = np.zeros((n, 2, 3))
        ankle_lift = np.zeros((n, 2))
        foot_u = np.zeros((n, 2, 3))
        foot_n = np.zeros((n, 2, 3))
        foot_yaw = np.zeros((n, 2))
        foot_pitch = np.zeros((n, 2))
        pitch_flat = np.zeros(2)
        for i, s in enumerate(SIDES):
            hip, knee, ankle, toe = (clip.j(f"{p}_{s}") for p in ("hip", "knee", "ankle", "toe"))
            ankle_hip[:, i] = to_local(R, ankle - hip) / leg_len[i]
            # A nearly straight knee gives no bend direction; there the pole leans toward the
            # foot's forward direction (knees over the toes).
            flex = 180.0 - inner_angle_deg(hip, knee, ankle)
            pitch = raw_pitch(ankle, toe)
            yaw_rel = np.arctan2(-to_local(R, toe - ankle)[:, 0], to_local(R, toe - ankle)[:, 2])
            default = np.stack([-np.sin(yaw_rel), np.zeros(n), np.cos(yaw_rel)], axis=1)
            knee_pole[:, i] = blend_pole(to_local(R, bend_pole(hip, knee, ankle)), default, flex)
            ankle_lift[:, i] = (above_floor(floor, ankle) - offsets[i, 0]) / leg_len[i]
            flat = flat_frames(clip, floor, s)
            contact = ann["contact"][:, i].all(axis=1)
            # The skeleton's ankle -> toe line is not parallel to the sole; flat stance reads 0.
            pitch_flat[i] = float(np.median(pitch[flat])) if flat.any() else (
                float(np.median(pitch[contact])) if contact.any() else 0.0)
            foot_u[:, i], foot_n[:, i], foot_yaw[:, i], foot_pitch[:, i] = sole_direction(ankle, toe, knee, pitch_flat[i], R)
        pelvis = clip.j("pelvis")
        out.update({
            "ankle_hip": ankle_hip, "knee_pole": knee_pole, "ankle_lift": ankle_lift, "foot_u": foot_u, "foot_n": foot_n, "foot_yaw": foot_yaw,
            "foot_pitch": foot_pitch, "foot_pitch_flat": pitch_flat,
            "contact": ann["contact"], "phase": ann["phase"],
            "pelvis_h": above_floor(floor, pelvis) / scale,
            "pelvis_vel": to_local(R, np.gradient(pelvis, 1.0 / clip.fs, axis=0)) / scale,
        })

    hand_shoulder = np.zeros((n, 2, 3))
    hand_shoulder_hf = np.zeros((n, 2, 3))
    elbow_pole = np.zeros((n, 2, 3))
    for i, s in enumerate(SIDES):
        shoulder, elbow, hand = (clip.j(f"{p}_{s}") for p in ("shoulder", "elbow", "hand"))
        hand_shoulder[:, i] = to_local(upper, hand - shoulder) / arm_len[i]
        hand_shoulder_hf[:, i] = to_local(R, hand - shoulder) / arm_len[i]
        # A nearly straight arm: the elbow points back and a little out (posterior = +Z).
        default = np.tile(unit_vector([ELBOW_DEFAULT_OUT * (1.0 if s == "r" else -1.0), 0.0, 1.0]), (n, 1))
        flex = 180.0 - inner_angle_deg(shoulder, elbow, hand)
        elbow_pole[:, i] = blend_pole(to_local(upper, bend_pole(shoulder, elbow, hand)), default, flex)
    out["hand_shoulder"], out["elbow_pole"] = hand_shoulder, elbow_pole
    out["hand_shoulder_hf"] = hand_shoulder_hf
    if not out["has_legs"] and "kneel" in card.get("activity_tags", []) and clip.has("knee_l", "knee_r"):
        knees = np.minimum(clip.j("knee_l")[:, 1], clip.j("knee_r")[:, 1])
        deck = float(np.percentile(knees, 5)) - KNEE_ABOVE_DECK_M
        out["hand_deck_h"] = np.stack([(clip.j(f"hand_{s}")[:, 1] - deck) / arm_len[i] for i, s in enumerate(SIDES)], axis=1)

    task, origin, axes = task_frame(clip, card, floor, heading)
    out["task"] = np.array(task)
    if origin is not None:
        out["hand_task"] = np.stack([to_local(axes, clip.j(f"hand_{s}") - origin) for s in SIDES], axis=1) / scale
        out["task_origin"] = origin[0] if task != "floor" else np.full(3, np.nan)

    s3 = card["s3"]
    meta = {
        "clip": name, "source": card["source"], "licence_class": card["licence"]["class"],
        "credit": card["licence"]["credit"], "fs": clip.fs, "frames": n, "height_est_m": clip.height,
        "activity_tags": card.get("activity_tags", []), "segments": s3["segments"], "bends": s3["bends"],
        "heel_strikes": s3["heel_strikes"], "wrist_decel_peaks": s3["wrist_decel_peaks"], "rung_contacts": s3["rung_contacts"],
        "frames_note": "hf = smoothed heading frame (right, up, forward); upper = clip segment 3, simulator convention",
    }
    out["meta"] = np.array(json.dumps(meta))
    CLIPS.mkdir(exist_ok=True)
    arrays = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v) for k, v in out.items()}
    np.savez_compressed(CLIPS / f"{name}.npz", **arrays)

    finite = {k: bool(np.isfinite(v).all()) for k, v in arrays.items()
              if isinstance(v, np.ndarray) and v.dtype.kind == "f" and k not in ("task_origin", "phase")}
    summary = {
        "file": f"clips/{name}.npz", "format": FORMAT_VERSION, "has_legs": bool(has_legs), "task": task,
        "leg_len_m": [round(float(x), 4) for x in leg_len], "arm_len_m": [round(float(x), 4) for x in arm_len],
        "foot_pitch_flat_deg": [round(float(np.degrees(x)), 1) for x in out.get("foot_pitch_flat", [np.nan, np.nan])],
        "finite": all(finite.values()),
        "non_finite": sorted(k for k, ok in finite.items() if not ok),
    }
    card["s4"] = summary
    save_card(card)
    return summary


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    for name in clip_names(prefix):
        s = build(name)
        print(f"{name:40s} legs {'yes' if s['has_legs'] else 'no ':3s} task {s['task']:12s} leg {s['leg_len_m']} arm {s['arm_len_m']}"
              f"{'' if s['finite'] else '  NON-FINITE: ' + ','.join(s['non_finite'])}")


if __name__ == "__main__":
    main()
