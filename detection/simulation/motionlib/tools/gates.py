# File: gates.py
# Purpose: Stage S5 of the motion library: quality gates for every clip. Each clip's S4 targets
#          are retargeted onto the viewer rig (rig.Dims of the default 1.75 m worker, rig.two_bone,
#          the rig's foot geometry) at the payload rate of 50 Hz, and the gate metrics are
#          written into the card. Any FAIL sets the card's status to "rejected".
# Contributors: Unassigned
"""Run from detection/simulation after features.py:

    .venv/bin/python motionlib/tools/gates.py             # every clip
    .venv/bin/python motionlib/tools/gates.py cmu_62      # clips whose id starts with this

What is measured. Stage S6 (thundersim/viz/mocap.py) does not exist yet, so there is no mocap
payload. The gates run on a preview that draws each clip the way the rig would:
  - the clip's own pelvis path, scaled uniformly by rig leg / source leg, its floor at y = 0 and
    shifted vertically so that a flat source foot stands flat on the rig floor;
  - legs: ankle target = hip + heading frame @ ankle_hip x rig leg, knee pole from the clip, then
    rig.two_bone (continuous bend plane) with the rig's thigh and shank; heel and toe from the
    rig's foot geometry (rig.build_legs / _foot_axes) with the clip's foot yaw and pitch;
  - arms: hand target = shoulder + upper-thoracic frame @ hand_shoulder x rig arm, elbow pole
    from the clip, rig.two_bone with the rig's upper arm and forearm.
The sync gates (simulated heel strike to touchdown, tool strike to impulse) need the simulator's
anchors and the S6 time warp, so they read PENDING here; stage S6 must measure them on the real
payload, together with every other gate, before a clip may be switched on.

Gates (limits from .claude/agents/mocap-curator.md):
  source      FK bone-length std < 1 mm and plausible foot/hand speeds (from S2 s2_checks)
  bones       |rig bone length error| < 1e-3 m
  penetration heel and toe at most 5 mm below the floor
  skate       during a labelled contact: drift < 1 cm per stance, median speed < 2 cm/s
  pops        per-frame move relative to the pelvis < 0.12 m (feet), < 0.20 m (hands); clips
              contain no fall phases, so no frames are excluded
  limits      knee 0-160 deg with no hyperextension, elbow 0-155 deg, hip flexion -30..+135 deg
              relative to the pelvis segment
  reach       < 5 % of frames clamped by two_bone, per limb
  sync        PENDING (stage S6)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # our own tools only, never raw/
from thundersim.viz.rig import MAX_FOOT_SPEED, Dims, speed_limited, two_bone  # noqa: E402

from annotate import above_floor  # noqa: E402
from cliplib import (ANNOTATIONS, CLIPS, SIDES, clip_names, heading_frame, inner_angle_deg, load_card,  # noqa: E402
                     load_clip, runs, save_card, segment_matrix, to_local, to_world)

PAYLOAD_FS = 50.0
REFERENCE_H, REFERENCE_MASS = 1.75, 80.0     # the simulator's default Worker (spine.Worker)

BONE_TOL_M = 1e-3
PENETRATION_M = 0.005
SKATE_DRIFT_M, SKATE_MEDIAN_MPS = 0.01, 0.02
POP_FOOT_M, POP_HAND_M = 0.12, 0.20
KNEE_MAX_DEG, ELBOW_MAX_DEG = 160.0, 155.0
HIP_FLEX_DEG = (-30.0, 135.0)
TOE_STRAIGHTEN_S = 0.15      # toes straighten this fast after lift-off (40 % of a ~0.4 s swing) [design]
HIP_IN_PLANE = 0.5           # hip flexion is checked where >= half the thigh lies in the sagittal plane [design]
HYPEREXT_MIN_DEG = 5.0
HYPEREXT_PLANE = 0.5       # a knee bent backwards by more than this counts as hyperextended [design]
REACH_MAX_SHARE = 0.05
CLAMP_TOL_M = 0.005          # a clamp counts when two_bone leaves the end joint this far from its target [design];
                             # two_bone always stops 0.05 % short of full reach, which straight limbs hit
GATE_ORDER = ("source", "bones", "penetration", "skate", "pops", "knee", "elbow", "hip", "reach", "sync")


def gate(value, limit: str, ok: bool | None) -> dict:
    """ok None = not applicable to this clip; the string results are what the cards show."""
    return {"value": value, "limit": limit, "result": "n/a" if ok is None else ("PASS" if ok else "FAIL")}


def toe_bends(toe_contact: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """rig._toe_bend_of: a foot on its toes keeps them flat on the ground; after lift-off the
    toes straighten over TOE_STRAIGHTEN_S (the rig: over the first 40 % of the swing)."""
    bend = np.zeros_like(pitch)
    for i in range(pitch.shape[1]):
        last, since = 0.0, 0.0
        for k in range(len(pitch)):
            if toe_contact[k, i]:
                last, since = max(-float(pitch[k, i]), 0.0), 0.0
                bend[k, i] = last
            else:
                since += 1.0 / PAYLOAD_FS
                bend[k, i] = last * max(0.0, 1.0 - since / TOE_STRAIGHTEN_S)
    return bend


def foot_points(ankle: np.ndarray, u: np.ndarray, nrm: np.ndarray, toe_bend: np.ndarray, d: Dims):
    """Heel, ball and toe tip of the rig foot (rig.build_legs)."""
    sole = ankle - nrm * d.ankle_h
    ball = sole + u * d.ball
    toe_dir = u * np.cos(toe_bend)[..., None] + nrm * np.sin(toe_bend)[..., None]
    return sole - u * d.heel, ball, ball + toe_dir * (d.toe - d.ball)


def preview(name: str) -> dict:
    """The clip drawn on the rig at 50 Hz: joints, IK clamps and what the gates need."""
    clip = load_clip(name)
    feat = np.load(CLIPS / f"{name}.npz")
    ann = np.load(ANNOTATIONS / f"{name}.npz")
    d = Dims(REFERENCE_H, REFERENCE_MASS)
    step = int(round(clip.fs / PAYLOAD_FS))
    pick = np.arange(0, clip.n, step)
    n = len(pick)
    has_legs = bool(feat["has_legs"])
    floor = ann["floor"].astype(float)
    k = d.leg / float(feat["leg_len"].mean())

    def place(p: np.ndarray) -> np.ndarray:
        """Source points (N,3) -> preview world: scaled by k, floor at y = 0."""
        q = p * k
        if has_legs:
            q[:, 1] = above_floor(floor, p) * k + lift
        return q

    lift = 0.0
    if has_legs:
        # A flat source foot stands flat on the rig floor: shift by the flat ankle heights.
        lift = d.ankle_h - k * float(ann["flat_offsets"][:, 0].mean())
    heading = feat["heading"].astype(float)[pick]
    R = heading_frame(heading)
    upper = segment_matrix(clip, 3)[pick]
    pelvis_frame = segment_matrix(clip, 0)[pick]
    pelvis = place(clip.j("pelvis")[pick])
    out = {"n": n, "pelvis": pelvis, "has_legs": has_legs, "pelvis_frame": pelvis_frame, "dims": d}

    if has_legs:
        hips = np.stack([place(clip.j(f"hip_{s}")[pick]) for s in SIDES], axis=1)
        # Ankle targets from the hip (each leg normalised by its own length, so reach ratios hold),
        # then one constant height correction per leg so that the frames where that foot stands
        # flat put the ankle at the rig's ankle height (the two source legs may differ slightly).
        target = np.stack([hips[:, i] + to_world(R, feat["ankle_hip"][pick, i].astype(float) * d.leg) for i in range(2)], axis=1)
        lift_flat = feat["ankle_lift"][pick].astype(float)
        correction = np.zeros(2)
        for i in range(2):
            flat = np.abs(lift_flat[:, i]) * d.leg < 0.005
            if flat.sum() >= 3:
                correction[i] = float(np.median(d.ankle_h - target[flat, i, 1]))
        if (correction == 0).any() and (correction != 0).any():
            correction[correction == 0] = correction[correction != 0][0]
        target[..., 1] += correction
        pole = np.stack([hips[:, i] + to_world(R, feat["knee_pole"][pick, i].astype(float)) for i in range(2)], axis=1)
        u = np.stack([to_world(R, feat["foot_u"][pick, i].astype(float)) for i in range(2)], axis=1)
        nrm = np.stack([to_world(R, feat["foot_n"][pick, i].astype(float)) for i in range(2)], axis=1)
        contact = feat["contact"][pick].astype(bool)
        pitch = np.arcsin(np.clip(u[..., 1], -1, 1))
        toe_bend = toe_bends(contact[..., 1], pitch)
        # rig.build_legs: lift the targets so heel, ball and toe stay on or above the floor
        # (rig._sole_lift, here with the flat floor y = 0), limit the foot speed relative to
        # the pelvis, then two-bone IK.
        heel, ball, toe = foot_points(target, u, nrm, toe_bend, d)
        lowest = np.minimum(np.minimum(heel[..., 1], ball[..., 1]), toe[..., 1])
        target[..., 1] += np.maximum(-lowest, 0.0)
        target = pelvis[:, None, :] + speed_limited(target - pelvis[:, None, :], MAX_FOOT_SPEED / PAYLOAD_FS)
        knee, ankle = two_bone(hips, target, pole, d.thigh, d.shank, continuous=True)
        heel, ball, toe = foot_points(ankle, u, nrm, toe_bend, d)
        # Source contact points (heel or toe joint) in the preview world, for the skate gate.
        src = np.stack([np.stack([place(clip.j(f"{q}_{s}")[pick]) for q in ("heel", "toe")], axis=1) for s in SIDES], axis=1)
        out.update({"hips": hips, "knee": knee, "ankle": ankle, "heel": heel, "ball": ball, "toe": toe, "foot_u": u,
                    "target": target, "contact": contact, "src_contact_pts": src, "leg_height_correction": correction,
                    "leg_clamp_m": np.linalg.norm(ankle - target, axis=-1)})

    shoulders = np.stack([place(clip.j(f"shoulder_{s}")[pick]) for s in SIDES], axis=1)
    target = np.stack([shoulders[:, i] + to_world(upper, feat["hand_shoulder"][pick, i].astype(float) * d.arm)
                       for i in range(2)], axis=1)
    pole = np.stack([shoulders[:, i] + to_world(upper, feat["elbow_pole"][pick, i].astype(float)) for i in range(2)], axis=1)
    elbow, hand = two_bone(shoulders, target, pole, d.upper_arm, d.forearm, continuous=True)
    out.update({"shoulders": shoulders, "elbow": elbow, "hand": hand, "arm_clamp_m": np.linalg.norm(hand - target, axis=-1)})
    return out


def skate_metrics(p: dict, source: bool) -> tuple[float, float, float, int]:
    """(worst drift, p90 drift, median speed, stances) of the heel and toe during their labelled
    contacts (horizontal, preview world). source=True measures the clip's own heel and toe
    joints (scaled), False the rig foot drawn from them."""
    drifts, speeds = [], []
    for i in range(2):
        for j, point in enumerate(("heel", "toe")):
            xz = (p["src_contact_pts"][:, i, j] if source else p[point][:, i])[:, [0, 2]]
            for a, b in runs(p["contact"][:, i, j]):
                if b - a < 2:
                    continue
                drifts.append(float(np.linalg.norm(xz[a:b] - xz[a], axis=1).max()))
                speeds.append(np.linalg.norm(np.diff(xz[a:b], axis=0), axis=1) * PAYLOAD_FS)
    if not drifts:
        return float("nan"), float("nan"), float("nan"), 0
    return max(drifts), float(np.percentile(drifts, 90)), float(np.median(np.concatenate(speeds))), len(drifts)


def evaluate(name: str) -> dict:
    card = load_card(name)
    p = preview(name)
    d = p["dims"]
    g: dict = {}
    s2 = card.get("s2_checks", {})
    bones_src = s2.get("bones")
    source_ok = bool(s2.get("speed")) and (bones_src is True or isinstance(bones_src, str))
    g["source"] = gate({"bone_std_mm": s2.get("bone_std_mm_max"), "foot_p99_mps": s2.get("foot_p99_mps"),
                        "hand_p99_mps": s2.get("hand_p99_mps"),
                        "note": bones_src if isinstance(bones_src, str) else None},
                       "FK bone std < 1 mm (n/a for position data); foot p99 < 12, hand p99 < 15 m/s", source_ok)

    errors = [np.abs(np.linalg.norm(p["elbow"] - p["shoulders"], axis=-1) - d.upper_arm).max(),
              np.abs(np.linalg.norm(p["hand"] - p["elbow"], axis=-1) - d.forearm).max()]
    if p["has_legs"]:
        errors += [np.abs(np.linalg.norm(p["knee"] - p["hips"], axis=-1) - d.thigh).max(),
                   np.abs(np.linalg.norm(p["ankle"] - p["knee"], axis=-1) - d.shank).max()]
    g["bones"] = gate(float(f"{max(errors):.2e}"), "< 1e-3 m", max(errors) < BONE_TOL_M)

    pel = p["pelvis"][:, None, :]
    hand_rel = p["hand"] - pel
    hand_pop = float(np.linalg.norm(np.diff(hand_rel, axis=0), axis=-1).max())
    elbow_flex = 180.0 - inner_angle_deg(p["shoulders"], p["elbow"], p["hand"])
    g["elbow"] = gate({"max_deg": round(float(elbow_flex.max()), 1), "min_deg": round(float(elbow_flex.min()), 1)},
                      "0-155 deg", bool(elbow_flex.max() <= ELBOW_MAX_DEG and elbow_flex.min() >= 0.0))
    arm_share = float((p["arm_clamp_m"] > CLAMP_TOL_M).mean(axis=0).max())

    if p["has_legs"]:
        low = np.minimum(np.minimum(p["heel"][..., 1], p["ball"][..., 1]), p["toe"][..., 1])
        worst = float(low.min())
        g["penetration"] = gate({"min_m": round(worst, 4), "frames_below": int((low < -PENETRATION_M).sum())},
                                ">= -0.005 m (heel, toe)", worst >= -PENETRATION_M)
        drift, drift90, median_speed, stances = skate_metrics(p, source=True)
        rig_drift, _, rig_speed, _ = skate_metrics(p, source=False)
        if stances:
            g["skate"] = gate({"max_drift_m": round(drift, 4), "p90_drift_m": round(drift90, 4),
                               "median_mps": round(median_speed, 4), "stances": stances,
                               "rig_foot_max_drift_m": round(rig_drift, 4), "rig_foot_median_mps": round(rig_speed, 4)},
                              "drift < 0.01 m every stance; median < 0.02 m/s",
                              drift < SKATE_DRIFT_M and median_speed < SKATE_MEDIAN_MPS)
        else:
            g["skate"] = gate({"stances": 0}, "no labelled contacts", None)
        foot_pts = np.stack([p["ankle"], p["heel"], p["toe"]], axis=2) - pel[:, :, None, :]
        foot_pop = float(np.linalg.norm(np.diff(foot_pts, axis=0), axis=-1).max())
        g["pops"] = gate({"feet_m": round(foot_pop, 4), "hands_m": round(hand_pop, 4)}, "< 0.12 m feet, < 0.20 m hands per frame",
                         foot_pop < POP_FOOT_M and hand_pop < POP_HAND_M)
        knee_flex = 180.0 - inner_angle_deg(p["hips"], p["knee"], p["ankle"])
        # Hyperextension: the knee bends the wrong way about the pelvis's lateral axis (+X). A
        # normal flexion turns the shank posterior of the thigh, so thigh x shank points to -X.
        # Checked where the knee's bend plane is roughly sagittal (|cos| >= HYPEREXT_PLANE).
        bend_axis = np.cross(p["knee"] - p["hips"], p["ankle"] - p["knee"])
        bend_axis /= np.maximum(np.linalg.norm(bend_axis, axis=-1, keepdims=True), 1e-12)
        lateral = p["pelvis_frame"][:, None, :, 0]
        cos = np.sum(bend_axis * lateral, axis=-1)
        backwards = (cos > HYPEREXT_PLANE) & (knee_flex > HYPEREXT_MIN_DEG)
        g["knee"] = gate({"max_deg": round(float(knee_flex.max()), 1), "hyperextended_frames": int(backwards.sum())},
                         "0-160 deg, no hyperextension", bool(knee_flex.max() <= KNEE_MAX_DEG and not backwards.any()))
        thigh = np.stack([to_local(p["pelvis_frame"], p["knee"][:, i] - p["hips"][:, i]) for i in range(2)], axis=1)
        # Flexion in the pelvis's sagittal plane (+Z posterior, so anterior = -Z), measured only
        # where the thigh lies mostly in that plane (a thigh swung out sideways has no flexion angle).
        hip_flex = np.degrees(np.arctan2(-thigh[..., 2], -thigh[..., 1]))
        hip_flex = np.where(hip_flex < -90.0, hip_flex + 360.0, hip_flex)   # past 180 deg flexion, not -180
        in_plane = np.hypot(thigh[..., 1], thigh[..., 2]) >= HIP_IN_PLANE * np.linalg.norm(thigh, axis=-1)
        flex = hip_flex[in_plane]
        g["hip"] = gate({"min_deg": round(float(flex.min()), 1), "max_deg": round(float(flex.max()), 1),
                         "out_of_plane_frames": int((~in_plane).any(axis=1).sum())},
                        "-30..+135 deg (pelvis segment, sagittal)",
                        bool(HIP_FLEX_DEG[0] <= flex.min() and flex.max() <= HIP_FLEX_DEG[1]))
        leg_share = float((p["leg_clamp_m"] > CLAMP_TOL_M).mean(axis=0).max())
        leg_clamp_max = float(p["leg_clamp_m"].max())
    else:
        reason = "no floor or no heel/toe joints: legs stay procedural for this clip"
        for key in ("penetration", "skate", "knee", "hip"):
            g[key] = gate(reason, "-", None)
        g["pops"] = gate({"hands_m": round(hand_pop, 4)}, "< 0.20 m hands per frame", hand_pop < POP_HAND_M)
        leg_share = leg_clamp_max = None
    g["reach"] = gate({"legs_share": None if leg_share is None else round(leg_share, 4), "arms_share": round(arm_share, 4),
                       "legs_max_clamp_m": None if leg_clamp_max is None else round(leg_clamp_max, 4),
                       "arms_max_clamp_m": round(float(p["arm_clamp_m"].max()), 4)},
                      "< 5 % of frames clamped (end joint > 5 mm from its target) per limb",
                      arm_share < REACH_MAX_SHARE and (leg_share is None or leg_share < REACH_MAX_SHARE))
    g["sync"] = {"value": None, "limit": "heel strike median < 30 ms, p90 < 60 ms; tool strike < 20 ms",
                 "result": "PENDING", "note": "needs the S6 time warp to simulator anchors; measured on the payload in S6"}

    failed = [key for key in GATE_ORDER if g[key]["result"] == "FAIL"]
    # Clips that fail only on skate would pass if stage S6 locks the labelled contacts; S6 must
    # then measure skate (and everything else) again on the real payload.
    only_skate = failed == ["skate"]
    card["gates"] = {"stage": "S5 preview", "payload_fs": PAYLOAD_FS,
                     "rig": {"H": REFERENCE_H, "mass_kg": REFERENCE_MASS, "scale": round(d.leg / float(np.load(CLIPS / f"{name}.npz")["leg_len"].mean()), 4)},
                     "method": "clip retargeted onto rig.Dims with rig.two_bone on its own pelvis path (see tools/gates.py)",
                     **{key: g[key] for key in GATE_ORDER}, "failed": failed, "fails_only_skate": only_skate}
    if not card.get("problems"):            # S2 problems keep a clip rejected whatever the gates say
        card["status"] = "rejected" if failed else "gated"
    save_card(card)
    return card


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    counts = {"gated": 0, "rejected": 0}
    for name in clip_names(prefix):
        card = evaluate(name)
        g = card["gates"]
        counts[card["status"]] = counts.get(card["status"], 0) + 1
        def show(key):
            v = g[key]["value"]
            return f"{key}={g[key]['result']}"
        print(f"{name:40s} {card['status']:8s} " + " ".join(show(k) for k in GATE_ORDER if g[k]["result"] in ("FAIL",)))
    print("gates:", counts)


if __name__ == "__main__":
    main()
