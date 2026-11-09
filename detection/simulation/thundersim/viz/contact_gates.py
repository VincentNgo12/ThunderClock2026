# File: contact_gates.py
# Purpose: Gates of the viewer's contact and balance pass (contact.py), measured on the real viewer
#          payload of the regression set: handholds, object penetration, foot skate, balance,
#          joint limits and frame pops.
# Contributors: Unassigned
"""Run from detection/simulation:

    .venv/bin/python -m thundersim.viz.contact_gates                      # contact pass as configured
    .venv/bin/python -m thundersim.viz.contact_gates --contact 0 --out before.json
    .venv/bin/python -m thundersim.viz.contact_gates --scenarios A-CARRY --seeds 0

Every metric is measured on active frames: ground-truth phases activity, calibration stand,
recovery and device off (contact_model.ACTIVE_PHASES), outside the ragdoll's windows
(meta.physics.windows). Gates (limits from the step 4 brief):
  handhold     hand to its handhold while an object is held <= 1 cm; frames where the hand's IK target is
               on the handhold but out of the arm's reach are counted separately and must be 0
  penetration  sack into the drawn thighs, shanks and torso from lift_grasp to put_down_release
               (carried or handled), and into the floor; the held stepladder's front rails into the
               legs and torso; <= 5 mm. The worker later walking through a sack left on the floor
               (the simulated path crosses it) is reported apart
  skate        stance = the rig's planted frames (rig.targets['planted']); every heel or toe contact
               in a stance (point within 1 cm of its support for >= 3 frames) drifts <= 1 cm in xz;
               heels within 1 cm of the floor on planted feet while squatting in a lift or put-down
               (pelvis at least 5 cm below its height at the lift's start). Sliding of a
               heel or toe on the floor outside stances (mode blends, kneeling) is reported
  floor_drag   a heel or toe within 5 mm of the floor (this frame and the previous one) moving faster
               than 0.5 m/s horizontally, in or out of a stance: 0 frames (a swing that skims the
               floor, review 1). Climbing frames (the rig's ladder steps) are reported apart
  balance      static balance: share of quasi-static frames (centre of mass slower than 0.15 m/s) with
               the centre of mass (segment masses + held load) outside the support polygon <= 2 %.
               The support polygon is the hull of every weight-bearing point (soles, a lying boot
               face, knees and shins when kneeling, hands on the floor). Reported apart, not gated:
               quasi-static frames with one foot in the air (a step: the simulated pelvis does not
               shift weight onto the stance foot first, so the centre of mass sits between the
               feet), moving frames (walking, running: the zero-moment point, which depends on the
               simulated trunk path), frames on a ladder (feet on rungs, hands on or reaching for the
               rails), frames without contact
  joints       knee 0-160 deg and not hyperextended, elbow 0-155 deg, hip -30..+135 deg (pelvis
               segment): 0 violating frames
  pops         per-frame move relative to the pelvis: feet < 0.12 m, hands < 0.20 m (the curator's
               limits), a held sack < 0.20 m; a resting sack does not move (< 1 mm). Knees and
               elbows moving 0.15 m or more in a frame are reported, not gated
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from ..model_api import GRAVITY
from .contact_model import (ACTIVE_PHASES, ELBOW_RANGE, HIP_RANGE, KNEE_RANGE, SACK_HALF, STEPLADDER_KG, body_com,
                            box_signed_distance, contact_points, joint_angles, leg_spheres, rail_distance, rail_point,
                            rung_tops, sack_axes, sack_handholds, sack_plan, stepladder_handhold, support_polygon_margin,
                            support_under, torso_spheres)
from .scene import body_scale

REGRESSION = ("A-HAMMER", "A-CARRY", "A-LADDER", "A-KNEEL", "A-REBAR", "A-RUN", "L-TRIP", "H-SCAF", "ARR")
SEEDS = (0, 1, 2)
HANDHOLD_M = 0.01
PENETRATION_M = 0.005
SKATE_M = 0.01
FLOOR_DRAG_H_M = 0.005         # a heel or toe this close to its support (this frame and the one before) ...
FLOOR_DRAG_SPEED = 0.5         # ... moving faster than this (m/s, horizontal) is dragged along the floor
                               # [design: review 1; real minimum toe clearance in swing is ~1.3 cm, Winter 1992]
CONTACT_H_M = 0.01             # a heel or toe tip this close to its support is on it (payload_gates.py)
HEEL_UP_M = 0.01
SQUAT_DROP_M = 0.05            # a lift's pelvis this far below its height at the lift's start = squatting
COM_OUT_SHARE = 0.02
STATIC_SPEED = 0.15            # centre of mass slower than this (m/s, horizontal) = quasi-static [design]
ZMP_CUTOFF_HZ = 6.0            # low-pass of the positions before differentiating (Winter 2009 style) [design]
POPS = {"feet": 0.12, "hands": 0.20, "knees_elbows": 0.15, "sack": 0.20}
FEET, HANDS, KNEES_ELBOWS = (4, 5, 6, 7, 8, 9), (14, 15), (2, 3, 12, 13)


def active_mask(payload) -> np.ndarray:
    n = payload["meta"]["frames"]
    t = np.arange(n) / payload["meta"]["fps"]
    phase = np.asarray(payload["frames"]["phase"])
    act = np.isin(phase, ACTIVE_PHASES)
    for t0, t1 in payload["meta"]["physics"].get("windows", []):
        act &= ~((t >= t0 - 1e-6) & (t <= t1 + 1e-6))
    return act


def rail_holds(rig, scene, t) -> list:
    """(lad, k0, k1) ranges where the arms are in 'ladder' mode, blended in, on a standing ladder."""
    out = []
    modes = rig.arm_modes
    for i, (t0, t1, mode) in enumerate(modes):
        if mode != "ladder":
            continue
        t_next = modes[i + 1][0] if i + 1 < len(modes) else t[-1] + 1.0
        lad = next((l for l in scene.ladders if l.t_up[0] - 1e-6 <= t0 <= (l.t_down[1] if l.t_down else np.inf)), None)
        if lad is None:
            continue
        k0, k1 = int(np.searchsorted(t, t1)), int(np.searchsorted(t, t_next))
        if k1 > k0:
            out.append((lad, k0, k1))
    return out


def measure(payload: dict, keep: dict) -> dict:
    """All gate metrics of one payload (with keep = run_payload's keep dict)."""
    rig, scene, ctx, frames = keep["rig"], keep["scene"], keep["ctx"], keep["frames"]
    n = payload["meta"]["frames"]
    fps = payload["meta"]["fps"]
    t = np.arange(n) / fps
    J = np.asarray(payload["frames"]["joints"], float).reshape(n, 18, 3)
    spine = np.asarray(payload["frames"]["spine"], float).reshape(n, 4, 3)
    pelvis = spine[:, 0]
    pelvis_R = Rotation.from_quat(np.asarray(payload["frames"]["segq"], float).reshape(n, 4, 4)[:, 0]).as_matrix()
    act = active_mask(payload)
    s, w = body_scale(ctx.worker)
    dyn = payload["dyn"]
    contact_meta = payload["meta"].get("contact") or {}
    out: dict = {"frames": n, "active_frames": int(act.sum())}

    # --- held objects ----------------------------------------------------------------------
    hold = {"max_m": 0.0, "frames": 0, "over_frames": 0, "unreachable_frames": 0, "objects": {}}
    hand_target = rig.targets["hand"][:n]

    def clamped(joint, handhold):
        """Frames where the hand's IK target is on the handhold but the arm cannot reach it."""
        side = joint - 14
        on = np.linalg.norm(hand_target[:, side] - handhold, axis=1) <= 0.002
        return on & (np.linalg.norm(J[:, joint] - hand_target[:, side], axis=1) > 0.005)

    def add_hold(name, dist, sel, unreachable=None):
        sel = sel & act[: len(sel)]
        if not sel.any():
            return
        un = np.zeros(len(sel), bool) if unreachable is None else unreachable
        judged = sel & ~un
        d = dist[judged] if judged.any() else np.zeros(0)
        o = hold["objects"].setdefault(name, {"max_m": 0.0, "frames": 0, "over_frames": 0, "unreachable_frames": 0})
        o["max_m"] = round(max(o["max_m"], float(d.max()) if len(d) else 0.0), 4)
        o["frames"] += int(sel.sum())
        o["over_frames"] += int((d > HANDHOLD_M).sum())
        o["unreachable_frames"] += int((sel & un).sum())
        hold["max_m"] = max(hold["max_m"], o["max_m"])
        hold["frames"] += int(sel.sum())
        hold["over_frames"] += int((d > HANDHOLD_M).sum())
        hold["unreachable_frames"] += int((sel & un).sum())

    sacks = sack_plan(ctx, scene)
    channels = contact_meta.get("sack_channels") or [sk.index for sk in sacks]
    sack_pose = {}                                  # channel -> (N, 4) pose
    sack_held = {}                                  # channel -> (N,) bool, in the air
    for sk in sacks:
        ch = channels[sk.index] if sk.index < len(channels) else sk.index
        key = f"bag{ch}"
        if key not in dyn:
            continue
        b = np.asarray(dyn[key], float).reshape(n, 4)
        sack_pose[ch] = b
        sack_held.setdefault(ch, np.zeros(n, bool))
        sack_held[ch] |= (t >= sk.t_off) & (t < sk.t_touch)
        k0, k1 = int(np.searchsorted(t, sk.t_grasp)), int(np.searchsorted(t, sk.t_release))
        sel = np.zeros(n, bool)
        sel[k0:k1] = True
        hh = sack_handholds(b[:, :3], b[:, 3], sk.grips(t))
        for side in (0, 1):
            add_hold("sack", np.linalg.norm(J[:, 14 + side] - hh[:, side], axis=1), sel, clamped(14 + side, hh[:, side]))
    grips = contact_meta.get("rail_grips")     # [[side, k0, k1, ladder index], ...] from the pass
    for lad, k0, k1 in rail_holds(rig, scene, t):
        li = next(i for i, l in enumerate(scene.ladders) if l is lad)
        for side in (0, 1):
            sel = np.zeros(n, bool)
            if grips is None:
                sel[k0:k1] = True
                windows = [(k0, k1)]
            else:
                windows = [(a, b) for sd, a, b, lj in grips if sd == side and lj == li and b > k0 and a < k1]
                for a, b in windows:
                    sel[max(a, k0):min(b, k1)] = True
            dist = rail_distance(lad, side, J[:, 14 + side])
            # a grip slides when the hand's height along the rail changes inside one grip window
            slide = np.zeros(n)
            for a, b in windows:
                a, b = max(a, k0), min(b, k1)
                if b > a:
                    y = J[a:b, 14 + side, 1]
                    slide[a:b] = np.abs(y - y[0]) / np.cos(np.arctan(0.25))
            on_line = rail_point(lad, side, hand_target[:, side, 1])
            add_hold("ladder_rails", np.hypot(dist, slide), sel, clamped(14 + side, on_line))
    if "stepladder" in dyn:
        st = np.asarray(dyn["stepladder"], float).reshape(n, 5)
        height = next((p["h"] for p in scene.props if p.get("k") == "stepladder"), 1.0)
        sel = st[:, 4] < 1.0 - 1e-6
        along = np.full(n, 0.5)
        for k_pick, a in contact_meta.get("stepladder_along", []):
            along[k_pick - int(round(1.0 * fps)):] = a           # from just before its pick-up
        hh = stepladder_handhold(st[:, :3], st[:, 3], st[:, 4], height, along)
        add_hold("stepladder", np.linalg.norm(J[:, 15] - hh, axis=1), sel, clamped(15, hh))
    hold["max_m"] = round(hold["max_m"], 4)
    out["handhold"] = hold

    # --- sack penetration ------------------------------------------------------------------
    pen = {"max_m": 0.0, "frames_over": 0, "floor_max_m": 0.0, "worst": None}
    if sack_pose:
        legs_c, legs_r, _ = leg_spheres(J, s, w)
        torso_c, torso_r = torso_spheres(ctx, spine, frames)
        cen = np.concatenate([legs_c, torso_c], axis=1)
        rad = np.concatenate([legs_r, torso_r])
        handled = {}
        for sk in sacks:
            ch = channels[sk.index] if sk.index < len(channels) else sk.index
            handled.setdefault(ch, np.zeros(n, bool))
            handled[ch] |= (t >= sk.t_grasp) & (t <= sk.t_release)
        pen["walk_through_max_m"], pen["walk_through_frames"] = 0.0, 0
        for idx, b in sack_pose.items():
            R = sack_axes(b[:, 3])
            depth = (rad[None, :] - box_signed_distance(cen, b[:, :3], R, SACK_HALF)).max(axis=1)
            mine = act & handled[idx]
            gated = np.where(mine, depth, -np.inf)
            k = int(np.argmax(gated))
            if gated[k] > pen["max_m"]:
                pen["max_m"], pen["worst"] = float(gated[k]), [idx, round(float(t[k]), 2)]
            pen["frames_over"] += int((gated > PENETRATION_M).sum())
            other = np.where(act & ~handled[idx], depth, -np.inf)
            pen["walk_through_max_m"] = round(max(pen["walk_through_max_m"], float(other.max())), 4)
            pen["walk_through_frames"] += int((other > PENETRATION_M).sum())
            bottom = b[:, 1] - SACK_HALF[1]
            sup = support_under(scene, np.column_stack([b[:, 0], bottom, b[:, 2]])[:, None, :], t)[:, 0]
            pen["floor_max_m"] = max(pen["floor_max_m"], float(np.where(act, sup - bottom, 0.0).max()))
    if "stepladder" in dyn:
        # the stepladder's front rails (capsules) into the drawn legs and torso while it is held
        from .contact import RAIL_R, ladder_rails, segment_distance

        st = np.asarray(dyn["stepladder"], float).reshape(n, 5)
        height = next((p["h"] for p in scene.props if p.get("k") == "stepladder"), 1.0)
        legs_c, legs_r, _ = leg_spheres(J, s, w)
        torso_c, torso_r = torso_spheres(ctx, spine, frames)
        cen = np.concatenate([legs_c, torso_c], axis=1)
        rad = np.concatenate([legs_r, torso_r])
        pen["stepladder_frames_over"] = 0
        for k in np.flatnonzero((st[:, 4] < 1.0 - 1e-6) & act):
            depth = max(float((RAIL_R + rad - segment_distance(cen[k], st[k, :3] + p0, st[k, :3] + p1)).max())
                        for p0, p1 in ladder_rails(st[k, 4], st[k, 3], height))
            if depth > pen["max_m"]:
                pen["max_m"], pen["worst"] = depth, ["stepladder", round(float(t[k]), 2)]
            pen["frames_over"] += int(depth > PENETRATION_M)
            pen["stepladder_frames_over"] += int(depth > PENETRATION_M)
    pen["max_m"] = round(pen["max_m"], 4)
    pen["floor_max_m"] = round(pen["floor_max_m"], 4)
    out["penetration"] = pen

    # --- skate and heels -----------------------------------------------------------------------
    rungs = rung_tops(scene, t)
    pts = J[:, [6, 7, 8, 9]]                      # heel l, heel r, toe l, toe r
    height = pts[..., 1] - support_under(scene, pts, t, rungs)
    planted = rig.targets["planted"][:n]
    drifts = []
    worst = (0.0, None)
    for p in range(4):
        side = p % 2
        # stance = the rig's planted frames; a heel or toe on its support inside one is a contact
        down = np.flatnonzero((height[:, p] <= CONTACT_H_M) & act & planted[:, side])
        for run in np.split(down, np.flatnonzero(np.diff(down) > 1) + 1):
            if len(run) < 3:
                continue
            xz = pts[run, p][:, [0, 2]]
            d = float(np.linalg.norm(xz - xz[0], axis=1).max())
            drifts.append(d)
            if d > worst[0]:
                worst = (d, [["heel_l", "heel_r", "toe_l", "toe_r"][p], round(float(t[run[0]]), 2), round(float(t[run[-1]]), 2)])
    # informational: a heel or toe sliding on the floor outside a stance (mode blends, kneeling)
    slide = 0.0
    for p in range(4):
        down = np.flatnonzero((height[:, p] <= CONTACT_H_M) & act & ~planted[:, p % 2])
        for run in np.split(down, np.flatnonzero(np.diff(down) > 1) + 1):
            if len(run) >= 3:
                xz = pts[run, p][:, [0, 2]]
                slide = max(slide, float(np.linalg.norm(xz - xz[0], axis=1).max()))
    # dragged: a heel or toe on the floor moving fast, in or out of a stance (a swing that skims
    # the floor, which the stance drift above never sees). Floors and walkables only. Climbing (the
    # rig's ladder steps, which the pass does not re-pose: toes brushing a rung as they leave it,
    # a toe leaving the floor for the first rung) is counted apart (ladder_drag_frames, reported).
    step = np.vstack([np.zeros((1, 4)), np.linalg.norm(np.diff(pts[..., [0, 2]], axis=0), axis=2)])
    fast = (step * fps > FLOOR_DRAG_SPEED) & act[:, None]
    climbing = (mode_per_frame(rig.leg_modes, t) == "ladder")[:, None]

    def low_twice(h):
        return (h <= FLOOR_DRAG_H_M) & np.vstack([np.ones((1, 4), bool), h[:-1] <= FLOOR_DRAG_H_M])
    on_floor = low_twice(pts[..., 1] - support_under(scene, pts, t, []))
    dragged = on_floor & fast & ~climbing
    ladder_dragged = low_twice(height) & fast & climbing
    lifting = np.zeros(n, bool)                   # squatting: in a lift, pelvis >= 5 cm below its start
    for c in ctx.rec.calls:
        if c.name in ("lift", "put_down"):
            inside = (t >= c.t0) & (t <= c.t1)
            if inside.any():
                y0 = pelvis[np.flatnonzero(inside)[0], 1]
                lifting |= inside & (pelvis[:, 1] <= y0 - SQUAT_DROP_M)
    heel_up = (planted & lifting[:, None] & act[:, None] & (height[:, :2] > HEEL_UP_M)).any(axis=1)
    out["skate"] = {"max_drift_m": round(max(drifts), 4) if drifts else 0.0, "contacts": len(drifts),
                    "over": int(sum(d > SKATE_M for d in drifts)), "worst": worst[1],
                    "off_stance_slide_m": round(slide, 4),
                    "floor_drag_frames": int(dragged.sum()), "floor_drag_m": round(float(step[dragged].sum()), 3),
                    "ladder_drag_frames": int(ladder_dragged.sum()),
                    "lift_heel_up_frames": int(heel_up.sum()), "lift_frames": int((lifting & act).sum())}

    # --- balance -----------------------------------------------------------------------------
    com = body_com(ctx, J, spine, frames)
    total = ctx.worker.mass_kg
    load_pos, load_m = np.zeros((n, 3)), np.zeros(n)
    for sk in sacks:
        ch = channels[sk.index] if sk.index < len(channels) else sk.index
        if ch in sack_pose:
            held = (t >= sk.t_off) & (t < sk.t_touch)
            load_pos[held] = sack_pose[ch][held, :3]
            load_m[held] = sk.mass
    if "stepladder" in dyn:
        st = np.asarray(dyn["stepladder"], float).reshape(n, 5)
        carried = st[:, 4] < 0.5
        load_pos[carried] = st[carried, :3]
        load_m[carried] = STEPLADDER_KG
    com_all = (com * total + load_pos * load_m[:, None]) / (total + load_m)[:, None]
    points, on = contact_points(scene, J, t, s, rungs)
    sos = butter(2, ZMP_CUTOFF_HZ, fs=fps, output="sos")
    smooth = sosfiltfilt(sos, com_all, axis=0) if n > 20 else com_all
    vel = np.gradient(smooth, 1.0 / fps, axis=0)
    zmp = zmp_points(ctx, J, spine, frames, load_pos, load_m, fps)
    speed = np.hypot(vel[:, 0], vel[:, 2])
    grasp = np.zeros(n, bool)
    for lad, k0, k1 in rail_holds(rig, scene, t):
        grasp[k0:k1] = True
    leg_mode = mode_per_frame(rig.leg_modes, t)
    margin = np.full(n, np.nan)
    kind = np.full(n, "", dtype=object)
    for k in np.flatnonzero(act):
        p = points[k, on[k]][:, [0, 2]]
        if not len(p):
            kind[k] = "flight"
            continue
        feet_on = [bool(on[k, 8 * side:8 * side + 8].any()) for side in (0, 1)]
        stepping = leg_mode[k] == "stand" and sum(feet_on) == 1 and not on[k, 16:].any()
        if grasp[k] or leg_mode[k] == "ladder":
            kind[k] = "grasp"
        elif speed[k] >= STATIC_SPEED:
            kind[k] = "dynamic"
        else:
            kind[k] = "step" if stepping else "static"
        point = zmp[k] if kind[k] == "dynamic" else com_all[k, [0, 2]]
        margin[k] = support_polygon_margin(list(p), point)
    judged = kind == "static"
    outside = judged & (margin < 0)
    bal = {"judged_frames": int(judged.sum()), "outside_frames": int(outside.sum()),
           "outside_share": round(float(outside.sum() / max(judged.sum(), 1)), 4)}
    for name in ("static", "step", "dynamic", "grasp"):
        sel = kind == name
        bal[name] = {"frames": int(sel.sum()), "outside": int((sel & (margin < 0)).sum()),
                     "worst_m": round(float(np.nanmin(margin[sel])), 3) if sel.any() else None}
    bal["flight_frames"] = int((kind == "flight").sum())
    worst_k = np.flatnonzero(outside)
    bal["outside_times"] = summarise_times(t[worst_k])
    out["balance"] = bal

    # --- joint limits --------------------------------------------------------------------------
    ang = joint_angles(J, pelvis_R)
    knee_bad = ((ang["knee"] > KNEE_RANGE[1]) | (ang["knee"] < KNEE_RANGE[0]) | ang["hyper"]).any(axis=1)
    elbow_bad = ((ang["elbow"] > ELBOW_RANGE[1]) | (ang["elbow"] < ELBOW_RANGE[0])).any(axis=1)
    hip = ang["hip"]
    hip_bad = ((hip < HIP_RANGE[0]) | (hip > HIP_RANGE[1])).any(axis=1)
    out["joints"] = {"knee_frames": int((knee_bad & act).sum()), "elbow_frames": int((elbow_bad & act).sum()),
                     "hip_frames": int((hip_bad & act).sum()),
                     "knee_max": round(float(np.nanmax(np.where(act[:, None], ang["knee"], np.nan))), 1),
                     "elbow_max": round(float(np.nanmax(np.where(act[:, None], ang["elbow"], np.nan))), 1),
                     "hip_min": round(float(np.nanmin(np.where(act[:, None], hip, np.nan))), 1),
                     "hip_max": round(float(np.nanmax(np.where(act[:, None], hip, np.nan))), 1),
                     "hip_times": summarise_times(t[hip_bad & act]),
                     "knee_times": summarise_times(t[knee_bad & act]),
                     "elbow_times": summarise_times(t[elbow_bad & act])}

    # --- pops -----------------------------------------------------------------------------------
    pair = np.zeros(n, bool)
    pair[1:] = act[1:] & act[:-1]
    rel = J - pelvis[:, None, :]
    step = np.zeros((n, 18))
    step[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=-1)
    pops = {}
    for name, idx in (("feet", FEET), ("hands", HANDS), ("knees_elbows", KNEES_ELBOWS)):
        m = step[:, idx].max(axis=1)
        pops[name] = {"max_m": round(float(m[pair].max()) if pair.any() else 0.0, 4),
                      "frames": int((pair & (m >= POPS[name])).sum()),
                      "times": summarise_times(t[pair & (m >= POPS[name])])}
    sack_step, rest_step = 0.0, 0.0
    for ch, b in sack_pose.items():
        held = sack_held[ch]
        d_rel = np.zeros(n)
        d_rel[1:] = np.linalg.norm(np.diff(b[:, :3] - pelvis, axis=0), axis=1)
        d_abs = np.zeros(n)
        d_abs[1:] = np.linalg.norm(np.diff(b[:, :3], axis=0), axis=1)
        sel = pair & held
        resting = pair & ~held
        resting[1:] &= ~held[:-1]
        sack_step = max(sack_step, float(d_rel[sel].max()) if sel.any() else 0.0)
        rest_step = max(rest_step, float(d_abs[resting].max()) if resting.any() else 0.0)
    pops["sack"] = {"held_max_m": round(sack_step, 4), "resting_max_m": round(rest_step, 4)}
    out["pops"] = pops
    return out


def mode_per_frame(modes: list, t: np.ndarray) -> np.ndarray:
    """Name of the latest rig timeline item (t0, t1, mode) started at each frame time."""
    starts = np.array([m[0] for m in modes])
    idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(modes) - 1)
    return np.array([modes[i][2] for i in idx], dtype=object)


def zmp_points(ctx, J, spine, frames, load_pos, load_m, fps) -> np.ndarray:
    """Zero-moment point (N, 2: x, z) of the drawn body plus a held load: point masses at the joints
    (de Leva) and the torso centroid, positions low-passed at ZMP_CUTOFF_HZ, ignoring segment
    rotational inertia (the multi-point cart-table formula)."""
    from .physics import JOINT_MASS, TRUNK_M, torso_centroid
    from .scene import spine_pose

    n = len(J)
    s, w = body_scale(ctx.worker)
    _, M = spine_pose(ctx, frames)
    mass = ctx.worker.mass_kg
    pos = np.concatenate([J, torso_centroid(spine, M, ctx.H, s, w)[:, None, :], load_pos[:, None, :]], axis=1)
    m = np.concatenate([np.broadcast_to(JOINT_MASS * mass, (n, 18)), np.full((n, 1), TRUNK_M * mass), load_m[:, None]], axis=1)
    if n > 20:
        pos = sosfiltfilt(butter(2, ZMP_CUTOFF_HZ, fs=fps, output="sos"), pos, axis=0)
    acc = np.gradient(np.gradient(pos, 1.0 / fps, axis=0), 1.0 / fps, axis=0)
    ground = np.min(J[:, 6:10, 1], axis=1)[:, None]
    fy = m * (acc[..., 1] + GRAVITY)
    den = fy.sum(axis=1)
    weak = den < 0.2 * m.sum(axis=1) * GRAVITY        # near free fall: no meaningful ZMP
    den = np.where(weak, np.nan, den)
    x = (fy * pos[..., 0] - m * acc[..., 0] * (pos[..., 1] - ground)).sum(axis=1) / den
    z = (fy * pos[..., 2] - m * acc[..., 2] * (pos[..., 1] - ground)).sum(axis=1) / den
    return np.column_stack([x, z])


def summarise_times(times: np.ndarray, gap: float = 0.1) -> list:
    """Compress frame times into [[t0, t1], ...] runs (at most 12)."""
    if len(times) == 0:
        return []
    runs, start, prev = [], times[0], times[0]
    for x in times[1:]:
        if x - prev > gap:
            runs.append([round(float(start), 2), round(float(prev), 2)])
            start = x
        prev = x
    runs.append([round(float(start), 2), round(float(prev), 2)])
    return runs[:12]


def verdict(runs: list) -> dict:
    """Aggregate per-run metrics into PASS/FAIL per gate."""
    def worst(path, fn=max):
        vals = []
        for r in runs:
            v = r
            for key in path:
                v = v.get(key) if isinstance(v, dict) else None
            if v is not None:
                vals.append(v)
        return fn(vals) if vals else 0
    hold_max, unreach = worst(("handhold", "max_m")), sum(r["handhold"]["unreachable_frames"] for r in runs)
    pen, floor = worst(("penetration", "max_m")), worst(("penetration", "floor_max_m"))
    skate, heel = worst(("skate", "max_drift_m")), sum(r["skate"]["lift_heel_up_frames"] for r in runs)
    drag = sum(r["skate"]["floor_drag_frames"] for r in runs)
    drag_m = round(sum(r["skate"]["floor_drag_m"] for r in runs), 3)
    ladder_drag = sum(r["skate"]["ladder_drag_frames"] for r in runs)
    judged = sum(r["balance"]["judged_frames"] for r in runs)
    outside = sum(r["balance"]["outside_frames"] for r in runs)
    share_runs = worst(("balance", "outside_share"))
    joints = sum(r["joints"][k] for r in runs for k in ("knee_frames", "elbow_frames", "hip_frames"))
    pops = sum(r["pops"][k]["frames"] for r in runs for k in ("feet", "hands"))
    knees_elbows = sum(r["pops"]["knees_elbows"]["frames"] for r in runs)
    sack_pop = worst(("pops", "sack", "held_max_m"))
    rest_pop = worst(("pops", "sack", "resting_max_m"))
    gates = {
        "handhold": {"value": {"max_m": hold_max, "unreachable_frames": unreach}, "limit": "<= 0.01 m, 0 unreachable",
                     "pass": hold_max <= HANDHOLD_M and unreach == 0},
        "penetration": {"value": {"body_max_m": pen, "floor_max_m": floor}, "limit": "<= 0.005 m",
                        "pass": pen <= PENETRATION_M and floor <= PENETRATION_M},
        "skate": {"value": {"max_drift_m": skate, "lift_heel_up_frames": heel}, "limit": "<= 0.01 m per contact; heels down",
                  "pass": skate <= SKATE_M and heel == 0},
        "floor_drag": {"value": {"frames": drag, "path_m": drag_m, "climbing_frames_reported": ladder_drag},
                       "limit": "0 frames with a heel or toe within 5 mm of the floor moving > 0.5 m/s",
                       "pass": drag == 0},
        "balance": {"value": {"outside_share_all": round(outside / max(judged, 1), 4), "worst_run_share": share_runs},
                    "limit": "<= 2 % of judged frames (every run)", "pass": share_runs <= COM_OUT_SHARE},
        "joints": {"value": joints, "limit": "0 frames", "pass": joints == 0},
        "pops": {"value": {"feet_hands_frames": pops, "knees_elbows_frames_reported": knees_elbows,
                           "sack_held_max_m": sack_pop, "sack_resting_max_m": rest_pop},
                 "limit": "0 frames; held sack < 0.20 m, resting < 0.001 m",
                 "pass": pops == 0 and sack_pop < POPS["sack"] and rest_pop < 0.001},
    }
    return {"gates": gates, "failed": [k for k, v in gates.items() if not v["pass"]]}


def run_one(job) -> dict:
    scenario, seed, contact = job
    from .export import run_payload

    keep: dict = {}
    kwargs = {"model": "none", "keep": keep}
    if contact is not None and "contact" in inspect.signature(run_payload).parameters:
        kwargs["contact"] = contact
    payload = run_payload(scenario, seed, **kwargs)
    result = {"scenario": scenario, "seed": seed, "contact": (payload["meta"].get("contact") or {}).get("enabled")}
    result.update(measure(payload, keep))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact and balance gates on the viewer payload")
    parser.add_argument("--scenarios", default=",".join(REGRESSION))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--contact", type=int, choices=(0, 1), default=None, help="force the pass off or on")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    contact = None if args.contact is None else bool(args.contact)
    jobs = [(s, int(seed), contact) for s in args.scenarios.split(",") for seed in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(run_one, jobs))
    summary = verdict(runs)
    for r in runs:
        print(f"{r['scenario']:9s} {r['seed']}: hold {r['handhold']['max_m']:.3f} m ({r['handhold']['unreachable_frames']} unreach) | "
              f"pen {r['penetration']['max_m']:.3f}/{r['penetration']['floor_max_m']:.3f} | skate {r['skate']['max_drift_m']:.3f} "
              f"heel-up {r['skate']['lift_heel_up_frames']} | COM out {r['balance']['outside_share']:.3f} | "
              f"joints {r['joints']['knee_frames']}/{r['joints']['elbow_frames']}/{r['joints']['hip_frames']} | "
              f"pops {r['pops']['feet']['frames']}/{r['pops']['hands']['frames']}/{r['pops']['knees_elbows']['frames']}")
    for name, g in summary["gates"].items():
        print(f"  {name:12s} {'PASS' if g['pass'] else 'FAIL'}  {g['value']}  ({g['limit']})")
    if args.out:
        Path(args.out).write_text(json.dumps({"runs": runs, "summary": summary}, indent=1))
    sys.exit(0)


if __name__ == "__main__":
    main()
