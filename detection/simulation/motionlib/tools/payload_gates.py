# File: payload_gates.py
# Purpose: Stage S6 gates of the motion library, measured on the real viewer payload (50 Hz):
#          every clip listed in motionlib/manifest.json is forced into its role on scenarios of the
#          regression set, and the gates are measured where the clip drives the limbs. The same
#          windows of the procedural payload (mocap off) are measured for comparison.
# Contributors: Unassigned
"""Run from detection/simulation after manifest.py:

    .venv/bin/python motionlib/tools/payload_gates.py                 # every listed clip
    .venv/bin/python motionlib/tools/payload_gates.py --role hammer   # one role
    .venv/bin/python motionlib/tools/payload_gates.py --jobs 8 --seeds 0,1,2

Writes card["s6"] for each measured clip and motionlib/s6_gates.json (all runs, plus the procedural
baseline), then rebuilds the manifest. A clip whose S6 gates all pass and whose S5 preview failed
nothing but skate gets status "gated"; any S6 FAIL sets "rejected". Neither switches a clip on.

Gates (limits from .claude/agents/mocap-curator.md), on frames inside the clip's use windows
(payload meta.mocap.uses) outside the fall_descent, free_fall and impact phases:
  bones        |thigh, shank, upper arm, forearm length - rig.Dims| < 1e-3 m
  penetration  heel and toe at most 5 mm below scene.support (legs)
  skate        clip-shaped stances (touchdown on the simulated strike to the clip's toe-off): every
               frame where the rig foot is planted and the heel (or the toe tip) is within 1 cm of
               the support counts as its contact; drift < 1 cm per stance, median speed < 2 cm/s
               (legs). Stance frames where the rig foot is not planted are counted and reported.
  pops         per-frame move relative to the pelvis < 0.12 m (ankle, heel, toe), < 0.20 m (hands)
  knee         0-160 deg flexion, no hyperextension (as gates.py)
  elbow        0-155 deg
  hip          -30..+135 deg flexion relative to the simulated pelvis segment (as gates.py)
  sync         walking: rig touchdown (heel within 3 mm of the support) vs the simulated heel
               strike (bob minimum of track.root_pos[:,1]): median < 30 ms, p90 < 60 ms.
               hammer: the striking hand's closest approach to the impact point vs the impulse
               time: median < 20 ms (p90 reported). kneel_work: share of work impulses matched to
               a clip wrist event (reported, not gated: the simulator's kneel_work impulses are
               not tool strikes).
  reach        < 5 % of frames clamped by two_bone (end joint > 5 mm from its target), per limb
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # our own tools only, never raw/
from cliplib import CARDS, MOTIONLIB  # noqa: E402
from manifest import build as build_manifest  # noqa: E402

ROLE_SCENARIOS = {
    "walk": ("A-HAMMER", "A-CARRY", "A-KNEEL", "A-LADDER", "A-RUN", "L-TRIP"),
    "carry_walk": ("A-CARRY",),
    "hammer": ("A-HAMMER",),
    "kneel_work": ("A-KNEEL",),
}
LEG_ROLES = ("walk", "carry_walk")
EXCLUDED_PHASES = ("fall_descent", "free_fall", "impact")
BONE_TOL_M = 1e-3
PENETRATION_M = 0.005
SKATE_DRIFT_M, SKATE_MEDIAN_MPS = 0.01, 0.02
CONTACT_H_M = 0.01           # a heel or toe tip this close to the support is on it [design]
TOUCH_H_M = 0.003            # the heel has touched down once it is this close [design]
POP_FOOT_M, POP_HAND_M = 0.12, 0.20
KNEE_MAX_DEG, ELBOW_MAX_DEG = 160.0, 155.0
HIP_FLEX_DEG = (-30.0, 135.0)
HYPEREXT_MIN_DEG, HYPEREXT_PLANE, HIP_IN_PLANE = 5.0, 0.5, 0.5   # as gates.py
SYNC_HEEL_MS = (30.0, 60.0)
SYNC_TOOL_MS = 20.0
REACH_MAX_SHARE, CLAMP_TOL_M = 0.05, 0.005
OUT = MOTIONLIB / "s6_gates.json"
J = {name: i for i, name in enumerate(("hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r",
                                       "toe_l", "toe_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "hand_l",
                                       "hand_r", "neck", "head"))}


def inner_angle_deg(a, mid, b):
    u, v = a - mid, b - mid
    cos = np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def bob_minima(track, t0: float, t1: float) -> np.ndarray:
    """Simulated heel strikes of one walk: local minima of the pelvis height (motions.walk puts the
    bob minima and the heel-strike pulses at theta = pi/2 + k pi), slow trend removed."""
    from scipy.signal import find_peaks

    fs = track.fs
    a, b = int(round(t0 * fs)), min(int(round(t1 * fs)), len(track.root_pos) - 1)
    y = np.asarray(track.root_pos[a:b, 1], float)
    if len(y) < int(0.6 * fs):
        return np.array([])
    k = int(0.6 * fs)
    trend = np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    idx, _ = find_peaks(-(y - trend), distance=int(0.25 * fs), prominence=0.003)
    return (a + idx) / fs


_CANDIDATES: dict = {}


def candidates() -> dict:
    """The manifest with S6 failures still listed, so every candidate can be (re)measured."""
    if not _CANDIDATES:
        _CANDIDATES.update(build_manifest(include_s6_failed=True))
    return _CANDIDATES


def measure(scenario: str, seed: int, role: str, clip: str | None) -> dict:
    """One payload (mocap on with `clip` forced into `role`, or off when clip is None, measured on
    the windows of a mocap-on run of the first listed clip) and its gate metrics."""
    from thundersim.spine import PHASES
    from thundersim.viz.export import run_payload
    from thundersim.viz.rig import Dims

    keep: dict = {}
    if clip is None:
        probe: dict = {}
        run_payload(scenario, seed, model="none", mocap=True, mocap_manifest=candidates(), keep=probe)
        uses = [u for u in probe["rig"].mocap["uses"] if u["role"] == role]
        plans = probe["session"]
        payload = run_payload(scenario, seed, model="none", mocap=False, keep=keep)
    else:
        payload = run_payload(scenario, seed, model="none", mocap=True, mocap_only={role: clip},
                              mocap_manifest=candidates(), keep=keep)
        uses = [u for u in keep["rig"].mocap["uses"] if u["role"] == role and u["clip"] == clip]
        plans = keep["session"]
    result = {"scenario": scenario, "seed": seed, "role": role, "clip": clip, "uses": len(uses)}
    if not uses:
        return result
    rig, scene, ctx = keep["rig"], keep["scene"], keep["ctx"]
    fps = payload["meta"]["fps"]
    joints = np.array(payload["frames"]["joints"]).reshape(-1, 18, 3)
    n = len(joints)
    t = np.arange(n) / fps
    phase = np.array(payload["frames"]["phase"])
    bad = np.isin(phase, [k for k, v in PHASES.items() if v in EXCLUDED_PHASES])
    pelvis = np.array(payload["frames"]["spine"]).reshape(n, 4, 3)[:, 0]
    pelvis_q = np.array(payload["frames"]["segq"]).reshape(n, 4, 4)[:, 0]
    from scipy.spatial.transform import Rotation
    pelvis_frame = Rotation.from_quat(pelvis_q).as_matrix()
    d = Dims(ctx.H, ctx.worker.mass_kg)
    legs = role in LEG_ROLES
    limbs = {u["limbs"] for u in uses}
    win = np.zeros(n, bool)
    for u in uses:
        win |= (t >= u["t0"]) & (t <= u["t1"])
    win &= ~bad
    result["frames"] = int(win.sum())
    g: dict = {}

    def jj(name):
        return joints[:, J[name]]

    # bones
    errs = []
    for s in "lr":
        if "legs" in limbs:
            errs += [np.abs(np.linalg.norm(jj(f"knee_{s}") - jj(f"hip_{s}"), axis=1) - d.thigh)[win],
                     np.abs(np.linalg.norm(jj(f"ankle_{s}") - jj(f"knee_{s}"), axis=1) - d.shank)[win]]
        if "arms" in limbs:
            errs += [np.abs(np.linalg.norm(jj(f"elbow_{s}") - jj(f"shoulder_{s}"), axis=1) - d.upper_arm)[win],
                     np.abs(np.linalg.norm(jj(f"hand_{s}") - jj(f"elbow_{s}"), axis=1) - d.forearm)[win]]
    g["bones_m"] = float(np.max(np.concatenate(errs))) if errs else None
    # pops (relative to the pelvis)
    ok_pair = np.zeros(n, bool)
    ok_pair[1:] = win[1:] & win[:-1]
    rel = joints - pelvis[:, None, :]
    step = np.zeros((n, 18))
    step[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=-1)
    feet = [J[f"{p}_{s}"] for p in ("ankle", "heel", "toe") for s in "lr"]
    hands = [J["hand_l"], J["hand_r"]]
    g["pop_feet_m"] = float(step[ok_pair][:, feet].max()) if "legs" in limbs and ok_pair.any() else None
    g["pop_hands_m"] = float(step[ok_pair][:, hands].max()) if ok_pair.any() else None
    # elbows
    if "arms" in limbs:
        flex = np.stack([180.0 - inner_angle_deg(jj(f"shoulder_{s}"), jj(f"elbow_{s}"), jj(f"hand_{s}")) for s in "lr"], 1)[win]
        g["elbow_deg"] = [float(flex.min()), float(flex.max())]
        hand_t = rig.targets["hand"][:n]
        clamp = np.linalg.norm(np.stack([jj("hand_l"), jj("hand_r")], 1) - hand_t, axis=-1)[win]
        g["reach_arms_share"] = float((clamp > CLAMP_TOL_M).mean(axis=0).max())
    if legs:
        hips = np.stack([jj("hip_l"), jj("hip_r")], 1)
        knee = np.stack([jj("knee_l"), jj("knee_r")], 1)
        ankle = np.stack([jj("ankle_l"), jj("ankle_r")], 1)
        heel = np.stack([jj("heel_l"), jj("heel_r")], 1)
        toe = np.stack([jj("toe_l"), jj("toe_r")], 1)
        support = np.zeros((n, 2, 2))
        for k in np.flatnonzero(win):
            for s in (0, 1):
                for q, pt in enumerate((heel[k, s], toe[k, s])):
                    support[k, s, q] = scene.support(pt[0], pt[2], pt[1] + 0.05)
        height = np.stack([heel[..., 1], toe[..., 1]], axis=2) - support
        g["penetration_m"] = float(height[win].min())
        knee_flex = 180.0 - inner_angle_deg(hips, knee, ankle)
        bend_axis = np.cross(knee - hips, ankle - knee)
        bend_axis /= np.maximum(np.linalg.norm(bend_axis, axis=-1, keepdims=True), 1e-12)
        cos = np.sum(bend_axis * pelvis_frame[:, None, :, 0], axis=-1)
        backwards = (cos > HYPEREXT_PLANE) & (knee_flex > HYPEREXT_MIN_DEG)
        g["knee_max_deg"] = float(knee_flex[win].max())
        g["knee_hyperext_frames"] = int(backwards[win].any(axis=1).sum())
        thigh = np.einsum("nji,nsj->nsi", pelvis_frame, knee - hips)
        hip_flex = np.degrees(np.arctan2(-thigh[..., 2], -thigh[..., 1]))
        hip_flex = np.where(hip_flex < -90.0, hip_flex + 360.0, hip_flex)
        in_plane = np.hypot(thigh[..., 1], thigh[..., 2]) >= HIP_IN_PLANE * np.linalg.norm(thigh, axis=-1)
        sel = in_plane & win[:, None]
        g["hip_deg"] = [float(hip_flex[sel].min()), float(hip_flex[sel].max())]
        clamp = np.linalg.norm(ankle - rig.targets["ankle"][:n], axis=-1)[win]
        g["reach_legs_share"] = float((clamp > CLAMP_TOL_M).mean(axis=0).max())
        # skate and sync over the clip-shaped steps of the plans used
        drifts, speeds, sync_ms = [], [], []
        planted = rig.targets["planted"][:n]
        unplanted = 0
        for plan in plans.walk_plans.values():
            if plan.role_name != role or (clip is not None and plan.clip.name != clip):
                continue
            times = sorted(plan.step_times, key=lambda x: x[2])
            strikes = bob_minima(ctx.track, plan.call.t0, plan.call.t1)
            for side, _, t_land in times:
                if t_land < plan.s[1] - 1e-6:
                    continue
                nxt = [lift for sd, lift, land in times if sd == side and lift > t_land]
                t_end = min(nxt) if nxt else t_land + 0.4
                in_st = (t >= t_land) & (t <= t_end) & win
                idx = np.flatnonzero(in_st)
                if len(idx) < 3:
                    continue
                unplanted += int((~planted[idx, side]).sum())
                for q, pts in enumerate((heel, toe)):
                    down = idx[(height[idx, side, q] <= CONTACT_H_M) & planted[idx, side]]
                    # One stance per run of consecutive contact frames (a step the rig took on its
                    # own inside the window splits it).
                    for run in np.split(down, np.flatnonzero(np.diff(down) > 1) + 1):
                        if len(run) < 2:
                            continue
                        xz = pts[run, side][:, [0, 2]]
                        drifts.append(float(np.linalg.norm(xz - xz[0], axis=1).max()))
                        speeds.append(np.linalg.norm(np.diff(xz, axis=0), axis=1) * fps)
                # touchdown: first frame from 0.15 s before the planned landing with the heel down
                look = np.flatnonzero((t >= t_land - 0.15) & (t <= t_land + 0.15) & win)
                down = look[height[look, side, 0] <= TOUCH_H_M] if len(look) else []
                if len(down) and len(strikes):
                    k = down[0]
                    # the heel crosses TOUCH_H_M between frames k-1 and k: interpolate
                    t_touch = t[k]
                    if k > 0 and height[k - 1, side, 0] > TOUCH_H_M:
                        h0, h1 = height[k - 1, side, 0], height[k, side, 0]
                        t_touch = t[k - 1] + (h0 - TOUCH_H_M) / max(h0 - h1, 1e-9) / fps
                    bob = strikes[np.argmin(np.abs(strikes - t_land))]
                    if abs(bob - t_land) < 0.15:
                        sync_ms.append(abs(t_touch - bob) * 1000.0)
        g["skate_max_drift_m"] = max(drifts) if drifts else None
        g["skate_median_mps"] = float(np.median(np.concatenate(speeds))) if speeds else None
        g["stances"] = len(drifts)
        g["stance_frames_not_planted"] = unplanted
        g["sync_heel_ms"] = sync_ms
    if role == "hammer":
        sync = []
        hand_r = jj("hand_r")
        for u in uses:
            for c in ctx.rec.find("hammer"):
                if not (c.t0 - 1e-6 <= u["t0"] <= c.t1 + 1e-6):
                    continue
                strikes = sorted(s for x in ctx.rec.find("impulses", inside=c) for s in x.args.get("times_abs", []))
                # The impact point depends only on the simulated shoulders and the scene, not on mocap.
                impact, _ = plans.rb._hammer_points(slice(0, n), overhead=False)
                dist = np.linalg.norm(hand_r - impact, axis=1)
                for s in strikes:
                    k = int(round(s * fps))
                    if k < 2 or k > n - 3 or not win[k]:
                        continue
                    lo = k - 2 + int(np.argmin(dist[k - 2:k + 3]))
                    lo = min(max(lo, 1), n - 2)
                    y0, y1, y2 = dist[lo - 1], dist[lo], dist[lo + 1]
                    den = y0 - 2 * y1 + y2
                    off = 0.5 * (y0 - y2) / den if den > 1e-12 else 0.0
                    sync.append(abs((lo + float(np.clip(off, -1, 1))) / fps - s) * 1000.0)
        g["sync_tool_ms"] = sync
    if role == "kneel_work" and clip is not None:
        g["kneel_events"] = sum(p.events for p in plans.kneel_plans)
        g["kneel_matched"] = sum(len(p.matched) for p in plans.kneel_plans)
        g["deck_unreached_share"] = max((p.deck_unreached for p in plans.kneel_plans), default=0.0)
    result["metrics"] = g
    return result


def verdict(runs: list, role: str) -> dict:
    """Aggregate the runs of one clip into gate results."""
    ms = [r["metrics"] for r in runs if r.get("metrics")]
    if not ms:
        return {"result": "n/a", "note": "the clip was not used in any measured run"}

    def worst(key, fn=max):
        vals = [m[key] for m in ms if m.get(key) is not None]
        return fn(vals) if vals else None
    gates = {}

    def put(name, value, limit, ok):
        gates[name] = {"value": value, "limit": limit, "result": "n/a" if ok is None else ("PASS" if ok else "FAIL")}
    b = worst("bones_m")
    put("bones", None if b is None else float(f"{b:.2e}"), "< 1e-3 m", None if b is None else b < BONE_TOL_M)
    pf, ph = worst("pop_feet_m"), worst("pop_hands_m")
    put("pops", {"feet_m": None if pf is None else round(pf, 4), "hands_m": None if ph is None else round(ph, 4)},
        "< 0.12 m feet, < 0.20 m hands per frame",
        (pf is None or pf < POP_FOOT_M) and (ph is None or ph < POP_HAND_M))
    el = [m["elbow_deg"] for m in ms if m.get("elbow_deg")]
    if el:
        lo, hi = min(e[0] for e in el), max(e[1] for e in el)
        put("elbow", {"min_deg": round(lo, 1), "max_deg": round(hi, 1)}, "0-155 deg", lo >= 0.0 and hi <= ELBOW_MAX_DEG)
    reach = {"arms_share": worst("reach_arms_share"), "legs_share": worst("reach_legs_share")}
    ok = all(v is None or v < REACH_MAX_SHARE for v in reach.values())
    put("reach", {k: None if v is None else round(v, 4) for k, v in reach.items()}, "< 5 % of frames clamped per limb", ok)
    if role in LEG_ROLES:
        pen = worst("penetration_m", min)
        put("penetration", round(pen, 4), ">= -0.005 m (heel, toe)", pen >= -PENETRATION_M)
        drift, med = worst("skate_max_drift_m"), worst("skate_median_mps")
        stances = sum(m.get("stances", 0) for m in ms)
        put("skate", {"max_drift_m": None if drift is None else round(drift, 4),
                      "median_mps": None if med is None else round(med, 4), "stances": stances},
            "drift < 0.01 m every stance; median < 0.02 m/s",
            None if drift is None else (drift < SKATE_DRIFT_M and med < SKATE_MEDIAN_MPS))
        km, hx = worst("knee_max_deg"), sum(m.get("knee_hyperext_frames", 0) for m in ms)
        put("knee", {"max_deg": round(km, 1), "hyperextended_frames": hx}, "0-160 deg, no hyperextension",
            km <= KNEE_MAX_DEG and hx == 0)
        hip = [m["hip_deg"] for m in ms if m.get("hip_deg")]
        lo, hi = min(h[0] for h in hip), max(h[1] for h in hip)
        put("hip", {"min_deg": round(lo, 1), "max_deg": round(hi, 1)}, "-30..+135 deg (pelvis segment)",
            HIP_FLEX_DEG[0] <= lo and hi <= HIP_FLEX_DEG[1])
        sync = np.concatenate([np.asarray(m.get("sync_heel_ms", []), float) for m in ms])
        if len(sync):
            med, p90 = float(np.median(sync)), float(np.percentile(sync, 90))
            put("sync", {"heel_median_ms": round(med, 1), "heel_p90_ms": round(p90, 1), "strikes": len(sync)},
                "heel strike median < 30 ms, p90 < 60 ms", med < SYNC_HEEL_MS[0] and p90 < SYNC_HEEL_MS[1])
        else:
            put("sync", None, "heel strike median < 30 ms, p90 < 60 ms", False)
    if role == "hammer":
        sync = np.concatenate([np.asarray(m.get("sync_tool_ms", []), float) for m in ms])
        med, p90 = float(np.median(sync)), float(np.percentile(sync, 90))
        put("sync", {"tool_median_ms": round(med, 1), "tool_p90_ms": round(p90, 1), "strikes": len(sync)},
            "tool strike to impulse: median < 20 ms (p90 reported)", med < SYNC_TOOL_MS)
    if role == "kneel_work":
        events, matched = sum(m.get("kneel_events", 0) for m in ms), sum(m.get("kneel_matched", 0) for m in ms)
        unreached = max(m.get("deck_unreached_share", 0.0) for m in ms)
        gates["deck"] = {"value": round(unreached, 3), "limit": "reported only: share of frames where the clip's hand "
                         "height above the deck is out of the rig arm's reach (hands then as low as reachable)",
                         "result": "n/a"}
        put("sync", {"work_impulses": events, "matched_to_clip_wrist_events": matched},
            "reported only: kneel_work impulses are trunk jolts, not tool strikes", None)
    failed = [k for k, v in gates.items() if v["result"] == "FAIL"]
    return {"gates": gates, "failed": failed, "result": "FAIL" if failed else "PASS",
            "runs": [f"{r['scenario']}:{r['seed']}" for r in runs if r.get("metrics")],
            "frames": int(sum(r.get("frames", 0) for r in runs))}


def s6_status(card: dict) -> str:
    """Card status once S6 measured the clip on the payload. S5's skate gate measured the clip's own
    heel and toe joints; the viewer locks the feet, so the payload measurement replaces it. Any other
    S5 failure, any S2 problem or any S6 failure keeps the clip rejected."""
    s5_failed = card.get("gates", {}).get("failed", [])
    s6 = card.get("s6", {}).get("result")
    if card.get("problems") or any(f != "skate" for f in s5_failed) or s6 == "FAIL":
        return "rejected"
    if s6 == "PASS":
        return "gated"
    return "rejected" if s5_failed else "gated"


def run_job(args):
    return measure(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description="S6 gates on the viewer payload")
    parser.add_argument("--role", default="")
    parser.add_argument("--clip", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    manifest = candidates()
    jobs = []
    for role, scenarios in ROLE_SCENARIOS.items():
        if args.role and role != args.role:
            continue
        names = [e["clip"] for e in manifest["clips"] if role in e["roles"] and (not args.clip or e["clip"] == args.clip)]
        for name in names + [None]:
            jobs += [(s, seed, role, name) for s in scenarios for seed in seeds]
    print(f"payload_gates: {len(jobs)} payloads")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_job, jobs))
    by_clip: dict = {}
    for r in results:
        by_clip.setdefault((r["role"], r["clip"]), []).append(r)
    summary = {}
    for (role, name), runs in sorted(by_clip.items(), key=lambda x: (x[0][0], x[0][1] or "")):
        v = verdict(runs, role)
        label = name or "procedural (mocap off)"
        summary.setdefault(role, {})[label] = v
        shown = " ".join(f"{k}={g['result']}" for k, g in v.get("gates", {}).items() if g["result"] != "PASS")
        print(f"{role:11s} {label:42s} {v['result']:5s} {shown}")
    if args.no_write:
        return
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    for role, entries in summary.items():
        previous.setdefault(role, {}).update(entries)
    OUT.write_text(json.dumps({"stage": "S6 payload", "fps": 50, **{k: v for k, v in previous.items() if k != "stage"}},
                              indent=1) + "\n")
    # Cards: one s6 entry per role the clip was measured in.
    for role, entries in summary.items():
        for name, v in entries.items():
            path = CARDS / f"{name}.json"
            if not path.exists():
                continue
            card = json.loads(path.read_text())
            s6 = card.get("s6", {"stage": "S6 payload", "roles": {}})
            s6.setdefault("roles", {})[role] = v
            results_all = [r["result"] for r in s6["roles"].values() if r["result"] in ("PASS", "FAIL")]
            s6["result"] = "FAIL" if "FAIL" in results_all else ("PASS" if results_all else "n/a")
            card["s6"] = s6
            card["status"] = s6_status(card)
            path.write_text(json.dumps(card, indent=2) + "\n")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "manifest.py")], check=True)


if __name__ == "__main__":
    main()
