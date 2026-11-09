# File: annotate.py
# Purpose: Stage S3 of the motion library: annotate every canonical clip with the clip floor,
#          foot contacts (heel and toe, with hysteresis), heel strikes and gait-cycle phase,
#          task events (wrist deceleration peaks, ladder rung contacts) and activity segments
#          named after the Builder primitives, including the lift technique (stoop, semi_squat,
#          squat) of every lift and put-down.
# Contributors: Unassigned
"""Run from detection/simulation after canonical.py (reads only our own canonical/*.npz):

    .venv/bin/python motionlib/tools/annotate.py             # every clip
    .venv/bin/python motionlib/tools/annotate.py cmu_115     # clips whose id starts with this

Writes motionlib/annotations/<clip>.npz (per-frame arrays, 100 Hz like the canonical clip):
  contact (N,2,2) bool   [foot l/r][heel, toe] on the clip floor
  rung (N,2) bool        foot standing still above the floor (ladder clips only)
  knee_down (N,2) bool   knee on the floor (kneeling, crawling)
  phase (N,2) float32    gait-cycle phase 0..1 from each heel strike to the next (NaN outside)
  floor (3,) float32     clip floor plane y = a + b x + c z as (a, b, c); NaN when the clip has no floor
  flat_offsets (2,3)     ankle, heel and toe height above the floor of each foot standing flat
  label (N,) str         activity per frame (Builder primitive names)
and the summary (floor, contacts, heel strikes, events, segments, activity tags) into the card
under "s3". Card "activity_tags" lists the primitives found, plus "lift:<technique>" and
"put_down:<technique>" for every lift and put-down.

All thresholds are [design] values unless a source is given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))   # our own tools only, never raw/
from cliplib import (ANNOTATIONS, SIDES, Clip, clip_names, inner_angle_deg, load_card, load_clip,  # noqa: E402
                     moving_average, runs, save_card, segment_matrix, trunk_tilt_deg)

# --- foot contacts (spec: toe or heel < 3 cm above the clip floor, < 0.25 m/s, >= 3 frames) ----
CONTACT_ENTER_M, CONTACT_ENTER_MPS = 0.03, 0.25
CONTACT_STAY_M = 0.045         # height hysteresis: a contact ends only above 4.5 cm [design]
CONTACT_MIN_FRAMES = 3
CONTACT_GAP_FRAMES = 2          # a contact interrupted for at most this many frames (speed noise) continues [design]
FLOOR_STILL_MPS = 0.10          # lowest foot points slower than this vote for the floor [design]
FLOOR_BAND_M = 0.03             # ... if within this of the lowest such points (ladder rungs vote out)
FLOOR_RIDGE_M2 = 0.05          # floor slope prior (m^2): small spreads pull the slope toward level [design]
FLOOR_MIN_SPREAD_M = 0.5        # votes spread less than this (std, m) give a level floor [design]
FLOOR_MAX_SLOPE = 0.08          # steepest floor accepted from the data (4.6 deg) [design]
FLAT_ANKLE_BAND_M = 0.015       # a still foot stands flat when its ankle is this close to its lowest [design]
RUNG_MIN_HEIGHT_M = 0.08        # a still foot this far above the floor stands on a rung [design]
RUNG_MIN_S = 0.15
KNEE_DOWN_H = 0.07              # knee joint centre within 0.07 H of the floor = knee on the floor [design]

# --- heel strikes and cycle phase -------------------------------------------------------------
STRIKE_AIRBORNE_S = 0.10        # the whole foot was off the floor this long before a heel strike [design]
CYCLE_S = (0.4, 2.0)            # a gait cycle between two heel strikes of one foot lasts this long [design]

# --- task events ------------------------------------------------------------------------------
DECEL_MIN = 4.0                 # wrist deceleration peak (m/s^2): the CMU subject-62 hammer taps reach 5-10 [design]
DECEL_MIN_SPEED = 0.4           # ... after the hand moved at least this fast (m/s) within 0.15 s
DECEL_STOP_SHARE = 0.5          # ... and it slows to half that speed within DECEL_STOP_S: an abrupt stop,
DECEL_STOP_S = 0.10             #     not the smooth turn of an arm swing [design]
DECEL_SPACING_S = 0.2

# --- activity segments ------------------------------------------------------------------------
LOW_HAND_H = 0.33               # a bend: the lower hand below 0.33 H above the floor ...
LOW_HAND_DEEP_H = 0.25          # ... and reaching below 0.25 H somewhere in it [design; knee ~0.285 H]
SQUAT_DROP_H, SQUAT_DEEP_H = 0.10, 0.16   # a squat without a low reach: pelvis drop (x H) [design]
WALK_MPS, RUN_MPS = 0.30, 2.0   # smoothed pelvis speed for walking / running [design]
SPEED_WINDOW_S = 0.5
LIE_TILT, LIE_PELVIS_H = 55.0, 0.22       # trunk tilt (deg) and pelvis height (x H) for lying [design]
QUAD_TILT, HAND_DOWN_H = 45.0, 0.10       # on hands and knees: trunk tilt and hands near the floor
MIN_SEGMENT_S = 0.3
# Lift technique by knee flexion at the deepest frame [src: Straker 2003 via Washmuth 2022: stoop
# < 45 deg knee flexion, semi-squat ~90 deg, squat ~135 deg]; the class boundary between semi-squat
# and squat is the midpoint 112.5 deg [design].
STOOP_MAX_KNEE, SEMI_MAX_KNEE = 45.0, 112.5
LIFT_TECHNIQUES = ("stoop", "semi_squat", "squat")
ASYMMETRIC_KNEE_DEG = 30.0      # knees differ by more than this at the deepest frame: a split or one-legged
                                # bend, which the simulator does not model (Geissinger 2020) [design]


def technique_of(knee_flexion_deg: float) -> str:
    if knee_flexion_deg < STOOP_MAX_KNEE:
        return "stoop"
    if knee_flexion_deg < SEMI_MAX_KNEE:
        return "semi_squat"
    return "squat"


def hysteresis(enter: np.ndarray, stay: np.ndarray, min_frames: int, gap: int = 0) -> np.ndarray:
    """True from a frame where `enter` holds for as long as `stay` holds. Gaps of at most `gap`
    frames are filled first, then runs shorter than min_frames are dropped."""
    out = np.zeros(len(enter), bool)
    active = False
    for k in range(len(enter)):
        if active:
            active = bool(stay[k])
        else:
            active = bool(enter[k])
        out[k] = active
    if gap:
        for a, b in runs(~out):
            if 0 < a and b < len(out) and b - a <= gap:
                out[a:b] = True
    for a, b in runs(out):
        if b - a < min_frames:
            out[a:b] = False
    return out


def point_speed(p: np.ndarray, fs: float) -> np.ndarray:
    return np.linalg.norm(np.gradient(p, 1.0 / fs, axis=0), axis=1)


def estimate_floor(clip: Clip, root_relative: bool, ladder: bool) -> np.ndarray:
    """Clip floor as a plane y = a + b x + c z, returned as (a, b, c); NaN without a floor.

    The lowest foot point of each frame votes when it is still. Ladder clips take the median of
    the votes within FLOOR_BAND_M of the lowest (so rung contacts do not lift the floor) and a
    level plane. Other clips fit a plane: some capture floors are not level in the data (ACCAD
    walking trials rise 1-2 cm per metre). The fit starts from every vote and then keeps those
    within FLOOR_BAND_M of the lowest residuals. The slope is ridge-regularised, so a clip that
    stands on one spot (FLOOR_MIN_SPREAD_M) gets a level floor, and limited to FLOOR_MAX_SLOPE."""
    if root_relative or not clip.has("heel_l", "toe_l", "heel_r", "toe_r"):
        return np.full(3, np.nan)
    points = np.stack([clip.j(f"{p}_{s}") for s in SIDES for p in ("heel", "toe")], axis=1)   # (N,4,3)
    speeds = np.stack([point_speed(points[:, i], clip.fs) for i in range(4)], axis=1)
    lowest = np.argmin(points[:, :, 1], axis=1)
    k = np.arange(clip.n)
    votes, v = points[k, lowest], speeds[k, lowest]
    if (v < FLOOR_STILL_MPS).sum() >= 10:
        votes = votes[v < FLOOR_STILL_MPS]
    y = votes[:, 1]
    base = np.percentile(y, 2)
    level = float(np.median(y[y <= base + FLOOR_BAND_M]))
    if ladder:
        return np.array([level, 0.0, 0.0])
    plane = np.array([level, 0.0, 0.0])
    keep = np.ones(len(y), bool)          # still lowest points of a clip without rungs are floor contacts
    for _ in range(6):
        X = np.c_[np.ones(keep.sum()), votes[keep, 0], votes[keep, 2]]
        ridge = np.diag([0.0, 1.0, 1.0]) * keep.sum() * FLOOR_RIDGE_M2
        plane = np.linalg.solve(X.T @ X + ridge, X.T @ y[keep])
        plane[1:] = np.clip(plane[1:], -FLOOR_MAX_SLOPE, FLOOR_MAX_SLOPE)
        resid = y - (plane[0] + plane[1] * votes[:, 0] + plane[2] * votes[:, 2])
        keep = resid <= np.percentile(resid, 2) + FLOOR_BAND_M
    spread = float(np.sqrt(votes[keep, 0].var() + votes[keep, 2].var()))
    if spread < FLOOR_MIN_SPREAD_M:
        # Feet that stay on one spot cannot tell a slope from foot roll: level floor.
        return np.array([float(np.median(y[keep])), 0.0, 0.0])
    return plane


def above_floor(floor: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Height of points p (...,3) above the floor plane (a, b, c)."""
    return p[..., 1] - (floor[0] + floor[1] * p[..., 0] + floor[2] * p[..., 2])


def floor_slope_deg(floor: np.ndarray) -> float:
    return float(np.degrees(np.arctan(np.hypot(floor[1], floor[2]))))


def flat_frames(clip: Clip, floor: np.ndarray, side: str) -> np.ndarray:
    """Frames where one foot stands flat and still: ankle, heel and toe slower than
    FLOOR_STILL_MPS and the ankle within FLAT_ANKLE_BAND_M of its lowest still height."""
    pts = [clip.j(f"{p}_{side}") for p in ("ankle", "heel", "toe")]
    still = np.all([point_speed(p, clip.fs) < FLOOR_STILL_MPS for p in pts], axis=0)
    if np.isnan(floor).any() or still.sum() < 5:
        return np.zeros(clip.n, bool)
    ankle_h = above_floor(floor, pts[0])
    return still & (ankle_h <= np.percentile(ankle_h[still], 5) + FLAT_ANKLE_BAND_M)


def flat_offsets(clip: Clip, floor: np.ndarray) -> np.ndarray:
    """(2,3) height above the clip floor of the ankle, heel and toe of each foot while it stands
    flat and still. Skeletons put these points at different heights (a CMU heel sits on the
    floor while its toe joint is 6 cm up; a 100STYLE toe touches the floor while its heel is
    7 cm up), so each point is judged against its own flat-foot height. Zero without a floor."""
    out = np.zeros((2, 3))
    found = [False, False]
    for i, s in enumerate(SIDES):
        flat = flat_frames(clip, floor, s)
        if flat.any():
            out[i] = [float(np.median(above_floor(floor, clip.j(f"{p}_{s}")[flat]))) for p in ("ankle", "heel", "toe")]
            found[i] = True
    # A foot that never stands flat and still (short walking trials) borrows the other foot's
    # heights: both feet share one skeleton.
    for i in range(2):
        if not found[i] and found[1 - i]:
            out[i] = out[1 - i]
    return out


def foot_contacts(clip: Clip, floor: np.ndarray, offsets: np.ndarray, ladder: bool):
    """Contact (N,2,2) on the floor and rung (N,2) above it. Heights are measured from each
    point's own flat-foot height (flat_offsets)."""
    contact = np.zeros((clip.n, 2, 2), bool)
    rung = np.zeros((clip.n, 2), bool)
    if np.isnan(floor).any():
        return contact, rung
    for i, s in enumerate(SIDES):
        still_above = np.ones(clip.n, bool)
        v_foot = np.zeros(clip.n)
        for k, part in enumerate(("heel", "toe")):
            p = clip.j(f"{part}_{s}")
            h = above_floor(floor, p) - offsets[i, k + 1]
            v = point_speed(p, clip.fs)
            v_foot = np.maximum(v_foot, v)
            contact[:, i, k] = hysteresis((h < CONTACT_ENTER_M) & (v < CONTACT_ENTER_MPS),
                                          (h < CONTACT_STAY_M) & (v < CONTACT_ENTER_MPS), CONTACT_MIN_FRAMES,
                                          CONTACT_GAP_FRAMES)
            still_above &= h > RUNG_MIN_HEIGHT_M
        if ladder:
            rung[:, i] = hysteresis(still_above & (v_foot < CONTACT_ENTER_MPS), still_above & (v_foot < CONTACT_ENTER_MPS),
                                    int(round(RUNG_MIN_S * clip.fs)), CONTACT_GAP_FRAMES)
    return contact, rung


def heel_strikes(clip: Clip, contact: np.ndarray) -> list[tuple[int, int]]:
    """(frame, side) where a heel lands after the whole foot was airborne for STRIKE_AIRBORNE_S."""
    gap = int(round(STRIKE_AIRBORNE_S * clip.fs))
    strikes = []
    for i in range(2):
        foot = contact[:, i].any(axis=1)
        for a, _ in runs(contact[:, i, 0]):
            if a >= gap and not foot[a - gap:a].any():
                strikes.append((a, i))
    return sorted(strikes)


def cycle_phase(clip: Clip, strikes: list[tuple[int, int]]) -> np.ndarray:
    phase = np.full((clip.n, 2), np.nan)
    for i in range(2):
        frames = [k for k, side in strikes if side == i]
        for a, b in zip(frames[:-1], frames[1:]):
            if CYCLE_S[0] * clip.fs <= b - a <= CYCLE_S[1] * clip.fs:
                phase[a:b, i] = np.arange(b - a) / (b - a)
    return phase


def wrist_peaks(clip: Clip) -> list[dict]:
    """Wrist deceleration peaks: the hand stops sharply after moving fast (tool strikes)."""
    events = []
    window = int(round(0.15 * clip.fs))
    for i, s in enumerate(SIDES):
        if not clip.has(f"hand_{s}"):
            continue
        speed = moving_average(point_speed(clip.j(f"hand_{s}"), clip.fs), 3)
        decel = -np.gradient(speed, 1.0 / clip.fs)
        peaks, props = find_peaks(decel, height=DECEL_MIN, distance=max(int(DECEL_SPACING_S * clip.fs), 1))
        stop = int(round(DECEL_STOP_S * clip.fs))
        for k, value in zip(peaks, props["peak_heights"]):
            before = speed[max(k - window, 0):k + 1].max()
            after = speed[k:k + stop + 1].min()
            if before >= DECEL_MIN_SPEED and after <= DECEL_STOP_SHARE * before:
                events.append({"t": round(k / clip.fs, 3), "frame": int(k), "side": s, "decel_mps2": round(float(value), 1),
                               "speed_before_mps": round(float(before), 2)})
    return sorted(events, key=lambda e: e["t"])


def bend_windows(low: np.ndarray, deep: np.ndarray) -> list[tuple[int, int]]:
    return [(a, b) for a, b in runs(low) if deep[a:b].any()]


def thorax_inclination(clip: Clip) -> np.ndarray:
    """Upper-thoracic segment axis (+Y) from vertical (degrees)."""
    up = segment_matrix(clip, 3)[:, :, 1]
    return np.degrees(np.arccos(np.clip(up[:, 1], -1, 1)))


def segment(clip: Clip, card: dict, floor: np.ndarray, contact, rung, knee_down, peaks) -> tuple[np.ndarray, list, list]:
    """Per-frame activity labels, the merged segments and the lifts with their technique."""
    H, fs, n = clip.height, clip.fs, clip.n
    tags = set(card["tags"])
    label = np.array(["stand"] * n, dtype=object)
    pelvis = clip.j("pelvis")
    vel = np.gradient(pelvis, 1.0 / fs, axis=0)
    speed = moving_average(np.linalg.norm(vel[:, [0, 2]], axis=1), int(SPEED_WINDOW_S * fs))
    tilt = trunk_tilt_deg(clip)
    knee_flex = np.stack([180.0 - inner_angle_deg(clip.j(f"hip_{s}"), clip.j(f"knee_{s}"), clip.j(f"ankle_{s}"))
                          for s in SIDES], axis=1)
    has_floor = not np.isnan(floor).any()
    lifts: list[dict] = []
    if not has_floor:
        # Pelvis-relative clips (NIOSH): the source tags describe the whole clip.
        whole = "kneel_work" if "kneel_work" in tags else ("kneel" if "kneel" in tags else "stand")
        label[:] = whole
        return label, merge_segments(label, fs), lifts

    pelvis_h = above_floor(floor, pelvis)
    hands_h = np.min([above_floor(floor, clip.j(f"hand_{s}")) for s in SIDES], axis=0)
    # Standing pelvis height: upright, straight-legged, both feet on the floor (not on a rung).
    on_floor = contact.any(axis=2).all(axis=1)
    upright = (tilt < 20.0) & (knee_flex.max(axis=1) < 30.0) & on_floor
    stand_h = float(np.percentile(pelvis_h[upright], 90)) if upright.sum() > 10 else float(np.percentile(pelvis_h, 95))
    drop = stand_h - pelvis_h
    knees = knee_down.any(axis=1)
    hands_down = hands_h < HAND_DOWN_H * H

    # Locomotion first, then postures that override it.
    label[speed > WALK_MPS] = "walk"
    label[speed > RUN_MPS] = "run"
    for a, b in bend_windows(drop > SQUAT_DROP_H * H, drop > SQUAT_DEEP_H * H):
        label[a:b] = "squat"
    for a, b in bend_windows(hands_h < LOW_HAND_H * H, hands_h < LOW_HAND_DEEP_H * H):
        k = a + int(np.argmin(hands_h[a:b]))
        label[a:b] = "bend"
        lifts.append({"frames": [a, b], "k_low": k})
    label[knees] = "kneel"
    label[knees & hands_down & (tilt > QUAD_TILT)] = "quadruped"
    crawl = knees & hands_down & (tilt > QUAD_TILT) & (speed > 0.15)
    label[crawl] = "crawl"
    label[(tilt > LIE_TILT) & (pelvis_h < LIE_PELVIS_H * H)] = "lie"
    if "ladder" in tags and rung.any():
        on = rung.any(axis=1)
        idx = np.flatnonzero(on)
        # Frames between rung contacts less than 1.5 s apart belong to the climb. [design]
        start = idx[0]
        for prev, nxt in zip(idx[:-1], idx[1:]):
            if nxt - prev > 1.5 * fs:
                label[start:prev + 1] = "ladder"
                start = nxt
        label[start:idx[-1] + 1] = "ladder"
    if "hammer" in tags and len(peaks) >= 3:
        k0, k1 = peaks[0]["frame"], peaks[-1]["frame"]
        span = slice(max(k0 - int(0.5 * fs), 0), min(k1 + int(0.5 * fs), n))
        label[span] = np.where(label[span] == "stand", "hammer", label[span])

    # Name each bend: lift / put_down from the source tags (alternating when a clip has both,
    # starting with the pick-up) [design]; otherwise a trunk-only stoop or a squat.
    order = []
    if ("pickup" in tags and "putdown" in tags) or ("carry" in tags and not tags & {"pickup", "putdown"}):
        order = ["lift", "put_down"]
    elif "pickup" in tags:
        order = ["lift"]
    elif "putdown" in tags:
        order = ["put_down"]
    for j, lift in enumerate(lifts):
        a, b = lift["frames"]
        k = lift["k_low"]
        flex = float(knee_flex[k].max())
        technique = technique_of(flex)
        kneeling = bool(knees[a:b].any())
        if kneeling:
            primitive = "kneel"
        elif order:
            primitive = order[j % len(order)] if len(order) == 2 else order[0]
        else:
            primitive = "stoop" if technique == "stoop" else "squat"
        label[a:b] = np.where(label[a:b] == "bend", primitive, label[a:b])
        lift.update({"primitive": primitive, "technique": None if kneeling else technique,
                     "t_low": round(k / fs, 3), "t0": round(a / fs, 3), "t1": round(b / fs, 3),
                     "knee_flex_deg": [round(float(x), 1) for x in knee_flex[k]],
                     "asymmetric": bool(abs(knee_flex[k, 0] - knee_flex[k, 1]) > ASYMMETRIC_KNEE_DEG),
                     "thorax_incl_deg": round(float(thorax_inclination(clip)[k]), 1),
                     "pelvis_drop_m": round(float(drop[k]), 3), "hand_height_m": round(float(hands_h[k]), 3)})
    label[label == "bend"] = "stoop"
    if "get_up" in tags:
        # Getting up: from the last frame lying, kneeling or on all fours to the first upright stand.
        low = np.flatnonzero(np.isin(label, ["lie", "kneel", "quadruped", "crawl"]))
        if len(low):
            later = np.flatnonzero((np.arange(n) > low[-1]) & upright)
            end = later[0] if len(later) else n
            label[low[-1] + 1:end] = "get_up"

    # A load is held between a lift and the next put-down; clips tagged as carrying hold one
    # throughout when they show no lift. Walking with it is "carry", standing with it "hold".
    loaded = np.zeros(n, bool)
    if tags & {"carry", "load", "hold"}:
        if any(l["primitive"] == "lift" for l in lifts) or any(l["primitive"] == "put_down" for l in lifts):
            state = not any(l["primitive"] == "lift" for l in lifts)   # starts loaded if it only puts down
            k_prev = 0
            for lift in lifts:
                a, b = lift["frames"]
                if state:
                    loaded[k_prev:a] = True
                if lift["primitive"] == "lift":
                    state = True
                elif lift["primitive"] == "put_down":
                    state = False
                k_prev = b
            if state:
                loaded[k_prev:] = True
        else:
            loaded[:] = True
    label[loaded & (label == "walk")] = "carry"
    label[loaded & (label == "stand")] = "hold"
    return label, merge_segments(label, fs), lifts


def merge_segments(label: np.ndarray, fs: float) -> list[dict]:
    """Runs of one label; runs shorter than MIN_SEGMENT_S join the previous run."""
    segs: list[list] = []
    for k, name in enumerate(label):
        if segs and segs[-1][2] == name:
            segs[-1][1] = k + 1
        else:
            segs.append([k, k + 1, name])
    merged: list[list] = []
    for a, b, name in segs:
        if merged and (b - a < MIN_SEGMENT_S * fs or merged[-1][2] == name):
            merged[-1][1] = b
        else:
            merged.append([a, b, name])
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_SEGMENT_S * fs:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [{"primitive": name, "t0": round(a / fs, 3), "t1": round(b / fs, 3)} for a, b, name in merged]


def annotate(name: str) -> dict:
    clip = load_clip(name)
    card = load_card(name)
    ladder = "ladder" in card["tags"]
    floor = estimate_floor(clip, bool(card.get("root_relative")), ladder)
    offsets = flat_offsets(clip, floor)
    contact, rung = foot_contacts(clip, floor, offsets, ladder)
    knee_down = np.zeros((clip.n, 2), bool)
    if not np.isnan(floor).any():
        knee_down = np.stack([above_floor(floor, clip.j(f"knee_{s}")) < KNEE_DOWN_H * clip.height for s in SIDES], axis=1)
    strikes = heel_strikes(clip, contact)
    phase = cycle_phase(clip, strikes)
    peaks = wrist_peaks(clip)
    label, segments, lifts = segment(clip, card, floor, contact, rung, knee_down, peaks)

    rung_events = []
    for i, s in enumerate(SIDES):
        for a, b in runs(rung[:, i]):
            height = float(np.min([above_floor(floor, clip.j(f"{p}_{s}")[a:b]).mean() - offsets[i, j + 1]
                                   for j, p in enumerate(("heel", "toe"))]))
            rung_events.append({"t0": round(a / clip.fs, 3), "t1": round(b / clip.fs, 3), "side": s,
                                "height_m": round(height, 3)})
    rung_events.sort(key=lambda e: e["t0"])

    ANNOTATIONS.mkdir(exist_ok=True)
    np.savez_compressed(ANNOTATIONS / f"{name}.npz", contact=contact, rung=rung, knee_down=knee_down,
                        flat_offsets=offsets.astype(np.float32),
                        phase=phase.astype(np.float32), floor=floor.astype(np.float32), label=label.astype(str))

    primitives = sorted({seg["primitive"] for seg in segments})
    if np.isnan(floor).any() and "kneel" in card["tags"]:
        primitives = sorted(set(primitives) | {"kneel"})     # kneeling floor work is kneeling too
    techniques = sorted({f"{l['primitive']}:{l['technique']}" for l in lifts
                         if l["primitive"] in ("lift", "put_down") and l["technique"]})
    stances = {s: len(runs(contact[:, i].any(axis=1))) for i, s in enumerate(SIDES)}
    summary = {
        "floor_m": None if np.isnan(floor).any() else round(float(floor[0]), 4),
        "floor_slope_deg": None if np.isnan(floor).any() else round(floor_slope_deg(floor), 2),
        "flat_offsets_m": {s: dict(zip(("ankle", "heel", "toe"), [round(float(x), 4) for x in offsets[i]]))
                           for i, s in enumerate(SIDES)},
        "contact_share": {s: round(float(contact[:, i].any(axis=1).mean()), 3) for i, s in enumerate(SIDES)},
        "stances": stances,
        "heel_strikes": [{"t": round(k / clip.fs, 3), "side": SIDES[i]} for k, i in strikes],
        "cycles": int(np.sum(np.diff(np.nan_to_num(phase, nan=1.0), axis=0) < -0.5)),
        "wrist_decel_peaks": peaks if len(peaks) <= 60 else peaks[:60],
        "wrist_decel_peak_count": len(peaks),
        "rung_contacts": rung_events,
        "bends": lifts,
        "segments": segments,
        "notes": [] if not np.isnan(floor).any() else ["no floor (pelvis-relative or no heel/toe): no foot contacts; "
                                                  "activity from the source tags"],
    }
    # Source tags (S2, from the trial names) that the S3 detectors did not find in the motion.
    expected = {"pickup": "lift", "putdown": "put_down", "walk": "walk", "run": "run", "kneel": "kneel",
                "kneel_work": "kneel_work", "ladder": "ladder", "hammer": "hammer", "lie": "lie", "crawl": "crawl",
                "squat": "squat", "stoop": "stoop", "get_up": "get_up"}
    for tag in card["tags"]:
        want = expected.get(tag)
        if want == "squat" and any(l["technique"] in ("semi_squat", "squat") for l in lifts):
            continue
        if want == "stoop" and any(l["technique"] == "stoop" for l in lifts):
            continue
        if want == "walk" and "carry" in primitives:
            continue
        if want and want not in primitives:
            summary["notes"].append(f"source tag '{tag}' not found in the motion by S3 (expected '{want}')")
    card["s3"] = summary
    card["activity_tags"] = primitives + techniques
    save_card(card)
    return card


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    for name in clip_names(prefix):
        card = annotate(name)
        s3 = card["s3"]
        print(f"{name:40s} floor {s3['floor_m'] if s3['floor_m'] is not None else '  n/a':>7} strikes {len(s3['heel_strikes']):3d} "
              f"peaks {s3['wrist_decel_peak_count']:3d} rungs {len(s3['rung_contacts']):3d}  {' '.join(card['activity_tags'])}")


if __name__ == "__main__":
    main()
