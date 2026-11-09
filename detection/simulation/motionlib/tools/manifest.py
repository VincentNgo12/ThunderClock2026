# File: manifest.py
# Purpose: Stage S6 of the motion library: list the clips the viewer may use (motionlib/manifest.json)
#          and the anchors each use needs: walking steps (heel strikes and toe-offs), hammer strikes
#          and wrist events of kneeling work. The viewer (thundersim/viz/mocap.py) reads only this
#          file and the clips it names; the bytes of this file are part of the viewer's run cache key.
# Contributors: Unassigned
"""Run from detection/simulation after gates.py (and again after payload_gates.py):

    .venv/bin/python motionlib/tools/manifest.py

Which clips are listed (all other clips stay out of the viewer):
  - licence class A (class B only with an entry in approvals/approved.json; there are none);
  - no S2 problems, and the S5 preview failed nothing except skate. Skate in S5 is measured on the
    clip's own heel and toe joints; the viewer locks the feet during labelled contacts, so skate is
    measured again on the real payload by payload_gates.py;
  - the card's S6 payload result (card["s6"]["result"]) is not FAIL.
Being listed does not switch a clip on: mocap stays off by default, and `default_on` is true only
for clips whose card has `reviewed_by` set by a person.

Roles and their anchors (clip times in seconds, sides 0 = left, 1 = right):
  walk, carry_walk   steps [t_strike, t_next_strike, landing side, toe-off fraction, pelvis speed / leg]
                     in runs of steady alternating steps
  hammer             right-hand strikes (wrist deceleration peaks) in runs of steady strokes
  kneel_work         wrist deceleration peaks (either hand) and the usable window
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # our own tools only, never raw/
from cliplib import CARDS, CLIPS, MOTIONLIB, load_clip  # noqa: E402

MANIFEST = MOTIONLIB / "manifest.json"
APPROVED = MOTIONLIB / "approvals" / "approved.json"
FORMAT_VERSION = 1

# Steady walking [design]: a step (one heel strike to the next, opposite foot) of 0.4-0.8 s, within
# 25 % of its run's median, the pelvis moving at least 0.8 leg lengths per second (about 0.7 m/s),
# and the landing foot's toe-off between 10 % before the other heel strike and 60 % into the step.
STEP_S = (0.4, 0.8)
STEP_SPREAD = 0.25
MIN_SPEED_LEG = 0.8
TOE_OFF_MAX = 0.6
MIN_STEPS = 4
# Steady hammering [design]: strokes of 0.2-0.9 s between right-hand strikes, at least 6 strikes.
STROKE_S = (0.2, 0.9)
MIN_STRIKES = 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approved_sources() -> set:
    if not APPROVED.exists():
        return set()
    data = json.loads(APPROVED.read_text())
    return {entry.get("source") for entry in data.get("approved", [])}


def eligible(card: dict, approved: set, include_s6_failed: bool = False) -> tuple[bool, str]:
    cls = card.get("licence", {}).get("class")
    if cls not in ("A", "B"):
        return False, f"licence class {cls}"
    if cls == "B" and card.get("source") not in approved:
        return False, "class B without approval"
    if card.get("problems"):
        return False, "S2 problems"
    failed = card.get("gates", {}).get("failed")
    if failed is None:
        return False, "no S5 gates"
    if any(name != "skate" for name in failed):
        return False, "S5 failed " + ", ".join(name for name in failed if name != "skate")
    if card.get("s6", {}).get("result") == "FAIL" and not include_s6_failed:
        return False, "S6 payload gates failed"
    return True, ""


def segments(card: dict, primitive: str) -> list:
    return [s for s in card.get("s4", {}).get("segments", card.get("segments", [])) if s.get("primitive") == primitive]


def gait_events(name: str, feat, meta: dict) -> tuple[list, dict]:
    """Heel strikes and toe-offs from the pelvis-relative foot path [src: Zeni 2008, Gait & Posture
    27:710-714]: a heel strike is the heel's most forward point relative to the pelvis, a toe-off
    the toe's most backward point. S3's strike labels (heel slow and low) come 50-110 ms later and
    S3's toe labels end too early on the toe tip, so neither marks the instant of the event.
    Returns the refined strikes [(t, side)] and, per side, the toe-off times."""
    clip = load_clip(name)
    heading = feat["heading"].astype(float)
    forward = np.stack([-np.sin(heading), np.zeros_like(heading), -np.cos(heading)], axis=1)
    pelvis = clip.j("pelvis")
    heel = [np.sum((clip.j(f"heel_{s}") - pelvis) * forward, axis=1) for s in "lr"]
    toe = [np.sum((clip.j(f"toe_{s}") - pelvis) * forward, axis=1) for s in "lr"]
    fs = clip.fs
    strikes = []
    for h in meta["heel_strikes"]:
        side = 0 if h["side"] == "l" else 1
        k = int(round(h["t"] * fs))
        a, b = max(k - int(0.2 * fs), 0), min(k + int(0.05 * fs), clip.n - 1)
        strikes.append((float(a + int(np.argmax(heel[side][a:b + 1]))) / fs, side))
    toe_offs: dict = {0: [], 1: []}
    for side in (0, 1):
        mine = [t for t, s in strikes if s == side]
        for t0, t1 in zip(mine[:-1], mine[1:], strict=True):     # one toe-off between two strikes of a foot
            a, b = int(round(t0 * fs)), int(round(t1 * fs))
            toe_offs[side].append(float(a + int(np.argmin(toe[side][a:b]))) / fs)
    return strikes, toe_offs


def walk_steps(name: str, meta: dict, fs: float, primitive: str) -> dict | None:
    """Steady steps of one clip inside its `primitive` segments (walk or carry)."""
    feat = np.load(CLIPS / f"{name}.npz")
    leg = float(feat["leg_len"].mean())
    pelvis = load_clip(name).j("pelvis")
    windows = [(s["t0"], s["t1"]) for s in meta["segments"] if s["primitive"] == primitive]
    strikes, toe_offs = gait_events(name, feat, meta)
    steps = []
    for (ta, sa), (tb, sb) in zip(strikes[:-1], strikes[1:], strict=True):
        if sa == sb or not any(w0 <= ta and tb <= w1 for w0, w1 in windows):
            continue
        dur = tb - ta
        if not STEP_S[0] <= dur <= STEP_S[1]:
            continue
        # The landing foot leaves the floor shortly after the other heel strikes.
        offs = [t for t in toe_offs[sb] if ta - 0.1 * dur <= t < tb]
        if len(offs) != 1:
            continue
        f_off = (offs[0] - ta) / dur
        if f_off > TOE_OFF_MAX:
            continue
        ka, kb = int(round(ta * fs)), int(round(tb * fs))
        travel = float(np.linalg.norm((pelvis[kb] - pelvis[ka])[[0, 2]]))
        speed = travel / dur / leg
        if speed < MIN_SPEED_LEG:
            continue
        steps.append([round(ta, 3), round(tb, 3), int(sb), round(f_off, 4), round(speed, 4)])
    # Runs of consecutive steps (each starts where the previous one ended) of similar duration.
    groups, current = [], []
    for st in steps:
        if current and abs(st[0] - current[-1][1]) < 1e-6:
            current.append(st)
        else:
            if current:
                groups.append(current)
            current = [st]
    if current:
        groups.append(current)
    kept, run_index = [], []
    for g in groups:
        med = float(np.median([s[1] - s[0] for s in g]))
        g = [s for s in g if abs((s[1] - s[0]) - med) <= STEP_SPREAD * med]
        # Removing a step can split a run: rebuild the consecutive pieces.
        piece: list = []
        for s in g + [None]:
            if s is not None and piece and abs(s[0] - piece[-1][1]) < 1e-6:
                piece.append(s)
                continue
            if len(piece) >= 2:
                run_index.append([len(kept), len(kept) + len(piece)])
                kept += piece
            piece = [s] if s is not None else []
    if len(kept) < MIN_STEPS:
        return None
    return {"steps": kept, "runs": run_index, "leg_len_m": round(leg, 4),
            "speed_leg_per_s": round(float(np.median([s[4] for s in kept])), 4)}


def hammer_strikes(meta: dict) -> dict | None:
    windows = [(s["t0"], s["t1"]) for s in meta["segments"] if s["primitive"] == "hammer"]
    times = sorted(float(p["t"]) for p in meta["wrist_decel_peaks"]
                   if p["side"] == "r" and any(w0 <= p["t"] <= w1 for w0, w1 in windows))
    runs_out, current = [], []
    for t in times:
        if current and STROKE_S[0] <= t - current[-1] <= STROKE_S[1]:
            current.append(t)
        else:
            if len(current) >= 2:
                runs_out.append(current)
            current = [t]
    if len(current) >= 2:
        runs_out.append(current)
    strikes, run_index = [], []
    for r in runs_out:
        run_index.append([len(strikes), len(strikes) + len(r)])
        strikes += [round(t, 3) for t in r]
    if len(strikes) < MIN_STRIKES:
        return None
    w0 = min(w[0] for w in windows)
    w1 = max(w[1] for w in windows)
    return {"strikes": strikes, "runs": run_index, "window": [round(w0, 3), round(w1, 3)]}


def kneel_events(meta: dict) -> dict | None:
    windows = [(s["t0"], s["t1"]) for s in meta["segments"] if s["primitive"] == "kneel_work"]
    if not windows:
        return None
    w0, w1 = windows[0][0] + 0.5, windows[0][1] - 0.5           # trim the trial's start and end [design]
    if w1 - w0 < 5.0:
        return None
    peaks = sorted({round(float(p["t"]), 3) for p in meta["wrist_decel_peaks"] if w0 <= p["t"] <= w1})
    return {"window": [round(w0, 3), round(w1, 3)], "peaks": peaks}


def build(include_s6_failed: bool = False) -> dict:
    """include_s6_failed=True lists S6 failures too: the candidates payload_gates.py measures."""
    approved = approved_sources()
    clips, skipped = [], {}
    for path in sorted(CARDS.glob("*.json")):
        card = json.loads(path.read_text())
        name = card["clip"]
        ok, why = eligible(card, approved, include_s6_failed)
        if not ok:
            skipped[name] = why
            continue
        feat_path = CLIPS / f"{name}.npz"
        if not feat_path.exists():
            skipped[name] = "no S4 clip"
            continue
        feat = np.load(feat_path)
        meta = json.loads(str(feat["meta"]))
        fs = float(feat["fs"])
        roles = {}
        if bool(feat["has_legs"]):
            for primitive, role in (("walk", "walk"), ("carry", "carry_walk")):
                found = walk_steps(name, meta, fs, primitive)
                if found:
                    roles[role] = found
        if str(feat["task"]) == "hammer_post":
            found = hammer_strikes(meta)
            if found:
                roles["hammer"] = found
        if "kneel_work" in card.get("activity_tags", []):
            found = kneel_events(meta)
            if found:
                roles["kneel_work"] = found
        if not roles:
            skipped[name] = "no role with enough steady events"
            continue
        reviewed = card.get("reviewed_by")
        clips.append({
            "clip": name, "file": f"clips/{name}.npz", "sha256": sha256(feat_path),
            "source": card["source"], "licence_class": card["licence"]["class"],
            "licence": card["licence"]["name"], "credit": card["licence"]["credit"],
            "card_status": card.get("status"), "s6_result": card.get("s6", {}).get("result", "PENDING"),
            "reviewed_by": reviewed, "default_on": bool(reviewed),
            "roles": roles,
        })
    return {"format": FORMAT_VERSION, "generated_by": "motionlib/tools/manifest.py",
            "note": "Clips the viewer may use with mocap on (thundersim viz with THUNDERSIM_MOCAP=1, or ?mocap=1). "
                    "default_on stays false until a person sets reviewed_by on the card.",
            "clips": clips, "skipped": skipped}


def main() -> None:
    manifest = build()
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    by_role: dict = {}
    for entry in manifest["clips"]:
        for role in entry["roles"]:
            by_role.setdefault(role, []).append(entry["clip"])
    for role, names in sorted(by_role.items()):
        print(f"{role:11s} {len(names):2d}: {', '.join(names)}")
    print(f"manifest: {len(manifest['clips'])} clips listed, {len(manifest['skipped'])} skipped -> {MANIFEST}")


if __name__ == "__main__":
    main()
