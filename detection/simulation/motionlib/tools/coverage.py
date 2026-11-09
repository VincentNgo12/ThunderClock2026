# File: coverage.py
# Purpose: Stage S0 inventory: map every recorded Builder primitive (recorder.WRAPPED, labelled in
#          export.LABELS) and every rig limb mode to the canonical clips that can drive it, or to
#          a gap. Writes motionlib/coverage.json.
# Contributors: Unassigned
"""Run from detection/simulation after canonical.py:

    .venv/bin/python motionlib/tools/coverage.py

After stage S3 the cards carry "activity_tags": the Builder primitives found in the motion
itself (annotate.py), plus "lift:<technique>" / "put_down:<technique>" for the lift technique
(stoop, semi_squat, squat; motions.LIFT_TECHNIQUES). Those are used when present; otherwise the
S2 source tags are mapped with TAG_TO_PRIMITIVES (pickup -> lift, putdown -> put_down, ...).
Each primitive lists every clip that shows it ("clips") and the ones that passed the S5 gates
("gated", card status "gated"). Stage S6 adds "viewer_roles": what the viewer's mocap mode
(thundersim/viz/mocap.py) can drive and with which clips of motionlib/manifest.json. Mocap stays
off by default; a clip is on by default only after a person sets reviewed_by (stage S7).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM_DIR))
from thundersim.viz.export import LABELS  # noqa: E402
from thundersim.viz.recorder import WRAPPED  # noqa: E402

MOTIONLIB = Path(__file__).resolve().parents[1]
RIG = SIM_DIR / "thundersim" / "viz" / "rig.py"

TAG_TO_PRIMITIVES = {
    "walk": ["walk"], "run": ["run"], "idle": ["stand", "hold"], "hold": ["hold"], "stoop": ["stoop", "lift"],
    "squat": ["squat", "lift"], "pickup": ["lift"], "putdown": ["put_down"], "carry": ["carry"], "load": ["carry"],
    "kneel": ["kneel"], "kneel_work": ["kneel_work"], "ladder": ["ladder"], "hammer": ["hammer"], "lie": ["lie"],
    "get_up": ["get_up", "stand_up"], "crawl": ["crawl", "quadruped"],
}

# Primitives that are not a limb activity, or whose limbs stay procedural by design.
NOT_APPLICABLE = {
    "impulses": "tool-strike timing inside hammer/rebar_tie: used as sync anchors, not a clip",
    "vibrate": "trunk vibration overlay on another activity",
    "strap_slip": "sensor mounting event, no body motion",
    "set_work_mode": "bookkeeping, no body motion",
    "hat_off": "short procedural arm grab; no open clip needed",
    "remove_strip": "end-of-shift procedural arm motion; no open clip needed",
}

# No open (class A) mocap; the team would record these (class E, consent id on the card).
TEAM_RECORDING = {
    "rebar_tie": "rebar tying (bent-over hand work at the floor)",
    "drill_overhead": "overhead drilling",
    "jackhammer": "jackhammer operation",
    "lying_activity": "work while lying (under a vehicle or slab)",
    "cab_ride": "equipment cab work",
}

# What the viewer's mocap mode drives (mocap.py), per manifest role, and what stays procedural.
VIEWER_ROLES = {
    "walk": {"primitive": "walk", "limbs": "legs (toe-off, swing path, foot roll, knee direction) and arms",
             "anchor": "simulated heel strikes (pelvis yaw extremes = bob minima)"},
    "carry_walk": {"primitive": "carry", "limbs": "legs while carrying (the arms hold the load: procedural)",
                   "anchor": "simulated heel strikes"},
    "hammer": {"primitive": "hammer", "limbs": "arms (waist-height hammering; overhead and on-ladder stay procedural)",
               "anchor": "Builder.impulses times"},
    "kneel_work": {"primitive": "kneel_work", "limbs": "arms", "anchor": "kneel_work impulses, near real speed"},
}
PROCEDURAL_IN_VIEWER = {
    "lift": "hands are fixed to the simulated grip point and the feet planted: contact-driven, no clip needed",
    "put_down": "as lift",
    "ladder": "ladder clips are not integrated: the rig follows the simulator's rung timing (rung-to-rung swing "
              "from a clip is future work)",
    "run": "one open running clip (1.3 s) only",
    "stand": "the gated standing clips add little to planted feet; not integrated",
}

# Gaps that open sources might fill later but are not in the current acquisition.
OPEN_GAP = {
    "sit": "sitting on a bucket or bench: no clip acquired yet (CMU has sit-down trials; not fetched)",
    "sit_posture": "seated posture hold: same gap as sit",
    "sit_ground": "sitting on the ground: no clip acquired yet",
    "jump_down": "planned jump from a step: no clip acquired yet (CMU has jump trials; not fetched)",
    "posture_hold": "static trunk hold: idle/stoop clips give the limbs only approximately",
}

FALL_MODES = {"slip", "air", "hang", "brace", "balance", "rail"}   # involuntary: rig + physics post-pass


def rig_modes() -> dict[str, list[str]]:
    """Mode names added to the leg and arm timelines in rig.py (read as text, not executed)."""
    text = RIG.read_text()
    out = {}
    for limb, start, stop in (("legs", "def leg_timeline", "def leg_target"), ("arms", "def arm_timeline", "def arm_target")):
        body = text[text.index(start):text.index(stop)]
        found = set(re.findall(r'tl\.add\([^"]*"([a-z_]+)"', body)) | {"stand" if limb == "legs" else "idle"}
        # The regex takes the first string in the call; one call computes its times with n == "quadruped".
        out[limb] = sorted({"quad" if name == "quadruped" else name for name in found})
    return out


LIFT_TECHNIQUES = ("stoop", "semi_squat", "squat")     # = motions.LIFT_TECHNIQUES (read, never imported)
USABLE = ("canonical", "gated")                         # rejected cards (S2 problems or S5 FAIL) are listed apart


def load_cards() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((MOTIONLIB / "cards").glob("*.json"))]


def card_primitives(card: dict) -> list[str]:
    """Builder primitives a clip shows: S3 activity tags, else the mapped S2 source tags."""
    if "activity_tags" in card:
        found = {t for t in card["activity_tags"] if ":" not in t}
        # A stoop lift shows the limbs of Builder.stoop() (trunk-only bend) as well.
        if any(t.endswith(":stoop") for t in card["activity_tags"]):
            found.add("stoop")
        return sorted(found)
    return sorted({p for tag in card["tags"] for p in TAG_TO_PRIMITIVES.get(tag, [])})


def main() -> None:
    cards = load_cards()
    by_primitive: dict[str, list[str]] = {}
    gated: dict[str, list[str]] = {}
    rejected: dict[str, list[str]] = {}
    techniques = {kind: {t: {"clips": [], "gated": []} for t in LIFT_TECHNIQUES} for kind in ("lift", "put_down")}
    for card in cards:
        for primitive in card_primitives(card):
            by_primitive.setdefault(primitive, []).append(card["clip"])
            if card.get("status") == "gated":
                gated.setdefault(primitive, []).append(card["clip"])
            elif card.get("status") == "rejected":
                rejected.setdefault(primitive, []).append(card["clip"])
        for tag in card.get("activity_tags", []):
            if ":" in tag:
                kind, technique = tag.split(":", 1)
                if kind in techniques and technique in techniques[kind]:
                    techniques[kind][technique]["clips"].append(card["clip"])
                    if card.get("status") == "gated":
                        techniques[kind][technique]["gated"].append(card["clip"])

    primitives = {}
    for name in WRAPPED:
        clips = sorted(set(by_primitive.get(name, [])))
        entry = {"label": LABELS.get(name), "clips": clips, "gated": sorted(set(gated.get(name, []))),
                 "rejected": sorted(set(rejected.get(name, [])))}
        if name in techniques:
            entry["techniques"] = techniques[name]
        if entry["gated"]:
            entry["status"] = "gated"
        elif clips:
            entry["status"] = "covered_all_rejected" if len(entry["rejected"]) == len(clips) else "covered"
        elif name in NOT_APPLICABLE:
            entry["status"], entry["note"] = "not_applicable", NOT_APPLICABLE[name]
        elif name in TEAM_RECORDING:
            entry["status"], entry["note"] = "gap_team_recording", TEAM_RECORDING[name]
        else:
            entry["status"] = "gap_open"
            entry["note"] = OPEN_GAP.get(name, "no clip")
        primitives[name] = entry

    # Rig modes: which primitives' clips would replace the procedural targets of each mode.
    mode_sources = {
        "legs": {"stand": ["walk", "run", "stand", "hold", "lift", "put_down", "carry", "hammer"], "ladder": ["ladder"],
                 "kneel": ["kneel", "kneel_work"], "quad": ["crawl", "quadruped"], "lie": ["lie"], "sitg": ["sit_ground"],
                 "sit": ["sit"]},
        "arms": {"idle": ["walk", "stand", "hold"], "run": ["run"], "hammer": ["hammer"], "grip": ["lift", "put_down"],
                 "hold": ["lift", "put_down"], "carry": ["carry"], "floor": ["kneel_work"], "ladder": ["ladder"],
                 "quad": ["crawl", "quadruped"], "lie": ["lie"], "sitg": ["sit_ground"], "sit": ["sit"],
                 "tie": ["rebar_tie"], "overhead": ["drill_overhead"], "jack": ["jackhammer"], "liework": ["lying_activity"],
                 "cab": ["cab_ride"], "remove": ["remove_strip"], "carrystrip": ["remove_strip"], "hatgrab": ["hat_off"],
                 "carryladder": ["ladder"]},
    }
    modes = {}
    gated_by_primitive = {name: entry["gated"] for name, entry in primitives.items()}
    for limb, names in rig_modes().items():
        modes[limb] = {}
        for mode in names:
            if mode in FALL_MODES:
                modes[limb][mode] = {"status": "procedural_by_design",
                                     "note": "falls and perturbations stay procedural, with the viz/physics.py post-pass"}
                continue
            sources = mode_sources[limb].get(mode, [])
            clips = sorted({c for p in sources for c in primitives.get(p, {}).get("clips", [])})
            passed = sorted({c for p in sources for c in gated_by_primitive.get(p, [])})
            if limb == "legs":
                # A clip without legs (no floor: NIOSH) cannot drive leg targets.
                passed = [c for c in passed if next(x for x in cards if x["clip"] == c).get("s4", {}).get("has_legs")]
            if passed:
                status = "gated"
            elif clips:
                status = "covered"
            elif sources and all(primitives.get(p, {}).get("status") == "not_applicable" for p in sources):
                status = "procedural_by_design"
            elif sources and all(p in TEAM_RECORDING for p in sources):
                status = "gap_team_recording"
            else:
                status = "gap_open"
            modes[limb][mode] = {"primitives": sources, "clips": clips, "gated": passed, "status": status}

    listed = json.loads((MOTIONLIB / "manifest.json").read_text())["clips"] if (MOTIONLIB / "manifest.json").exists() else []
    viewer = {role: {**info, "clips": sorted(e["clip"] for e in listed if role in e["roles"]),
                     "default_on": sorted(e["clip"] for e in listed if role in e["roles"] and e["default_on"])}
              for role, info in VIEWER_ROLES.items()}
    out = {
        "note": "Coverage after stage S6. 'clips': clips showing the primitive (S3 activity tags); 'gated': clips whose "
                "card status is gated (S5, and for clips used by the viewer the S6 payload gates). status: gated > "
                "covered (clips, none gated) > covered_all_rejected > gap. viewer_roles: what mocap.py drives with which "
                "manifest clips (mocap off by default; none on by default until reviewed, S7). "
                "Generated by motionlib/tools/coverage.py.",
        "rig": {"joints": 18, "limb_modes_from": "thundersim/viz/rig.py leg_timeline / arm_timeline",
                "primitives_from": "thundersim/viz/recorder.py WRAPPED, labels from thundersim/viz/export.py LABELS"},
        "primitives": primitives,
        "rig_modes": modes,
        "gaps_team_recording": sorted(TEAM_RECORDING),
        "gaps_open": sorted(n for n, e in primitives.items() if e["status"] == "gap_open"),
        "lift_techniques": {kind: {t: {"clips": len(v["clips"]), "gated": len(v["gated"])} for t, v in techs.items()}
                            for kind, techs in techniques.items()},
        "viewer_roles": viewer,
        "procedural_in_viewer": PROCEDURAL_IN_VIEWER,
    }
    (MOTIONLIB / "coverage.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, entry in primitives.items():
        print(f"{name:15s} {entry['status']:22s} {len(entry['clips']):3d} clip(s), {len(entry['gated']):3d} gated")
    for kind, techs in techniques.items():
        print(kind, {t: f"{len(v['clips'])} clips / {len(v['gated'])} gated" for t, v in techs.items()})


if __name__ == "__main__":
    main()
