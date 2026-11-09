# File: check_canonical.py
# Purpose: Check every canonical clip (stage S2 output) against the canonical spec and the
#          source-data gates, and confirm the trunk orientations follow the simulator's own
#          rotation recipe (spine.root_rotation / spine.joint_rotation).
# Contributors: Unassigned
"""Run from detection/simulation (reads only our own canonical/*.npz, never raw/):

    .venv/bin/python motionlib/tools/check_canonical.py            # every clip, writes cards
    .venv/bin/python motionlib/tools/check_canonical.py cmu_115    # clips starting with this

Writes the results into each card under "s2_checks" and prints one line per clip.
Checks:
  shape      joints (N,20,3), 100 Hz, finite where valid, unit quaternions
  scale      estimated height 1.4-2.1 m (metres)
  up         neck base > 0.3 m above the pelvis in the clip's most upright frame (+Y up); for clips
             with no frame within 45 deg of upright (lying, crawling) the pelvis stays within 0.6 m
             of the clip floor instead
  bones      thigh, shank, upper-arm length std < 1 mm (rigid FK sources only: CMU, ACCAD, 100STYLE)
  speed      feet and hands below 12 and 15 m/s at the 99th percentile [design]
  sim_conv   seg_q rebuilt from trunk_angles with the simulator's builders (< 1e-3 deg)
  neutral    every trunk angle except heading ~0 in the neutral window (< 1 deg; the window mean
             of angles differs slightly from the rotation mean used to calibrate when it is not still)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SIM_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM_DIR))
from thundersim.spine import joint_rotation, root_rotation  # noqa: E402

MOTIONLIB = Path(__file__).resolve().parents[1]
FK_SOURCES = ("cmu", "accad", "100style")        # skeleton + joint angles, so bone lengths are rigid
# Single rigid bones only. elbow->hand and ankle->toe span the wrist and the toe joint (the canonical
# hand is the wrist-to-fingertip midpoint, the toe the toe tip), so their lengths vary by design;
# they are reported as information.
BONES = (("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("hip_r", "knee_r"), ("knee_r", "ankle_r"),
         ("shoulder_l", "elbow_l"), ("shoulder_r", "elbow_r"))
MULTI_BONE = (("elbow_l", "hand_l"), ("elbow_r", "hand_r"), ("ankle_l", "toe_l"), ("ankle_r", "toe_r"))
FOOT_P99_MAX, HAND_P99_MAX = 12.0, 15.0           # m/s, generous plausibility bounds [design]


def check_clip(path: Path, card: dict) -> dict:
    d = np.load(path)
    names = list(d["joint_names"])
    joints = d["joints"].astype(float)
    result = {}

    q = d["seg_q"].astype(float)
    shape_ok = (joints.ndim == 3 and joints.shape[1:] == (20, 3) and float(d["fs"]) == 100.0
                and np.isfinite(joints[:, d["valid"]]).all() and np.abs(np.linalg.norm(q, axis=-1) - 1).max() < 1e-4)
    result["shape"] = bool(shape_ok)
    result["height_m"] = round(float(d["height_est_m"]), 3)
    result["scale"] = 1.4 <= result["height_m"] <= 2.1

    k0, k1 = (int(x) for x in d["neutral"])
    pelvis, neck = joints[:, names.index("pelvis")], joints[:, names.index("neck")]
    trunk = neck - pelvis
    tilt = np.degrees(np.arccos(np.clip(trunk[:, 1] / np.linalg.norm(trunk, axis=1), -1, 1)))
    if tilt.min() < 45.0:
        k = int(np.argmin(tilt))      # the clip's most upright frame
        result["up"] = bool(trunk[k, 1] > 0.3)
    else:
        # No upright frame (lying, crawling): +Y up means the pelvis stays low over the clip floor.
        floor = np.nanmin(joints[:, d["valid"], 1])
        result["up"] = bool(np.median(pelvis[:, 1]) - floor < 0.6)

    stds = []
    for a, b in BONES:
        if d["valid"][names.index(a)] and d["valid"][names.index(b)]:
            stds.append(np.linalg.norm(joints[:, names.index(a)] - joints[:, names.index(b)], axis=1).std() * 1000)
    result["bone_std_mm_max"] = round(float(max(stds)), 3)
    multi = [np.linalg.norm(joints[:, names.index(a)] - joints[:, names.index(b)], axis=1).std() * 1000
             for a, b in MULTI_BONE if d["valid"][names.index(a)] and d["valid"][names.index(b)]]
    result["hand_toe_span_std_mm_max"] = round(float(max(multi)), 2) if multi else None
    if card["source"] in FK_SOURCES:
        result["bones"] = result["bone_std_mm_max"] < 1.0
    else:
        result["bones"] = "n/a (position data, not a rigid skeleton)"

    def p99_speed(joint_names) -> float:
        speeds = [np.linalg.norm(np.diff(joints[:, names.index(j)], axis=0), axis=1) * 100.0
                  for j in joint_names if d["valid"][names.index(j)]]
        return float(np.percentile(np.concatenate(speeds), 99))

    result["foot_p99_mps"] = round(p99_speed(("ankle_l", "ankle_r", "toe_l", "toe_r")), 2)
    result["hand_p99_mps"] = round(p99_speed(("hand_l", "hand_r")), 2)
    result["speed"] = result["foot_p99_mps"] < FOOT_P99_MAX and result["hand_p99_mps"] < HAND_P99_MAX

    # Rebuild the four segments from the stored angles with the simulator's own builders.
    angles = d["trunk_angles"].astype(float)
    seg = [Rotation.from_quat(q[:, i]) for i in range(4)]
    current = root_rotation(angles[:, 0], angles[:, 1], angles[:, 2])
    errors = [(current.inv() * seg[0]).magnitude().max()]
    for j in range(3):
        current = current * joint_rotation(angles[:, 3 + j], angles[:, 6 + j], angles[:, 9 + j])
        errors.append((current.inv() * seg[j + 1]).magnitude().max())
    result["sim_conv_err_deg"] = float(f"{np.degrees(max(errors)):.2e}")
    result["sim_conv"] = result["sim_conv_err_deg"] < 1e-3

    neutral = np.degrees(np.abs(angles[k0:k1, 1:].mean(axis=0))).max()
    result["neutral_resid_deg"] = round(float(neutral), 3)
    if bool(d["neutral_from_reference"]):
        result["neutral"] = "n/a (calibrated to another clip of the subject)"
    else:
        result["neutral"] = result["neutral_resid_deg"] < 1.0

    deg = np.degrees(angles)
    result["trunk_p5_p95_deg"] = {
        "pitch": [round(float(x), 1) for x in np.percentile(deg[:, 1], [5, 95])],
        "flex_sum": [round(float(x), 1) for x in np.percentile(deg[:, 3:6].sum(axis=1), [5, 95])],
        "roll": [round(float(x), 1) for x in np.percentile(deg[:, 2], [5, 95])],
        "twist_sum": [round(float(x), 1) for x in np.percentile(deg[:, 9:12].sum(axis=1), [5, 95])],
    }
    checks = ("shape", "scale", "up", "bones", "speed", "sim_conv", "neutral")
    result["pass"] = all(result[c] is True or isinstance(result[c], str) for c in checks)
    return result


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    failed = 0
    for path in sorted((MOTIONLIB / "canonical").glob("*.npz")):
        clip = path.stem
        if not clip.startswith(prefix):
            continue
        card_path = MOTIONLIB / "cards" / f"{clip}.json"
        card = json.loads(card_path.read_text())
        result = check_clip(path, card)
        card["s2_checks"] = result
        card_path.write_text(json.dumps(card, indent=2) + "\n")
        failed += not result["pass"]
        bad = [c for c in ("shape", "scale", "up", "bones", "speed", "sim_conv", "neutral") if result[c] is False]
        print(f"{clip:40s} {'PASS' if result['pass'] else 'FAIL ' + ','.join(bad):18s} bone std {result['bone_std_mm_max']:7.2f} mm  "
              f"foot p99 {result['foot_p99_mps']:5.2f}  hand p99 {result['hand_p99_mps']:5.2f} m/s  "
              f"conv {result['sim_conv_err_deg']:.1e} deg  neutral {result['neutral_resid_deg']:.2f} deg")
    print(f"check_canonical: {failed} clip(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
