# File: analyze_recording.py
# Purpose: Explain a "Collect data" recording layer by layer: raw IMU health (firmware), fused tilt vs
#          measured gravity and heading drift (firmware fusion), and the heading-free segment model
#          (backend/app/segments.py) on every labelled hold, with an expected-direction check.
# Contributors: Unassigned
"""Run from backend/:

    python tools/analyze_recording.py recordings/<session>.jsonl
    python tools/analyze_recording.py <rec> --export-mounting mounting.json   # use live: MOUNTING_FILE=mounting.json
    python tools/analyze_recording.py <rec> --export-holds tests/fixtures/<name>_holds.json

The mounting is fitted from stand#1 and the round-1 bends; round 2 and the lying holds are therefore
checks the fit never saw.
"""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.models import REGIONS  # noqa: E402
from app.recording_io import fit_model, hold_up_vectors, read_recording  # noqa: E402
from app.segments import apply_mounting, dot, from_to, rotate, unit, world_up_in_sensor, UP  # noqa: E402

NAMES = list(REGIONS.values())
# Firmware toProtocolFrame / RAW_IMU_AXES default "-y,z,-x": protocol axes from board axes.
BOARD_TO_PROTOCOL = ((0, -1, 0), (0, 0, 1), (-1, 0, 0))

# Expected sign of each segment's angle per movement ("flex" + = forward, "lat" + = right).
EXPECTED = {"forward": ("flex", 1), "backward": ("flex", -1), "left": ("lat", -1), "right": ("lat", 1),
            "lie_back": ("flex", -1), "lie_front": ("flex", 1), "lie_left": ("lat", -1), "lie_right": ("lat", 1)}
WRONG_IF_BEYOND_DEG = 10.0


def segment_angles(model, up_by_region):
    out = {}
    for region, up in up_by_region.items():
        mounting = model.mountings.get(region)
        if mounting is None:
            continue
        g = unit(apply_mounting(mounting, up))
        a = rotate(from_to(g, UP), UP)
        out[region] = (math.degrees(math.atan2(-a[2], a[1])), math.degrees(math.atan2(a[0], a[1])))
    return out


def direction_check(holds, model):
    """[(hold, {region: (flex, lat)}, wrong_count)]; wrong = clearly the opposite way from the label."""
    rows = []
    for name, ups in holds.items():
        angles = segment_angles(model, ups)
        movement = name.split("#")[0]
        wrong = None
        if movement in EXPECTED:
            kind, sign = EXPECTED[movement]
            wrong = 0
            for flex, lat in angles.values():
                value = flex if kind == "flex" else lat
                if value * sign < 0 and abs(value) > WRONG_IF_BEYOND_DEG:
                    wrong += 1
        rows.append((name, angles, wrong))
    return rows


def windows_mean_accel(recording):
    """{hold: {region: mean accel (board axes)}} from raw samples."""
    out = {}
    raw = [(t, s) for t, origin, s in recording.samples if s.format == "raw" and origin == "device"]
    for name, t0, t1 in recording.windows():
        if not name.endswith(":hold"):
            continue
        sums = {r: [0.0, 0.0, 0.0, 0] for r in NAMES}
        for t, s in raw:
            if t0 + 0.7 <= t < t1 - 0.2:
                for imu in s.imus:
                    if imu.accel and imu.valid:
                        acc = sums[REGIONS[imu.sensor_id]]
                        for k in range(3):
                            acc[k] += imu.accel[k]
                        acc[3] += 1
        out[name] = {r: (v[0] / v[3], v[1] / v[3], v[2] / v[3]) for r, v in sums.items() if v[3]}
    return out


def heading_deg(q):
    """Heading of a sensor: its protocol +Z axis projected on the horizontal plane (world +Y up)."""
    f = rotate(q, (0.0, 0.0, 1.0))
    return math.degrees(math.atan2(f[0], f[2]))


def report(path):
    rec = read_recording(path)
    holds = hold_up_vectors(rec)
    print(f"# {rec.meta['session']}  ({rec.meta['source']}, format {rec.meta['device_format']})")
    samples = rec.samples
    duration = rec.end["t"] if rec.end else samples[-1][0]
    for fmt in ("raw", "quaternion"):
        n = sum(1 for _, o, s in samples if s.format == fmt and o == "device")
        print(f"{fmt:10s} {n:7d} samples  {n / duration:6.1f} Hz")

    # ---- firmware: raw health at the first stand
    accel = windows_mean_accel(rec)
    if "stand#1:hold" in accel:
        print("\n## Firmware: board orientation at stand#1 (gravity in board axes) and |a|")
        for region, a in accel["stand#1:hold"].items():
            n = math.sqrt(dot(a, a))
            print(f"  {region:15s} up = board {tuple(round(v / n, 2) for v in a)}   |a| {n:5.2f} m/s2")

    # ---- firmware fusion: tilt vs measured gravity
    if accel:
        print("\n## Firmware fusion: angle between fused up and measured gravity per hold (deg)")
        print(f"  {'hold':22s}" + "".join(f"{r[:10]:>11s}" for r in NAMES))
        worst = {r: 0.0 for r in NAMES}
        for name, ups in holds.items():
            cells = []
            for region in NAMES:
                a = accel.get(name, {}).get(region)
                if a is None or region not in ups:
                    cells.append("     -")
                    continue
                a_p = unit(tuple(dot(row, a) for row in BOARD_TO_PROTOCOL))
                err = math.degrees(math.acos(max(-1.0, min(1.0, dot(a_p, ups[region])))))
                worst[region] = max(worst[region], err)
                cells.append(f"{err:11.1f}")
            print(f"  {name:22s}" + "".join(cells))
        print("  worst: " + ", ".join(f"{r} {v:.1f}" for r, v in worst.items()))

    # ---- firmware fusion: heading drift between the first and last stand
    stands = [n for n in ("stand#1:hold", "stand_end#1:hold") if n in holds]
    if len(stands) == 2:
        print("\n## Firmware fusion: heading change between stand#1 and stand_end (deg) -- drift")
        quats = rec.quaternion_samples()
        spans = {name: (t0, t1) for name, t0, t1 in rec.windows()}
        gap = spans[stands[1]][0] - spans[stands[0]][0]
        for region_id, region in REGIONS.items():
            mean = []
            for name in stands:
                t0, t1 = spans[name]
                hs = [heading_deg(next(i.quaternion for i in s.imus if i.sensor_id == region_id))
                      for t, s in quats if t0 + 0.7 <= t < t1 - 0.2]
                mean.append(math.degrees(math.atan2(sum(math.sin(math.radians(h)) for h in hs),
                                                    sum(math.cos(math.radians(h)) for h in hs))))
            change = (mean[1] - mean[0] + 180) % 360 - 180
            print(f"  {region:15s} {change:+7.1f} deg in {gap:.0f} s  ({change / gap * 60:+.0f} deg/min)")

    # ---- heading-free segment model
    model = fit_model(holds, source=str(path))
    print("\n## Segment model: mounting fitted from stand#1 + round-1 bends")
    for region, info in model.report.items():
        print(f"  {region:15s} bend tilt seen {info['tilt_deg']}  direction error {info['direction_error_deg']} deg")
    rows = direction_check(holds, model)
    print("\n## Segment model on every hold: flexion/lateral per segment (deg; +forward, +right)")
    print(f"  {'hold':22s}" + "".join(f"{r[:10]:>16s}" for r in NAMES) + "   wrong")
    total = checked = 0
    for name, angles, wrong in rows:
        cells = "".join(f"{angles[r][0]:+7.0f}/{angles[r][1]:+6.0f}  " if r in angles else f"{'-':>16s}" for r in NAMES)
        print(f"  {name:22s}{cells} {'-' if wrong is None else wrong}")
        if wrong is not None:
            total += wrong
            checked += len(angles)
    print(f"\nsegments bending the wrong way: {total} of {checked}")
    return rec, holds, model, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording")
    parser.add_argument("--export-mounting", help="write the fitted mounting (MOUNTING_FILE for live use)")
    parser.add_argument("--export-holds", help="write per-hold world-up vectors (regression fixture)")
    args = parser.parse_args()
    rec, holds, model, wrong = report(args.recording)
    if args.export_mounting:
        pathlib.Path(args.export_mounting).write_text(json.dumps(model.to_json(), indent=2))
        print(f"wrote mounting to {args.export_mounting}")
    if args.export_holds:
        fixture = {"session": rec.meta["session"], "source": rec.meta["source"],
                   "device_format": rec.meta["device_format"],
                   "note": "mean world-up in each sensor's own axes per labelled hold (from fused quaternions)",
                   "holds": {name: {r: [round(v, 6) for v in up] for r, up in ups.items()} for name, ups in holds.items()}}
        pathlib.Path(args.export_holds).write_text(json.dumps(fixture, indent=1))
        print(f"wrote holds fixture to {args.export_holds}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
