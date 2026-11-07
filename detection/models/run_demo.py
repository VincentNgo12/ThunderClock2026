"""Replay fixture recordings through the streaming fall detector, sample by sample.

    .venv/bin/python run_demo.py                          # all target-profile fixtures
    .venv/bin/python run_demo.py fixtures/target/fall_syncope.jsonl
    .venv/bin/python run_demo.py --profile current        # degraded (today's firmware) fixtures
    .venv/bin/python run_demo.py --backend logreg_json    # numpy-only model (no scikit-learn)

For each fixture it prints every event the stream emits (Stage-1 trigger, Stage-2
decision) with its time, then compares the final decision with the true label.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from falldet import FallDetectorStream, detect, fixture_arrays, get_detector, load_fixture  # noqa: E402

SEVERITY = {"normal": 0, "near_fall": 1, "fall_confirmed": 2}


def replay(path, profile, backend="auto"):
    header, samples = load_fixture(path)
    det = get_detector(profile, backend)
    stream = FallDetectorStream(detector=det)
    print(f"\n== {os.path.basename(path)}  [{header.get('activity')}]  true: {header.get('true_label')}"
          f"  ({len(samples) / 100:.1f} s, profile {profile})")
    worst = "normal"
    for s in samples:                       # one 4-IMU telemetry sample at a time
        for ev in stream.push(s):
            worst = _report(ev, worst)
    for ev in stream.flush():               # end of file: decide anything still pending
        worst = _report(ev, worst)
    acc, gyro = fixture_arrays(samples)     # same data through the one-shot API
    one = detect((acc, gyro), detector=det)
    print(f"   stream verdict: {worst:15s} | detect(): {one['state']} "
          f"(conf {one['confidence']:.2f}, stage1_triggered={one['stage1_triggered']})")
    return header.get("true_label"), worst


def _report(ev, worst):
    if ev["type"] == "stage1_trigger":
        print(f"   t={ev['t_ms'] / 1000:6.2f}s  STAGE 1 trigger -> suspected "
              f"(svm {ev['svm_g']:.2f} g, rot {ev['rot_rad_s']:.2f} rad/s)")
    else:
        f = ev["features"]
        print(f"   t={ev['decided_at_ms'] / 1000:6.2f}s  STAGE 2 -> {ev['state']:15s} conf {ev['confidence']:.2f}  "
              f"(for trigger at {ev['t_ms'] / 1000:.2f}s; anchors lying {int(f['n_anchors_lying'])}/4, "
              f"still {f['longest_still_s']:.1f}s, impact {f['impact_peak_g']:.2f} g)")
        if SEVERITY[ev["state"]] > SEVERITY[worst]:
            worst = ev["state"]
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--profile", default="target", choices=["target", "current"])
    ap.add_argument("--backend", default="auto", choices=["auto", "gbt", "logreg_json"],
                    help="logreg_json = numpy-only model, no scikit-learn needed")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(os.path.join(HERE, "fixtures", a.profile, "*.jsonl")))
    rows = [(os.path.basename(f),) + replay(f, a.profile, a.backend) for f in files]
    print("\nfile                                        true label                     stream verdict")
    for name, true, pred in rows:
        mark = "ok" if true and true.startswith(pred) else "--"
        print(f"{mark} {name:42s} {str(true):30s} {pred}")


if __name__ == "__main__":
    main()
