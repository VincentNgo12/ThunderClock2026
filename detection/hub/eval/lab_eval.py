# File: lab_eval.py
# Purpose: Run PostureTrack + T3 + C3 (thresholds frozen in hub/config.json) over the harmonised
#          Erciyes/VTT recordings of the sibling research/fall_model checkout (read-only) and report
#          trigger / confirm rates per activity code with Wilson CIs, per subject (leave-one-subject-out).
# Contributors: Unassigned
"""PYTHONPATH=.:simulation simulation/.venv/bin/python -m hub.eval.lab_eval [--data DIR] [--out FILE.json]

No threshold is fitted to these recordings: every value comes from config.json (chosen on ThunderSim
dev seeds), so the training-subject threshold choice of LOSO is the identity and each held-out
subject is scored with the same rules. Results are pooled over held-out subjects.
Limits: ~8 s after the event, acted falls, harmonised to +-2 g / 100 Hz, lumbar and thoracic are
synthetic (anchor_is_real = pelvis, upper thoracic only).
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from hub.collapse import CollapseConfirmer
from hub.config import load_config
from hub.eval.summarize import wilson
from hub.posture_track import PostureTrack
from hub.topple import ToppleTrigger

DEFAULT = Path(__file__).resolve().parents[4] / "research" / "fall_model" / "data" / "processed"
CODES = ("919", "920", "815", "816", "917")


def run_record(acc, gyro, cfg):
    track = PostureTrack(cfg["posture"])
    trig = ToppleTrigger(cfg["topple"])
    c3 = CollapseConfirmer(cfg["collapse"])
    fires, decisions, peaks = [], [], 0.0
    al, gl = acc.tolist(), gyro.tolist()
    for i in range(len(al)):
        t = i / 100.0
        info, closed = track.step(t, al[i], gl[i], [True] * 4)
        peaks = max(peaks, info["rot_pl"])
        if trig.step(t, info["rot_pl"], info["down"], info["not_down"]):
            fires.append(t)
            c3.open(t)
        d = c3.on_sample(t, track)
        if d:
            decisions.append(d)
        for b in closed:
            d = c3.on_bin(b, track, t)
            if d:
                decisions.append(d)
    return fires, decisions, peaks, (c3.pending is not None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. topple.rot_rad_s=3.3 (sweeps only)")
    args = ap.parse_args(argv)
    data = Path(args.data)
    if not (data / "canonical_current.npz").exists():
        raise SystemExit(f"LOSO could not run: {data}/canonical_current.npz is missing")
    overrides = {}
    for item in args.set:
        path, value = item.split("=")
        section, name = path.split(".")
        overrides.setdefault(section, {})[name] = json.loads(value)
    cfg = load_config(overrides=overrides or None)
    z = np.load(data / "canonical_current.npz")
    acc, gyro = z["acc"], z["gyro"]
    rows = list(csv.DictReader(open(data / "meta.csv")))
    started = time.time()
    per_code = defaultdict(lambda: {"n": 0, "t3": 0, "confirmed": 0, "undecided": 0, "subjects": set(), "peaks": []})
    adl_hours, adl_fires = 0.0, 0
    for r in rows:
        a, b = int(r["start"]), int(r["stop"])
        fires, decisions, peak, undecided = run_record(acc[a:b].astype(float), gyro[a:b].astype(float), cfg)
        key = r["code"]
        s = per_code[key]
        s["n"] += 1
        s["subjects"].add(r["subject"])
        s["t3"] += bool(fires)
        s["confirmed"] += any(d["kind"] == "fall_confirmed" for d in decisions)
        s["undecided"] += bool(fires) and undecided
        s["peaks"].append(round(peak, 2))
        s["label"], s["activity"], s["dataset"] = r["label"], r["activity"], r["dataset"]
        if r["label"] == "normal":
            adl_hours += (b - a) / 100.0 / 3600.0
            adl_fires += len(fires)
    out = {"elapsed_s": round(time.time() - started, 1), "adl_hours": round(adl_hours, 3), "adl_t3_fires": adl_fires,
           "adl_t3_per_h": round(adl_fires / max(adl_hours, 1e-9), 2), "codes": {}}
    for code, s in sorted(per_code.items()):
        lo, hi = wilson(s["t3"], s["n"])
        clo, chi = wilson(s["confirmed"], s["n"])
        peaks = np.array(s["peaks"])
        out["codes"][code] = {"dataset": s["dataset"], "activity": s["activity"], "label": s["label"], "n": s["n"],
                              "subjects": len(s["subjects"]), "t3": s["t3"], "t3_ci": [round(lo, 3), round(hi, 3)],
                              "confirmed": s["confirmed"], "confirmed_ci": [round(clo, 3), round(chi, 3)],
                              "undecided_at_end": s["undecided"],
                              "rot_peak_p50_p90": [round(float(np.median(peaks)), 2), round(float(np.quantile(peaks, 0.9)), 2)],
                              "rot_peak_ge_3": int((peaks >= 3.0).sum())}
    text = json.dumps(out, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "codes"}))
    for code in CODES:
        if code in out["codes"]:
            print(code, out["codes"][code])
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
