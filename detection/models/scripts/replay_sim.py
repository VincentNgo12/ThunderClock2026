"""Replay recorded ThunderSim streams (export .npz runs, recorded shifts) through falldet offline.

    .venv/bin/python -I scripts/replay_sim.py --export sim_runs/export_dev --shifts sim_runs/shifts_dev_rec \
        --nearfall v1 v2 --out reports/replay_dev.json

Scoring follows ThunderSim's runner.classify: any fall_confirmed -> fall_confirmed, else any
near_fall -> near_fall, else normal. Shifts: near_fall and fall_confirmed per hour.
SIMULATED DATA ONLY.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run_one(args):
    path, nearfall, batch_s = args
    import warnings
    warnings.simplefilter("ignore")
    from falldet.sim_adapter import SpineFallModel
    z = np.load(path)
    m = SpineFallModel(profile="target", nearfall=nearfall)
    m.reset(None)
    t = z["t"]
    lw = z["lying_work"] if "lying_work" in z.files else np.zeros(len(t), bool)
    evs = []
    step = int(batch_s * 100)
    for a in range(0, len(t), step):
        b = a + step
        evs += m.update_arrays(t[a:b], z["acc"][a:b], z["gyro"][a:b], z["slot_mask"][a:b], z["sat_mask"][a:b],
                               "lying_work" if lw[a:b].any() else "normal")
    kinds = [e.kind for e in evs]
    out = "fall_confirmed" if "fall_confirmed" in kinds else ("near_fall" if "near_fall" in kinds else "normal")
    nf_times = [round(e.t, 2) for e in evs if e.kind == "near_fall"]
    paths = [e.detail.get("path", "fall_path") for e in evs if e.kind == "near_fall"]
    return {"file": os.path.basename(path), "nearfall": nearfall, "output": out,
            "n_near_fall": kinds.count("near_fall"), "n_fall_confirmed": kinds.count("fall_confirmed"),
            "n_suspected": kinds.count("suspected_fall"), "near_fall_t": nf_times, "near_fall_paths": paths,
            "hours": float((t[-1] - t[0]) / 3600) if len(t) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="*", default=[])
    ap.add_argument("--shifts", nargs="*", default=[])
    ap.add_argument("--nearfall", nargs="+", default=["v1", "v2"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks, meta = [], {}
    for d in a.export:
        man = json.load(open(os.path.join(d, "manifest.json")))
        for r in man["runs"]:
            meta[r["file"]] = {"scenario": r["scenario"], "seed": r["seed"], "expected": r["expected"]}
            tasks += [(os.path.join(d, r["file"]), nf, 0.5) for nf in a.nearfall]
    for d in a.shifts:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            meta[os.path.basename(p)] = {"scenario": "SHIFT", "seed": os.path.basename(p), "expected": "normal"}
            tasks += [(p, nf, 0.5) for nf in a.nearfall]
    with ProcessPoolExecutor(a.jobs) as pool:
        res = list(pool.map(run_one, tasks, chunksize=4))
    for r in res:
        r.update(meta[r["file"]])
    summary = {}
    for nf in a.nearfall:
        rows = [r for r in res if r["nearfall"] == nf]
        sc = {}
        for r in rows:
            if r["scenario"] == "SHIFT":
                continue
            s = sc.setdefault(r["scenario"], {"expected": r["expected"], "n": 0, "fall_confirmed": 0, "near_fall": 0, "normal": 0})
            s["n"] += 1
            s[r["output"]] += 1
        sh = [r for r in rows if r["scenario"] == "SHIFT"]
        hrs = sum(r["hours"] for r in sh)
        summary[nf] = {"scenarios": sc, "shifts": {"hours": round(hrs, 2),
                                                   "near_fall_per_h": round(sum(r["n_near_fall"] for r in sh) / hrs, 3) if hrs else None,
                                                   "fall_confirmed_per_h": round(sum(r["n_fall_confirmed"] for r in sh) / hrs, 3) if hrs else None}}
    json.dump({"note": "SIMULATED (ThunderSim) streams replayed offline", "summary": summary, "runs": res},
              open(a.out, "w"), indent=1)
    names = sorted({k for nf in summary for k in summary[nf]["scenarios"]})
    print(f"{'scenario':14s} {'expected':15s} " + "  ".join(f"{nf}: fc/nf/no" for nf in a.nearfall))
    for k in names:
        cells = []
        for nf in a.nearfall:
            s = summary[nf]["scenarios"].get(k)
            cells.append(f"{s['fall_confirmed']:3d}/{s['near_fall']:3d}/{s['normal']:3d} (n={s['n']})" if s else "-")
        exp = next(summary[nf]["scenarios"][k]["expected"] for nf in a.nearfall if k in summary[nf]["scenarios"])
        print(f"{k:14s} {exp:15s} " + "   ".join(cells))
    for nf in a.nearfall:
        print(nf, "shifts", summary[nf]["shifts"])


if __name__ == "__main__":
    main()
