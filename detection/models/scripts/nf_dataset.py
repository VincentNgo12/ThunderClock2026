"""Build the near-fall v2 candidate table (features at every loose perturbation trigger).

    .venv/bin/python -I scripts/nf_dataset.py [--sim-export DIR ...] [--shift-rec DIR ...] --out CSV

Sources and labels (label column):
  public  Erciyes 808 stumble     -> 'nf' for the candidate with the largest trunk rotation in the
                                     recording (no onset labels exist), others 'ignore'
          Erciyes/VTT normal      -> 'neg'
          Erciyes falls / recovered falls (905, 906, 914, 916) -> 'fall' (not used for training)
  sim     NF-STUMBLE, NF-RAIL, STRUCK-NF -> 'nf' if within [event - 0.5 s, event + 2 s] of the
                                     scripted near_fall_onset / struck event, else 'neg'
          A-*, HAT-OFF, STRAP-SLIP, DEVICE-OFF -> 'neg'; all other scenarios -> 'fall'
  shift   fall-free shift recordings -> 'neg'
The trigger here is deliberately LOOSE so tighter settings can be emulated from the features.
All simulator data are simulated (ThunderSim), dev seeds only.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from falldet.nearfall import nf_features, nf_window, run_perturbation_trigger  # noqa: E402
from falldet.resample import Preprocessor  # noqa: E402

LOOSE = {"dip_g": 0.92, "dip_ms": 20, "jolt_g": 1.5, "rot_rad_s": 0.9, "coincidence_ms": 500,
         "refractory_ms": 1500}
NF_SCEN = {"NF-STUMBLE", "NF-RAIL", "STRUCK-NF"}
NEG_SCEN_PREFIX = ("A-",)
NEG_SCEN = {"HAT-OFF", "STRAP-SLIP", "DEVICE-OFF"}


def to_grid(t, acc, gyro, slot_mask):
    """Simulator frames -> concatenated 100 Hz grid (gaps closed up; times kept)."""
    pre = Preprocessor()
    segs = pre.process(t, acc, gyro, slot_mask)
    if not segs:
        return None
    return (np.concatenate([s.t for s in segs]), np.concatenate([s.acc for s in segs]),
            np.concatenate([s.gyro for s in segs]))


def rows_for(acc, gyro, times, base):
    out = []
    for i in run_perturbation_trigger(acc, gyro, LOOSE):
        aw, gw, j = nf_window(acc, gyro, i)
        f = nf_features(aw, gw, j)
        out.append({**base, "t": float(times[i]) if times is not None else i / 100.0, **f})
    return out


def public_rows():
    z = np.load(os.path.join(ROOT, "data/processed/canonical_target.npz"))
    A, W = z["acc"].astype(np.float64), z["gyro"].astype(np.float64)
    meta = pd.read_csv(os.path.join(ROOT, "data/processed/meta.csv"))
    rows = []
    for _, m in meta.iterrows():
        a, w = A[m.start:m.stop], W[m.start:m.stop]
        base = {"source": "public", "dataset": m.dataset, "subject": m.subject, "rec": f"pub{m.rec}",
                "code": str(m.code), "activity": m.activity, "hours": m.duration_s / 3600}
        rr = rows_for(a, w, None, base)
        if m.code == 808:
            if rr:
                k = int(np.argmax([r["rot_peak"] for r in rr]))
                for j, r in enumerate(rr):
                    r["label"] = "nf" if j == k else "ignore"
        else:
            lab = "neg" if m.label == "normal" else "fall"
            for r in rr:
                r["label"] = lab
        if not rr:  # keep a placeholder so recordings without candidates are counted
            rr = [{**base, "label": "none_nf" if m.code == 808 else "none"}]
        rows += rr
    return rows


def sim_rows(export_dir):
    man = json.load(open(os.path.join(export_dir, "manifest.json")))
    rows = []
    for r in man["runs"]:
        z = np.load(os.path.join(export_dir, r["file"]))
        g = to_grid(z["t"], z["acc"], z["gyro"], z["slot_mask"])
        sid = r["scenario"]
        base = {"source": "sim", "dataset": "thundersim", "subject": f"sim{r['seed']}", "rec": f"{sid}_{r['seed']}",
                "code": sid, "activity": sid, "hours": float((z["t"][-1] - z["t"][0]) / 3600)}
        if g is None:
            rows.append({**base, "label": "none"})
            continue
        t, a, w = g
        rr = rows_for(a, w, t, base)
        ev = [e["t"] for e in r["events"] if e["name"] in ("near_fall_onset", "struck")]
        for x in rr:
            if sid in NF_SCEN:
                x["label"] = "nf" if ev and (min(ev) - 0.5 <= x["t"] <= max(ev) + 2.0) else "neg"
            elif sid.startswith(NEG_SCEN_PREFIX) or sid in NEG_SCEN:
                x["label"] = "neg"
            else:
                x["label"] = "fall"
            x["event_t"] = min(ev) if ev else np.nan
        if sid in NF_SCEN and not any(x["label"] == "nf" for x in rr):
            rr.append({**base, "label": "none_nf"})
        rows += rr or [{**base, "label": "none"}]
    return rows


def shift_rows(rec_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(rec_dir, "*.npz"))):
        z = np.load(p)
        g = to_grid(z["t"], z["acc"], z["gyro"], z["slot_mask"])
        name = os.path.basename(p)[:-4]
        base = {"source": "shift", "dataset": "thundersim", "subject": f"shift_{name}", "rec": name,
                "code": "shift", "activity": "shift", "hours": float((z["t"][-1] - z["t"][0]) / 3600)}
        if g is None:
            continue
        t, a, w = g
        rr = rows_for(a, w, t, base)
        for x in rr:
            x["label"] = "neg"
        rows += rr or [{**base, "label": "none"}]
        print("shift", name, len(rr), file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--sim-export", nargs="*", default=[])
    ap.add_argument("--shift-rec", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = public_rows() if a.public else []
    for d in a.sim_export:
        rows += sim_rows(d)
    for d in a.shift_rec:
        rows += shift_rows(d)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
