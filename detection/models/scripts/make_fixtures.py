"""Write simulator fixtures in the canonical 4-anchor format (JSON lines + CSV).

    .venv/bin/python -I scripts/make_fixtures.py

fixtures/<profile>/<name>.jsonl   line 1: {"type": "header", ...metadata...}
                                  then:   {"type": "sample", "time": ms, "data": [4 IMUs]}
fixtures/<profile>/<name>.csv     same samples, one row each (t_ms + 24 columns)
fixtures/<profile>/index.json     list of fixtures with true labels and detector outputs
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from falldet.constants import ANCHORS, DEVICE_PROFILES, FS_HZ  # noqa: E402
from falldet.detector import Detector, FallDetectorStream, detect  # noqa: E402

SUBJECT = "E209"   # one Erciyes volunteer for all Erciyes fixtures
PICK = [  # (name, dataset subject or None, code)
    ("fall_front_lying", None, 901), ("fall_back_lying", None, 910),
    ("fall_right_sideway", None, 913), ("fall_syncope", None, 919),
    ("near_fall_front_quick_recovery", None, 905), ("near_fall_stumble", None, 808),
    ("adl_walking", None, 801), ("adl_jogging", None, 803),
    ("adl_sit_chair", None, 811), ("adl_bending_pick_up", None, 806),
    ("construction_jumping_down", "V03", 1007), ("construction_kneel_floor", "V03", 1012),
]
COMPOSITE = [801, 806, 811, 803, 905, 901]   # one continuous "mini shift" ending in a fall

LICENCES = {
    "erciyes": dict(name="Simulated Falls and Daily Living Activities (Erciyes / UCI 455)",
                    license="CC BY 4.0",
                    url="https://archive.ics.uci.edu/dataset/455/simulated+falls+and+daily+living+activities+data+set",
                    citation="Özdemir A.T., Barshan B. Detecting Falls with Wearable Sensors Using "
                             "Machine Learning Techniques. Sensors 2014, 14, 10691-10708."),
    "vtt": dict(name="VTT-ConIot", license="CC BY 4.0", url="https://zenodo.org/records/4683703",
                citation="Mäkela S.-M. et al. Introducing VTT-ConIot: A Realistic Dataset for Activity "
                         "Recognition of Construction Workers Using IMU Devices. Sustainability 2022, 14(1), 220."),
}


def header(meta_rows, profile, extra):
    m = meta_rows.iloc[0]
    ds = m["dataset"]
    return {
        "type": "header", "format": "thunderclock-4imu-canonical-v0", "fs_hz": FS_HZ,
        "units": {"accel": "m/s^2", "gyro": "rad/s", "time": "ms"},
        "frame": "segment frame per Hackathon2026/protocol/ORIENTATION.md (proposed): "
                 "+X wearer-right, +Y up, +Z posterior; accel is specific force (+9.81 on +Y when upright)",
        "anchors": [{"sensor_id": i + 1, "name": a} for i, a in enumerate(ANCHORS)],
        "anchor_is_real": {"pelvis": True, "lumbar": False, "thoracic": False, "upper_thoracic": True},
        "anchor_source": {
            "pelvis": f"REAL: {m['pelvis_src']}",
            "lumbar": "SYNTHETIC: slerp(pelvis->upper_thoracic, f=0.30), rigid-trunk accel interpolation",
            "thoracic": "SYNTHETIC: slerp(pelvis->upper_thoracic, f=0.65), rigid-trunk accel interpolation",
            "upper_thoracic": f"REAL: {m['ut_src']}" + (" + lever arm to spine" if ds == "erciyes" else ""),
        },
        "source_rate_hz": float(m["source_fs_hz"]),
        "effective_bandwidth_hz": float(m["source_fs_hz"]) / 2,
        "device_profile": profile, "device": DEVICE_PROFILES[profile],
        "dataset": LICENCES[ds], **extra,
    }


def sample_lines(acc, gyro, t0_ms=0):
    for i in range(len(acc)):
        yield {"type": "sample", "time": int(t0_ms + round(i * 1000 / FS_HZ)),
               "data": [{"sensor_id": k + 1, "name": ANCHORS[k],
                         "accel": dict(zip("xyz", np.round(acc[i, k], 4).tolist())),
                         "gyro": dict(zip("xyz", np.round(gyro[i, k], 5).tolist()))}
                        for k in range(4)]}


def write(path_base, hdr, acc, gyro):
    with open(path_base + ".jsonl", "w") as f:
        f.write(json.dumps(hdr) + "\n")
        for s in sample_lines(acc, gyro):
            f.write(json.dumps(s) + "\n")
    cols = {"t_ms": (np.arange(len(acc)) * 1000 / FS_HZ).round().astype(int)}
    for k, a in enumerate(ANCHORS):
        for j, ax in enumerate("xyz"):
            cols[f"{a}_a{ax}_ms2"] = np.round(acc[:, k, j], 4)
        for j, ax in enumerate("xyz"):
            cols[f"{a}_g{ax}_rads"] = np.round(gyro[:, k, j], 5)
    pd.DataFrame(cols).to_csv(path_base + ".csv", index=False)


def stream_summary(det, acc, gyro):
    st = FallDetectorStream(detector=det)
    evs = []
    for i in range(len(acc)):
        evs += st.push((acc[i], gyro[i]))
    evs += st.flush()
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()
             if k in ("type", "state", "confidence", "t_ms", "decided_at_ms")} for e in evs]


def main():
    meta = pd.read_csv(os.path.join(ROOT, "data/processed/meta.csv"))
    loso = pd.read_csv(os.path.join(ROOT, "reports/loso_predictions_target.csv"))
    best = json.load(open(os.path.join(ROOT, "reports/metrics_target.json")))["best_model"]
    for profile in ("target", "current"):
        z = np.load(os.path.join(ROOT, f"data/processed/canonical_{profile}.npz"))
        det = Detector(profile)
        outdir = os.path.join(ROOT, "fixtures", profile)
        os.makedirs(outdir, exist_ok=True)
        index = []
        for name, subj, code in PICK:
            sel = meta[(meta["subject"] == (subj or SUBJECT)) & (meta["code"] == code)]
            m = sel.iloc[[0]]
            r = int(m["rec"].iloc[0])
            acc, gyro = z["acc"][m["start"].iloc[0]:m["stop"].iloc[0]], z["gyro"][m["start"].iloc[0]:m["stop"].iloc[0]]
            out = detect((acc, gyro), detector=det)
            extra = dict(subject=m["subject"].iloc[0], activity_code=int(code),
                         activity=m["activity"].iloc[0], trial=m["trial"].iloc[0],
                         true_label=m["label"].iloc[0], duration_s=float(m["duration_s"].iloc[0]),
                         loso_prediction=(loso.loc[loso["rec"] == r, f"pred_{best}"].iloc[0]
                                          if profile == "target" else None),
                         note="This recording WAS in the training data of the shipped model "
                              "(trained on all subjects). The LOSO prediction is the honest one.")
            write(os.path.join(outdir, name), header(m, profile, extra), acc, gyro)
            index.append(dict(file=f"{name}.jsonl", true_label=extra["true_label"],
                              activity=extra["activity"], duration_s=extra["duration_s"],
                              loso_prediction=extra["loso_prediction"],
                              shipped_detect_state=out["state"],
                              shipped_detect_confidence=round(out["confidence"], 3),
                              stream_events=stream_summary(det, acc, gyro)))
        # composite: one subject's recordings back to back with 0.5 s cross-fades
        pieces, timeline, t = [], [], 0
        for code in COMPOSITE:
            m = meta[(meta["subject"] == SUBJECT) & (meta["code"] == code)].iloc[0]
            a = z["acc"][m["start"]:m["stop"]].astype(np.float64)
            g = z["gyro"][m["start"]:m["stop"]].astype(np.float64)
            if pieces:
                n = int(0.5 * FS_HZ)
                w = np.linspace(0, 1, n)[:, None, None]
                pa, pg = pieces[-1]
                a[:n] = (1 - w) * pa[-n:] + w * a[:n]
                g[:n] = (1 - w) * pg[-n:] + w * g[:n]
                pieces[-1] = (pa[:-n], pg[:-n])
                t -= n
            timeline.append(dict(start_ms=int(t * 1000 / FS_HZ), activity=m["activity"],
                                 code=int(code), true_label=m["label"]))
            pieces.append((a, g))
            t += len(a)
        acc = np.concatenate([p[0] for p in pieces]).astype(np.float32)
        gyro = np.concatenate([p[1] for p in pieces]).astype(np.float32)
        m = meta[(meta["subject"] == SUBJECT) & (meta["code"] == COMPOSITE[0])].iloc[[0]]
        extra = dict(subject=SUBJECT, activity="composite mini-shift", true_label="fall_confirmed (last segment)",
                     timeline=timeline, duration_s=len(acc) / FS_HZ,
                     note="Recordings joined with 0.5 s cross-fades; seams are artificial.")
        write(os.path.join(outdir, "composite_shift_ending_in_fall"), header(m, profile, extra), acc, gyro)
        index.append(dict(file="composite_shift_ending_in_fall.jsonl", true_label=extra["true_label"],
                          activity=extra["activity"], duration_s=extra["duration_s"], timeline=timeline,
                          stream_events=stream_summary(det, acc, gyro)))
        with open(os.path.join(outdir, "index.json"), "w") as f:
            json.dump(index, f, indent=1)
        print(profile, "fixtures:", len(index))


if __name__ == "__main__":
    main()
