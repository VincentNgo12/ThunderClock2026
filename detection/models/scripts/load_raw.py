"""Load the two public datasets into one plain-Python list of trials (sensor frame, SI units).

Run from the detection/models/ folder:
    .venv/bin/python -I scripts/load_raw.py

Datasets are untrusted downloads, so this script only *reads* text/CSV files from
data/raw/ and never imports or executes anything from there. Output:
    data/interim/raw_trials.pkl   (list of dicts, see `make_trial` below)

Units after loading: accelerometer m/s^2, gyroscope rad/s, sensor's own axes.
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 9.80665  # m/s^2 per g

# ---------------------------------------------------------------- Erciyes (UCI 455)
ERCIYES_DIR = os.path.join(ROOT, "data/raw/erciyes_uci455/subjects")
# Information.pdf lists 340527 = chest and 340539 = right thigh, but in all 17 subjects the
# data behave the other way round: in sit-chair (811) 340527 ends ~70 deg from vertical
# (a horizontal thigh) while 340539 stays upright; in bending (805) 340539 tilts ~90 deg
# (the trunk) while 340527 stays straight. Head/ankle IDs are swapped the same way.
# We therefore use 340539 as the chest. Waist 340535 behaves as documented.
ERCIYES_SENSORS = {"waist": "340535", "chest": "340539"}
ERCIYES_FS = 25.0  # Hz, "// Update Rate: 25.0Hz" in every file header

# Class labels used everywhere in this project.
#   fall_confirmed: fall that ends on the floor without getting up
#   near_fall:      fall/stumble followed by recovery (the person gets back up)
#   normal:         everything else
ERCIYES_NEAR_FALL = {905, 906, 914, 916, 808}  # falls with recovery + "stumble with recovery"
ERCIYES_NAMES = {
    801: "walking-fw", 802: "walking-bw", 803: "jogging", 804: "squatting-down",
    805: "bending", 806: "bending-pick-up", 807: "limp", 808: "stumble",
    809: "trip-over", 810: "coughing-sneezing", 811: "sit-chair", 812: "sit-sofa",
    813: "sit-air", 814: "sit-bed", 815: "lying-bed", 816: "rising-bed",
    901: "front-lying", 902: "front-protecting-lying", 903: "front-knees",
    904: "front-knees-lying", 905: "front-quick-recovery", 906: "front-slow-recovery",
    907: "front-right", 908: "front-left", 909: "back-sitting", 910: "back-lying",
    911: "back-right", 912: "back-left", 913: "right-sideway", 914: "right-recovery",
    915: "left-sideway", 916: "left-recovery", 917: "rolling-out-bed", 918: "podium",
    919: "syncope", 920: "syncope-wall",
}


def erciyes_label(code):
    if code in ERCIYES_NEAR_FALL:
        return "near_fall"
    return "fall_confirmed" if code >= 900 else "normal"


def read_xsens_txt(path):
    """Return (counter, acc[m/s^2], gyr[rad/s]) from one Xsens MTw export file."""
    df = pd.read_csv(path, sep="\t", comment="/", engine="c")
    df.columns = [c.strip() for c in df.columns]
    # A few rows have missing values ("Missing Attribute Values: Rarely"): drop them,
    # align_on_counter() then fills the hole by linear interpolation.
    df = df.dropna(subset=["Counter", "Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"])
    cnt = df["Counter"].to_numpy(np.int64)
    acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(np.float64)
    gyr = df[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(np.float64)
    return cnt, acc, gyr


def align_on_counter(parts):
    """Put several sensors on one common 25 Hz grid using the sample counter.

    Missing samples (rare, per the dataset notes) are linearly interpolated.
    """
    lo = max(p[0][0] for p in parts)
    hi = min(p[0][-1] for p in parts)
    grid = np.arange(lo, hi + 1)
    out = []
    for cnt, acc, gyr in parts:
        # counters are 16-bit in some exports; unwrap just in case
        cnt = np.unwrap(cnt, period=65536).astype(np.int64) if np.any(np.diff(cnt) < 0) else cnt
        cols = []
        for arr in (acc, gyr):
            cols.append(np.stack([np.interp(grid, cnt, arr[:, k]) for k in range(3)], axis=1))
        out.append(cols)
    return out, len(grid)


def load_erciyes():
    trials = []
    for subj_dir in sorted(glob.glob(os.path.join(ERCIYES_DIR, "*"))):
        subject = os.path.basename(subj_dir)
        for act_dir in sorted(glob.glob(os.path.join(subj_dir, "Testler Export", "*"))):
            m = re.match(r"(\d{3})", os.path.basename(act_dir))
            if not m:
                continue
            code = int(m.group(1))
            for test_dir in sorted(glob.glob(os.path.join(act_dir, "Test_*"))):
                files = {k: os.path.join(test_dir, v + ".txt") for k, v in ERCIYES_SENSORS.items()}
                if not all(os.path.exists(f) for f in files.values()):
                    print("missing sensor file, skipped:", test_dir, file=sys.stderr)
                    continue
                try:
                    parts = [read_xsens_txt(files["waist"]), read_xsens_txt(files["chest"])]
                    (w, c), n = align_on_counter(parts)
                except Exception as e:  # malformed file: skip, but say so
                    print("unreadable, skipped:", test_dir, e, file=sys.stderr)
                    continue
                if n < 2 * ERCIYES_FS:
                    continue
                trials.append(dict(
                    dataset="erciyes", subject=f"E{subject}", code=code,
                    activity=ERCIYES_NAMES.get(code, str(code)),
                    trial=os.path.basename(test_dir), label=erciyes_label(code), fs=ERCIYES_FS,
                    pelvis_src="erciyes:waist(340535)", ut_src="erciyes:chest(340539, see ID note)",
                    acc_p=w[0], gyr_p=w[1], acc_ut=c[0], gyr_ut=c[1]))
        print("erciyes", subject, len(trials), file=sys.stderr)
    return trials


# ---------------------------------------------------------------- VTT-ConIot
VTT_DIR = os.path.join(ROOT, "data/raw/vtt_coniot/extracted/VTT_ConIot_Dataset/VTT-ConIot-IMU-Data")
VTT_FS = 100.0  # rows are on a 10 ms grid ("datetime" column); sensor ts spacing ~9.7 ms
VTT_NAMES = {1: "roll-painting", 2: "spraying-paint", 3: "levelling-paint", 4: "vacuum-cleaning",
             5: "picking-objects", 6: "climbing-stairs", 7: "jumping-down", 8: "laying-back",
             9: "hands-up-high", 10: "hands-up-low", 11: "crouch-floor", 12: "kneel-floor",
             13: "walk-straight", 14: "walk-winding", 15: "pushing-cart", 16: "stairs-up-down"}


def vtt_xyz(df, sensor, kind):
    if kind == "mag":  # microtesla, only used to align the hip sensor's heading
        cols = [f"{sensor}_M{a}_uT" for a in "xyz"]
        scale = 1.0
    elif kind == "acc":  # file unit is "half-g": |a| at rest reads ~2.0, so /2 -> g -> m/s^2
        cols = [f"{sensor}_A{a}_g" for a in "xyz"]
        scale = G / 2.0
    else:              # deg/s -> rad/s
        cols = [f"{sensor}_G{a}_dps" for a in "xyz"]
        scale = np.pi / 180.0
    x = df[cols].astype(float).interpolate(limit_direction="both").to_numpy()
    return x * scale


def load_vtt():
    trials = []
    for path in sorted(glob.glob(os.path.join(VTT_DIR, "activity_*_user_*_combined.csv"))):
        m = re.search(r"activity_(\d+)_user_(\d+)_combined", path)
        act, user = int(m.group(1)), int(m.group(2))
        df = pd.read_csv(path)
        acc_p, gyr_p = vtt_xyz(df, "trousers", "acc"), vtt_xyz(df, "trousers", "gyr")
        acc_u, gyr_u = vtt_xyz(df, "back", "acc"), vtt_xyz(df, "back", "gyr")
        if any(np.isnan(a).any() for a in (acc_p, gyr_p, acc_u, gyr_u)):
            print("all-NaN sensor, skipped:", path, file=sys.stderr)
            continue
        trials.append(dict(
            dataset="vtt", subject=f"V{user:02d}", code=1000 + act, activity=VTT_NAMES[act],
            trial="1", label="normal", fs=VTT_FS,
            pelvis_src="vtt:trousers(hip)", ut_src="vtt:back",
            acc_p=acc_p, gyr_p=gyr_p, acc_ut=acc_u, gyr_ut=gyr_u,
            mag_p=vtt_xyz(df, "trousers", "mag"), mag_ut=vtt_xyz(df, "back", "mag")))
    print("vtt", len(trials), file=sys.stderr)
    return trials


if __name__ == "__main__":
    trials = load_erciyes() + load_vtt()
    out = os.path.join(ROOT, "data/interim/raw_trials.pkl")
    with open(out, "wb") as f:
        pickle.dump(trials, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", out, len(trials), "trials")
