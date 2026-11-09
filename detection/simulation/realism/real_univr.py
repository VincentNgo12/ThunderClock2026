# File: realism/real_univr.py
# Purpose: Load UNIVRFall (Zenodo 18346755, CC BY 4.0): one lower-back IMU (L1-L2), 100 Hz,
#          video-labelled fall onset and impact frames, plus construction-site recordings.
# Contributors: Unassigned
"""UNIVRFall reader. The archive is untrusted data: it is only read as CSV and as spreadsheet XML
(standard library zipfile + ElementTree); nothing in it is executed.

Measured from the files (not the dataset page):
- accelerometer in milli-g, sensor +X is up when standing; values clip per axis at +/-3997 mg
  (so the effective range is +/-4 g, not the +/-16 g the page states);
- gyroscope in milli-deg/s with an uncalibrated bias of a few deg/s (removed per recording);
- 10 ms rows (100 Hz), FrameCounter matches the label spreadsheet's onset/impact frames;
- tasks 41/42 are backward falls while climbing up/down STAIRS (labels), not a ladder.

Frame: only the up axis is calibrated (per recording, from the first second), because the event
features use magnitudes and inclination only. Activity features that need the horizontal axes
are dropped for UNIVRFall categories (see mapping.ActivityCategory.drop).
"""

from __future__ import annotations

import glob
import os
import pickle
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .paths import DATA_ROOT, G

UNIVR_ROOT = DATA_ROOT / "raw" / "univrfall" / "extracted" / "UniVrFall_Dataset"
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MAX_XLSX_BYTES = 5_000_000  # label sheets are tiny; refuse anything unexpectedly large


def read_xlsx_rows(path) -> list[list]:
    """First worksheet of an .xlsx as rows of strings (values only, no formulas evaluated)."""
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.file_size > MAX_XLSX_BYTES:
                raise ValueError(f"{path}: member {info.filename} too large")
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.iter("{%s}row" % _NS["m"]):
        row = []
        for c in r.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            val = None if v is None else v.text
            if c.get("t") == "s" and val is not None:
                val = shared[int(val)]
            row.append(val)
        rows.append(row)
    return rows


def _to_segment(acc_mg: np.ndarray, gyr_mdps: np.ndarray, up_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so the mean specific force over the first up_n samples is +Y; gyro bias from the same
    samples. Returns SI units (m/s^2, rad/s)."""
    acc = acc_mg / 1000.0 * G
    gyr = np.deg2rad(gyr_mdps / 1000.0)
    up = acc[:up_n].mean(0)
    up /= np.linalg.norm(up)
    gyr = gyr - np.median(gyr[:up_n], axis=0)
    # shortest rotation taking `up` to +Y
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    s, c = np.linalg.norm(v), float(up @ y)
    if s < 1e-9:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = v / s
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        ang = np.arctan2(s, c)
        rot = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx
    return acc @ rot.T, gyr @ rot.T


@dataclass
class UnivrEvent:
    subject: str
    task: int
    trial: int
    fs: float
    acc: np.ndarray     # (N,2,3): the one lower-back sensor duplicated into both slots
    gyro: np.ndarray
    onset: int          # sample index of the labelled fall onset
    impact: int         # sample index of the labelled impact


def _read_csv(path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(8), dtype=np.float64)


def load_events(tasks: set[int]) -> list[UnivrEvent]:
    out = []
    for lab in sorted(glob.glob(str(UNIVR_ROOT / "laboratory" / "labels_data" / "SA*_label.xlsx"))):
        subj = os.path.basename(lab)[2:4]
        for r in read_xlsx_rows(lab)[1:]:
            if len(r) < 5 or r[0] is None or r[3] is None or r[4] is None or "(" not in r[0]:
                continue
            task = int(r[0].split("(")[1].rstrip(")"))
            if task not in tasks:
                continue
            trial = int(float(r[2]))
            path = UNIVR_ROOT / "laboratory" / "sensors_data" / f"SA{subj}" / f"S{subj}T{task:02d}R{trial:02d}.csv"
            if not path.exists():
                continue
            a = _read_csv(path)
            if len(a) < 300:
                continue
            fc = a[:, 1]
            onset = int(np.searchsorted(fc, float(r[3])))
            impact = int(np.searchsorted(fc, float(r[4])))
            if not (0 < onset < impact < len(a)):
                continue
            acc, gyr = _to_segment(a[:, 2:5], a[:, 5:8], up_n=min(100, onset))
            out.append(UnivrEvent(f"U{subj}", task, trial, 100.0, np.stack([acc, acc], 1).astype(np.float32),
                                  np.stack([gyr, gyr], 1).astype(np.float32), onset, impact))
    return out


def load_site_windows(chunks_per_file: int = 6, chunk_s: float = 300.0, seed: int = 0):
    """Construction-site recordings (OnField): a random sample of 5-minute chunks per file.

    Format differs from the lab files: columns id, timestamp, type, 3 reserved, accX..Z (mg),
    gyrX..Z (mdps), ...; only type == 1 rows are IMU samples (10 ms apart). The worker id is the
    file-name suffix (e.g. 20250109_1003.csv -> worker 1003) and is the grouping unit."""
    rng = np.random.default_rng(seed)
    out = []
    for path in sorted(glob.glob(str(UNIVR_ROOT / "OnField" / "*" / "*.csv"))):
        worker = "W" + os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(12), dtype=np.float64)
        a = a[a[:, 2] == 1]
        chunk = int(chunk_s * 100)
        starts = np.arange(0, max(len(a) - chunk, 1), chunk)
        for s0 in rng.choice(starts, size=min(len(starts), chunks_per_file), replace=False):
            seg = a[s0:s0 + chunk]
            if len(seg) < 1000:
                continue
            acc_mg, gyr = seg[:, 6:9], seg[:, 9:12]
            pad_a, pad_g = np.tile(np.median(acc_mg, 0), (100, 1)), np.tile(np.median(gyr, 0), (100, 1))
            acc, gyr = _to_segment(np.vstack([pad_a, acc_mg]), np.vstack([pad_g, gyr]), 100)
            acc, gyr = acc[100:], gyr[100:]
            out.append((worker, np.stack([acc, acc], 1).astype(np.float32), np.stack([gyr, gyr], 1).astype(np.float32)))
    return out


def cached(name: str, cache_dir, fn):
    path = cache_dir / name
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return value
