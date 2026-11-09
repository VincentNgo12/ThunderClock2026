# File: realism/real.py
# Purpose: Load the real recordings (Erciyes waist+chest, VTT hip+back) into the simulator's
#          segment frame (+X right, +Y up, +Z posterior) at their native rate.
# Contributors: Unassigned
"""Real side of the comparison.

raw_trials.pkl was written by research-95's load_raw.py from the open-licence downloads (sensor
frame, SI units). The sensor-to-segment rotation per subject and sensor is the data-driven one
from harmonize.calibrate() (up from quiet standing, posterior from forward bends; VTT hip
heading from the magnetometers). harmonize.py is imported read-only and not modified. Unlike
canonical_target.npz, nothing here is resampled, lever-arm corrected or device-emulated: the
real signal stays as recorded and the SIMULATED body is made to look like the real sensor.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from dataclasses import dataclass

import numpy as np

from .paths import FALLDET_ROOT, HARMONIZE_PY, RAW_TRIALS


@dataclass
class Recording:
    dataset: str          # erciyes | vtt
    subject: str          # group id for subject-grouped statistics
    code: int
    activity: str
    trial: str
    fs: float
    acc: np.ndarray       # (N, 2, 3) m/s^2, segment frame; slot 0 waist/hip, slot 1 chest/back
    gyro: np.ndarray      # (N, 2, 3) rad/s


def _harmonize():
    if str(FALLDET_ROOT) not in sys.path:
        sys.path.insert(0, str(FALLDET_ROOT))
    spec = importlib.util.spec_from_file_location("_tc_harmonize", HARMONIZE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_recordings(cache_path=None) -> list[Recording]:
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    with open(RAW_TRIALS, "rb") as handle:  # local research-95 output, not a download
        trials = pickle.load(handle)
    cal, _notes = _harmonize().calibrate(trials)
    out = []
    for t in trials:
        mp, mu = cal[(t["subject"], "p")], cal[(t["subject"], "ut")]
        acc = np.stack([t["acc_p"] @ mp.T, t["acc_ut"] @ mu.T], axis=1)
        gyro = np.stack([t["gyr_p"] @ mp.T, t["gyr_ut"] @ mu.T], axis=1)
        if t["dataset"] == "vtt":  # the first 3 s hold two synchronisation jumps (harmonize.VTT_TRIM_S)
            acc, gyro = acc[300:], gyro[300:]
        out.append(Recording(t["dataset"], t["subject"], int(t["code"]), t["activity"], t["trial"],
                             float(t["fs"]), acc.astype(np.float32), gyro.astype(np.float32)))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out
