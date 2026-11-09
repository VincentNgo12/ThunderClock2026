# File: realism/paths.py
# Purpose: Locations of the real datasets (never inside the repo) and their sensor specs.
# Contributors: Unassigned
"""Real data stays under research/fall_model/data (outside the team repo, never committed).

Set THUNDERSIM_REALDATA to that data folder if it is elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]                 # detection/simulation
REPO_ROOT = SIM_ROOT.parents[1]                                # ThunderClock2026
FALLDET_ROOT = REPO_ROOT / "detection" / "models"              # read-only
HARMONIZE_PY = FALLDET_ROOT / "scripts" / "harmonize.py"       # read-only
DATA_ROOT = Path(os.environ.get(
    "THUNDERSIM_REALDATA", REPO_ROOT.parent / "research" / "fall_model" / "data"))
RAW_TRIALS = DATA_ROOT / "interim" / "raw_trials.pkl"          # made by load_raw.py (research-95)
PROCESSED = DATA_ROOT / "processed"                            # made by harmonize.py (research-95)

G = 9.80665


@dataclass(frozen=True)
class SensorSpec:
    """What one real dataset's trunk sensors can see. Used to emulate them on the sim body."""

    name: str
    fs: float              # output rate, Hz
    lowpass_hz: float      # effective bandwidth (anti-alias) applied to the simulated signal
    acc_clip: float        # per-axis full scale, m/s^2
    gyro_clip: float       # per-axis full scale, rad/s
    acc_noise: float       # white noise std per sample, m/s^2 (design estimate)
    gyro_noise: float      # rad/s (design estimate)
    slots: tuple[str, str]


SPECS = {
    # Xsens MTw, 25 Hz output (file header), +/-16 g and +/-1200 deg/s (MTw spec sheet [verify]).
    "erciyes": SensorSpec("erciyes", 25.0, 11.0, 16 * G, np.deg2rad(1200.0), 0.01, 0.003, ("waist", "chest")),
    # VTT-ConIot: 100 Hz rows; recorded values clip at exactly +/-2 g and ~+/-250 deg/s (4.4 rad/s),
    # measured from the files (see REPORT.md), so the sensors ran at +/-2 g / +/-250 deg/s.
    "vtt": SensorSpec("vtt", 100.0, 40.0, 2 * G, np.deg2rad(250.0), 0.02, 0.005, ("hip", "back")),
    # UNIVRFall: one lower-back sensor (L1-L2), 100 Hz; the recorded accelerometer clips at +/-3997 mg
    # per axis (measured), gyro +/-2000 deg/s; bandwidth assumed ~40 Hz [verify]. Both slots = that sensor.
    "univr": SensorSpec("univr", 100.0, 40.0, 3.997 * G, np.deg2rad(2000.0), 0.02, 0.005, ("lumbar", "lumbar")),
}
