# File: realism/site_impacts.py
# Purpose: Hazard-relevant side effect of contact-model changes: pelvis impact peaks of SITE falls
#          (their own surfaces, target-firmware bandwidth 44 Hz at 100 Hz), old vs current constants.
# Contributors: Unassigned
"""Usage: .venv/bin/python -m realism.site_impacts [--n 12]"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import signal

from thundersim import falls
from thundersim.scenarios import build_run

from . import emulate

OLD = {"CONTACT_RESTITUTION": None, "SET_RECOVERY_S": 1.5, "CONTACT_HC_ALPHA": None, "ROT_CONTACT_ZETA": 0.4, "ROT_CONTACT_RAMP_S": 0.0, "TRUNK_TOPPLE_GAIN": 1.0,
       "SYNC_SUPPORT": None, "SYNC_BUCKLE_S": (1.0, 2.0), "PIVOT_MOMENTUM_KEEP": 1.0, "PIVOT_RESIST": 0.0}
SCENARIOS = ("L-TRIP", "L-SLIP", "STRUCK-KD", "SYNC", "H-LAD", "H-SCAF", "STAIR")


def peaks(scenario: str, n: int) -> list[float]:
    out = []
    sos = signal.butter(2, 44.0, fs=1000.0, output="sos")
    for i in range(n):
        run = build_run(scenario, emulate.dev_seed(70, i), obs_s=10)
        acc, _ = emulate.body_signals(run.track, run.worker, run.soft_fn, run.soft_zeta, "vtt", four=True)
        a = np.linalg.norm(signal.sosfilt(sos, acc[:, 0], axis=0), axis=1)[::10] / 9.80665
        t0 = run.ref_time or 0.0
        lo = max(int((t0 - 2.0) * 100), 0)
        out.append(float(a[lo:lo + 600].max()))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--keys", default="", help="comma-separated constants to revert (default: all in OLD)")
    args = p.parse_args(argv)
    old = {k: v for k, v in OLD.items() if not args.keys or k in args.keys.split(",")}
    current = {k: getattr(falls, k) for k in old}
    print("| scenario | old median (IQR) g | current median (IQR) g |\n|---|---|---|")
    for sc in SCENARIOS:
        res = {}
        for label, values in (("old", old), ("cur", current)):
            for k, v in values.items():
                setattr(falls, k, v)
            q = np.percentile(peaks(sc, args.n), [25, 50, 75])
            res[label] = f"{q[1]:.1f} ({q[0]:.1f}-{q[2]:.1f})"
        for k, v in current.items():
            setattr(falls, k, v)
        print(f"| {sc} | {res['old']} | {res['cur']} |", flush=True)


if __name__ == "__main__":
    main()
