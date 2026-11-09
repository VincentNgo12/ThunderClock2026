# File: realism/fit.py
# Purpose: Grid-fit simulator parameters to the real data on FIT seeds (dev seeds disjoint from
#          the scorecard's), by the summed normalised Wasserstein distance of chosen features.
# Contributors: Unassigned
"""Usage: .venv/bin/python -m realism.fit contact --out <dir>   (see GRIDS)

Each grid point patches module constants in the worker processes, regenerates the event
categories on fit seeds (block 50+), and scores the chosen features against the cached real side.
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import emulate
from .features import trace_features
from .mapping import EVENTS
from .score import _event_job, real_side
from .stats import feature_gap

GRIDS = {
    "contact": {
        "module": "thundersim.falls",
        "params": {"CONTACT_HC_ALPHA": [None, 0.6, 1.2], "ROT_CONTACT_ZETA": [0.4, 0.8, 1.2],
                   "TRUNK_TOPPLE_GAIN": [1.0, 0.5]},
        "fixed": {"ROT_CONTACT_RAMP_S": 0.05},
        "categories": ["fall_forward", "fall_backward", "fall_side"],
        "features": ["postmin_w", "gyro_post_c", "rebound_w", "npk_w", "peak_c", "settle_s", "gpk_c", "peak_w"],
    },
    "contact2": {
        "module": "thundersim.falls",
        "params": {"CONTACT_HC_ALPHA": [1.2, 2.0], "TRUNK_TOPPLE_GAIN": [0.5, 0.25], "MAT_K_SCALE": [1.0, 0.5, 0.3, 0.2]},
        "fixed": {"ROT_CONTACT_RAMP_S": 0.05, "ROT_CONTACT_ZETA": 1.2},
        "categories": ["fall_forward", "fall_backward", "fall_side"],
        "features": ["postmin_w", "gyro_post_c", "rebound_w", "npk_w", "peak_c", "settle_s", "gpk_c", "peak_w", "jerk_w"],
    },
    "contact3": {
        "module": "thundersim.falls",
        "params": {"CONTACT_HC_ALPHA": [0.3, 0.6, 0.9]},
        "fixed": {},
        "categories": ["fall_forward", "fall_backward", "fall_side"],
        "features": ["postmin_w", "gyro_post_c", "rebound_w", "npk_w", "peak_c", "settle_s", "gpk_c", "peak_w", "jerk_w",
                     "post_incl_w", "post_incl_c", "rot_total"],
    },
    "contact4": {
        "module": "thundersim.falls",
        "params": {"CONTACT_RESTITUTION": [0.1, 0.25, 0.4]},
        "fixed": {},
        "categories": ["fall_forward", "fall_backward", "fall_side"],
        "features": ["postmin_w", "gyro_post_c", "rebound_w", "npk_w", "peak_c", "settle_s", "gpk_c", "peak_w", "jerk_w",
                     "post_incl_w", "post_incl_c", "rot_total"],
    },
    "anny": {
        "module": "thundersim.falls",
        "params": {"ANNY_GEOM": [None, "com", "com+spacing"]},
        "fixed": {},
        "categories": ["fall_forward", "fall_backward", "fall_side"],
        "features": ["descent_s", "rot_total", "gpk_c", "gpk_w", "tgpk_c", "ffmin_w", "ffdur_w", "peak_w", "peak_c",
                     "postmin_w", "gyro_post_c", "post_incl_w", "post_incl_c"],
    },
    "descent_keep": {
        "module": "thundersim.falls",
        "params": {"PIVOT_MOMENTUM_KEEP": [1.0, 0.5, 0.2]},
        "fixed": {},
        "categories": ["fall_forward", "fall_side", "fall_backward"],
        "features": ["descent_s", "rot_total", "gpk_c", "gpk_w", "tgpk_c", "ffmin_w", "ffdur_w", "peak_w", "jerk_w",
                     "postmin_w"],
    },
    "descent_resist": {
        "module": "thundersim.falls",
        "params": {"PIVOT_RESIST": [0.0, 2.0, 4.0, 6.0]},
        "fixed": {"PIVOT_MOMENTUM_KEEP": 0.5},
        "categories": ["fall_forward", "fall_side", "fall_backward"],
        "features": ["descent_s", "rot_total", "gpk_c", "gpk_w", "tgpk_c", "ffmin_w", "ffdur_w", "peak_w", "jerk_w",
                     "postmin_w"],
    },
    "descent2": {
        "module": "thundersim.falls",
        "params": {"PIVOT_RESIST": [2.0, 3.0, 4.0, 5.0]},
        "fixed": {"PIVOT_MOMENTUM_KEEP": 0.5},
        "categories": ["fall_forward", "fall_side", "fall_recover", "u_trip"],
        "features": ["descent_s", "label_dur_s", "rot_total", "gpk_c", "gpk_w", "tgpk_c", "ffmin_w", "ffdur_w", "peak_w",
                     "jerk_w", "postmin_w"],
    },
    "sync3": {
        "module": "thundersim.falls",
        "params": {"SYNC_SUPPORT": [0.5, 0.6, 0.7, 0.8], "SYNC_BUCKLE_S": [(0.4, 0.8), (0.8, 1.2)]},
        "fixed": {},
        "categories": ["syncope", "u_faint_walk"],
        "features": ["peak_w", "ffmin_w", "ffdur_w", "gpk_w", "tgpk_c", "descent_s", "label_dur_s", "rot_total",
                     "npk_w", "jerk_w", "postmin_w", "gyro_post_c", "rebound_w", "settle_s"],
    },
    "topple": {
        "module": "thundersim.falls",
        "params": {"TRUNK_TOPPLE_GAIN": [0.5, 0.35, 0.25]},
        "fixed": {},
        "categories": ["fall_backward", "u_slip_back", "fall_forward", "fall_side"],
        "features": ["gyro_post_c", "rebound_w", "postmin_w", "settle_s", "label_dur_s", "peak_w", "post_incl_w",
                     "dincl_w", "rot_total", "npk_w"],
    },
    "sync2": {
        "module": "thundersim.falls",
        "params": {"SYNC_SUPPORT": [0.7, 0.8], "SYNC_BUCKLE_S": [(0.4, 0.8)]},
        "fixed": {},
        "categories": ["syncope"],
        "features": ["peak_w", "peak_c", "ffmin_w", "ffdur_w", "gpk_w", "gpk_c", "tgpk_c", "descent_s", "rot_total",
                     "npk_w", "jerk_w", "postmin_w", "gyro_post_c"],
    },
    "sync": {
        "module": "thundersim.falls",
        "params": {"SYNC_SUPPORT": [None, 0.6, 0.45, 0.3], "SYNC_BUCKLE_S": [(1.0, 2.0), (0.4, 0.8), (0.2, 0.4)]},
        "fixed": {},
        "categories": ["syncope"],
        "features": ["peak_w", "peak_c", "ffmin_w", "ffdur_w", "gpk_w", "gpk_c", "tgpk_c", "descent_s", "rot_total",
                     "npk_w", "jerk_w", "postmin_w", "gyro_post_c"],
    },
}


def _init(module: str, values: dict) -> None:
    import importlib

    from dataclasses import replace

    from thundersim import site

    mod = importlib.import_module(module)
    for k, v in values.items():
        if k == "ANNY_GEOM":  # candidate Anny-derived body geometry (avg male, scaled by height)
            if v is None:
                continue
            from dataclasses import replace as dc_replace

            from . import emulate as em
            base = em.erciyes_worker

            def anny_worker(rng, clothing, _base=base, _mode=v):
                w = _base(rng, clothing)
                h = w.height_m
                extra = {"pelvis_height_m": 0.57475 * h, "com_to_pelvis_m": (0.0, 0.00416 * h, 0.02752 * h + 0.04)}
                if "spacing" in _mode:
                    extra["spacing_m"] = (0.04975 * h, 0.07597 * h, 0.13403 * h)
                return dc_replace(w, **extra)

            em.erciyes_worker = anny_worker
        elif k == "MAT_K_SCALE":  # crash-mat stiffness range multiplier (a [design] value in site.py)
            mat = site.SURFACES["crash_mat"]
            site.SURFACES["crash_mat"] = replace(mat, stiffness_range=tuple(x * v for x in mat.stiffness_range))
        else:
            setattr(mod, k, v)


def evaluate(grid: dict, values: dict, real: dict, n: int, jobs: int) -> dict:
    tasks = []
    for block, cat in enumerate(EVENTS):
        if cat.name in grid["categories"]:
            tries = n * (3 if cat.scenario == "STRUCK-KD" else 1)
            tasks += [(cat.name, emulate.dev_seed(50 + block, i), False) for i in range(tries)]
    sims = {c: [] for c in grid["categories"]}
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(grid["module"], {**grid["fixed"], **values})) as pool:
        for item in pool.map(_event_job, tasks, chunksize=2):
            if item and len(sims[item[0]]) < n:
                sims[item[0]].append(item[1])
    detail, total = {}, 0.0
    for cat in grid["categories"]:
        rr = real["events"][cat]
        for feat in grid["features"]:
            get = lambda row: {**trace_features(row["trace"], row.get("fs", 25.0)), **row["f"]}.get(feat, np.nan)
            g = feature_gap(np.array([get(r) for r in rr]), np.array([r["group"] for r in rr]),
                            np.array([get(r) for r in sims[cat]]), np.array([r["group"] for r in sims[cat]]))
            if not np.isfinite(g.get("nW", np.nan)):
                continue
            detail[f"{cat}/{feat}"] = round(g["nW"], 3)
            total += g["nW"]
        speeds = [r["params"]["hip_speed_ms"] for r in sims[cat] if r["params"].get("hip_speed_ms") is not None]
        if speeds:
            detail[f"{cat}/hip_speed_median_ms"] = round(float(np.median(speeds)), 2)
        detail[f"{cat}/descent_sim_median_s"] = round(float(np.nanmedian([r["f"]["descent_s"] for r in sims[cat]])), 3)
        if any("label_dur_s" in r["f"] for r in sims[cat]):
            detail[f"{cat}/label_dur_sim_median_s"] = round(float(np.nanmedian([r["f"].get("label_dur_s", np.nan) for r in sims[cat]])), 3)
        detail[f"{cat}/gpk_c_sim_median"] = round(float(np.nanmedian([r["f"]["gpk_c"] for r in sims[cat]])), 2)
    return {"values": values, "objective": round(total, 3), "detail": detail}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("grid", choices=list(GRIDS))
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = GRIDS[args.grid]
    real = real_side(out.parent / "cache", True)
    names = list(grid["params"])
    rows = []
    for combo in itertools.product(*grid["params"].values()):
        values = dict(zip(names, combo))
        row = evaluate(grid, values, real, args.n, args.jobs)
        rows.append(row)
        print(row["objective"], values, flush=True)
        (out / f"fit_{args.grid}.json").write_text(json.dumps(rows, indent=1))
    best = min(rows, key=lambda r: r["objective"])
    print("BEST", best["objective"], best["values"])
    print(json.dumps(best["detail"], indent=1))


if __name__ == "__main__":
    main()
