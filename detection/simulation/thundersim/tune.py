# File: tune.py
# Purpose: Record canonical frames once, then replay them to tune baseline thresholds and to
#          export labelled synthetic datasets for model developers.
# Contributors: Unassigned
"""Threshold tuning on DEV seeds and dataset export.

Tuning never touches TEST seeds (>= TEST_SEED_BASE): pick a configuration on dev runs, then
judge it once with `suite --seed-base 1000000`. Exported datasets are simulated; use them to
test and augment models, never as their only training data or as a claim of performance.
"""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .baseline import BaselineConfig, BaselineRules
from .model_api import ALERT_KINDS, FrameBatch, StreamContext
from .runner import CALIBRATION_S, StreamOptions, run_scenario, run_shift
from .scenarios import CATALOGUE, build_run

TEST_SEED_BASE = 1_000_000


class Recorder:
    """FallModel that keeps every batch (used to cache frames for fast replay)."""

    name = "recorder"

    def reset(self, context: StreamContext) -> None:
        self.context = context
        self.batches: list[FrameBatch] = []

    def update(self, batch: FrameBatch) -> list:
        self.batches.append(batch)
        return []


def record(scenario_id: str, seed: int, profile: str, obs_s: float, fast: bool = True) -> dict:
    rec = Recorder()
    result = run_scenario(scenario_id, seed, rec, StreamOptions(profile=profile, fast=fast), obs_s=obs_s)
    return {"scenario": scenario_id, "seed": seed, "expected": result.expected, "report_only": result.report_only,
            "ref_time": result.ref_time, "context": rec.context, "batches": rec.batches}


def replay(entry: dict, model) -> list[dict]:
    model.reset(entry["context"])
    events = []
    for batch in entry["batches"]:
        for ev in model.update(batch) or []:
            events.append({"t": ev.t - 5.0, "kind": ev.kind, "detail": ev.detail})  # device boot offset 5 s
    return events


def outcome(events: list[dict]) -> str:
    kinds = {e["kind"] for e in events if e["t"] > CALIBRATION_S}
    return "fall_confirmed" if kinds & set(ALERT_KINDS) else ("near_fall" if "near_fall" in kinds else "normal")


_CACHE: list = []


def _init(cache_path: str) -> None:
    global _CACHE
    with open(cache_path, "rb") as handle:
        _CACHE = pickle.load(handle)


def evaluate(config: BaselineConfig) -> dict:
    falls_hit = falls_n = nf_ok = nf_n = nf_fa = neg_fa = neg_n = 0
    shift_fa, shift_h = 0, 0.0
    for entry in _CACHE:
        events = replay(entry, BaselineRules(config))
        if entry["scenario"] == "__shift__":
            shift_fa += sum(e["kind"] in ALERT_KINDS for e in events if e["t"] > CALIBRATION_S)
            shift_h += entry["hours"]
            continue
        out = outcome(events)
        if entry["report_only"]:
            continue
        if entry["expected"] == "fall_confirmed":
            falls_n += 1
            falls_hit += out == "fall_confirmed"
        elif entry["expected"] == "near_fall":
            nf_n += 1
            nf_ok += out == "near_fall"
            nf_fa += out == "fall_confirmed"
        else:
            neg_n += 1
            neg_fa += out == "fall_confirmed"
    sens = falls_hit / max(falls_n, 1)
    fa8 = shift_fa / max(shift_h / 8, 1e-9) if shift_h else 0.0
    score = sens + 0.3 * nf_ok / max(nf_n, 1) - 2.0 * neg_fa / max(neg_n, 1) - 0.5 * nf_fa / max(nf_n, 1) - 0.25 * fa8
    return {"config": asdict(config), "sensitivity": sens, "near_fall_acc": nf_ok / max(nf_n, 1),
            "near_fall_fa": nf_fa / max(nf_n, 1), "negative_fa": neg_fa / max(neg_n, 1), "shift_fa_per_8h": fa8, "score": score}


def build_cache(path: Path, seeds_per_scenario: int, profile: str, obs_s: float, shift_hours: float, jobs: int) -> None:
    tasks = [(sid, seed) for sid in CATALOGUE for seed in range(seeds_per_scenario)]
    with ProcessPoolExecutor(jobs) as pool:
        entries = list(pool.map(_record_task, [(sid, seed, profile, obs_s) for sid, seed in tasks]))
    if shift_hours > 0:
        rec = Recorder()
        result = run_shift(424_242, shift_hours, rec, StreamOptions(profile=profile, fast=True))
        entries.append({"scenario": "__shift__", "seed": 424_242, "hours": result["hours"], "context": rec.context,
                        "batches": rec.batches, "expected": "normal", "report_only": False, "ref_time": None})
    with open(path, "wb") as handle:
        pickle.dump(entries, handle)


def _record_task(args) -> dict:
    return record(*args)


def random_configs(n: int, seed: int = 0) -> list[BaselineConfig]:
    rng = np.random.default_rng(seed)
    base = BaselineConfig()
    configs = [base]
    for _ in range(n - 1):
        configs.append(replace(
            base,
            impact_g=float(rng.uniform(2.0, 4.0)), rotation_rad_s=float(rng.uniform(2.0, 4.5)),
            free_fall_g=float(rng.uniform(0.4, 0.75)), lying_deg=float(rng.uniform(50, 75)),
            collapse_still_s=float(rng.uniform(2, 10)), height_free_fall_s=float(rng.uniform(0.2, 0.45)),
            arrest_g=float(rng.uniform(1.5, 3.0)), still_acc_sd_g=float(rng.uniform(0.015, 0.05)),
            near_rotation_rad_s=float(rng.uniform(1.0, 2.5)), near_dip_g=float(rng.uniform(0.75, 0.9)),
            lying_pelvis_deg=float(rng.uniform(40, 70)), warning_s=float(rng.choice([20.0, 30.0, 45.0])),
            confirm_window_s=float(rng.choice([8.0, 10.0, 15.0]))))
    return configs


def tune(out_dir: Path, trials: int, seeds_per_scenario: int, profile: str, obs_s: float, shift_hours: float, jobs: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"tune_cache_{profile}.pkl"
    if not cache.exists():
        build_cache(cache, seeds_per_scenario, profile, obs_s, shift_hours, jobs)
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(str(cache),)) as pool:
        results = list(pool.map(evaluate, random_configs(trials)))
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    (out_dir / "tuned_baseline.json").write_text(json.dumps(best["config"], indent=2))
    (out_dir / "tune_results.json").write_text(json.dumps(results, indent=2))
    return {"best": best, "default": next(r for r in results if r["config"] == asdict(BaselineConfig())), "trials": len(results)}


def export(out_dir: Path, scenarios: list[str], seeds: list[int], profile: str, obs_s: float, jobs: int) -> int:
    """Write one .npz per run (canonical frames + per-frame ground-truth phase) and a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(sid, seed, profile, obs_s, str(out_dir)) for sid in scenarios for seed in seeds]
    with ProcessPoolExecutor(jobs) as pool:
        manifest = list(pool.map(_export_task, tasks))
    (out_dir / "manifest.json").write_text(json.dumps({"profile": profile, "simulated": True, "runs": manifest,
                                                      "phases": __import__("thundersim.spine", fromlist=["PHASES"]).PHASES}, indent=1))
    return len(manifest)


def _export_task(args) -> dict:
    sid, seed, profile, obs_s, out_dir = args
    entry = record(sid, seed, profile, obs_s)
    run = build_run(sid, seed, obs_s=obs_s)  # same seed -> same ground truth
    t = np.concatenate([b.t for b in entry["batches"]]) - 5.0
    idx = np.clip((t * run.track.fs).astype(int), 0, len(run.track) - 1)
    name = f"{sid}_{seed}.npz"
    np.savez_compressed(Path(out_dir) / name, t=t, acc=np.concatenate([b.acc for b in entry["batches"]]),
                        gyro=np.concatenate([b.gyro for b in entry["batches"]]),
                        slot_mask=np.concatenate([b.slot_mask for b in entry["batches"]]),
                        sat_mask=np.concatenate([b.sat_mask for b in entry["batches"]]), phase=run.track.phase[idx])
    return {"file": name, "scenario": sid, "seed": seed, "expected": run.expected, "report_only": run.report_only,
            "ref_time": run.ref_time, "params": run.params,
            "events": [{"t": round(tt, 3), "name": n} for tt, n, _ in run.events]}
