# File: realism/kinematics.py
# Purpose: Scorecard provider for voluntary-motion kinematics: runs the motionval metrics
#          (thundersim/motion_validation.py) on simulator runs and on the open mocap clips, and
#          returns them as scorecard rows, so `python -m realism score` reports trunk kinematics
#          next to the worn-IMU features.
# Contributors: Unassigned
"""Voluntary-motion kinematics in the realism scorecard.

What it compares: the same per-activity metrics that `thundersim motionval` reports (gait, lifts,
kneeling, ladder, quiet standing), simulator against the canonical class A mocap clips in
motionlib/canonical. The literature ranges motionval cites are carried along as reference ranges.

Row format (realism/score.py, register_provider): each row is one (activity, metric):
  real         one value per mocap clip (lifts: one per bend), the median of that clip's windows
  real_groups  the clip's performer (cmu_07, accad_female1, niosh_s3, ...), so the bootstrap and the
               split-half noise floor resample people, as the IMU rows resample subjects
  sim          one value per simulator run (lifts: one per bend)
  sim_groups   the run ("A-CARRY/3")
The scorecard turns these into nW, CI, null95 and excess like every other row.

Adapted here because score.py has no slot for them (noted in eval/GAPS.md):
  - reference ranges: score.py only scores two samples. A metric with fewer than 5 mocap clips (or a
    literature range only) gets nW = nan there and is left out of the ranking. The motionval verdict
    (sim median against the mocap 10-90 % spread, else the literature range) and the distance outside
    that range are therefore put in each row's description, and the full table, with ranges, is
    written by this module to <out>/kinematics.md and <out>/kinematics.json.
  - score.py's markdown shows provider rows only inside the overall ranked table, so the same
    kinematics.md also lists every kinematics row.

Standalone (fast, no real IMU data needed):
    .venv/bin/python -m realism.kinematics --out <dir> [--seeds 10] [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thundersim import motion_validation as mv

from .score import register_provider, score_provider_rows

LEVEL = "kinematics"
SEEDS = 10              # simulator runs per scenario, as `thundersim motionval` defaults
SEED_BASE = 0           # motionval's default dev seeds 0-9 (never >= 1000000)
MIN_REAL = 5            # stats.feature_gap needs at least 5 values on each side

# Downstream relevance 0..1 per metric, the same meaning as in score.py's ranking: how much the
# metric feeds what the detectors and posture rules read. [design]
RELEVANCE = {
    "trunk_rate_p95_rads": 0.8,        # falldet near-fall context (pre_rot_p95 reads trunk rotation rate)
    "imu_step_peak_g": 0.7, "imu_step_peak_10hz_g": 0.7, "imu_vert_rms_g": 0.7, "imu_peak_norm_g": 0.7,
    "imu_share_above_5hz": 0.6, "imu_share_above_10hz": 0.6,
    "pelvis_yaw_pp_deg": 0.6, "pelvis_roll_pp_deg": 0.6, "pelvis_pitch_pp_deg": 0.6,
    "thorax_pelvis_yaw_pp_deg": 0.6,   # posture twist_hold reads thorax-pelvis twist
    "pelvis_pitch_mean_deg": 0.6, "upper_incl_mean_deg": 0.6,
    "peak_upper_deg": 0.7, "peak_thoracic_deg": 0.7, "spine_flex_at_peak_deg": 0.7,   # severe_flexion
    "peak_rate_thoracic_dps": 0.6, "hold_s": 0.6,
    "speed_mps": 0.3, "step_hz": 0.4, "step_length_m": 0.3, "bob_cm": 0.4, "ml_cm": 0.4,
    "rung_hz": 0.3, "climb_speed_mps": 0.3, "rise_per_rung_m": 0.3,
}
DEFAULT_RELEVANCE = 0.5   # [design]


def performer(clip: str) -> str:
    """Performer id from a canonical clip name: clips of one person are one bootstrap group."""
    parts = clip.split("_")
    if parts[0] == "cmu" and len(parts) >= 2:
        return f"cmu_{parts[1]}"                 # cmu_<subject>_<trial>
    if parts[0] in ("accad", "niosh", "cml") and len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"          # accad_female1_*, niosh_s3_t1, cml_27movingobject_*
    if parts[0] == "100style":
        return "100style"                        # 100STYLE is one performer
    return clip


def clip_values(rows: list[dict], activity: str, source: str, metric: str) -> tuple[np.ndarray, list[str]]:
    """Values and their clip ids, one per clip (the median of its windows), as motionval's
    per_clip_values; lifts keep one value per bend."""
    by_clip: dict[str, list[float]] = {}
    for r in rows:
        value = r.get(metric)
        if r["_activity"] != activity or r["_source"] != source or value is None or not np.isfinite(value):
            continue
        key = f"{r['_clip']}#{len(by_clip)}" if activity == "lift" else r["_clip"]
        by_clip.setdefault(key, []).append(float(value))
    clips = [key.split("#")[0] for key in by_clip]
    return np.array([np.median(v) for v in by_clip.values()], float), clips


def reference_band(entry: dict) -> tuple[float, float, str] | None:
    """The range motionval's verdict judges against: the vest or mocap 10-90 % spread, else the
    literature range."""
    for source in ("vest", "mocap"):
        if source in entry:
            return entry[source]["p10"], entry[source]["p90"], f"{source} p10-p90"
    if "lit" in entry:
        return entry["lit"]["low"], entry["lit"]["high"], "literature"
    return None


def range_distance(value: float, low: float, high: float) -> float:
    """How far a value lies outside [low, high], in widths of the range; 0 where motionval's verdict
    says ok (it pads the range by 5 % of its largest end). The width is floored at twice that pad, so a
    one-clip "range" (low == high) does not blow the distance up."""
    pad = 0.05 * max(abs(low), abs(high), 1e-6)
    width = max(high - low, 2 * pad)
    if value < low - pad:
        return (low - pad - value) / width
    if value > high + pad:
        return (value - high - pad) / width
    return 0.0


def build_rows(metric_rows: list[dict]) -> list[dict]:
    """Scorecard rows from motionval metric rows (mv.collect output)."""
    table = mv.summarise(metric_rows)
    rows = []
    for activity in mv.ACTIVITIES:
        for metric in mv.METRICS[activity]:
            entry = table.get((activity, metric))
            if entry is None or "sim" not in entry:
                continue
            real, real_clips = clip_values(metric_rows, activity, "mocap", metric)
            sim, sim_clips = clip_values(metric_rows, activity, "sim", metric)
            band = reference_band(entry)
            caveat = mv.REFERENCE_CAVEATS.get((activity, metric))
            distance = range_distance(entry["sim"]["median"], band[0], band[1]) if band else float("nan")
            words = [f"motionval {entry['verdict']}"]
            if band:
                words.append(f"sim median {entry['sim']['median']:.3g} vs {band[2]} {band[0]:.3g}-{band[1]:.3g}"
                             f" ({distance:.2f} range widths outside)")
            if "lit" in entry:
                words.append(f"literature {entry['lit']['low']:g}-{entry['lit']['high']:g}")
            if len(real) < MIN_REAL:
                words.append(f"only {len(real)} mocap clips: range check only")
            if caveat:
                words.append(f"not judged: {caveat}")
            confidence = "low" if metric.startswith("imu_") else "medium"   # mocap IMU = S1 path differentiated twice
            rows.append({
                "level": LEVEL, "category": activity, "feature": metric,
                "real": real, "real_groups": np.array([performer(c) for c in real_clips]),
                "sim": sim, "sim_groups": np.array(sim_clips),
                # Not-judged rows stay in the table with no ranking weight, as score.py does for protocol features.
                "relevance": 0.0 if caveat else RELEVANCE.get(metric, DEFAULT_RELEVANCE),
                "confidence": confidence,
                "description": "; ".join(words),
                # Extra keys (score.py ignores them; kept in provider_rows.pkl and kinematics.json).
                "verdict": entry["verdict"], "range_distance": distance,
                "reference": {key: entry[key] for key in ("mocap", "vest", "lit") if key in entry},
                "sim_summary": entry["sim"],
            })
    return rows


def collect_rows(seeds: int = SEEDS, seed_base: int = SEED_BASE, jobs: int = 4) -> list[dict]:
    """Run motionval's simulator scenarios and mocap windows; return scorecard rows."""
    assert seed_base + seeds <= 1_000_000, "held-out seeds are never used here"
    metric_rows = mv.collect(mv.DEFAULT_SCENARIOS, seeds, seed_base, jobs=jobs)
    return build_rows(metric_rows)


def _fmt(x, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{nd}f}"


def _stats(s: dict | None) -> str:
    if not s:
        return "-"
    return f"{s['median']:.3g} ({s['p10']:.3g}-{s['p90']:.3g}) n={s['n']}"


def _clean(x):
    """JSON-safe copy: numpy scalars to Python, nan to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_tables(rows: list[dict], gaps: dict, out: Path, seeds: int = SEEDS, seed_base: int = SEED_BASE) -> Path:
    """kinematics.md and kinematics.json in out: every kinematics row with its gap and reference range."""
    out.mkdir(parents=True, exist_ok=True)
    cards = gaps.get(LEVEL, {})
    lines = ["# Voluntary-motion kinematics (realism provider)", "",
             f"Simulator: {', '.join(mv.DEFAULT_SCENARIOS)}, seeds {seed_base}-{seed_base + seeds - 1}, target profile. "
             "Real: canonical class A mocap clips (motionlib/canonical), grouped by performer. Values are median "
             "(10th-90th percentile) over clips / runs. nW, CI and null95 as in stats.py (nW needs >= 5 clips per side). "
             "'outside' = range widths the sim median lies outside the motionval reference range. "
             "Simulated and recorded data only: no medical claim.", "",
             "| activity | metric | sim | mocap | literature | verdict | outside | nW | nW 95 % CI | null95 | excess |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def gap_of(r: dict) -> dict:
        return cards.get(r["category"], {}).get(r["feature"], {})

    def size(r: dict) -> tuple[bool, float, float]:
        # Judged rows first; largest excess first, then rows with a range check only, by distance outside it.
        excess = gap_of(r).get("excess", float("nan"))
        distance = r["range_distance"]
        return (r["verdict"] == "not judged", -(excess if np.isfinite(excess) else 0.0),
                -(distance if np.isfinite(distance) else 0.0))

    plain = []
    for r in sorted(rows, key=size):
        g = gap_of(r)
        ref = r["reference"]
        lit = f"{ref['lit']['low']:g}-{ref['lit']['high']:g}" if "lit" in ref else "-"
        ci = g.get("ci")
        lines.append(f"| {r['category']} | {r['feature']} | {_stats(r['sim_summary'])} | {_stats(ref.get('mocap'))} | {lit} | "
                     f"{r['verdict']} | {_fmt(r['range_distance'])} | {_fmt(g.get('nW'))} | "
                     f"{(_fmt(ci[0]) + '-' + _fmt(ci[1])) if ci else '-'} | {_fmt(g.get('null95'))} | {_fmt(g.get('excess'), 1)} |")
        plain.append({"activity": r["category"], "metric": r["feature"], "verdict": r["verdict"],
                      "range_distance": r["range_distance"], "relevance": r["relevance"], "confidence": r["confidence"],
                      "sim": r["sim_summary"], "reference": ref, "gap": g, "description": r["description"]})
    path = out / "kinematics.md"
    path.write_text("\n".join(lines) + "\n")
    (out / "kinematics.json").write_text(json.dumps(_clean(plain), indent=1))
    return path


@register_provider
def kinematics_provider(context: dict) -> list[dict]:
    """Scorecard hook: rows for every motionval metric, plus kinematics.md/json in the run's out dir."""
    if not mv.MOCAP_DIR.exists():
        print("kinematics provider: no mocap clips in", mv.MOCAP_DIR, "- skipped")
        return []
    rows = collect_rows(jobs=int(context.get("jobs", 4)))
    out = context.get("out")
    if out is not None:
        write_tables(rows, score_provider_rows(rows), Path(out))
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="realism.kinematics", description="kinematics rows only (no IMU scorecard)")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args(argv)
    rows = collect_rows(args.seeds, args.seed_base, args.jobs)
    path = write_tables(rows, score_provider_rows(rows), Path(args.out), args.seeds, args.seed_base)
    print("kinematics table:", path)


if __name__ == "__main__":
    main()
