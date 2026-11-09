# File: realism/score.py
# Purpose: Build the realism scorecard: real vs emulated-sim features per category, gap
#          statistics, classifier two-sample tests, downstream (falldet) impact, ranked gaps.
# Contributors: Unassigned
"""Orchestration. The real side is cached (it never changes between simulator iterations);
the sim side is regenerated on every run from dev seeds."""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import emulate
from .features import (ACT_RELEVANCE, ACT_BASE, EVENT_FEATURES, activity_feature_names, activity_windows,
                       event_features, event_trace, find_event, trace_features)
from .mapping import ACTIVITIES, EVENTS
from .paths import FALLDET_ROOT, PROCESSED, SPECS
from .real import load_recordings
from .stats import c2st, feature_gap

REAL_CACHE_VERSION = "v2"  # bump when real-side features change
FALLDET_CACHE_VERSION = "v1"
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
FALLDET_FEATURES = ("impact_peak_g", "ff_min_g", "jerk_peak_g_s", "rot_peak_rad_s", "pre_std_g", "post_std_g",
                    "post_gyro_mean", "incl_post_pelvis", "incl_post_upper_thoracic", "n_anchors_lying",
                    "still_frac", "longest_still_s")


# --- plug-in feature providers ----------------------------------------------------------------
# A provider adds rows to the same per-feature gap table and ranking. Signature:
#     provider(context: dict) -> list[dict]
# context: {"out", "cache_dir", "jobs", "n_events", "n_act"}. Each row:
#     {"level": str, "category": str, "feature": str, "real": array, "real_groups": array,
#      "sim": array, "sim_groups": array, "relevance": 0..1, "confidence": "high"|"medium"|"low",
#      "description": str}
# Groups are subject ids (real) and seeds (sim). Register with @register_provider in a module
# listed in PROVIDER_MODULES (imported if it exists), e.g. realism/kinematics.py (voluntary-motion
# kinematics against open mocap, owned by the motionval work).
PROVIDERS: list = []
PROVIDER_MODULES = ("realism.kinematics",)


def register_provider(fn):
    PROVIDERS.append(fn)
    return fn


def provider_rows(context: dict) -> list[dict]:
    import importlib

    for name in PROVIDER_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    rows = []
    for fn in PROVIDERS:
        rows += fn(context)
    return rows


def score_provider_rows(rows: list[dict]) -> dict:
    """{level: {category: {feature: gap dict + relevance/confidence/description}}}"""
    out: dict = {}
    for j, r in enumerate(rows):
        g = feature_gap(np.asarray(r["real"], float), np.asarray(r["real_groups"]), np.asarray(r["sim"], float),
                        np.asarray(r["sim_groups"]), seed=j)
        g.update(relevance=float(r.get("relevance", 0.5)), confidence=r.get("confidence", "medium"),
                 description=r.get("description", ""))
        out.setdefault(r["level"], {}).setdefault(r["category"], {})[r["feature"]] = g
    return out


# --- falldet (read-only) ------------------------------------------------------------------------
def _falldet():
    if str(FALLDET_ROOT) not in sys.path:
        sys.path.insert(0, str(FALLDET_ROOT))
    from falldet.detector import detect, get_detector  # noqa: E402

    return detect, get_detector


def _detect(acc4: np.ndarray, gyro4: np.ndarray) -> dict:
    detect, get_detector = _falldet()
    out = detect((acc4, gyro4), detector=get_detector("target"), force_stage2=True)
    f = out.get("features") or {}
    return {"stage1": bool(out["stage1_triggered"]), "p_fall": float(out["probabilities"]["fall_confirmed"]),
            "state": out["state"], **{k: float(f[k]) for k in FALLDET_FEATURES if k in f}}


# --- real side ---------------------------------------------------------------------------------
def real_side(cache_dir: Path, use_falldet: bool) -> dict:
    out = _erciyes_vtt_side(cache_dir, use_falldet)
    out["events"].update(_univr_events(cache_dir))
    out["activities"].update(_univr_activities(cache_dir))
    return out


def _erciyes_vtt_side(cache_dir: Path, use_falldet: bool) -> dict:
    path = cache_dir / f"real_side_{REAL_CACHE_VERSION}.pkl"
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    recs = load_recordings(cache_dir / "real_recordings.pkl")
    out = {"events": {}, "activities": {}, "falldet": {}}
    for cat in EVENTS:
        if cat.dataset != "erciyes":
            continue
        rows = []
        for r in recs:
            if r.dataset != "erciyes" or r.code not in cat.real_codes:
                continue
            acc, gyro = r.acc.astype(float), r.gyro.astype(float)
            i0 = find_event(acc, r.fs)
            rows.append({"group": r.subject, "key": (r.subject, r.code, r.trial),
                         "f": event_features(acc, gyro, r.fs, i0), "trace": event_trace(acc, gyro, r.fs, i0)})
        out["events"][cat.name] = rows
    for cat in ACTIVITIES:
        if cat.dataset not in ("erciyes", "vtt"):
            continue
        X, groups = [], []
        for r in recs:
            if r.dataset != cat.dataset or r.code not in cat.real_codes:
                continue
            w = activity_windows(r.acc.astype(float), r.gyro.astype(float), r.fs)
            X += w
            groups += [r.subject] * len(w)
        out["activities"][cat.name] = (np.array(X), np.array(groups))
    fd_path = cache_dir / f"real_falldet_{FALLDET_CACHE_VERSION}.pkl"
    if use_falldet and fd_path.exists():
        with open(fd_path, "rb") as handle:
            out["falldet"] = pickle.load(handle)
    elif use_falldet:
        data = np.load(PROCESSED / "canonical_target.npz")
        acc_all, gyro_all = data["acc"], data["gyro"]
        codes = {c: cat.name for cat in EVENTS for c in cat.real_codes}
        with open(PROCESSED / "meta.csv") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] != "erciyes" or int(row["code"]) not in codes:
                    continue
                s, e = int(row["start"]), int(row["stop"])
                res = _detect(acc_all[s:e].astype(float), gyro_all[s:e].astype(float))
                out["falldet"].setdefault(codes[int(row["code"])], []).append({"group": row["subject"], **res})
        with open(fd_path, "wb") as handle:
            pickle.dump(out["falldet"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out


UNIVR_CACHE_VERSION = "v1"


def _univr_events(cache_dir: Path) -> dict:
    """UNIVRFall event rows per category; aligned on the |a| peak within -0.3..+0.5 s of the
    video-labelled impact. label_dur_s = labelled onset -> impact."""
    from . import real_univr

    def build():
        cats = [c for c in EVENTS if c.dataset == "univr"]
        events = real_univr.load_events({t for c in cats for t in c.real_codes})
        out = {}
        for cat in cats:
            rows = []
            for e in events:
                if e.task not in cat.real_codes:
                    continue
                acc, gyro = e.acc.astype(float), e.gyro.astype(float)
                i0 = find_event(acc, e.fs, (e.impact - 30) / e.fs, (e.impact + 50) / e.fs)
                f = event_features(acc, gyro, e.fs, i0)
                f["label_dur_s"] = (e.impact - e.onset) / e.fs
                rows.append({"group": e.subject, "key": (e.subject, e.task, e.trial), "f": f, "fs": e.fs,
                             "trace": event_trace(acc, gyro, e.fs, i0)})
            out[cat.name] = rows
        return out

    if not real_univr.UNIVR_ROOT.exists():
        return {}
    return real_univr.cached(f"real_univr_events_{UNIVR_CACHE_VERSION}.pkl", cache_dir, build)


def _univr_activities(cache_dir: Path) -> dict:
    from . import real_univr

    def build():
        X, groups = [], []
        for worker, acc, gyro in real_univr.load_site_windows():
            w = activity_windows(acc.astype(float), gyro.astype(float), 100.0)
            X += w
            groups += [worker] * len(w)
        return {c.name: (np.array(X), np.array(groups)) for c in ACTIVITIES if c.dataset == "univr"}

    if not any(c.dataset == "univr" for c in ACTIVITIES) or not real_univr.UNIVR_ROOT.exists():
        return {}
    return real_univr.cached(f"real_univr_site_{UNIVR_CACHE_VERSION}.pkl", cache_dir, build)


# --- sim side (worker processes) --------------------------------------------------------------
def _event_job(args):
    cat_name, seed, use_falldet = args
    cat = next(c for c in EVENTS if c.name == cat_name)
    run = emulate.event_run(cat, seed)
    if run is None:
        return None
    rng = np.random.default_rng(seed + 99)
    spec = SPECS[cat.dataset]
    acc, gyro = emulate.body_signals(run["track"], run["worker"], run["soft_fn"], run["soft_zeta"], cat.dataset)
    acc, gyro = emulate.sample_like(acc, gyro, run["track"].fs, spec, rng)
    t0, t1 = run["window"]
    i0 = find_event(acc, spec.fs, t0, t1)
    res = {"group": seed, "f": event_features(acc, gyro, spec.fs, i0), "trace": event_trace(acc, gyro, spec.fs, i0),
           "fs": spec.fs, "params": {k: v for k, v in run["params"].items() if isinstance(v, (int, float, str, bool))}}
    events = run.get("events") or []
    onset = next((t for t, n, _ in events if n == cat.onset_event), None)
    impact = next((t for t, n, _ in events if n == "impact" and onset is not None and t >= onset), None)
    res["f"]["label_dur_s"] = (impact - onset) if (onset is not None and impact is not None) else float("nan")
    if use_falldet and cat.dataset == "erciyes":
        acc4, gyro4 = emulate.falldet_like_erciyes(run["track"], run["worker"], run["soft_fn"], run["soft_zeta"], rng)
        j0 = i0 * 4
        lo, hi = max(0, j0 - 800), min(len(acc4), j0 + 900)
        res["falldet"] = _detect(acc4[lo:hi], gyro4[lo:hi])
    return cat_name, res


def _activity_job(args):
    cat_name, seed = args
    cat = next(c for c in ACTIVITIES if c.name == cat_name)
    spec = SPECS[cat.dataset]
    rng = np.random.default_rng(seed + 99)
    if cat.sim_activity == "shift":
        rows = []
        for run in emulate.shift_run(seed):
            acc, gyro = emulate.body_signals(run["track"], run["worker"], run["soft_fn"], run["soft_zeta"], cat.dataset)
            acc, gyro = emulate.sample_like(acc, gyro, run["track"].fs, spec, rng)
            t0, t1 = run["window"]
            rows += activity_windows(acc[int(t0 * spec.fs):int(t1 * spec.fs)], gyro[int(t0 * spec.fs):int(t1 * spec.fs)],
                                     spec.fs, max_windows=12)
        return cat_name, seed, rows
    run = emulate.activity_run(cat.sim_activity, seed, cat.dataset)
    acc, gyro = emulate.body_signals(run["track"], run["worker"], run["soft_fn"], run["soft_zeta"], cat.dataset)
    acc, gyro = emulate.sample_like(acc, gyro, run["track"].fs, spec, rng)
    t0, t1 = run["window"]
    a, b = int(t0 * spec.fs), int(t1 * spec.fs)
    return cat_name, seed, activity_windows(acc[a:b], gyro[a:b], spec.fs)


def apply_overrides(overrides: dict) -> None:
    """Set module constants, e.g. {"thundersim.falls.PIVOT_RESIST": 0.0}, in this process (used as a
    worker initializer so A/B runs can score old constants on the current code)."""
    import importlib

    for key, value in (overrides or {}).items():
        module, name = key.rsplit(".", 1)
        mod = importlib.import_module(module)
        if not hasattr(mod, name):
            raise AttributeError(f"{module} has no {name}")
        setattr(mod, name, value)


def sim_side(n_events: int, n_act: int, jobs: int, use_falldet: bool, only: set | None = None,
             overrides: dict | None = None, seed_offset: int = 0) -> dict:
    out = {"events": {c.name: [] for c in EVENTS}, "activities": {c.name: ([], []) for c in ACTIVITIES}}
    tasks = []
    for block, cat in enumerate(EVENTS):
        if only and cat.name not in only:
            continue
        tries = n_events * (3 if cat.scenario == "STRUCK-KD" else 1)
        tasks += [(cat.name, emulate.dev_seed(block, seed_offset + i), use_falldet) for i in range(tries)]
    act_tasks = []
    for block, cat in enumerate(ACTIVITIES):
        if only and cat.name not in only:
            continue
        act_tasks += [(cat.name, emulate.dev_seed(100 + block, seed_offset + i)) for i in range(n_act)]
    with ProcessPoolExecutor(max_workers=jobs, initializer=apply_overrides, initargs=(overrides or {},)) as pool:
        for item in pool.map(_event_job, tasks, chunksize=2):
            if item is None:
                continue
            name, res = item
            if len(out["events"][name]) < n_events:
                out["events"][name].append(res)
        for name, seed, rows in pool.map(_activity_job, act_tasks, chunksize=2):
            X, g = out["activities"][name]
            X += rows
            g += [seed] * len(rows)
    out["activities"] = {k: (np.array(X), np.array(g)) for k, (X, g) in out["activities"].items()}
    return out


# --- scoring -----------------------------------------------------------------------------------
def _rate_ci(values: np.ndarray, groups: np.ndarray, rng, reps: int = 200):
    parts = [values[groups == g] for g in np.unique(groups)]
    boots = [np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))]).mean() for _ in range(reps)]
    return float(values.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def score(real: dict, sim: dict, only: set | None = None) -> dict:
    card = {"events": {}, "activities": {}, "downstream": {}, "c2st": {}}
    names = list(EVENT_FEATURES)
    for cat in EVENTS:
        if only and cat.name not in only:
            continue
        r_rows, s_rows = real["events"].get(cat.name, []), sim["events"].get(cat.name, [])
        if not r_rows or len(s_rows) < 5:
            continue
        for row in r_rows + s_rows:
            row["f"] = {**trace_features(row["trace"], row.get("fs", 25.0)), **row["f"]}
        names = list(EVENT_FEATURES) + (["label_dur_s"] if cat.dataset == "univr" else [])
        get = lambda row, n: float("nan") if n in cat.drop else row["f"].get(n, float("nan"))
        Xr = np.array([[get(row, n) for n in names] for row in r_rows])
        gr = np.array([row["group"] for row in r_rows])
        Xs = np.array([[get(row, n) for n in names] for row in s_rows])
        gs = np.array([row["group"] for row in s_rows])
        keep = [j for j, n in enumerate(names) if n not in cat.drop]
        card["events"][cat.name] = {names[j]: feature_gap(Xr[:, j], gr, Xs[:, j], gs, seed=j) for j in keep}
        card["c2st"][cat.name] = c2st(Xr[:, keep], gr, Xs[:, keep], gs, [names[j] for j in keep])
        rf = real.get("falldet", {}).get(cat.name, [])
        sf = [row["falldet"] for row in s_rows if "falldet" in row]
        if rf and sf:
            rng = np.random.default_rng(1)
            rgr = np.array([row["group"] for row in rf])
            sgr = np.array([row["group"] for row in s_rows if "falldet" in row])
            d = {}
            for key in ("stage1", "p_fall"):
                rv = np.array([float(row[key]) for row in rf])
                sv = np.array([float(row[key]) for row in sf])
                d[key] = {"real": _rate_ci(rv, rgr, rng), "sim": _rate_ci(sv, sgr, rng)}
            d["state_fall_confirmed"] = {"real": float(np.mean([row["state"] == "fall_confirmed" for row in rf])),
                                         "sim": float(np.mean([row["state"] == "fall_confirmed" for row in sf]))}
            d["features"] = {k: feature_gap(np.array([row.get(k, np.nan) for row in rf]), rgr,
                                            np.array([row.get(k, np.nan) for row in sf]), sgr, seed=7)
                             for k in FALLDET_FEATURES}
            card["downstream"][cat.name] = d
    act_names = activity_feature_names()
    for cat in ACTIVITIES:
        if only and cat.name not in only:
            continue
        if cat.name not in real["activities"] or cat.name not in sim["activities"]:
            continue
        Xr, gr = real["activities"][cat.name]
        Xs, gs = sim["activities"][cat.name]
        if cat.active_only and len(Xr) and len(Xs):
            mr, ms = active_mask(Xr), active_mask(Xs)
            Xr, gr, Xs, gs = Xr[mr], gr[mr], Xs[ms], gs[ms]
        if len(Xr) < 10 or len(Xs) < 10:
            continue
        keep = [j for j, n in enumerate(act_names) if n not in cat.drop]
        card["activities"][cat.name] = {act_names[j]: feature_gap(Xr[:, j], gr, Xs[:, j], gs, seed=j) for j in keep}
        card["c2st"][cat.name] = c2st(Xr[:, keep], gr, Xs[:, keep], gs, [act_names[j] for j in keep])
    card["ranked"] = rank(card)
    return card


ACTIVE_GYRO_RAD_S = 0.15   # a window is quiet if pelvis |w| mean < this AND |a| std < ACTIVE_ACC_STD_G [design]
ACTIVE_ACC_STD_G = 0.03


def active_mask(X: np.ndarray) -> np.ndarray:
    names = activity_feature_names()
    return (X[:, names.index("gmag_mean_w")] >= ACTIVE_GYRO_RAD_S) | (X[:, names.index("amag_std_w")] >= ACTIVE_ACC_STD_G)


def _act_relevance(feature: str) -> float:
    base = feature.rsplit("_", 1)[0] if feature != "incl_diff" else "incl_diff"
    return ACT_RELEVANCE.get(base, 0.5)


def rank(card: dict) -> list[dict]:
    rows = []
    conf = {c.name: c.confidence for c in EVENTS + ACTIVITIES}
    protocol = {c.name: c.protocol for c in EVENTS}
    for level in ("events", "activities"):
        for cat, feats in card[level].items():
            for feat, g in feats.items():
                if not np.isfinite(g.get("nW", np.nan)) or not np.isfinite(g.get("excess", np.nan)):
                    continue
                relevance = EVENT_FEATURES.get(feat, ("", 1.0))[1] if level == "events" else _act_relevance(feat)
                if feat in protocol.get(cat, ()):
                    relevance = 0.0  # set by the real recording protocol, not by the simulator
                weight = (1.0 if level == "events" else 0.6) * relevance * CONFIDENCE_WEIGHT[conf[cat]]
                rows.append({"level": level, "category": cat, "feature": feat, "nW": g["nW"], "ci": g["ci"],
                             "null95": g["null95"], "excess": g["excess"], "impact": g["excess"] * weight,
                             "real_med": g["real_med"], "sim_med": g["sim_med"], "outside": g["ci"][0] > g["null95"], "protocol": feat in protocol.get(cat, ())})
    for level, cats in card.get("providers", {}).items():
        for cat, feats in cats.items():
            for feat, g in feats.items():
                if not np.isfinite(g.get("nW", np.nan)) or not np.isfinite(g.get("excess", np.nan)):
                    continue
                weight = g["relevance"] * CONFIDENCE_WEIGHT.get(g["confidence"], 0.7)
                rows.append({"level": level, "category": cat, "feature": feat, "nW": g["nW"], "ci": g["ci"],
                             "null95": g["null95"], "excess": g["excess"], "impact": g["excess"] * weight,
                             "real_med": g["real_med"], "sim_med": g["sim_med"], "outside": g["ci"][0] > g["null95"],
                             "protocol": False})
    rows.sort(key=lambda r: -r["impact"])
    return rows


# --- output ------------------------------------------------------------------------------------
def _fmt(x, nd=2):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def tables(card: dict, top: int = 30) -> str:
    lines = ["## Classifier two-sample tests (AUC 0.5 = indistinguishable)", "",
             "| category | n real | n sim | AUC | 95 % CI | most separating features (univariate AUC) |", "|---|---|---|---|---|---|"]
    for cat, c in card["c2st"].items():
        if not np.isfinite(c.get("auc", np.nan)):
            continue
        tops = ", ".join(f"{n} {v:.2f}" for n, v in c["top_univariate"][:3])
        lines.append(f"| {cat} | {c['n_real']} | {c['n_sim']} | {c['auc']:.2f} | {c['ci'][0]:.2f}-{c['ci'][1]:.2f} | {tops} |")
    lines += ["", f"## Ranked gaps (top {top} by impact = excess x downstream relevance x mapping confidence)", "",
              "| # | level | category | feature | real median | sim median | nW | nW 95 % CI | real null95 | excess | impact |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(card["ranked"][:top], 1):
        lines.append(f"| {i} | {r['level'][:3]} | {r['category']} | {r['feature']} | {_fmt(r['real_med'])} | {_fmt(r['sim_med'])} | "
                     f"{_fmt(r['nW'])} | {_fmt(r['ci'][0])}-{_fmt(r['ci'][1])} | {_fmt(r['null95'])} | {_fmt(r['excess'], 1)} | {_fmt(r['impact'], 1)} |")
    lines += ["", "## Downstream: falldet (target model) on real vs emulated-sim events", "",
              "| category | Stage-1 rate real | Stage-1 rate sim | P(fall) real | P(fall) sim | confirmed real | confirmed sim | largest input gaps (nW) |",
              "|---|---|---|---|---|---|---|---|"]
    for cat, d in card["downstream"].items():
        fg = sorted(((k, v["nW"]) for k, v in d["features"].items() if np.isfinite(v.get("nW", np.nan))), key=lambda kv: -kv[1])[:3]
        s1r, s1s = d["stage1"]["real"], d["stage1"]["sim"]
        pr, ps = d["p_fall"]["real"], d["p_fall"]["sim"]
        lines.append(f"| {cat} | {s1r[0]:.2f} ({s1r[1][0]:.2f}-{s1r[1][1]:.2f}) | {s1s[0]:.2f} ({s1s[1][0]:.2f}-{s1s[1][1]:.2f}) | "
                     f"{pr[0]:.2f} | {ps[0]:.2f} | {d['state_fall_confirmed']['real']:.2f} | {d['state_fall_confirmed']['sim']:.2f} | "
                     + ", ".join(f"{k} {v:.1f}" for k, v in fg) + " |")
    lines += ["", "## Event features per category (nW; * = outside real-data noise, i.e. CI low > null95)", ""]
    for cat, feats in card["events"].items():
        lines.append(f"**{cat}**: " + ", ".join(
            f"{n} {_fmt(g.get('nW'))}{'*' if g.get('ci') and g['ci'][0] > g['null95'] else ''} "
            f"(r {_fmt(g.get('real_med'))} / s {_fmt(g.get('sim_med'))})" for n, g in feats.items()))
        lines.append("")
    return "\n".join(lines)


def plots(real: dict, sim: dict, card: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = [c.name for c in EVENTS if c.name in card["events"]]
    fig, axes = plt.subplots(len(cats), 3, figsize=(14, 2.6 * len(cats)), squeeze=False)
    for i, cat in enumerate(cats):
        for side, rows, colour in (("real", real["events"][cat], "C0"), ("sim", sim["events"][cat], "C3")):
            fs = rows[0].get("fs", 25.0)
            tr = np.array([r["trace"] for r in rows])
            t = (np.arange(tr.shape[1]) - int(3 * fs)) / fs
            for j, label in enumerate(("waist |a| g", "chest |a| g", "chest |w| rad/s")):
                ax = axes[i, j]
                med = np.nanmedian(tr[:, :, j], axis=0)
                lo, hi = np.nanpercentile(tr[:, :, j], [25, 75], axis=0)
                ax.plot(t, med, colour, label=side)
                ax.fill_between(t, lo, hi, color=colour, alpha=0.2)
                ax.set_title(f"{cat}: {label}", fontsize=9)
        axes[i, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "event_traces.png", dpi=90)
    plt.close(fig)
    top = [r for r in card["ranked"] if r["level"] == "events"][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, r in zip(axes.flat, top):
        rv = np.array([row["f"][r["feature"]] for row in real["events"][r["category"]]])
        sv = np.array([row["f"][r["feature"]] for row in sim["events"][r["category"]]])
        both = np.concatenate([rv[np.isfinite(rv)], sv[np.isfinite(sv)]])
        bins = np.linspace(np.percentile(both, 1), np.percentile(both, 99), 25)
        ax.hist(rv[np.isfinite(rv)], bins, density=True, alpha=0.5, label="real")
        ax.hist(sv[np.isfinite(sv)], bins, density=True, alpha=0.5, color="C3", label="sim")
        ax.set_title(f"{r['category']}: {r['feature']} (nW {r['nW']:.1f})", fontsize=9)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "top_event_gaps.png", dpi=90)
    plt.close(fig)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_score(out: Path, n_events: int = 40, n_act: int = 20, jobs: int = 4, use_falldet: bool = True,
              only: set | None = None, cache_dir: Path | None = None, overrides: dict | None = None,
              seed_offset: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or out.parent / "cache"
    t0 = time.time()
    real = real_side(cache_dir, use_falldet)
    t1 = time.time()
    sim = sim_side(n_events, n_act, jobs, use_falldet, only, overrides, seed_offset)
    t2 = time.time()
    card = score(real, sim, only)
    t3 = time.time()
    extra = provider_rows({"out": out, "cache_dir": cache_dir, "jobs": jobs, "n_events": n_events, "n_act": n_act})
    if extra:
        card["providers"] = score_provider_rows(extra)
        card["ranked"] = rank(card)
        with open(out / "provider_rows.pkl", "wb") as handle:
            pickle.dump(extra, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t4 = time.time()
    card["meta"] = {"n_events": n_events, "n_act": n_act, "real_s": round(t1 - t0, 1), "sim_s": round(t2 - t1, 1),
                    "score_s": round(t3 - t2, 1), "providers_s": round(t4 - t3, 1), "total_s": round(t4 - t0, 1),
                    "providers": [getattr(fn, "__name__", str(fn)) for fn in PROVIDERS],
                    "seed_base": emulate.SEED_BASE, "seed_offset": seed_offset, "overrides": {k: repr(v) for k, v in (overrides or {}).items()},
                    "git_note": "dev seeds only"}
    (out / "scorecard.json").write_text(json.dumps(_jsonable(card), indent=1))
    (out / "scorecard.md").write_text(tables(card))
    with open(out / "sim_side.pkl", "wb") as handle:
        pickle.dump(sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
    plots(real, sim, card, out)
    return card


def compare(before: Path, after: Path, top: int = 25) -> str:
    """Before/after table for the changelog (keyed by category and feature)."""
    a = json.loads((before / "scorecard.json").read_text())
    b = json.loads((after / "scorecard.json").read_text())
    lines = ["| category | feature | nW before | nW after | null95 | CI after | change |", "|---|---|---|---|---|---|---|"]
    keyed = {(r["category"], r["feature"]): r for r in b["ranked"]}
    rows = []
    for r in a["ranked"][:top]:
        s = keyed.get((r["category"], r["feature"]))
        if s:
            rows.append((r, s))
    for r, s in rows:
        flag = "shrank" if s["nW"] < r["ci"][0] else ("grew" if s["nW"] > r["ci"][1] else "within CI")
        lines.append(f"| {r['category']} | {r['feature']} | {r['nW']:.2f} | {s['nW']:.2f} | {s['null95']:.2f} | "
                     f"{s['ci'][0]:.2f}-{s['ci'][1]:.2f} | {flag} |")
    grew = []
    for r in a["ranked"]:
        k = (r["category"], r["feature"])
        s = keyed.get(k)
        if s and s["nW"] > r["ci"][1] and s["ci"][0] > s["null95"]:
            grew.append((k, r, s))
    lines += ["", "Gaps that grew beyond their before-CI and are outside real-data noise after:"]
    for k, r, s in grew:
        lines.append(f"- {k[0]} / {k[1]}: {r['nW']:.2f} -> {s['nW']:.2f} (before CI {r['ci'][0]:.2f}-{r['ci'][1]:.2f})")
    ca, cb = a["c2st"], b["c2st"]
    lines += ["", "| category | AUC before | AUC after |", "|---|---|---|"]
    for k in ca:
        if k in cb and ca[k].get("auc") is not None and cb[k].get("auc") is not None:
            lines.append(f"| {k} | {ca[k]['auc']:.2f} | {cb[k]['auc']:.2f} |")
    return "\n".join(lines)


def rescore(out: Path, cache_dir: Path | None = None) -> dict:
    """Recompute the scorecard from a saved sim_side.pkl (after adding features or changing stats)."""
    cache_dir = cache_dir or out.parent / "cache"
    real = real_side(cache_dir, True)
    with open(out / "sim_side.pkl", "rb") as handle:
        sim = pickle.load(handle)
    card = score(real, sim)
    if (out / "provider_rows.pkl").exists():
        with open(out / "provider_rows.pkl", "rb") as handle:
            card["providers"] = score_provider_rows(pickle.load(handle))
        card["ranked"] = rank(card)
    old = json.loads((out / "scorecard.json").read_text()).get("meta", {}) if (out / "scorecard.json").exists() else {}
    card["meta"] = {**old, "rescored": True}
    (out / "scorecard.json").write_text(json.dumps(_jsonable(card), indent=1))
    (out / "scorecard.md").write_text(tables(card))
    plots(real, sim, card, out)
    return card
