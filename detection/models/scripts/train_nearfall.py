"""Train and choose the near-fall v2 classifier + trigger settings with grouped cross-validation.

    .venv/bin/python -I scripts/train_nearfall.py --tables data/interim/nf_public.csv \
        data/interim/nf_sim_dev.csv data/interim/nf_shift_dev.csv --out models/nearfall_v2.json

Groups (leave-one-group-out): public subject (LOSO, as before), simulator dev seed % 5,
shift recording. Every prediction is made by a model that never saw that group.
All simulator rows are ThunderSim dev seeds (0-19 scenarios, 500000-500014 shifts).

Recording-level scoring at each setting:
  public  stumble recall (808 recordings with >= 1 positive), near-fall flags per hour on
          Erciyes ADL + VTT normal recordings
  sim     NF-STUMBLE / NF-RAIL / STRUCK-NF run pass rate (>= 1 positive candidate in the run),
          A-* runs with a positive, flags per hour on dev shifts
Selection: largest worst-case sim NF pass rate (then stumble recall) subject to
  dev-shift flags <= MAX_SHIFT_FPH and public ADL flags <= MAX_PUBLIC_FPH.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from falldet.nearfall import DEFAULT_NF_CONFIG, NF_FEATURE_NAMES  # noqa: E402

MAX_SHIFT_FPH = 0.5      # leaves room for the fall path's own near_fall flags (bar: <= 1 / h)
MAX_PUBLIC_FPH = 1.5     # before: 1.28 / h (fall path only)
NF_SCEN = ("NF-STUMBLE", "NF-RAIL", "STRUCK-NF")
GRID = dict(dip_g=[0.85, 0.8, 0.75], jolt_g=[1.8, 2.2, 2.6], rot_rad_s=[1.2, 1.5, 2.0])
P_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
FEATURE_SETS = {
    "all": list(NF_FEATURE_NAMES),
    "no_yaw_roll": [n for n in NF_FEATURE_NAMES if n not in ("yaw_rate_peak", "roll_rate_peak")],
}


def load(tables):
    d = pd.concat([pd.read_csv(t) for t in tables], ignore_index=True)
    def group(r):
        if r.source == "public":
            return r.subject
        if r.source == "sim":
            return f"simfold{int(r.subject[3:]) % 5}"
        return r.rec
    d["group"] = d.apply(group, axis=1)
    return d


def trig_mask(d, g):
    return (((d.dip_min_g < g["dip_g"]) | (d.peak_g > g["jolt_g"]) | (d.ut_peak_g > g["jolt_g"]))
            & (d.rot_peak > g["rot_rad_s"])).to_numpy()


FINAL_TRIGGER = {"dip_g": 0.75, "jolt_g": 1.8, "rot_rad_s": 1.5}
FINAL_P_GBT = 0.6
FINAL_P_LR = 0.9


def make_model(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if kind == "gbt":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.1,
                                              class_weight="balanced", random_state=0)
    raise ValueError(kind)


def cv_probs(d, cand, feats, model="lr"):
    """Leave-one-group-out p(near_fall) for candidate rows `cand` (bool mask)."""
    p = np.full(len(d), np.nan)
    train_lab = d.label.isin(["nf", "neg"]).to_numpy() & cand
    X = d[feats].to_numpy(float)
    y = (d.label == "nf").to_numpy()
    for g in d.group[cand].unique():
        te = cand & (d.group == g).to_numpy()
        tr = train_lab & (d.group != g).to_numpy()
        if model == "gbt":
            p[te] = make_model("gbt").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def score(d, pos, recs):
    """pos: bool per candidate row (emitted near_fall). recs: per-recording table."""
    hit = d[pos].groupby("rec").size()
    r = recs.copy()
    r["nflags"] = r.rec.map(hit).fillna(0)
    out = {}
    pub808 = r[(r.source == "public") & (r.code == "808")]
    out["public_stumble_recall"] = float((pub808.nflags > 0).mean())
    pubneg = r[(r.source == "public") & (r.kind == "neg")]
    out["public_flags_per_h"] = float(pubneg.nflags.sum() / pubneg.hours.sum())
    for s in NF_SCEN:
        x = r[r.code == s]
        out[f"sim_{s}"] = float((x.nflags > 0).mean()) if len(x) else float("nan")
    a = r[(r.source == "sim") & (r.kind == "neg")]
    out["sim_adl_runs_flagged"] = float((a.nflags > 0).mean())
    sh = r[r.source == "shift"]
    out["shift_flags_per_h"] = float(sh.nflags.sum() / sh.hours.sum()) if len(sh) else float("nan")
    out["sim_nf_min"] = min(out[f"sim_{s}"] for s in NF_SCEN)
    return out


def recordings(d):
    r = d.groupby("rec").agg(source=("source", "first"), code=("code", "first"), hours=("hours", "first"),
                             labels=("label", lambda s: set(s))).reset_index()
    r["code"] = r.code.astype(str)
    r["kind"] = ["nf" if ("nf" in L or "none_nf" in L) else ("fall" if "fall" in L else "neg") for L in r.labels]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=os.path.join(ROOT, "reports/nearfall_v2_cv.json"))
    a = ap.parse_args()
    d = load(a.tables)
    recs = recordings(d)
    isc = d.label.isin(["nf", "neg", "fall", "ignore"]).to_numpy()   # real candidate rows
    results = []
    for fs_name, feats in FEATURE_SETS.items():
        for vals in itertools.product(*GRID.values()):
            g = dict(zip(GRID, vals))
            cand = isc & trig_mask(d, g)
            p = cv_probs(d, cand, feats)
            for thr in P_GRID:
                pos = cand & (p >= thr) & (d.incl_post_max < DEFAULT_NF_CONFIG["upright_deg"]).to_numpy()
                s = score(d, pos, recs)
                results.append({"features": fs_name, **g, "p_threshold": thr, **s})
        print("done", fs_name, file=sys.stderr)
    R = pd.DataFrame(results)
    R.to_csv(a.report.replace(".json", "_grid.csv"), index=False)
    # ---- final choice (see README "near-fall v2"): the LR grid showed that every setting meeting
    # the simulator bars trades Erciyes stumble recall against Erciyes lab flags (mostly the
    # 'trip-over' and 'limp' activities). Gradient-boosted trees gave a better trade-off, so the
    # trigger is fixed at FINAL_TRIGGER and the GBT threshold at FINAL_P_GBT; the numpy-only LR
    # fallback uses FINAL_P_LR (similar public flag rate). Chosen on DEV data only.
    g = FINAL_TRIGGER
    feats = FEATURE_SETS["all"]
    cand = isc & trig_mask(d, g)
    p_gbt = cv_probs(d, cand, feats, model="gbt")
    p_lr = cv_probs(d, cand, feats, model="lr")
    gate = (d.incl_post_max < DEFAULT_NF_CONFIG["upright_deg"]).to_numpy()
    cv = {}
    for name, pp, thr in (("gbt", p_gbt, FINAL_P_GBT), ("logreg_json", p_lr, FINAL_P_LR)):
        cv[name] = {"p_threshold": thr, **score(d, cand & (pp >= thr) & gate, recs)}
        for t2 in P_GRID:
            cv[f"{name}@{t2}"] = score(d, cand & (pp >= t2) & gate, recs)
    print(json.dumps({k: v for k, v in cv.items() if "@" not in k}, indent=1))
    tr = cand & d.label.isin(["nf", "neg"]).to_numpy()
    X = d.loc[tr, feats].to_numpy(float)
    y = (d.label[tr] == "nf").to_numpy()
    gbt = make_model("gbt").fit(X, y)
    import joblib
    import sklearn
    joblib.dump({"model": gbt, "feature_names": list(NF_FEATURE_NAMES), "sklearn_version": sklearn.__version__},
                os.path.join(ROOT, "models/nearfall_v2_gbt.joblib"))
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    cfg = {**DEFAULT_NF_CONFIG, **g, "p_threshold": FINAL_P_GBT, "p_threshold_logreg": FINAL_P_LR}
    with open(a.out, "w") as f:
        json.dump({"_comment": "Near-fall v2 (falldet/nearfall.py). Trigger + thresholds chosen by grouped CV on "
                               "public data (Erciyes/VTT, LOSO by subject) and ThunderSim DEV seeds only (scenarios "
                               "0-19, shifts 500000-500014). Main model: models/nearfall_v2_gbt.joblib "
                               "(p >= p_threshold). Numpy-only fallback: this logistic regression, "
                               "p = sigmoid(w.((x-mean)/scale)+b) >= p_threshold_logreg.",
                   "feature_names": list(NF_FEATURE_NAMES), "mean": sc.mean_.round(6).tolist(),
                   "scale": sc.scale_.round(6).tolist(), "w": lr.coef_[0].round(6).tolist(),
                   "b": round(float(lr.intercept_[0]), 6), "config": cfg,
                   "n_train": {"nf": int(y.sum()), "neg": int((~y).sum())}}, f, indent=1)
    with open(a.report, "w") as f:
        json.dump({"note": "Grouped CV (public subjects LOSO; ThunderSim DEV seeds). Simulated rows are simulated.",
                   "trigger": g, "cv": cv, "lr_grid_csv": os.path.basename(a.report.replace(".json", "_grid.csv"))},
                  f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
