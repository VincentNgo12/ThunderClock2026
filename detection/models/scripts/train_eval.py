"""Tune Stage 1, train/evaluate Stage 2 with leave-one-subject-out (LOSO), save final models.

    .venv/bin/python -I scripts/train_eval.py

Reads  data/processed/canonical_{target,current}.npz + meta.csv
Writes models/stage1_thresholds[_current].json, models/stage2_model[_current].joblib,
       models/stage2_logreg_coefficients[_current].json,
       reports/metrics_*.json, reports/per_type_*.csv, reports/stage1_grid_*.csv,
       reports/loso_predictions_*.csv

Evaluation unit = one recording. A recording is predicted
    fall_confirmed  if any Stage-1 trigger in it is classified fall_confirmed,
    near_fall       else if any trigger is classified near_fall,
    normal          otherwise (including "Stage 1 never fired").
False alarms per hour = Stage-2 'fall_confirmed' decisions inside `normal` recordings
divided by the total duration of those recordings.
"""
import csv
import itertools
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from falldet.constants import CLASSES, DEVICE_PROFILES, FS_HZ, G  # noqa: E402
from falldet.detector import window_around  # noqa: E402
from falldet.features import FEATURE_NAMES, feature_vector, rule_based_stage2, stage2_features  # noqa: E402
from falldet.stage1 import DEFAULT_THRESHOLDS, Stage1Trigger  # noqa: E402

RECALL_TARGET = 0.95   # Stage 1 must catch >= 95 % of fall + near-fall recordings (training subjects)
SEVERITY = {"normal": 0, "near_fall": 1, "fall_confirmed": 2}

GRIDS = {
    "target": dict(freefall_g=[0.5, 0.6, 0.7, 0.8], impact_g=[1.8, 2.0, 2.25, 2.5, 3.0],
                   rotation_rad_s=[1.0, 1.5, 2.0, 3.0], min_conditions=[2, 3]),
    # +-2 g clips each axis, so |a| can still reach ~3.5 g only when all axes saturate;
    # +-250 deg/s clips each gyro axis at 4.36 rad/s
    "current": dict(freefall_g=[0.5, 0.6, 0.7, 0.8], impact_g=[1.5, 1.7, 1.9, 2.25],
                    rotation_rad_s=[1.0, 1.5, 2.0, 3.0], min_conditions=[2, 3]),
}


def make_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000)),
        "svm": make_pipeline(StandardScaler(), SVC(
            C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=0)),
        "gbt": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=200, class_weight="balanced", random_state=0),
    }


# ------------------------------------------------------------------ data
def load(profile):
    z = np.load(os.path.join(ROOT, f"data/processed/canonical_{profile}.npz"))
    meta = pd.read_csv(os.path.join(ROOT, "data/processed/meta.csv"))
    acc, gyro = z["acc"], z["gyro"]
    recs = [(acc[s:e].astype(np.float64), gyro[s:e].astype(np.float64))
            for s, e in zip(meta["start"], meta["stop"])]
    return meta, recs


def stage1_grid(recs, grid):
    """Run every threshold combination over all recordings in lock-step.
    Returns list of (config dict, list-of-trigger-index-lists per recording)."""
    K = len(recs)
    N = max(len(a) for a, _ in recs)
    svm = np.ones((K, N), dtype=np.float32)
    rot = np.zeros((K, N), dtype=np.float32)
    for k, (a, w) in enumerate(recs):
        s, r = Stage1Trigger.signals(a[:, :2], w[:, :2])
        svm[k, :len(s)], rot[k, :len(r)] = s, r
    t_ms = (np.arange(N) * 1000.0 / FS_HZ).round().astype(np.int64)
    results = []
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        th = {**DEFAULT_THRESHOLDS, **dict(zip(keys, vals))}
        trig = Stage1Trigger(th, k=K)
        hits = [[] for _ in range(K)]
        for i in range(N):
            fire = trig.step(t_ms[i], svm[:, i], rot[:, i])
            for k in np.flatnonzero(fire):
                hits[k].append(i)
        results.append((th, hits))
    return results


def stage1_score(hits, meta, mask):
    ev = mask & meta["label"].isin(["fall_confirmed", "near_fall"]).to_numpy()
    adl = mask & (meta["label"] == "normal").to_numpy()
    n_trig = np.array([len(h) for h in hits])
    recall = float((n_trig[ev] > 0).mean()) if ev.any() else float("nan")
    hours = meta.loc[adl, "duration_s"].sum() / 3600.0
    rate = float(n_trig[adl].sum() / hours) if hours > 0 else float("nan")
    return recall, rate


def choose_stage1(grid_results, meta, mask):
    scored = [(stage1_score(h, meta, mask), j) for j, (_, h) in enumerate(grid_results)]
    ok = [(rate, -rec, j) for (rec, rate), j in scored if rec >= RECALL_TARGET]
    if ok:
        return min(ok)[2]
    return min((-rec, rate, j) for (rec, rate), j in scored)[2]


# ------------------------------------------------------------------ Stage-2 rows
class FeatureCache:
    def __init__(self, recs):
        self.recs, self.cache = recs, {}

    def get(self, r, i0):
        key = (r, int(i0))
        if key not in self.cache:
            a, w = self.recs[r]
            aw, gw, j0 = window_around(a, w, int(i0))
            v = feature_vector(stage2_features(aw, gw, j0))
            assert np.isfinite(v).all(), (r, i0, v)
            self.cache[key] = v
        return self.cache[key]


def rows_for(hits, recs, meta, rec_mask, augment):
    """(rec, i0, is_trigger) rows. Augmentation adds one window at the largest |a| of each
    recording (as if Stage 1 had fired there) so Stage 2 sees every fall and many hard ADLs."""
    rows = []
    for r in np.flatnonzero(rec_mask):
        for i0 in hits[r]:
            rows.append((r, i0, True))
        if augment:
            a = recs[r][0]
            pk = int(np.argmax(np.linalg.norm(a[:, :2], axis=-1).mean(-1)))
            if all(abs(pk - i0) > FS_HZ for i0 in hits[r]):
                rows.append((r, pk, False))
    return rows


def predict_probs(model_name, model, X, F):
    if model_name == "rule":
        p = np.zeros((len(F), 3))
        for i, f in enumerate(F):
            p[i, CLASSES.index(rule_based_stage2(dict(zip(FEATURE_NAMES, f))))] = 1.0
        return p
    pr = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(c)] = pr[:, j]
    return out


# ------------------------------------------------------------------ metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def metrics(meta, rec_pred, trig_preds, hits):
    """rec_pred: array of predicted class per recording; trig_preds: list of per-trigger classes."""
    out = {}
    E = (meta["dataset"] == "erciyes").to_numpy()
    y = meta["label"].to_numpy()
    pos, pred_pos = (y == "fall_confirmed"), (rec_pred == "fall_confirmed")
    tp = int((pos & pred_pos & E).sum()); fn = int((pos & ~pred_pos & E).sum())
    fp = int((~pos & pred_pos & E).sum()); tn = int((~pos & ~pred_pos & E).sum())
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1); pr = tp / max(tp + fp, 1)
    out["fall_confirmed_vs_rest"] = dict(
        tp=tp, fn=fn, fp=fp, tn=tn, sensitivity=round(se, 4), sensitivity_ci95=wilson(tp, tp + fn),
        specificity=round(sp, 4), specificity_ci95=wilson(tn, tn + fp), precision=round(pr, 4),
        f1=round(2 * pr * se / max(pr + se, 1e-9), 4))
    ev = np.isin(y, ["fall_confirmed", "near_fall"]) & E
    out["any_event_detected_rate"] = round(float((rec_pred[ev] != "normal").mean()), 4)
    cm = pd.crosstab(pd.Series(y[E], name="true"), pd.Series(rec_pred[E], name="pred")) \
        .reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    out["confusion_erciyes"] = {t: {p: int(cm.loc[t, p]) for p in CLASSES} for t in CLASSES}
    f1s = []
    for c in CLASSES:
        tpc = cm.loc[c, c]; prc = tpc / max(cm[c].sum(), 1); rcc = tpc / max(cm.loc[c].sum(), 1)
        f1s.append(2 * prc * rcc / max(prc + rcc, 1e-9))
    out["macro_f1_3class"] = round(float(np.mean(f1s)), 4)
    # false alarms per hour on normal recordings
    for name, m in (("erciyes_adl", E & (y == "normal")), ("vtt_construction", ~E),
                    ("all_normal", y == "normal")):
        hours = meta.loc[m, "duration_s"].sum() / 3600.0
        n_fa = sum(sum(p == "fall_confirmed" for p in trig_preds[r]) for r in np.flatnonzero(m))
        n_near = sum(sum(p == "near_fall" for p in trig_preds[r]) for r in np.flatnonzero(m))
        n_tr = sum(len(hits[r]) for r in np.flatnonzero(m))
        out[f"fa_{name}"] = dict(hours=round(hours, 3), false_fall_confirmed=int(n_fa),
                                 fa_per_hour=round(n_fa / hours, 3),
                                 near_fall_flags=int(n_near), stage1_triggers=int(n_tr),
                                 stage1_triggers_per_hour=round(n_tr / hours, 2))
    s1 = np.array([len(h) > 0 for h in hits])
    out["stage1_recall_fall_confirmed"] = round(float(s1[pos & E].mean()), 4)
    out["stage1_recall_near_fall"] = round(float(s1[(y == "near_fall") & E].mean()), 4)
    return out


def per_type(meta, rec_pred, hits):
    rows = []
    s1 = np.array([len(h) > 0 for h in hits])
    for (ds, code, act, lab), g in meta.groupby(["dataset", "code", "activity", "label"]):
        idx = g.index.to_numpy()
        rows.append(dict(dataset=ds, code=code, activity=act, true_label=lab, n=len(idx),
                         stage1_fired=round(float(s1[idx].mean()), 3),
                         pred_fall_confirmed=round(float((rec_pred[idx] == "fall_confirmed").mean()), 3),
                         pred_near_fall=round(float((rec_pred[idx] == "near_fall").mean()), 3),
                         pred_normal=round(float((rec_pred[idx] == "normal").mean()), 3),
                         hours=round(g["duration_s"].sum() / 3600, 3)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ LOSO
def run_loso(profile, meta, recs, grid_results, eval_on=None):
    """eval_on: optional (meta, recs) of ANOTHER profile to test the fold models on (mismatch)."""
    subjects = sorted(meta["subject"].unique())
    subj = meta["subject"].to_numpy()
    cache = FeatureCache(recs)
    names = list(make_models()) + ["rule"]
    trig_preds = {m: [[] for _ in recs] for m in names}
    fold_cfg = {}
    mism = None
    if eval_on is not None:
        m_meta, m_recs = eval_on
        m_cache = FeatureCache(m_recs)
        mism = {m: [[] for _ in m_recs] for m in names}
        mism_hits = [[] for _ in m_recs]
    for s in subjects:
        train_mask, test_mask = subj != s, subj == s
        j = choose_stage1(grid_results, meta, train_mask)
        th, hits = grid_results[j]
        fold_cfg[s] = j
        tr_rows = rows_for(hits, recs, meta, train_mask, augment=True)
        Xtr = np.array([cache.get(r, i) for r, i, _ in tr_rows])
        ytr = meta["label"].to_numpy()[[r for r, _, _ in tr_rows]]
        te_rows = rows_for(hits, recs, meta, test_mask, augment=False)
        Xte = np.array([cache.get(r, i) for r, i, _ in te_rows]) if te_rows else np.zeros((0, len(FEATURE_NAMES)))
        if eval_on is not None:
            from falldet.stage1 import run_stage1
            m_rows = []
            for r in np.flatnonzero(m_meta["subject"].to_numpy() == s):
                a, w = m_recs[r]
                trg = run_stage1(a, w, th)
                mism_hits[r] = trg
                m_rows += [(r, i) for i in trg]
            Xm = np.array([m_cache.get(r, i) for r, i in m_rows]) if m_rows else None
        for name in names:
            model = None
            if name != "rule":
                model = make_models()[name].fit(Xtr, ytr)
            if len(te_rows):
                P = predict_probs(name, model, Xte, Xte)
                for (r, i, _), p in zip(te_rows, P):
                    trig_preds[name][r].append(CLASSES[int(np.argmax(p))])
            if eval_on is not None and Xm is not None:
                P = predict_probs(name, model, Xm, Xm)
                for (r, i), p in zip(m_rows, P):
                    mism[name][r].append(CLASSES[int(np.argmax(p))])
        print(f"[{profile}] fold {s}: stage1 cfg #{j} "
              f"(ff<{th['freefall_g']} imp>{th['impact_g']} rot>{th['rotation_rad_s']} "
              f"k={th['min_conditions']}), train rows {len(tr_rows)}, test triggers {len(te_rows)}",
              file=sys.stderr)
    # recording-level hits for the per-fold configs
    hits_eval = [grid_results[fold_cfg[subj[r]]][1][r] for r in range(len(recs))]
    res = {}
    for name in names:
        rec_pred = np.array([max(p, key=SEVERITY.get) if p else "normal" for p in trig_preds[name]])
        res[name] = dict(metrics=metrics(meta, rec_pred, trig_preds[name], hits_eval),
                         rec_pred=rec_pred, trig_preds=trig_preds[name])
    out_m = None
    if eval_on is not None:
        out_m = {}
        for name in names:
            rec_pred = np.array([max(p, key=SEVERITY.get) if p else "normal" for p in mism[name]])
            out_m[name] = dict(metrics=metrics(m_meta, rec_pred, mism[name], mism_hits),
                               rec_pred=rec_pred)
    return res, fold_cfg, hits_eval, out_m


def pick_best(res):
    # highest fall_confirmed F1; ties broken by fewer false alarms per hour
    cands = [n for n in res if n != "rule"]
    return max(cands, key=lambda n: (res[n]["metrics"]["fall_confirmed_vs_rest"]["f1"],
                                     -res[n]["metrics"]["fa_all_normal"]["fa_per_hour"]))


def save_final(profile, meta, recs, grid_results, best_name):
    suffix = "" if profile == "target" else f"_{profile}"
    allmask = np.ones(len(meta), bool)
    j = choose_stage1(grid_results, meta, allmask)
    th, hits = grid_results[j]
    rec, rate = stage1_score(hits, meta, allmask)
    th_out = dict(th)
    th_out["_comment"] = (f"Stage-1 thresholds for device profile '{profile}' "
                          f"{DEVICE_PROFILES[profile]}. Signals: svm_g = mean(|a_pelvis|,|a_lumbar|)/g, "
                          "rot = max(|w_pelvis|,|w_lumbar|) rad/s. Tuned on pooled Erciyes+VTT "
                          f"(recall {rec:.3f} on fall+near-fall recordings, {rate:.1f} triggers/h on "
                          "normal recordings). Erciyes is 25 Hz upsampled: real 100 Hz impact peaks "
                          "will be higher; re-tune on team recordings. rotation_rad_s should become the "
                          "worker's own 99.9th percentile once personal data exist.")
    with open(os.path.join(ROOT, f"models/stage1_thresholds{suffix}.json"), "w") as f:
        json.dump(th_out, f, indent=2)
    cache = FeatureCache(recs)
    rows = rows_for(hits, recs, meta, allmask, augment=True)
    X = np.array([cache.get(r, i) for r, i, _ in rows])
    y = meta["label"].to_numpy()[[r for r, _, _ in rows]]
    model = make_models()[best_name].fit(X, y)
    joblib.dump(dict(model=model, model_type=best_name, classes=list(model.classes_),
                     feature_names=list(FEATURE_NAMES), profile=profile,
                     device_profile=DEVICE_PROFILES[profile], stage1_thresholds=th,
                     n_training_rows=len(rows), sklearn_version=sklearn.__version__,
                     trained_on="Erciyes UCI-455 (17 subjects) + VTT-ConIot (13 users), "
                                "harmonised to 4 anchors; pelvis+upper_thoracic real, "
                                "lumbar+thoracic synthetic"),
                os.path.join(ROOT, f"models/stage2_model{suffix}.joblib"))
    # portable logistic-regression version (plain arithmetic) for C++/JS ports
    lr = make_models()["logreg"].fit(X, y)
    sc, clf = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
    with open(os.path.join(ROOT, f"models/stage2_logreg_coefficients{suffix}.json"), "w") as f:
        json.dump(dict(
            _comment="p = softmax(W @ ((x - mean) / scale) + b); x in feature_names order",
            feature_names=list(FEATURE_NAMES), classes=list(clf.classes_),
            mean=sc.mean_.round(6).tolist(), scale=sc.scale_.round(6).tolist(),
            W=clf.coef_.round(6).tolist(), b=clf.intercept_.round(6).tolist()), f, indent=1)
    print(f"[{profile}] final: stage1 cfg {th}, model {best_name}, rows {len(rows)}")


def strip(res):
    return {n: r["metrics"] for n, r in res.items()}


def main():
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    data, grids, loso = {}, {}, {}
    for profile in ("target", "current"):
        t0 = time.time()
        meta, recs = load(profile)
        data[profile] = (meta, recs)
        grids[profile] = stage1_grid(recs, GRIDS[profile])
        g = []
        for th, hits in grids[profile]:
            rec, rate = stage1_score(hits, meta, np.ones(len(meta), bool))
            g.append({**{k: th[k] for k in GRIDS[profile]}, "recall_events": rec, "adl_triggers_per_h": rate})
        pd.DataFrame(g).to_csv(os.path.join(ROOT, f"reports/stage1_grid_{profile}.csv"), index=False)
        print(f"[{profile}] stage-1 grid done in {time.time() - t0:.0f}s", file=sys.stderr)

    # target: LOSO + test the same fold models on current-config data (mismatch)
    for profile in ("target", "current"):
        meta, recs = data[profile]
        eval_on = data["current"] if profile == "target" else None
        res, fold_cfg, hits_eval, mism = run_loso(profile, meta, recs, grids[profile], eval_on)
        best = pick_best(res)
        loso[profile] = (res, best)
        with open(os.path.join(ROOT, f"reports/metrics_{profile}.json"), "w") as f:
            json.dump(dict(profile=profile, device=DEVICE_PROFILES[profile], best_model=best,
                           stage1_fold_configs={s: grids[profile][j][0] for s, j in fold_cfg.items()},
                           models=strip(res)), f, indent=1)
        per_type(meta, res[best]["rec_pred"], hits_eval).to_csv(
            os.path.join(ROOT, f"reports/per_type_{profile}.csv"), index=False)
        pd.DataFrame(dict(rec=meta["rec"], subject=meta["subject"], code=meta["code"],
                          activity=meta["activity"], label=meta["label"],
                          **{f"pred_{n}": r["rec_pred"] for n, r in res.items()})).to_csv(
            os.path.join(ROOT, f"reports/loso_predictions_{profile}.csv"), index=False)
        if mism is not None:
            with open(os.path.join(ROOT, "reports/metrics_mismatch_target_model_on_current_data.json"), "w") as f:
                json.dump(dict(note="Models and Stage-1 thresholds trained for the TARGET profile, "
                                    "tested (LOSO) on the same recordings degraded to the CURRENT profile",
                               models={n: r["metrics"] for n, r in mism.items()}), f, indent=1)
        save_final(profile, meta, recs, grids[profile], best)

    for profile, (res, best) in loso.items():
        print(f"\n=== {profile}: best Stage-2 model = {best}")
        for n, r in res.items():
            m = r["metrics"]
            b = m["fall_confirmed_vs_rest"]
            print(f"  {n:7s} Se {b['sensitivity']:.3f} Sp {b['specificity']:.3f} "
                  f"Prec {b['precision']:.3f} F1 {b['f1']:.3f} macroF1 {m['macro_f1_3class']:.3f} "
                  f"FA/h all {m['fa_all_normal']['fa_per_hour']:.2f} "
                  f"(ADL {m['fa_erciyes_adl']['fa_per_hour']:.2f}, VTT {m['fa_vtt_construction']['fa_per_hour']:.2f}) "
                  f"S1 recall {m['stage1_recall_fall_confirmed']:.3f}")


if __name__ == "__main__":
    main()
