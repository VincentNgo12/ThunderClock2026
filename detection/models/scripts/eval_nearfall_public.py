"""Public-data (Erciyes + VTT) leave-one-subject-out, near-fall BEFORE (v1) vs AFTER (v2).

    .venv/bin/python -I scripts/eval_nearfall_public.py

Fall path: the shipped Stage-1 thresholds; Stage-2 GBT refit without the held-out subject;
post-rules applied (as in the stream). Near-fall v2: p(near_fall) for each perturbation
candidate from a model that never saw the held-out subject (grouped CV, same as
train_nearfall.py: GBT, trigger FINAL_TRIGGER, p >= FINAL_P_GBT, upright gate).
The two are combined with the stream's v2 rules:
  * a fall-path near_fall is kept only with ground contact or a v2 positive within 3 s;
  * a v2 positive is dropped if a fall-path fall_confirmed trigger lies within 8 s;
  * at most one near_fall per 10 s.
Output: reports/nearfall_v2_public_loso.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_eval as TE  # noqa: E402
import train_nearfall as TN  # noqa: E402
from falldet.constants import CLASSES  # noqa: E402
from falldet.detector import DEFAULT_POST_RULES, apply_rules, window_around  # noqa: E402
from falldet.features import aux_event_features, ground_contact, stage2_features  # noqa: E402
from falldet.stage1 import load_thresholds, run_stage1  # noqa: E402

FS = 100.0


def main():
    meta, recs = TE.load("target")
    th = load_thresholds(os.path.join(ROOT, "models/stage1_thresholds.json"))
    hits = [run_stage1(a, w, th) for a, w in recs]
    cache = TE.FeatureCache(recs)
    subj = meta["subject"].to_numpy()
    y_all = meta["label"].to_numpy()
    # ---- fall path, per trigger, LOSO
    fall_ev = {r: [] for r in range(len(recs))}   # r -> [(t_s, state, ground_contact)]
    for s in sorted(set(subj)):
        tr = TE.rows_for(hits, recs, meta, subj != s, augment=True)
        Xtr = np.array([cache.get(r, i) for r, i, _ in tr])
        model = TE.make_models()["gbt"].fit(Xtr, y_all[[r for r, _, _ in tr]])
        for r in np.flatnonzero(subj == s):
            a, w = recs[r]
            for i0 in hits[r]:
                aw, gw, j0 = window_around(a, w, i0)
                f = stage2_features(aw, gw, j0)
                p = TE.predict_probs("gbt", model, cache.get(r, i0)[None, :], None)[0]
                state = CLASSES[int(np.argmax(p))]
                state, _, _ = apply_rules(state, float(p.max()), f, aux_event_features(aw, j0), DEFAULT_POST_RULES)
                gc = ground_contact(aw, gw, j0) if state == "near_fall" else None
                fall_ev[r].append((i0 / FS, state, gc))
        print("fall fold", s, file=sys.stderr)
    # ---- near-fall v2 candidates, grouped CV (public subjects held out; sim/shift always in train)
    tables = [os.path.join(ROOT, "data/interim", f) for f in ("nf_public.csv", "nf_sim_dev.csv", "nf_shift_dev.csv")]
    d = TN.load(tables)
    isc = d.label.isin(["nf", "neg", "fall", "ignore"]).to_numpy()
    cand = isc & TN.trig_mask(d, TN.FINAL_TRIGGER)
    p = TN.cv_probs(d, cand, TN.FEATURE_SETS["all"], model="gbt")
    pos = cand & (p >= TN.FINAL_P_GBT) & (d.incl_post_max < 45).to_numpy() & (d.source == "public").to_numpy()
    nf_t = {}
    for _, row in d[pos].iterrows():
        nf_t.setdefault(int(row.rec[3:]), []).append(float(row.t))
    # ---- combine per recording
    out = {"v1": [], "v2": []}
    for r in range(len(recs)):
        ev = fall_ev[r]
        fc = [t for t, st, _ in ev if st == "fall_confirmed"]
        v1 = "fall_confirmed" if fc else ("near_fall" if any(st == "near_fall" for _, st, _ in ev) else "normal")
        n1 = sum(st == "near_fall" for _, st, _ in ev)
        emitted = []
        nfp = nf_t.get(r, [])
        for t, st, gc in ev:
            if st == "near_fall" and (gc or any(abs(t - x) <= 3.0 for x in nfp)):
                emitted.append(t + 8.0)
        for x in nfp:
            if not any(abs(x - t) <= 8.0 for t in fc):
                emitted.append(x + 3.0)
        kept, last = [], -1e9
        for t in sorted(emitted):
            if t - last >= 10.0:
                kept.append(t)
                last = t
        v2 = "fall_confirmed" if fc else ("near_fall" if kept else "normal")
        out["v1"].append((v1, n1))
        out["v2"].append((v2, len(kept)))
    res = {}
    for ver in ("v1", "v2"):
        pred = np.array([o for o, _ in out[ver]])
        nflags = np.array([n for _, n in out[ver]])
        m = meta.assign(pred=pred, nflags=nflags)
        r = {}
        for code in (808, 905, 906, 914, 916):
            x = m[m.code == code]
            r[f"{code}_{x.activity.iloc[0]}_near_fall_rate"] = round(float((x.pred == "near_fall").mean()), 3)
        x = m[m.label == "near_fall"]
        r["all_near_fall_label_recall"] = round(float((x.pred == "near_fall").mean()), 3)
        x = m[m.label == "fall_confirmed"]
        r["fall_confirmed_sensitivity"] = round(float((x.pred == "fall_confirmed").mean()), 4)
        for name, mm in (("erciyes_adl", (m.dataset == "erciyes") & (m.label == "normal")),
                         ("vtt_construction", m.dataset == "vtt"), ("all_normal", m.label == "normal")):
            hrs = m.loc[mm, "duration_s"].sum() / 3600
            r[f"near_fall_flags_per_h_{name}"] = round(float(m.loc[mm, "nflags"].sum() / hrs), 3)
        er = m[(m.dataset == "erciyes") & (m.label == "normal") & (m.code != 809)]
        r["near_fall_flags_per_h_erciyes_adl_excl_809_trip_over"] = round(float(er.nflags.sum() / (er.duration_s.sum() / 3600)), 3)
        top = m[m.label == "normal"].groupby("activity").nflags.sum().sort_values(ascending=False)
        r["flags_by_activity_top"] = {k: int(v) for k, v in top[top > 0].head(8).items()}
        res[ver] = r
    path = os.path.join(ROOT, "reports/nearfall_v2_public_loso.json")
    json.dump({"note": "Public data (Erciyes UCI-455 + VTT-ConIot), leave-one-subject-out. All falls simulated "
                       "by young volunteers; 'near-fall flags' = near_fall outputs during normal activity.",
               "v1_before": res["v1"], "v2_after": res["v2"]}, open(path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
