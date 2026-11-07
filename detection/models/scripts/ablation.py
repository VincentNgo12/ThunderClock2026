"""Which features carry the result? LOSO with the shipped Stage-1 config and GBT, feature subsets.

    .venv/bin/python -I scripts/ablation.py      -> reports/ablation_target.json

Because lumbar and thoracic are synthesised from pelvis + upper thoracic, any gain from
"four anchors" over "two real anchors" on this data is NOT evidence that the real
4-IMU strip will do better. This script makes that visible.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_eval as T  # noqa: E402
from falldet.constants import CLASSES  # noqa: E402
from falldet.features import FEATURE_NAMES  # noqa: E402
from falldet.stage1 import load_thresholds, run_stage1  # noqa: E402

MULTI = ["ff_min_g", "impact_peak_g", "jerk_peak_g_s", "rot_peak_rad_s", "pre_std_g",
         "post_std_g", "post_std_late_g", "post_periodicity", "post_gyro_mean"]
SYNTH = [n for n in FEATURE_NAMES if n.endswith(("_lumbar", "_thoracic")) and "upper" not in n]
CROSS = ["n_anchors_lying", "incl_post_min", "incl_post_range", "incl_post_std", "still_frac",
         "longest_still_s"]  # computed over all 4 anchors, so they mix real + synthetic
SUBSETS = {
    "all_features": FEATURE_NAMES,
    "multiphase_only": MULTI,
    "posture_only_4_anchors": [n for n in FEATURE_NAMES if n not in MULTI],
    "real_anchors_only (no lumbar/thoracic, no cross-anchor)":
        [n for n in FEATURE_NAMES if n not in SYNTH and n not in CROSS],
    "pelvis_only": MULTI + [n for n in FEATURE_NAMES if n.endswith("_pelvis")],
}


def main():
    meta, recs = T.load("target")
    th = load_thresholds(os.path.join(ROOT, "models/stage1_thresholds.json"))
    hits = [run_stage1(a, w, th) for a, w in recs]
    cache = T.FeatureCache(recs)
    subj = meta["subject"].to_numpy()
    y_all = meta["label"].to_numpy()
    out = {}
    for name, cols in SUBSETS.items():
        idx = [FEATURE_NAMES.index(c) for c in cols]
        trig_preds = [[] for _ in recs]
        for s in sorted(set(subj)):
            tr = T.rows_for(hits, recs, meta, subj != s, augment=True)
            te = T.rows_for(hits, recs, meta, subj == s, augment=False)
            Xtr = np.array([cache.get(r, i) for r, i, _ in tr])[:, idx]
            m = T.make_models()["gbt"].fit(Xtr, y_all[[r for r, _, _ in tr]])
            if te:
                Xte = np.array([cache.get(r, i) for r, i, _ in te])[:, idx]
                for (r, _, _), p in zip(te, T.predict_probs("gbt", m, Xte, Xte)):
                    trig_preds[r].append(CLASSES[int(np.argmax(p))])
        rec_pred = np.array([max(p, key=T.SEVERITY.get) if p else "normal" for p in trig_preds])
        met = T.metrics(meta, rec_pred, trig_preds, hits)
        b = met["fall_confirmed_vs_rest"]
        out[name] = dict(n_features=len(cols), sensitivity=b["sensitivity"], specificity=b["specificity"],
                         f1=b["f1"], macro_f1_3class=met["macro_f1_3class"],
                         fa_per_hour_all_normal=met["fa_all_normal"]["fa_per_hour"])
        print(f"{name:58s} n={len(cols):2d} Se {b['sensitivity']:.3f} Sp {b['specificity']:.3f} "
              f"F1 {b['f1']:.3f} macroF1 {met['macro_f1_3class']:.3f} FA/h {met['fa_all_normal']['fa_per_hour']:.2f}")
    with open(os.path.join(ROOT, "reports/ablation_target.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
