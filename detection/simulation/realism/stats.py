# File: realism/stats.py
# Purpose: Gap statistics between real and simulated feature samples, grouped by subject/seed.
# Contributors: Unassigned
"""Per feature:
  nW     Wasserstein-1 distance divided by a robust scale of the real data (IQR/1.349, floored);
         0 = identical distributions, 1 = shifted by about one real standard deviation.
  CI     95 % bootstrap interval of nW, resampling whole groups (real subjects, sim seeds).
  null95 95th percentile of nW between two random halves of the REAL subjects: how much the
         real data disagrees with itself. A gap is "inside real-data noise" if nW <= null95.
  excess nW / null95 (the size ranking).
  KS     two-sample KS statistic, with its p-value.
Per category, a classifier two-sample test (C2ST): gradient-boosted trees learn real vs sim
with subject/seed-grouped 5-fold CV; out-of-fold AUC 0.5 = indistinguishable. The AUC CI
bootstraps groups. Univariate AUCs name the features that separate them.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

B = 200


def _scale(real: np.ndarray) -> float:
    q75, q25 = np.percentile(real, [75, 25])
    # Floors keep near-constant real features (e.g. counts, quiet standing) from exploding nW.
    return float(max((q75 - q25) / 1.349, 0.1 * np.std(real), 0.05 * abs(np.median(real)), 1e-3))


def _groups(values: np.ndarray, groups: np.ndarray) -> list[np.ndarray]:
    return [values[groups == g] for g in np.unique(groups)]


def _boot(parts: list[np.ndarray], rng) -> np.ndarray:
    pick = rng.integers(0, len(parts), len(parts))
    return np.concatenate([parts[i] for i in pick])


def feature_gap(real: np.ndarray, real_g: np.ndarray, sim: np.ndarray, sim_g: np.ndarray, seed: int = 0) -> dict:
    ok_r, ok_s = np.isfinite(real), np.isfinite(sim)
    real, real_g, sim, sim_g = real[ok_r], real_g[ok_r], sim[ok_s], sim_g[ok_s]
    if len(real) < 5 or len(sim) < 5:
        return {"n_real": int(len(real)), "n_sim": int(len(sim)), "nW": float("nan")}
    rng = np.random.default_rng(seed)
    scale = _scale(real)
    nw = stats.wasserstein_distance(real, sim) / scale
    rp, sp = _groups(real, real_g), _groups(sim, sim_g)
    boots = []
    for _ in range(B):
        r, s = _boot(rp, rng), _boot(sp, rng)
        boots.append(stats.wasserstein_distance(r, s) / scale)
    nulls = []
    ug = np.unique(real_g)
    for _ in range(B):
        perm = rng.permutation(ug)
        half = set(perm[: len(ug) // 2])
        mask = np.array([g in half for g in real_g])
        if mask.sum() < 3 or (~mask).sum() < 3:
            continue
        nulls.append(stats.wasserstein_distance(real[mask], real[~mask]) / scale)
    ks = stats.ks_2samp(real, sim)
    null95 = float(np.percentile(nulls, 95)) if nulls else float("nan")
    return {"n_real": int(len(real)), "n_sim": int(len(sim)), "nW": float(nw),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "null95": null95, "excess": float(nw / null95) if null95 > 0 else float("nan"),
            "ks": float(ks.statistic), "ks_p": float(ks.pvalue),
            "real_med": float(np.median(real)), "sim_med": float(np.median(sim)),
            "real_iqr": [float(v) for v in np.percentile(real, [25, 75])],
            "sim_iqr": [float(v) for v in np.percentile(sim, [25, 75])]}


def c2st(X_real: np.ndarray, g_real: np.ndarray, X_sim: np.ndarray, g_sim: np.ndarray, names: list[str],
         seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = np.vstack([X_real, X_sim])
    y = np.r_[np.ones(len(X_real)), np.zeros(len(X_sim))]
    groups = np.r_[np.array([f"R{g}" for g in g_real]), np.array([f"S{g}" for g in g_sim])]
    if len(np.unique(groups[y == 1])) < 5 or len(np.unique(groups[y == 0])) < 5:
        return {"auc": float("nan")}
    weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    oof = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.1, random_state=seed)
        clf.fit(X[train], y[train], sample_weight=weight[train])
        oof[test] = clf.predict_proba(X[test])[:, 1]
    auc = roc_auc_score(y, oof)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    real_g = [g for g in ug if g.startswith("R")]
    sim_g = [g for g in ug if g.startswith("S")]
    boots = []
    for _ in range(B):
        pick = np.concatenate([idx_by[g] for g in rng.choice(real_g, len(real_g))] +
                              [idx_by[g] for g in rng.choice(sim_g, len(sim_g))])
        boots.append(roc_auc_score(y[pick], oof[pick]))
    uni = {}
    for j, name in enumerate(names):
        col = X[:, j]
        ok = np.isfinite(col)
        if ok.sum() > 10 and len(np.unique(y[ok])) == 2:
            uni[name] = float(roc_auc_score(y[ok], col[ok]))
    top = sorted(uni.items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]
    return {"auc": float(auc), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_real": int(len(X_real)), "n_sim": int(len(X_sim)), "top_univariate": top}
