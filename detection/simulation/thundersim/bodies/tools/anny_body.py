#!/usr/bin/env python
"""Build a ThunderSim worker body description from an Anny body (anny 0.6.2, torch CPU).

Run with the Anny venv:
    <anny-venv>/bin/python tools/anny_body.py --preset avg_male --out anny_avg_male.json
    <anny-venv>/bin/python tools/anny_body.py --gender 1 --age-years 35 \
        --muscle 0.5 --weight 0.5 --proportions 0.5 --height-m 1.60 --out anny_female_160.json
(offline only; the anny venv is separate from the simulator's .venv. ANNY_WORK_DIR sets the cache.)

Output frame ("sim frame", identical to thundersim/spine.py): metres, right-handed, +X wearer-right,
+Y up, +Z posterior (the body faces -Z). Feet (lowest rest vertex) on y = 0. Horizontal origin:
x = 0 on Anny's mid-sagittal plane, z = 0 at the S1 spinal-midline point, so the sim's pelvis
anchor root_pos is (0, pelvis_height, 0).
    p_sim = M p_anny + t,  M = [[-1,0,0],[0,0,1],[0,1,0]]  (Anny native: Z up, faces -Y, left +X)
M = Ry(pi) . Rx(-90deg): Rx(-90) is extract.py's Z-up -> Y-up map (faces +Z, left +X), Ry(pi) turns
that into the sim's facing. det M = +1, so mesh winding (CCW = outward) is preserved.

Deterministic: no random numbers; Anny's forward pass is deterministic; phenotype values are
rounded to 4 decimals before the final forward pass (as in anny/extract.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ANNY_DIR = os.environ.get("ANNY_WORK_DIR", os.path.join(os.path.expanduser("~"), ".cache", "anny"))
os.environ.setdefault("ANNY_CACHE_DIR", os.path.join(ANNY_DIR, "cache"))
os.environ.setdefault("WARP_CACHE_PATH", os.path.join(ANNY_DIR, "cache", "warp"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import torch  # noqa: E402
import anny  # noqa: E402
from anny.anthropometry import Anthropometry  # noqa: E402
from anny.shape_distribution import SimpleShapeDistribution  # noqa: E402

torch.set_grad_enabled(False)

FORMAT = "anny-worker-body/1"
M_ANNY_TO_SIM = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
DENSITY = 980.0  # kg/m^3: anny/anthropometry.py Anthropometry.mass() ("density of 980 kg/m^3 for simplicity")

# Spinous-level mapping (anny_lift_spec.md section 5 step 1 / D7). BodyParts3D spinous-tip heights in the
# BP3D page frame (feet at y = 0), from insite/anatomy-bp3d.json landmarks.spine[].tip[1]; hip and C7
# reference heights as typed in the spec (hip = mean femoral-head centre, C7 = C7 landmark).
BP3D_TIP_Y = {"S1": 0.9708, "L3": 1.0560, "T10": 1.1836, "T2": 1.4093}
BP3D_HIP_Y, BP3D_C7_Y = 0.8915, 1.457
LEVELS = ["S1", "L3", "T10", "T2"]
POD_FOOT = (0.034, 0.044)      # pod footprint x, y (m), spec 5
POD_STANDOFF, POD_THICK = 0.002, 0.012  # pod bottom 2 mm off the skin, 12 mm tall (spec 5)
SENSOR_BEHIND_MIDLINE = 0.04   # thundersim Worker.sensor_offset_m default (spine.py)

PRESETS = {
    # == anny/extract.py average male: male anchor, 35 y, average muscle/weight/proportions, 1.765 m
    "avg_male": dict(gender=0.0, age_years=35.0, muscle=0.5, weight=0.5, proportions=0.5, height_m=1.765),
    "female_160": dict(gender=1.0, age_years=35.0, muscle=0.5, weight=0.5, proportions=0.5, height_m=1.60),
}

# de Leva 1996, J Biomech 29(9):1223-1230, Table 4: segment mass, % of body mass (adjusted Zatsiorsky-Seluyanov).
# head = head + neck; trunk subsegments: upper (suprasternale-xiphoid), middle (xiphoid-omphalion),
# lower (omphalion-hip joint centres).
DE_LEVA = {
    "male": {"head_neck": 6.94, "trunk": 43.46, "upper_arm": 2.71, "forearm": 1.62, "hand": 0.61,
             "thigh": 14.16, "shank": 4.33, "foot": 1.37,
             "trunk_upper": 15.96, "trunk_middle": 16.33, "trunk_lower": 11.17},
    "female": {"head_neck": 6.68, "trunk": 42.57, "upper_arm": 2.55, "forearm": 1.38, "hand": 0.56,
               "thigh": 14.78, "shank": 4.81, "foot": 1.29,
               "trunk_upper": 15.45, "trunk_middle": 14.65, "trunk_lower": 12.47},
}

# Skin-weight regions: every one of the 104 bones goes to exactly one region.
TRUNK_BONES = {
    "pelvis": ["root", "pelvis.L", "pelvis.R", "spine05"],
    "lumbar": ["spine04", "spine03"],
    "thoracic": ["spine02"],
    "upper_thoracic": ["spine01", "clavicle.L", "clavicle.R", "shoulder01.L", "shoulder01.R"],
}
MIDLINE_CHAIN = ["root", "spine04", "spine03", "spine02", "spine01", "neck01", "neck02", "neck03", "head"]

WORKER_DEFAULTS = {  # thundersim/spine.py Worker dataclass defaults (read-only reference)
    "height_m": 1.75, "mass_kg": 80.0, "load_kg": 8.0, "spacing_m": None,
    "neutral_curvature_deg": [-12.0, -6.0, 10.0, 18.0], "mount_error_deg": 6.0, "sensor_offset_m": 0.04,
}


def bone_region(name):
    for seg, names in TRUNK_BONES.items():
        if name in names:
            return "trunk_" + seg
    if name.startswith(("neck", "head", "eye")):
        return "head_neck"
    side = name.rsplit(".", 1)[-1]
    if name.startswith("upperarm"):
        return "upper_arm_" + side
    if name.startswith("lowerarm"):
        return "forearm_" + side
    if name.startswith(("wrist", "finger", "metacarpal")):
        return "hand_" + side
    if name.startswith("upperleg"):
        return "thigh_" + side
    if name.startswith("lowerleg"):
        return "shank_" + side
    if name.startswith(("foot", "toe")):
        return "foot_" + side
    raise KeyError(name)


REGIONS = ["head_neck", "trunk_pelvis", "trunk_lumbar", "trunk_thoracic", "trunk_upper_thoracic"] + [
    f"{r}_{s}" for r in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot") for s in ("L", "R")]


# ------------------------------------------------------------------------------------------------- geometry
def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area-weighted, outward for CCW
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(n > 0, n, 1.0)


def ray_hits(V, F, o, d):
    """All ray-triangle hit distances t > 0 (Moller-Trumbore), sorted."""
    d = unit(d)
    A = V[F[:, 0]]
    e1, e2 = V[F[:, 1]] - A, V[F[:, 2]] - A
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - A
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ d) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    idx = np.nonzero(good)[0]
    o_ = np.argsort(t[idx])
    return t[idx][o_], idx[o_]


def back_skin(V, F, y, x=0.0):
    """Posterior-most skin crossing of the horizontal +Z ray at (x, y): farthest hit from in front."""
    o = np.array([x, y, V[:, 2].min() - 0.1])
    t, fi = ray_hits(V, F, o, np.array([0.0, 0.0, 1.0]))
    if len(t) == 0:
        raise ValueError(f"ray missed the body at y={y}")
    return o + t[-1] * np.array([0.0, 0.0, 1.0]), int(fi[-1])


def signed_distance(V, F, P):
    """Signed distance (negative inside) of points P to the closed mesh: exact |d|, winding-number sign."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ab, ac = B - A, C - A
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    out = []
    for p in np.atleast_2d(P):
        a, b, c = A - p, B - p, C - p
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la + \
            np.einsum("ij,ij->i", c, a) * lb
        w = np.sum(2 * np.arctan2(num, den)) / (4 * np.pi)
        ap = p - A
        u = np.einsum("ij,ij->i", np.cross(ap, ac), n) / nn
        v = np.einsum("ij,ij->i", np.cross(ab, ap), n) / nn
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        dpl = np.abs(np.einsum("ij,ij->i", ap, n)) / np.sqrt(nn)

        def seg(X, Y):
            dd = Y - X
            tt = np.clip(np.einsum("ij,ij->i", p - X, dd) / np.maximum(np.einsum("ij,ij->i", dd, dd), 1e-30), 0, 1)
            return np.linalg.norm(p - (X + tt[:, None] * dd), axis=1)
        de = np.minimum(np.minimum(seg(A, B), seg(B, C)), seg(C, A))
        dist = np.where(inside, dpl, de).min()
        out.append(-dist if w > 0.5 else dist)
    return np.array(out)


def interior_voxels(V, F, h):
    """Centres of an h-grid inside the closed mesh: +Y scan-line parity per (x, z) column."""
    lo, hi = V.min(0) - h, V.max(0) + h
    xs = np.arange(lo[0], hi[0], h) + h / 2
    ys = np.arange(lo[1], hi[1], h) + h / 2
    zs = np.arange(lo[2], hi[2], h) + h / 2
    nz = len(zs)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    T = np.stack([a, b, c], 1)
    i0 = np.ceil((T[:, :, 0].min(1) - xs[0]) / h).astype(np.int64)
    i1 = np.floor((T[:, :, 0].max(1) - xs[0]) / h).astype(np.int64)
    j0 = np.ceil((T[:, :, 2].min(1) - zs[0]) / h).astype(np.int64)
    j1 = np.floor((T[:, :, 2].max(1) - zs[0]) / h).astype(np.int64)
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    cnt = ni * nj
    fidx = np.repeat(np.arange(len(F)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    I = i0[fidx] + off // np.maximum(nj[fidx], 1)
    J = j0[fidx] + off % np.maximum(nj[fidx], 1)
    px, pz = xs[I], zs[J]
    A = a[fidx]
    e1, e2 = b[fidx] - A, c[fidx] - A
    det = e1[:, 0] * e2[:, 2] - e1[:, 2] * e2[:, 0]
    ok = np.abs(det) > 1e-14
    ds = np.where(ok, det, 1.0)
    dx, dz = px - A[:, 0], pz - A[:, 2]
    u = (dx * e2[:, 2] - dz * e2[:, 0]) / ds
    v = (e1[:, 0] * dz - e1[:, 2] * dx) / ds
    ok &= (u >= 0) & (v >= 0) & (u + v <= 1)
    col = (I * nz + J)[ok]
    y = (A[:, 1] + u * e1[:, 1] + v * e2[:, 1])[ok]
    o = np.lexsort((y, col))
    col, y = col[o], y[o]
    starts = np.r_[0, np.nonzero(np.diff(col))[0] + 1, len(col)]
    out, odd = [], 0
    for s, e in zip(starts[:-1], starts[1:]):
        yy = y[s:e]
        if len(yy) % 2:
            odd += 1
            continue
        Ii, Jj = divmod(int(col[s]), nz)
        inside = ((ys[None, :] > yy[0::2, None]) & (ys[None, :] < yy[1::2, None])).any(0)
        if inside.any():
            sel = ys[inside]
            out.append(np.stack([np.full(len(sel), xs[Ii]), sel, np.full(len(sel), zs[Jj])], 1))
    return np.concatenate(out), odd


def r(x, n=5):
    if isinstance(x, dict):
        return {k: r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, n) for v in np.asarray(x).tolist()] if isinstance(x, np.ndarray) else [r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ------------------------------------------------------------------------------------------------- main
def solve_height_param(model, anth, base, target_m, iters=40):
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        h = float(anth.height(model(phenotype_kwargs={**base, "height": mid})["rest_vertices"])[0])
        lo, hi = (mid, hi) if h < target_m else (lo, mid)
    return 0.5 * (lo + hi)


def build(args):
    model = anny.Anny()  # rig 'anny' (104 bones), triangle topology, float64
    assert model.bone_count == 104
    labels = list(model.bone_labels)
    li = {n: i for i, n in enumerate(labels)}
    anth = Anthropometry(model)

    # ---- phenotype
    p = dict(PRESETS[args.preset]) if args.preset else {}
    for k in ("gender", "age", "age_years", "muscle", "weight", "proportions", "height", "height_m"):
        v = getattr(args, k)
        if v is not None:
            p[k] = v
            if k == "height":
                p.pop("height_m", None)
            if k == "height_m":
                p.pop("height", None)
            if k == "age":
                p.pop("age_years", None)
            if k == "age_years":
                p.pop("age", None)
    phen = {k: float(p.get(k, 0.5)) for k in ("gender", "muscle", "weight", "proportions")}
    if "age_years" in p:
        sd = SimpleShapeDistribution(model)
        phen["age"] = float(sd.morphological_age_mapping.morphological_to_anny_age(
            torch.tensor([p["age_years"]], dtype=model.dtype))[0])
    else:
        phen["age"] = float(p.get("age", 0.5))
    if "height_m" in p:
        phen["height"] = solve_height_param(model, anth, phen, p["height_m"])
    else:
        phen["height"] = float(p.get("height", 0.5))
    phen = {k: round(float(phen[k]), 4) for k in ("gender", "age", "muscle", "weight", "height", "proportions")}
    out = model(phenotype_kwargs=phen)
    meas = {k: float(v[0]) for k, v in anth(out["rest_vertices"]).items()}

    # ---- to the sim frame
    Va = out["rest_vertices"][0].numpy()
    Ha = out["rest_bone_heads"][0].numpy()
    V = Va @ M_ANNY_TO_SIM.T
    Hb = Ha @ M_ANNY_TO_SIM.T
    F = model.faces.numpy().astype(np.int64)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    ncomp, lab = connected_components(sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V),) * 2),
                                      directed=False)
    body_v = lab == np.bincount(lab).argmax()
    body_f = body_v[F[:, 0]]
    Fb = F[body_f]

    # midline polyline (rig spinal column: bone heads root -> spine04..01 -> neck01..03 -> head), then origin
    chain = np.array([Hb[li[n]] for n in MIDLINE_CHAIN])
    assert np.all(np.diff(chain[:, 1]) > 0), "midline chain must rise monotonically"
    sole = V[:, 1].min()
    hip_y = 0.5 * (Hb[li["upperleg01.L"], 1] + Hb[li["upperleg01.R"], 1]) - sole
    c7_y = Hb[li["neck01"], 1] - sole
    scale = (c7_y - hip_y) / (BP3D_C7_Y - BP3D_HIP_Y)
    lev_y = {k: hip_y + (BP3D_TIP_Y[k] - BP3D_HIP_Y) * scale for k in LEVELS}

    def midline_at(y_abs):
        return np.array([np.interp(y_abs, chain[:, 1], chain[:, j]) for j in range(3)])
    s1_mid = midline_at(lev_y["S1"] + sole)
    t = np.array([0.0, -sole, -s1_mid[2]])  # x stays on the mid-sagittal plane (x = 0)
    assert abs(s1_mid[0]) < 1e-4, s1_mid
    V = V + t
    Hb = Hb + t
    chain = chain + t
    N = vertex_normals(V, Fb)

    def midline(y):
        return np.array([0.0, y, np.interp(y, chain[:, 1], chain[:, 2])])

    # ---- spinous anchors, sensors
    anchors, sensors = {}, {}
    post = np.array([0.0, 0.0, 1.0])
    for k in LEVELS:
        y = lev_y[k]
        skin, tri = back_skin(V, Fb, y)
        sel = body_v & (np.abs(V[:, 0]) < POD_FOOT[0] / 2) & (np.abs(V[:, 1] - y) < POD_FOOT[1] / 2) & \
            (V[:, 2] > skin[2] - 0.02)
        vids = np.nonzero(sel)[0]
        n = unit(N[vids].mean(0)) if len(vids) else unit(N[F[body_f][tri]].mean(0))
        m = midline(y)
        sens = m + SENSOR_BEHIND_MIDLINE * post
        pod_c = skin + (POD_STANDOFF + POD_THICK / 2) * n
        anchors[k] = {"skin": skin, "normal": n, "footprint_vertices": int(len(vids)), "midline": m,
                      "midline_to_skin_depth_m": float(skin[2] - m[2]), "pod_centre": pod_c,
                      "midline_to_pod_centre_horizontal_m": float(pod_c[2] - m[2]),
                      "neutral_pitch_deg": float(np.degrees(np.arctan2(n[1], n[2])))}
        sensors[k] = sens
    sd_sens = signed_distance(V, Fb, np.array([sensors[k] for k in LEVELS]))
    for i, k in enumerate(LEVELS):
        anchors[k]["sensor_gap_to_skin_along_posterior_m"] = float(sensors[k][2] - anchors[k]["skin"][2])
        anchors[k]["sensor_signed_distance_to_skin_m"] = float(sd_sens[i])

    # ---- links (thundersim: link i joins spine-line point i to i+1; upper thoracic has no length in the sim)
    mids = np.array([anchors[k]["midline"] for k in LEVELS])
    skins = np.array([anchors[k]["skin"] for k in LEVELS])
    c7 = Hb[li["neck01"]]
    ys = np.unique(np.r_[np.arange(lev_y["S1"], lev_y["T2"], 0.002), [lev_y[k] for k in LEVELS]])  # levels sampled exactly
    prof = np.array([back_skin(V, Fb, y)[0] for y in ys])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    arc_at = [float(np.interp(lev_y[k], ys, arc)) for k in LEVELS]
    names = ["pelvis", "lumbar", "thoracic"]
    links = {}
    for i, nm in enumerate(names):
        links[nm] = {"from": LEVELS[i], "to": LEVELS[i + 1],
                     "length_m": float(np.linalg.norm(mids[i + 1] - mids[i])),
                     "midline_chord_m": float(np.linalg.norm(mids[i + 1] - mids[i])),
                     "skin_chord_m": float(np.linalg.norm(skins[i + 1] - skins[i])),
                     "skin_arc_m": arc_at[i + 1] - arc_at[i],
                     "vertical_m": float(lev_y[LEVELS[i + 1]] - lev_y[LEVELS[i]])}
    links["upper_thoracic"] = {"from": "T2", "to": "C7 (neck01 bone head)", "length_m": float(np.linalg.norm(c7 - mids[3])),
                               "midline_chord_m": float(np.linalg.norm(c7 - mids[3])),
                               "vertical_m": float(c7[1] - lev_y["T2"]),
                               "used_by_sim": False}
    spacing = tuple(links[nm]["length_m"] for nm in names)

    # ---- volume, mass, skin-weight partition
    vol_exact_all = meas["volume"]
    total_mass = meas["mass"]
    a, b, c = V[Fb[:, 0]], V[Fb[:, 1]], V[Fb[:, 2]]
    vol_body = float((np.einsum("ij,ij->i", np.cross(a, b), c)).sum() / 6.0)
    h = args.voxel
    P, odd = interior_voxels(V, Fb, h)
    vox_vol = len(P) * h ** 3
    bidx = np.nonzero(body_v)[0]
    wi = model.vertex_bone_indices.numpy().astype(np.int64)
    ww = model.vertex_bone_weights.numpy()
    ww = ww / ww.sum(1, keepdims=True)
    reg_of_bone = np.array([REGIONS.index(bone_region(n)) for n in labels])
    RW = np.zeros((len(V), len(REGIONS)))
    for j in range(wi.shape[1]):
        np.add.at(RW, (np.arange(len(V)), reg_of_bone[wi[:, j]]), ww[:, j])
    tree = cKDTree(V[bidx])
    dist, nn = tree.query(P, k=args.knn)
    alpha = 1.0 / np.maximum(dist, 1e-4)
    alpha /= alpha.sum(1, keepdims=True)
    VR = np.einsum("nk,nkr->nr", alpha, RW[bidx][nn])  # (voxels, regions), rows sum to 1
    frac = VR.sum(0) / VR.sum()
    reg_mass = {REGIONS[i]: float(frac[i] * total_mass) for i in range(len(REGIONS))}
    com = (VR.sum(1)[:, None] * P).sum(0) / VR.sum()
    # trunk split by horizontal planes at the sim link boundaries (L3, T10, T2)
    trunk_w = VR[:, [REGIONS.index("trunk_" + s) for s in TRUNK_BONES]].sum(1)
    cuts = [-np.inf, lev_y["L3"], lev_y["T10"], lev_y["T2"], np.inf]
    plane_split = {}
    for i, s in enumerate(TRUNK_BONES):
        m_ = (P[:, 1] >= cuts[i]) & (P[:, 1] < cuts[i + 1])
        plane_split["trunk_" + s] = float(trunk_w[m_].sum() / VR.sum() * total_mass)

    g = phen["gender"]
    dl = {k: (1 - g) * DE_LEVA["male"][k] + g * DE_LEVA["female"][k] for k in DE_LEVA["male"]}

    def pair(name):
        return reg_mass[name + "_L"] + reg_mass[name + "_R"]
    trunk = sum(reg_mass["trunk_" + s] for s in TRUNK_BONES)
    cmp_rows = {"head_neck": (reg_mass["head_neck"], dl["head_neck"], 1), "trunk": (trunk, dl["trunk"], 1)}
    for nm in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot"):
        cmp_rows[nm + " (each side, mean L/R)"] = (pair(nm) / 2, dl[nm], 1)
    comparison = {k: {"anny_kg": v, "anny_pct": 100 * v / total_mass, "de_leva_pct": d,
                      "de_leva_kg": d / 100 * total_mass, "anny_minus_de_leva_pct_points": 100 * v / total_mass - d}
                  for k, (v, d, _) in cmp_rows.items()}

    # ---- worker fields
    curv = tuple(anchors[k]["neutral_pitch_deg"] for k in LEVELS)
    height_m = meas["height"]
    worker_fields = {
        "height_m": height_m,
        "mass_kg": total_mass,
        "load_kg": WORKER_DEFAULTS["load_kg"],
        "spacing_m": list(spacing),
        "neutral_curvature_deg": list(curv),
        "mount_error_deg": WORKER_DEFAULTS["mount_error_deg"],
        "sensor_offset_m": SENSOR_BEHIND_MIDLINE,
    }
    sim_default_spacing = [f * height_m for f in (0.07, 0.09, 0.11)]
    com_to_s1 = mids[0] - com

    # ---- what spine.py actually produces at neutral (all angles 0: one straight vertical spine line)
    sens_arr = np.array([sensors[k] for k in LEVELS])
    straight = np.array([[0.0, mids[0][1] + s, SENSOR_BEHIND_MIDLINE] for s in np.r_[0.0, np.cumsum(spacing)]])
    # curved-neutral alternative: segment i long axis along Anny's midline chord i (seg 3: T2 -> C7);
    # spine.py up-axis of Rx(-a) is (0, cos a, -sin a), so the forward-lean angle of a chord is atan2(-dz, dy).
    chords = [mids[1] - mids[0], mids[2] - mids[1], mids[3] - mids[2], c7 - mids[3]]
    lean = np.array([np.arctan2(-v[2], v[1]) for v in chords])
    curv_curved = [float(np.degrees(np.radians(curv[i]) - lean[i])) for i in range(4)]
    curved_pts = [mids[0]]
    for i in range(3):
        curved_pts.append(curved_pts[-1] + spacing[i] * np.array([0.0, np.cos(lean[i]), -np.sin(lean[i])]))
    curved_sens = np.array([p + SENSOR_BEHIND_MIDLINE * np.array([0.0, np.sin(lean[i]), np.cos(lean[i])])
                            for i, p in enumerate(curved_pts)])
    neutral_check = {
        "note": "spine.py's neutral stand (pitch = flex = 0) puts all four anchors on ONE vertical line at "
                "z = sensor_offset_m. Anny's midline is curved (lordosis/kyphosis), so the sim's neutral anchors differ "
                "from sensor_positions by the residual below. Neither Worker kwarg can fix this; the curved-neutral "
                "alternative needs a neutral pose offset in the sim's PoseTrack (calibration stand), not a Worker field.",
        "straight_neutral_anchor_minus_sensor_positions_m": {k: r(straight[i] - sens_arr[i]) for i, k in enumerate(LEVELS)},
        "curved_neutral_alternative": {
            "definition": "PoseTrack at stand: pitch = lean_0, flex[j] = lean_{j+1} - lean_j (sagittal only), with "
                          "lean_i = forward lean of Anny midline chord i (S1-L3, L3-T10, T10-T2, T2-C7) from vertical, "
                          "+ = top towards -Z. Use together with neutral_curvature_deg below (pod normal pitch minus "
                          "lean_i) and the same spacing_m: anchors then lie on Anny's midline points + 0.04 m along "
                          "each segment's own posterior axis, and chip z axes equal the pod normals.",
            "pitch_rad": float(lean[0]), "flex_rad": [float(lean[i + 1] - lean[i]) for i in range(3)],
            "segment_lean_deg": [float(np.degrees(a)) for a in lean],
            "neutral_curvature_deg": curv_curved,
            "anchor_minus_sensor_positions_m": {k: r(curved_sens[i] - sens_arr[i]) for i, k in enumerate(LEVELS)},
        },
    }

    doc = {
        "format": FORMAT,
        "generator": {"script": "anny_worker/anny_body.py", "anny_version": anny.__version__,
                      "torch": torch.__version__, "args": {k: v for k, v in vars(args).items() if k != "out"},
                      "deterministic": "no timestamps or timings are written; reruns give byte-identical files"},
        "frame": {
            "name": "thundersim spine.py frame",
            "units": "metres",
            "axes": "+X wearer-right, +Y up, +Z posterior; the body faces -Z (same as thundersim/spine.py)",
            "origin": "y = 0 at the lowest rest vertex (soles); x = 0 on Anny's mid-sagittal plane; z = 0 at the S1 "
                      "spinal-midline point, so the sim pelvis anchor root_pos = (0, pelvis_height, 0)",
            "from_anny_native": {"M": M_ANNY_TO_SIM.tolist(), "t": r(t, 6),
                                 "rule": "p_sim = M p_anny + t; Anny native = Z up, faces -Y, left +X. "
                                         "M = Ry(180) * Rx(-90) (extract.py's Y-up map then turn to face -Z)"},
            "pose": "Anny rest (bind) pose: MakeHuman A-pose, standing, arms ~45 deg down-out",
        },
        "phenotype": {"values": phen, "requested": p,
                      "note": "Anny phenotype parameters in [0,1]; gender 0 = male anchor, 1 = female anchor. age from "
                              "years via Anny's MorphologicalAgeMapping; height param solved by bisection on rest "
                              "crown-to-sole height when height_m is given; values rounded to 4 decimals."},
        "height_m": height_m,
        "total_mass_kg": total_mass,
        "mass_method": {"density_kg_m3": DENSITY,
                        "density_source": "anny/anthropometry.py Anthropometry.mass(): volume x 980 kg/m^3 "
                                          "('for simplicity'); matches the extract.py / anny_avg_male.json value",
                        "volume_m3_whole_mesh": vol_exact_all,
                        "volume_m3_body_component": vol_body,
                        "note": "whole closed mesh (body + eyeballs + tongue, divergence theorem). Whole-body density "
                                "of living adults is ~1010-1070 kg/m^3 depending on fat and lung air, so the true mass "
                                "for this volume could be 3-9% higher; kept at Anny's 980 for consistency."},
        "bmi": meas["bmi"],
        "bone_heads": {n: r(Hb[i]) for i, n in enumerate(labels)},
        "spinal_midline": {
            "definition": "Anny rig spinal column: polyline through the bone heads " + " -> ".join(MIDLINE_CHAIN) +
                          " (rest pose), linearly interpolated at each level height; x = 0",
            "polyline": r(chain),
            "skin_depth_note": "Anny's spine bones lie 3-5 cm anterior to the back skin, i.e. close to the "
                               "vertebral canal / lamina depth, so the sim's 'spine line' maps onto them directly.",
        },
        "spinous_anchors": {
            "method": "anny_lift_spec.md section 5 / D7: level height y = hip_y + (y_tip_BP3D - 0.8915) * s, "
                      "s = (C7_y - hip_y) / (1.457 - 0.8915); hip_y = mean upperleg01.L/R head height, C7_y = neck01 "
                      "head height; horizontal +Z (posterior) ray at x = 0, farthest hit = back skin; normal = mean "
                      "area-weighted vertex normal over the 34 x 44 mm pod footprint (outward)",
            "bp3d_tip_heights_m": BP3D_TIP_Y, "hip_y_m": hip_y, "c7_y_m": c7_y, "spinous_scale": scale,
            "levels": {k: r({"y": lev_y[k], **anchors[k]}) for k in LEVELS},
        },
        "sensor_positions": {
            "definition": "sensor_k = midline(y_k) + 0.04 m * (0, 0, +1): 0.04 m behind the spinal midline, "
                          "measured horizontally along the sim's posterior axis +Z in the standing rest pose "
                          "(in the sim's neutral stand every segment frame equals world, so the OFFSET equals "
                          "spine.py's seg_i.apply([0, 0, sensor_offset_m]); the BASE points differ because the sim's "
                          "neutral spine line is straight: see the top-level sim_neutral_check)",
            "offset_m": SENSOR_BEHIND_MIDLINE,
            "positions": {k: r(sensors[k]) for k in LEVELS},
            "gap_to_skin_m": {k: r(anchors[k]["sensor_gap_to_skin_along_posterior_m"]) for k in LEVELS},
            "signed_distance_to_skin_m": {k: r(anchors[k]["sensor_signed_distance_to_skin_m"]) for k in LEVELS},
            "gap_sign": "+ = outside (behind) the skin, - = inside the body",
            "pod_centre_note": "a physical 12 mm pod with a 2 mm stand-off has its centre 8 mm off the skin along "
                               "the normal (spinous_anchors.levels.*.pod_centre)",
        },
        "thunder_links": {
            "definition": "thundersim spine.py: spine_line[i+1] = spine_line[i] + spacing[i] * seg_i up-axis, "
                          "spine_line[0] = root_pos (pelvis anchor). Link i joins the spinal-midline points at "
                          "levels (S1, L3, T10, T2)[i] and [i+1]; length_m = straight midline chord. "
                          "Upper thoracic has no length in the sim (only its sensor); T2 -> C7 is reported for info.",
            **r(links),
            "sim_default_spacing_m_for_this_height": r(sim_default_spacing),
            "anny_minus_sim_default_m": r([s - d for s, d in zip(spacing, sim_default_spacing)]),
        },
        "region_masses": {
            "method": f"interior of the closed body component sampled on a {h * 1000:.0f} mm voxel grid (scan-line "
                      f"parity); each voxel takes the inverse-distance blend of the full skin weights of its {args.knn} "
                      "nearest body vertices; bone weights summed into regions; masses = region volume fraction x "
                      "total_mass_kg (uniform density). Bone -> region: trunk_pelvis = root, pelvis.L/R, spine05; "
                      "trunk_lumbar = spine04, spine03; trunk_thoracic = spine02; trunk_upper_thoracic = spine01, "
                      "clavicles, shoulder01s; head_neck = neck01-03, head, eyes; upper_arm = upperarm01-02; forearm = "
                      "lowerarm01-02; hand = wrist, metacarpals, fingers; thigh = upperleg01-02; shank = lowerleg01-02; "
                      "foot = foot, toes.",
            "voxel_volume_m3": vox_vol, "voxel_vs_exact_body_volume": vox_vol / vol_body,
            "odd_parity_columns_skipped": odd, "voxels": int(len(P)),
            "kg": reg_mass,
            "trunk_kg_by_skin_weights": {"trunk_" + s: reg_mass["trunk_" + s] for s in TRUNK_BONES},
            "trunk_kg_by_planes_at_sim_link_levels": {
                "definition": "trunk-weighted volume cut by horizontal planes at the L3, T10, T2 level heights: "
                              "pelvis < L3 <= lumbar < T10 <= thoracic < T2 <= upper_thoracic",
                **plane_split},
            "whole_body_com": com,
            "de_leva_1996_check": {
                "source": "de Leva P. Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. J Biomech "
                          "1996;29(9):1223-1230, Table 4 (male/female, % body mass); linearly interpolated by Anny "
                          f"gender = {g}",
                "fractions_pct": dl,
                "comparison": comparison,
                "caveat": "uniform density and skin-weight boundaries (not de Leva's joint-centre planes) shift mass "
                          "from the limbs to the trunk: de Leva limb masses come from denser tissue than the trunk "
                          "(which holds the lungs). Treat de Leva as the reference for per-segment masses.",
            },
        },
        "worker_fields": r(worker_fields),
        "worker_fields_sources": {
            "height_m": "Anny rest crown-to-sole height (anthropometry.height)",
            "mass_kg": "Anny total_mass_kg (body only; the sim adds load_kg)",
            "load_kg": "NOT DERIVED: sim default (tool belt / harness / PPE)",
            "spacing_m": "Anny spinal-midline chords S1-L3, L3-T10, T10-T2 (thunder_links.*.length_m)",
            "neutral_curvature_deg": "Anny pod normal pitch at S1/L3/T10/T2: c = atan2(n_y, n_z) in the sim frame, "
                                     "so that spine.py's chip = seg * Rx(-c) has its z axis on the skin normal",
            "mount_error_deg": "NOT DERIVED: sim default (strap misalignment)",
            "sensor_offset_m": "kept at the sim default 0.04 m so that sensor_positions match spine.py; the "
                               "Anny-measured alternatives are in worker_fields_alternatives",
        },
        "worker_fields_not_derived": ["load_kg", "mount_error_deg", "sensor_offset_m (kept at 0.04 by design)"],
        "worker_fields_alternatives": {
            "sensor_offset_m_midline_to_skin_per_level": r([anchors[k]["midline_to_skin_depth_m"] for k in LEVELS]),
            "sensor_offset_m_midline_to_skin_mean": r(float(np.mean([anchors[k]["midline_to_skin_depth_m"] for k in LEVELS]))),
            "sensor_offset_m_midline_to_pod_centre_mean": r(float(np.mean([anchors[k]["midline_to_pod_centre_horizontal_m"] for k in LEVELS]))),
            "spacing_m_skin_chord": r([links[n]["skin_chord_m"] for n in names]),
            "spacing_m_skin_arc_tape": r([links[n]["skin_arc_m"] for n in names]),
            "spacing_m_vertical": r([links[n]["vertical_m"] for n in names]),
            "spacing_note": "spine.py stacks the links on one straight neutral axis, so with the midline chords its "
                            "neutral anchor heights rise above Anny's by the cumulative chord-minus-vertical (a few "
                            "mm at T2); spacing_m_vertical reproduces Anny's level heights exactly instead. Worker's "
                            "own comment calls spacings 'surface distances' (tape): spacing_m_skin_arc_tape is that "
                            "reading; it differs from the midline chords by <= 5 mm (avg_male +2.1/-0.4/+2.2, female_160 +4.1/-0.1/+2.4 mm)",
        },
        "sim_neutral_check": neutral_check,
        "worker_properties_not_settable": {
            "note": "computed properties in spine.py / falls.py that a Worker(**kwargs) cannot override",
            "pelvis_height": {"sim_formula": "0.53 * height_m", "sim_value": r(0.53 * height_m),
                              "anny_value_S1_midline_y": r(lev_y["S1"]), "anny_fraction_of_height": r(lev_y["S1"] / height_m)},
            "falls_r_p_com_to_pelvis_anchor": {"sim_formula": "(0, -0.03 H, 0.10)", "sim_value": r([0.0, -0.03 * height_m, 0.10]),
                                               "anny_com_to_S1_midline": r(com_to_s1),
                                               "anny_com_to_S1_sensor": r(sensors["S1"] - com)},
            "total_mass": "mass_kg + load_kg (unchanged)",
        },
        "worker_defaults_reference": WORKER_DEFAULTS,
        "licence": {
            "anny": "Anny code (C) 2025 NAVER Corp., Apache-2.0 (https://github.com/naver/anny)",
            "assets": "body mesh, blend shapes, rig and skin weights derived from MakeHuman / MPFB2 assets, CC0 1.0",
            "bp3d": "spinous-tip heights from BodyParts3D, (c) The Database Center for Life Science, CC BY 4.0 "
                    "(attribution required)",
            "de_leva": "segment fractions cited from de Leva 1996 (facts, cited)",
        },
    }
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), help="start from a named phenotype; flags override it")
    ap.add_argument("--gender", type=float, help="0 = male anchor, 1 = female anchor")
    ap.add_argument("--age", type=float, help="Anny age parameter [0,1]")
    ap.add_argument("--age-years", type=float, help="age in years (Anny MorphologicalAgeMapping)")
    ap.add_argument("--muscle", type=float)
    ap.add_argument("--weight", type=float)
    ap.add_argument("--proportions", type=float)
    ap.add_argument("--height", type=float, help="Anny height parameter [0,1]")
    ap.add_argument("--height-m", type=float, help="target crown-to-sole height in metres (bisection)")
    ap.add_argument("--voxel", type=float, default=0.005, help="voxel size for the mass partition (m)")
    ap.add_argument("--knn", type=int, default=8, help="skin vertices blended per voxel")
    ap.add_argument("--out", default=os.path.join(HERE, "anny_worker_body.json"))
    args = ap.parse_args()
    T0 = time.time()
    if not args.preset and args.height is None and args.height_m is None:
        ap.error("give --preset, --height or --height-m")
    doc = build(args)
    with open(args.out, "w") as f:
        json.dump(r(doc), f, indent=1)
    w = doc["worker_fields"]
    print(f"[anny_body] {time.time() - T0:.1f}s {args.out}: H {doc['height_m']:.4f} m, M {doc['total_mass_kg']:.2f} kg, "
          f"spacing {np.round(w['spacing_m'], 4).tolist()}, curvature {np.round(w['neutral_curvature_deg'], 1).tolist()}")


if __name__ == "__main__":
    sys.exit(main())
