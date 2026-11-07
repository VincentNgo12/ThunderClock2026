"""Turn public trunk-IMU recordings into ThunderClock-shaped 4-anchor recordings.

    .venv/bin/python -I scripts/harmonize.py

Input : data/interim/raw_trials.pkl  (from load_raw.py; sensor frame, SI units)
Output: data/processed/canonical_target.npz, canonical_current.npz, meta.csv

Steps per recording (all in one direction: source -> device, never the reverse):
 1. Sensor-to-segment calibration (per subject, per sensor), data-driven:
      +Y (up)        = mean accelerometer direction while standing still
      +Z (posterior) = direction gravity tilts towards during a forward bend
                       (forward flexion makes world-up lean towards segment +Z)
      +X (right)     = Y x Z
    VTT hip sensor only: its bend direction is ambiguous (squats tilt the pelvis
    either way), so its heading is set from the magnetometers so that hip and back
    agree on magnetic north while walking.
 2. Resample to 100 Hz (Erciyes is 25 Hz -> 4x polyphase; true bandwidth stays 12.5 Hz).
 3. Erciyes chest (sternum) -> upper-thoracic anchor: rigid-body lever arm
      a_T = a_chest + alpha x r + omega x (omega x r),  r = (0, 0.08, 0.20) m  (estimate)
 4. 6-axis Mahony fusion per real anchor (same filter for every source) -> segment
    orientation in a Y-up world with arbitrary but shared heading.
 5. Synthesise LUMBAR (f=0.30) and THORACIC (f=0.65) between pelvis (f=0) and
    upper thoracic (f=1):
      orientation   q_f = q_p * slerp(I, q_p^-1 q_ut, f)
      accel (world) a_f = (1-f) a_p + f a_ut    (exact for a rigid trunk: rigid-body
                                                 acceleration is affine in position)
      gyro  (world) w_f = (1-f) w_p + f w_ut    (approximation; exact when the trunk
                                                 does not bend while rotating)
    then rotate a_f, w_f into the synthetic segment frame.
 6. MPU-6050 emulation per device profile: bias + white noise (datasheet densities),
    clip to full scale, causal 2nd-order low-pass at the DLPF corner, clip, quantise.

Anchors 0 (pelvis) and 3 (upper thoracic) are REAL measurements; 1 and 2 are SYNTHETIC.
"""
import csv
import os
import pickle
import sys
import zlib

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, resample_poly
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # -I drops the script dir from sys.path; add our own package root
from falldet.constants import DEVICE_PROFILES, FS_HZ, G  # noqa: E402

FRACS = {"lumbar": 0.30, "thoracic": 0.65}  # fractional position S1 (0) -> T1-T3 (1); estimates
CHEST_TO_UT_M = np.array([0.0, 0.08, 0.20])  # sternum sensor -> upper-thoracic spine, segment frame
VTT_TRIM_S = 3.0  # VTT files start with two synchronisation jumps; drop them
MAHONY_KP = 2.0
MAHONY_GATE_G = 0.2


def nz(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ------------------------------------------------------------------ 1. calibration
def quiet(acc, gyr):
    return (np.linalg.norm(gyr, axis=1) < 0.3) & (np.abs(np.linalg.norm(acc, axis=1) - G) < 0.5)


def up_axis_from_starts(trials, key, n_start):
    """Median accelerometer direction over the first quiet samples of each upright-start trial."""
    vs = []
    for t in trials:
        a, g = t["acc_" + key][:n_start], t["gyr_" + key][:n_start]
        q = quiet(a, g)
        if q.sum() >= n_start // 2:
            vs.append(nz(a[q].mean(0)))
    return nz(np.median(np.array(vs), axis=0)), len(vs)


def posterior_from_bends(trials, key, u, trim=0):
    """Mean direction gravity tilts towards (sensor frame) during forward bends."""
    ps = []
    for t in trials:
        a, g = t["acc_" + key][trim:], t["gyr_" + key][trim:]
        an = nz(a)
        ang = np.degrees(np.arccos(np.clip(an @ u, -1, 1)))
        m = (ang > 30) & (ang < 120) & (np.linalg.norm(g, axis=1) < 0.6) \
            & (np.abs(np.linalg.norm(a, axis=1) - G) < 1.5)
        if m.sum():
            ps.append(an[m] - np.outer(an[m] @ u, u))
    p = np.concatenate(ps).mean(0)
    p = nz(p - (p @ u) * u)
    return p


def frame_matrix(u, p):
    """Rows = segment axes (X right, Y up, Z posterior) in sensor coordinates: v_seg = M @ v_sensor."""
    x = nz(np.cross(u, p))
    return np.stack([x, u, p])


def heading_from_mag(walk_trials, u_p, M_ut, trim):
    """Choose the hip sensor's posterior axis (perpendicular to u_p) so that hip and back
    agree on the heading of magnetic north, averaged over walking."""
    e1 = nz(np.cross(u_p, [1.0, 0, 0]) if abs(u_p[0]) < 0.9 else np.cross(u_p, [0, 1.0, 0]))
    e2 = np.cross(u_p, e1)
    best = None
    for gam in np.radians(np.arange(0, 360, 2)):
        z = np.cos(gam) * e1 + np.sin(gam) * e2
        Mp = frame_matrix(u_p, z)
        err = []
        for t in walk_trials:
            mp = t["mag_p"][trim:] @ Mp.T
            mu = t["mag_ut"][trim:] @ M_ut.T
            # heading of north in each segment frame: atan2(m.x, m.z)
            d = np.arctan2(mp[:, 0], mp[:, 2]) - np.arctan2(mu[:, 0], mu[:, 2])
            err.append(np.exp(1j * d))
        c = np.mean(np.concatenate(err))
        score = np.real(c)  # 1 = perfect agreement
        if best is None or score > best[0]:
            best = (score, Mp, np.abs(c))
    return best[1], best[0], best[2]


def calibrate(trials):
    """Return {(subject, 'p'|'ut'): M} and a list of calibration notes."""
    cal, notes = {}, []
    by_subj = {}
    for t in trials:
        by_subj.setdefault(t["subject"], []).append(t)
    for subj, ts in sorted(by_subj.items()):
        if ts[0]["dataset"] == "erciyes":
            upright = [t for t in ts if t["code"] not in (815, 816, 917)]
            bends = [t for t in ts if t["code"] in (805, 806)]
            for key in ("p", "ut"):
                u, n = up_axis_from_starts(upright, key, n_start=12)  # first 0.5 s at 25 Hz
                p = posterior_from_bends(bends, key, u)
                cal[(subj, key)] = frame_matrix(u, p)
        else:
            trim = int(VTT_TRIM_S * 100)
            walks = [t for t in ts if t["code"] in (1013, 1014)]
            u_ut = nz(np.mean([nz(t["acc_ut"][trim:].mean(0)) for t in walks], 0))
            u_p = nz(np.mean([nz(t["acc_p"][trim:].mean(0)) for t in walks], 0))
            bends = [t for t in ts if t["code"] in (1005, 1011, 1012)]
            M_ut = frame_matrix(u_ut, posterior_from_bends(bends, "ut", u_ut, trim))
            M_p, score, conc = heading_from_mag(walks, u_p, M_ut, trim)
            cal[(subj, "ut")], cal[(subj, "p")] = M_ut, M_p
            notes.append(f"{subj}: hip heading from magnetometer, agreement cos={score:.2f}, "
                         f"concentration={conc:.2f}")
    return cal, notes


# ------------------------------------------------------------------ quaternion helpers (w,x,y,z)
def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qrot_inv(q, v):
    """Rotate world vector v into the body frame of q (q maps body -> world)."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    qc = q * np.array([1, -1, -1, -1])
    return qmul(qmul(qc, qv), q)[..., 1:]


def shortest_arc(a, b):
    """Unit quaternion rotating unit vector a onto unit vector b (batched)."""
    w = 1.0 + np.sum(a * b, axis=-1, keepdims=True)
    xyz = np.cross(a, b)
    return nz(np.concatenate([w, xyz], axis=-1))


def mahony_batch(acc, gyr, fs):
    """Lock-step 6-axis Mahony filter. acc, gyr: (K, N, 3) segment frame. Returns q (K, N, 4)."""
    K, N, _ = acc.shape
    dt = 1.0 / fs
    yhat = np.array([0.0, 1.0, 0.0])
    q = shortest_arc(nz(acc[:, : int(0.25 * fs)].mean(1)), np.broadcast_to(yhat, (K, 3)))
    out = np.empty((K, N, 4))
    for i in range(N):
        a, w = acc[:, i], gyr[:, i]
        n = np.linalg.norm(a, axis=1, keepdims=True)
        v = qrot_inv(q, np.broadcast_to(yhat, (K, 3)))     # predicted up, body frame
        e = np.cross(a / np.maximum(n, 1e-9), v)
        e *= (np.abs(n - G) < MAHONY_GATE_G * G)
        wc = w + MAHONY_KP * e
        q = q + 0.5 * qmul(q, np.concatenate([np.zeros((K, 1)), wc], axis=1)) * dt
        q = nz(q)
        out[:, i] = q
    return out


# ------------------------------------------------------------------ 2-5 per-recording geometry
def to_100hz(x, fs):
    if fs == FS_HZ:
        return x
    up = int(round(FS_HZ / fs))
    return resample_poly(x, up, 1, axis=0)


def lever_arm(acc, gyr, r, fs):
    alpha = np.gradient(gyr, 1.0 / fs, axis=0)
    return acc + np.cross(alpha, r) + np.cross(gyr, np.cross(gyr, r))


def segment_streams(t, cal):
    """Calibrated 100 Hz segment-frame acc/gyro for the two real anchors."""
    trim = int(VTT_TRIM_S * t["fs"]) if t["dataset"] == "vtt" else 0
    out = {}
    for key in ("p", "ut"):
        M = cal[(t["subject"], key)]
        a = to_100hz(t["acc_" + key][trim:], t["fs"]) @ M.T
        g = to_100hz(t["gyr_" + key][trim:], t["fs"]) @ M.T
        if key == "ut" and t["dataset"] == "erciyes":
            a = lever_arm(a, g, CHEST_TO_UT_M, FS_HZ)
        out["acc_" + key], out["gyr_" + key] = a, g
    return out


def synthesize(acc_p, gyr_p, acc_ut, gyr_ut, q_p, q_ut):
    """Build the 4-anchor arrays (N, 4, 3) from the two real anchors and their orientations."""
    Rp = Rotation.from_quat(q_p, scalar_first=True)
    Ru = Rotation.from_quat(q_ut, scalar_first=True)
    rel = Rp.inv() * Ru
    a_wp, a_wu = Rp.apply(acc_p), Ru.apply(acc_ut)
    w_wp, w_wu = Rp.apply(gyr_p), Ru.apply(gyr_ut)
    acc = [acc_p]
    gyr = [gyr_p]
    for f in (FRACS["lumbar"], FRACS["thoracic"]):
        Rf = Rp * (rel ** f)
        acc.append(Rf.inv().apply((1 - f) * a_wp + f * a_wu))
        gyr.append(Rf.inv().apply((1 - f) * w_wp + f * w_wu))
    acc.append(acc_ut)
    gyr.append(gyr_ut)
    return np.stack(acc, axis=1), np.stack(gyr, axis=1)


# ------------------------------------------------------------------ 6. device emulation
ACC_NOISE_UG_RTHZ = 400.0     # MPU-6050 datasheet, accel noise density
GYRO_NOISE_DPS_RTHZ = 0.005   # MPU-6050 datasheet, gyro rate noise density
ACC_BIAS_SD_G = 0.02          # residual after a simple calibration (uncalibrated: +-50-80 mg)
GYRO_BIAS_SD_DPS = 0.5        # residual after the start-up gyro-bias capture


def emulate_device(acc, gyr, profile, seed):
    """acc, gyr: (N, 4, 3) clean segment-frame signals -> what an MPU-6050 with this profile outputs."""
    p = DEVICE_PROFILES[profile]
    rng = np.random.default_rng(seed)
    bw = 1.57 * min(p["dlpf_hz"], p["fs_hz"] / 2)  # equivalent noise bandwidth of a 2nd-order LPF
    a_rng, g_rng = p["acc_range_g"] * G, np.radians(p["gyro_range_dps"])
    out = []
    for x, rng_fs, bias_sd, noise_sd in (
            (acc, a_rng, ACC_BIAS_SD_G * G, ACC_NOISE_UG_RTHZ * 1e-6 * G * np.sqrt(bw)),
            (gyr, g_rng, np.radians(GYRO_BIAS_SD_DPS), np.radians(GYRO_NOISE_DPS_RTHZ) * np.sqrt(bw))):
        y = x + rng.normal(0, bias_sd, size=(1,) + x.shape[1:]) + rng.normal(0, noise_sd, size=x.shape)
        y = np.clip(y, -rng_fs, rng_fs)                       # ADC saturates first
        if p["dlpf_hz"] < p["fs_hz"] / 2:
            b, a = butter(2, p["dlpf_hz"], fs=p["fs_hz"])
            zi = lfilter_zi(b, a)[:, None, None] * y[0][None]
            y, _ = lfilter(b, a, y, axis=0, zi=zi)            # causal, like the on-chip DLPF
        lsb = rng_fs / 32768.0
        y = np.clip(np.round(y / lsb) * lsb, -rng_fs, rng_fs - lsb)
        out.append(y.astype(np.float32))
    return out


# ------------------------------------------------------------------ driver
def main():
    with open(os.path.join(ROOT, "data/interim/raw_trials.pkl"), "rb") as f:
        trials = pickle.load(f)
    cal, notes = calibrate(trials)
    for n in notes:
        print(n)

    # geometry (independent of device profile), processed in length-sorted chunks
    streams = [segment_streams(t, cal) for t in trials]
    lengths = np.array([len(s["acc_p"]) for s in streams])
    order = np.argsort(lengths)
    clean = [None] * len(trials)
    for c0 in range(0, len(order), 400):
        idx = order[c0:c0 + 400]
        Nmax = lengths[idx].max()

        def pad(key):
            arr = np.zeros((len(idx), Nmax, 3))
            for j, i in enumerate(idx):
                x = streams[i][key]
                arr[j, :len(x)] = x
                arr[j, len(x):] = x[-1]
            return arr
        acc2 = np.concatenate([pad("acc_p"), pad("acc_ut")])
        gyr2 = np.concatenate([pad("gyr_p"), pad("gyr_ut")])
        q = mahony_batch(acc2, gyr2, FS_HZ)
        for j, i in enumerate(idx):
            n = lengths[i]
            s = streams[i]
            clean[i] = synthesize(s["acc_p"], s["gyr_p"], s["acc_ut"], s["gyr_ut"],
                                  q[j, :n], q[len(idx) + j, :n])
        print(f"fused {min(c0 + 400, len(order))}/{len(order)}", file=sys.stderr)

    # sanity check of the calibration on Erciyes end postures (body-frame up direction)
    checks = {"prone (901,902) expect +Z": (901, 902), "supine (910) expect -Z": (910,),
              "right side (913) expect -X": (913,), "left side (915) expect +X": (915,)}
    for name, codes in checks.items():
        v = [nz(clean[i][0][-50:].mean(0)) for i, t in enumerate(trials)
             if t["dataset"] == "erciyes" and t["code"] in codes]
        v = np.array(v)  # (n, 4, 3)
        print(f"calibration check {name}: pelvis {v[:, 0].mean(0).round(2)}  "
              f"upper_thoracic {v[:, 3].mean(0).round(2)}")

    os.makedirs(os.path.join(ROOT, "data/processed"), exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(c[0]) for c in clean])])
    with open(os.path.join(ROOT, "data/processed/meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rec", "dataset", "subject", "code", "activity", "trial", "label",
                    "start", "stop", "duration_s", "source_fs_hz", "pelvis_src", "ut_src"])
        for i, t in enumerate(trials):
            w.writerow([i, t["dataset"], t["subject"], t["code"], t["activity"], t["trial"],
                        t["label"], offsets[i], offsets[i + 1],
                        f"{(offsets[i + 1] - offsets[i]) / FS_HZ:.2f}", t["fs"],
                        t["pelvis_src"], t["ut_src"]])
    for profile in DEVICE_PROFILES:
        A, W = [], []
        for i, t in enumerate(trials):
            seed = zlib.crc32(f"{t['subject']}|{t['code']}|{t['trial']}|{profile}".encode())
            a, g = emulate_device(clean[i][0], clean[i][1], profile, seed)
            A.append(a)
            W.append(g)
        np.savez_compressed(os.path.join(ROOT, f"data/processed/canonical_{profile}.npz"),
                            acc=np.concatenate(A), gyro=np.concatenate(W), offsets=offsets,
                            anchor_is_real=np.array([True, False, False, True]))
        print("wrote canonical", profile)


if __name__ == "__main__":
    main()
