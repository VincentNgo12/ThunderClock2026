"""Smoke test of falldet.sim_adapter.SpineFallModel with FrameBatch-shaped input.

    .venv/bin/python tests/test_sim_adapter.py

Uses ThunderSim's own FrameBatch / StreamContext / load_model when the simulator package is
importable (read-only import, no bytecode written), otherwise identical stand-in dataclasses.
Covers: 100 Hz with jitter, lost packets, NaN and a missing slot; 0.5 Hz "current" firmware;
a synthetic harness arrest (hangs upright); lying_work suppression; a walking hard negative.
"""
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(os.path.dirname(ROOT), "simulation")
sys.path.insert(0, ROOT)
if os.path.isdir(SIM):
    sys.path.append(SIM)
try:
    from thundersim.model_api import FallModel, FrameBatch, StreamContext, load_model  # type: ignore
    HAVE_SIM = True
except Exception:
    HAVE_SIM = False

    @dataclass(frozen=True)
    class StreamContext:
        fs_hz: float
        profile: str
        accel_range_g: float
        gyro_range_dps: float
        seed: int = 0

    @dataclass(frozen=True)
    class FrameBatch:
        t: np.ndarray
        acc: np.ndarray
        gyro: np.ndarray
        slot_mask: np.ndarray
        sat_mask: np.ndarray
        fs_hz: float
        work_mode: str = "normal"

from falldet import fixture_arrays, load_fixture  # noqa: E402
from falldet.sim_adapter import SpineFallModel  # noqa: E402

G = 9.80665
PROFILES = {"target": (100.0, 16.0, 2000.0), "current": (0.5, 2.0, 250.0),
            "current_100hz": (100.0, 2.0, 250.0), "target_8g": (100.0, 8.0, 2000.0)}


def make_model(profile):
    m = load_model("falldet.sim_adapter:SpineFallModel") if HAVE_SIM else SpineFallModel()
    if HAVE_SIM:
        assert isinstance(m, FallModel)
    fs, ar, gr = PROFILES[profile]
    m.reset(StreamContext(fs_hz=fs, profile=profile, accel_range_g=ar, gyro_range_dps=gr))
    return m


def feed(model, t, acc, gyro, fs, batch_s=0.5, slot_mask=None, work_mode="normal", t0=1000.0):
    """Chop into FrameBatch chunks like the simulator and collect events."""
    n = len(t)
    slot_mask = np.ones((n, 4), bool) if slot_mask is None else slot_mask
    sat = (np.abs(acc) >= 16 * G * 0.999).any(-1)
    events = []
    step = max(1, int(round(batch_s * fs)))
    for a in range(0, n, step):
        b = min(n, a + step)
        fb = FrameBatch(t=t0 + t[a:b], acc=acc[a:b].astype(np.float32), gyro=gyro[a:b].astype(np.float32),
                        slot_mask=slot_mask[a:b], sat_mask=sat[a:b], fs_hz=fs, work_mode=work_mode)
        evs = model.update(fb)
        for e in evs:
            assert e.t <= float(fb.t[-1]) + 1e-6, "event in the future"
        events += evs
    return events


def verdict(events):
    kinds = [e.kind for e in events]
    for k in ("fall_confirmed", "near_fall", "recovered"):
        if k in kinds:
            return k
    return "normal"


def fixture(name, profile="target"):
    _, s = load_fixture(os.path.join(ROOT, "fixtures", profile, name + ".jsonl"))
    return fixture_arrays(s)


def messy_100hz(acc, gyro, rng):
    """Jitter timestamps, drop 5 % of frames in bursts, NaN one slot, remove lumbar for 3 s."""
    n = len(acc)
    t = np.arange(n) / 100.0 + rng.uniform(-0.002, 0.002, n)
    t[0] = 0.0
    keep = np.ones(n, bool)
    for start in rng.choice(n - 20, size=max(1, n // 400), replace=False):
        keep[start:start + rng.integers(2, 15)] = False
    acc, gyro = acc.copy(), gyro.copy()
    acc[rng.choice(n, 30, replace=False), 2, 0] = np.nan
    mask = np.ones((n, 4), bool)
    mask[300:600, 1] = False
    acc[300:600, 1] = np.nan
    return t[keep], acc[keep], gyro[keep], mask[keep]


# ---------------------------------------------------------------- synthetic scenarios
def harness_arrest(fs=100.0):
    """5 s standing, 0.5 s free fall (~1.2 m), 0.3 s arrest at 6 g along +Y, swing, then
    hanging 25 deg forward and still for 30 s."""
    t = np.arange(0, 45, 1 / fs)
    acc = np.zeros((len(t), 4, 3)); gyro = np.zeros_like(acc)
    acc[..., 1] = G
    ff = (t >= 5.0) & (t < 5.5)
    acc[ff] = 0.0
    gyro[(t >= 5.0) & (t < 5.8), :, 0] = 1.5
    arr = (t >= 5.5) & (t < 5.8)
    acc[arr, :, 1] = 6 * G
    tilt = np.radians(25.0)
    after = t >= 5.8
    swing = 0.15 * np.exp(-(t - 5.8) / 4.0) * np.sin(2 * np.pi * (t - 5.8) / 2.4)
    for k in range(4):
        acc[after, k, 1] = G * np.cos(tilt) * (1 + swing[after])
        acc[after, k, 2] = G * np.sin(tilt)
        gyro[after, k, 0] = 0.3 * np.exp(-(t[after] - 5.8) / 4.0) * np.cos(2 * np.pi * (t[after] - 5.8) / 2.4)
    rng = np.random.default_rng(1)
    return t, acc + rng.normal(0, 0.03, acc.shape), gyro + rng.normal(0, 0.005, gyro.shape)


def lying_bump(fs=100.0):
    """Lying supine (working under something) for 10 s, a knock (3 g + rotation), still lying."""
    t = np.arange(0, 30, 1 / fs)
    acc = np.zeros((len(t), 4, 3)); gyro = np.zeros_like(acc)
    acc[..., 2] = -G
    hit = (t >= 10.0) & (t < 10.06)
    acc[hit, :, 2] = -3.2 * G
    gyro[(t >= 9.8) & (t < 10.3), :, 1] = 3.5
    rng = np.random.default_rng(2)
    return t, acc + rng.normal(0, 0.03, acc.shape), gyro + rng.normal(0, 0.005, gyro.shape)


def walking(fs=100.0, secs=60):
    t = np.arange(0, secs, 1 / fs)
    acc = np.zeros((len(t), 4, 3)); gyro = np.zeros_like(acc)
    acc[..., 1] = G * (1 + 0.25 * np.sin(2 * np.pi * 1.8 * t))[:, None]
    acc[..., 2] = 0.15 * G * np.sin(2 * np.pi * 0.9 * t)[:, None]
    gyro[..., 1] = 0.4 * np.sin(2 * np.pi * 0.9 * t)[:, None]
    return t, acc, gyro


def batches(t, acc, gyro, mask, fs, batch_s=0.5):
    step = max(1, int(round(batch_s * fs)))
    for a in range(0, len(t), step):
        yield t[a:a + step], acc[a:a + step], gyro[a:a + step], mask[a:a + step]


def stream_verdict(events):
    st = [e["state"] for e in events if e["type"] == "stage2_result"]
    for k in ("fall_confirmed", "near_fall", "normal"):
        if k in st:
            return k
    return "no decision"


def stream_batch_checks(results, rng):
    """FallDetectorStream.push_batch directly (dict events), FrameBatch-shaped arrays."""
    from falldet import FallDetectorStream
    a, g = fixture("fall_back_lying")
    # 100 Hz, jitter + lost packets + NaN + lumbar missing
    t, a2, g2, mask = messy_100hz(a, g, rng)
    st = FallDetectorStream(profile="target")
    ev = [e for b in batches(t, a2, g2, mask, 100.0) for e in st.push_batch(*b[:3], slot_mask=b[3])]
    results.append(("push_batch 100 Hz messy fall", stream_verdict(ev), "fall_confirmed"))
    # lumbar missing for the whole recording + 2 % NaN everywhere
    a3, g3 = a.copy(), g.copy()
    a3[rng.random(a3.shape[:2]) < 0.02] = np.nan
    mask = np.ones((len(a), 4), bool); mask[:, 1] = False
    st = FallDetectorStream(profile="target")
    ev = [e for b in batches(np.arange(len(a)) / 100.0, a3, g3, mask, 100.0)
          for e in st.push_batch(*b[:3], slot_mask=b[3])]
    frac = [e["slots_missing_frac"] for e in ev if e["type"] == "stage2_result"]
    results.append((f"push_batch lumbar missing all along + NaN (missing frac {frac})",
                    stream_verdict(ev), "fall_confirmed"))
    # 0.5 Hz: one insufficient_rate diagnostic, no detections, no exception
    idx = np.arange(0, len(a), 200)
    st = FallDetectorStream(profile="current")
    ev = [e for b in batches(idx / 100.0, a[idx], g[idx], np.ones((len(idx), 4), bool), 0.5, 2.0)
          for e in st.push_batch(*b[:3], slot_mask=b[3])]
    codes = [e.get("code") for e in ev]
    results.append((f"push_batch 0.5 Hz -> events {codes}",
                    "ok" if codes == ["insufficient_rate"] else str(codes), "ok"))
    # gap handling: free fall, 0.5 s gap, impact -> must NOT trigger; without the gap it does
    for gap, want in ((0.5, 0), (0.0, 1)):
        tt = np.arange(0, 12, 0.01)
        A = np.zeros((len(tt), 4, 3)); A[..., 1] = G
        A[(tt >= 5.0) & (tt < 5.3)] = 0.0
        A[(tt >= 5.3) & (tt < 5.35), :, 1] = 4 * G
        W = np.zeros_like(A)
        tt = tt + np.where(tt >= 5.3, gap, 0.0)
        drop = (tt > 5.3) & (tt < 5.3 + gap) if gap else np.zeros(len(tt), bool)
        st = FallDetectorStream(profile="target")
        ev = [e for b in batches(tt[~drop], A[~drop], W[~drop], np.ones((int((~drop).sum()), 4), bool), 100.0)
              for e in st.push_batch(*b[:3], slot_mask=b[3])]
        n = sum(e["type"] == "stage1_trigger" for e in ev)
        results.append((f"free fall | {gap} s gap | impact: triggers", str(n), str(want)))
    # numpy-only backend
    st = FallDetectorStream(profile="target", backend="logreg_json")
    ev = [e for b in batches(np.arange(len(a)) / 100.0, a, g, np.ones((len(a), 4), bool), 100.0)
          for e in st.push_batch(*b[:3], slot_mask=b[3])] + st.flush()
    results.append((f"backend {st.det.backend} on fall fixture", stream_verdict(ev), "fall_confirmed"))
    # no scikit-learn / joblib installed at all: auto backend must fall back, not crash
    import subprocess
    code = ("import sys; sys.dont_write_bytecode=True\n"
            "class B:\n def find_spec(self, n, p=None, t=None):\n"
            "  if n.split('.')[0] in ('sklearn','joblib'): raise ImportError('blocked '+n)\n"
            "sys.meta_path.insert(0, B()); sys.path.insert(0, %r)\n"
            "import warnings; warnings.simplefilter('ignore')\n"
            "from falldet import FallDetectorStream; import numpy as np\n"
            "s=FallDetectorStream(); print(s.det.backend)" % ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    results.append(("auto backend with sklearn+joblib blocked", (r.stdout.strip() or r.stderr[-200:]), "logreg_json"))


def main():
    rng = np.random.default_rng(0)
    print("ThunderSim importable:", HAVE_SIM)
    results = []

    # 1. 100 Hz, real fall fixture, messy timing + NaN + missing slot
    a, g = fixture("fall_front_lying")
    t, a2, g2, mask = messy_100hz(a, g, rng)
    ev = feed(make_model("target"), t, a2, g2, 100.0, slot_mask=mask)
    results.append(("100 Hz fall, jitter+loss+NaN+missing lumbar", verdict(ev), "fall_confirmed"))

    # 2. 100 Hz ADL fixture (jogging), messy
    a, g = fixture("adl_jogging")
    t, a2, g2, mask = messy_100hz(a, g, rng)
    ev = feed(make_model("target"), t, a2, g2, 100.0, slot_mask=mask)
    results.append(("100 Hz jogging, messy", verdict(ev), "normal|recovered"))

    # 3. 0.5 Hz current firmware: subsample the fall fixture (current profile), must not raise
    a, g = fixture("fall_front_lying", "current")
    idx = np.arange(0, len(a), 200)
    m = make_model("current")
    ev = feed(m, idx / 100.0, a[idx], g[idx], 0.5, batch_s=2.0)
    results.append((f"0.5 Hz current firmware ({m.status})", verdict(ev), "normal"))

    # 3b. 0.5 Hz data announced as 100 Hz (wrong context): measured-rate guard
    m = make_model("target")
    ev = feed(m, idx / 100.0, a[idx], g[idx], 0.5, batch_s=2.0)
    results.append((f"0.5 Hz data, context says 100 Hz ({m.status})", verdict(ev), "normal"))

    # 4. current_100hz profile on the degraded fall fixture
    ev = feed(make_model("current_100hz"), np.arange(len(a)) / 100.0, a, g, 100.0)
    results.append(("current_100hz fall fixture", verdict(ev), "fall_confirmed"))

    # 5. harness arrest: hangs upright and still
    t, a, g = harness_arrest()
    ev = feed(make_model("target"), t, a, g, 100.0)
    rule = [e.detail.get("rule") for e in ev if e.kind != "suspected_fall"]
    results.append((f"harness arrest, hanging upright (rule {rule})", verdict(ev), "fall_confirmed"))

    # 5b. harness arrest, then struggling intermittently (model says near_fall; post-rule confirms)
    t, a, g = harness_arrest()
    r = np.random.default_rng(3)
    m = t >= 8
    on = (np.sin(2 * np.pi * 0.3 * t[m]) > 0)[:, None, None]
    a[m] += r.normal(0, 0.4 * G, a[m].shape) * on
    g[m] += r.normal(0, 1.0, g[m].shape) * on
    ev = feed(make_model("target"), t, a, g, 100.0)
    rule = [e.detail.get("rule") for e in ev if e.kind != "suspected_fall"]
    results.append((f"harness arrest, struggling (rule {rule})", verdict(ev), "fall_confirmed"))

    # 5c. post-rules directly: lying_work suppression and free-fall rule
    from falldet.detector import DEFAULT_POST_RULES, apply_rules
    f = {f"incl_pre_{x}": 85.0 for x in ("pelvis", "lumbar", "thoracic", "upper_thoracic")}
    no_ff = {"free_fall_s": 0.0, "walking_after": False}
    results.append(("apply_rules: lying_work + already lying",
                    apply_rules("fall_confirmed", 0.9, f, no_ff, DEFAULT_POST_RULES, "lying_work")[0], "normal"))
    results.append(("apply_rules: normal mode keeps confirmation",
                    apply_rules("fall_confirmed", 0.9, f, no_ff, DEFAULT_POST_RULES, "normal")[0], "fall_confirmed"))
    results.append(("apply_rules: 0.4 s free fall then walking",
                    apply_rules("near_fall", 0.9, f, {"free_fall_s": 0.4, "walking_after": True},
                                DEFAULT_POST_RULES)[0], "near_fall"))

    # 6. lying work: knock while already lying
    t, a, g = lying_bump()
    v_norm = verdict(feed(make_model("target"), t, a, g, 100.0, work_mode="normal"))
    v_lw = verdict(feed(make_model("target"), t, a, g, 100.0, work_mode="lying_work"))
    results.append((f"knock while lying, work_mode=normal (got {v_norm}) vs lying_work", v_lw, "normal|recovered"))

    # 7. walking 60 s, all frames one slot missing (thoracic) and NaN bursts
    t, a, g = walking()
    mask = np.ones((len(t), 4), bool); mask[:, 2] = False
    ev = feed(make_model("target"), t, a, g, 100.0, slot_mask=mask)
    results.append(("synthetic walking, thoracic slot missing throughout", verdict(ev), "normal|recovered"))

    # 8. everything NaN for a while, then data again: must not raise
    t, a, g = walking(secs=10)
    a[200:500] = np.nan
    feed(make_model("target"), t, a, g, 100.0)
    results.append(("3 s of all-NaN frames", "no exception", "no exception"))

    stream_batch_checks(results, rng)

    fails = 0
    for name, got, want in results:
        ok = got in want.split("|")
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got}, expected {want}")
    assert fails == 0, f"{fails} smoke checks failed"


if __name__ == "__main__":
    main()
