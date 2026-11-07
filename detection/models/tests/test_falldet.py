"""Small self-checks. Run:  .venv/bin/python tests/test_falldet.py   (or pytest if installed)."""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from falldet import FallDetectorStream, detect, fixture_arrays, get_detector, load_fixture, parse_sample  # noqa: E402
from falldet.constants import G  # noqa: E402
from falldet.features import stage2_features  # noqa: E402
from falldet.stage1 import run_stage1  # noqa: E402
from falldet.tilt import inclination_deg, run_tilt  # noqa: E402


def test_tilt_upright_and_prone():
    n = 300
    up = np.zeros((n, 4, 3)); up[..., 1] = G           # upright: +Y up
    prone = np.zeros((n, 4, 3)); prone[..., 2] = G     # face down: world up = segment +Z (posterior)
    z = np.zeros((n, 4, 3))
    assert np.allclose(inclination_deg(run_tilt(up, z))[-1], 0, atol=1)
    assert np.allclose(inclination_deg(run_tilt(prone, z))[-1], 90, atol=1)


def test_tilt_follows_gyro():
    # start upright, rotate about +X at pi/2 rad/s for 1 s with a useless accelerometer
    n = 100
    acc = np.zeros((n + 25, 4, 3)); acc[:25, :, 1] = G       # 0.25 s upright to initialise
    gyro = np.zeros_like(acc); gyro[25:, :, 0] = np.pi / 2
    inc = inclination_deg(run_tilt(acc, gyro))
    assert abs(inc[-1, 0] - 90) < 3, inc[-1]


def synthetic_fall(n_before=300, n_after=900):
    """Standing, 0.3 s free fall, 4 g impact while rotating, then lying still on the back."""
    acc = np.zeros((n_before + 30 + 5 + n_after, 4, 3)); gyro = np.zeros_like(acc)
    acc[:n_before, :, 1] = G
    acc[n_before:n_before + 30, :, 1] = 0.2 * G
    gyro[n_before - 20:n_before + 35, :, 0] = 3.0
    acc[n_before + 30:n_before + 35, :, 2] = -4 * G
    acc[n_before + 35:, :, 2] = -G                     # supine: world up = segment -Z
    return acc, gyro


def test_stage1_fires_on_fall_not_on_standing():
    acc, gyro = synthetic_fall()
    det = get_detector("target")
    assert run_stage1(acc, gyro, det.thresholds), "should trigger"
    still = np.zeros((1000, 4, 3)); still[..., 1] = G
    assert not run_stage1(still, np.zeros_like(still), det.thresholds)


def test_stream_matches_detect_on_fixtures():
    det = get_detector("target")
    for path in sorted(glob.glob(os.path.join(ROOT, "fixtures/target/*.jsonl")))[:4]:
        _, samples = load_fixture(path)
        acc, gyro = fixture_arrays(samples)
        one = detect((acc, gyro), detector=det)
        st = FallDetectorStream(detector=det)
        evs = []
        for s in samples:
            evs += st.push(s)
        evs += st.flush()
        res = [e for e in evs if e["type"] == "stage2_result"]
        if one["stage1_triggered"]:
            a = np.array(list(one["features"].values()))
            b = np.array(list(res[0]["features"].values()))
            assert np.allclose(a, b, atol=1e-6), path
            assert res[0]["state"] == one["state"]
        else:
            assert not res


def test_parse_sample_formats():
    acc = np.arange(12.0).reshape(4, 3); gyro = -acc
    tele = {"time": 5, "data": [{"sensor_id": 4 - k, "accel": dict(zip("xyz", acc[3 - k])),
                                 "gyro": dict(zip("xyz", gyro[3 - k]))} for k in range(4)]}
    for s in (tele, {"t_ms": 5, "acc": acc.tolist(), "gyro": gyro.tolist()}, (acc, gyro)):
        _, a, g = parse_sample(s)
        assert np.allclose(a, acc) and np.allclose(g, gyro)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
