# File: recording_io.py
# Purpose: Read "Collect data" recordings (recording.py format) and derive what the segment model needs:
#          per-hold world-up vectors per sensor and a mounting fitted from the stand + bend holds.
# Contributors: Unassigned
import json

from .models import Imu, REGIONS, Sample
from .segments import SegmentModel, fit_mounting, mean_vector, world_up_in_sensor

HOLD_TRIM_START_S = 0.7     # skip the settling at the start of a hold
HOLD_TRIM_END_S = 0.2
BENDS = ("forward", "backward", "right", "left")


class Recording:
    def __init__(self, meta, labels, samples, end):
        self.meta, self.labels, self.samples, self.end = meta, labels, samples, end

    def label_at(self, t):
        current = None
        for start, name in self.labels:
            if start > t:
                break
            current = name
        return current

    def windows(self):
        """[(name, t0, t1)] for every label, t1 = next label start."""
        times = [t for t, _ in self.labels[1:]] + [float("inf")]
        return [(name, t, t1) for (t, name), t1 in zip(self.labels, times)]

    def quaternion_samples(self):
        return [(t, s) for t, origin, s in self.samples if s.format == "quaternion" and origin == "device"]


def read_recording(path):
    meta, end, labels, samples = None, None, [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            kind = row["type"]
            if kind == "meta":
                meta = row
            elif kind == "end":
                end = row
            elif kind == "label":
                labels.append((row["t"], row["name"]))
            elif kind == "sample":
                imus = tuple(Imu(i["id"], accel=tuple(i["a"]) if i["a"] else None, gyro=tuple(i["g"]) if i["g"] else None,
                                 quaternion=tuple(i["q"]) if i["q"] else None, valid=i["valid"])
                             for i in sorted(row["imus"], key=lambda imu: imu["id"]))
                samples.append((row["t"], row["origin"], Sample(row["format"], row["sample_time_ms"], imus, row["sequence"])))
    if meta is None:
        raise ValueError(f"{path}: no meta line; not a recording")
    return Recording(meta, labels, samples, end)


def hold_up_vectors(recording):
    """{hold label: {region: mean world-up in that sensor's axes}} over each ':hold' window."""
    quats = recording.quaternion_samples()
    out = {}
    for name, t0, t1 in recording.windows():
        if not name.endswith(":hold"):
            continue
        per_region = {region: [] for region in REGIONS.values()}
        for t, sample in quats:
            if t0 + HOLD_TRIM_START_S <= t < t1 - HOLD_TRIM_END_S:
                for imu in sample.imus:
                    if imu.valid and imu.quaternion is not None:
                        per_region[REGIONS[imu.sensor_id]].append(world_up_in_sensor(imu.quaternion))
        out[name] = {region: mean_vector(vs) for region, vs in per_region.items() if vs}
    return out


def fit_model(holds, run="1", stand="stand#1:hold", source="recording"):
    """SegmentModel from a stand hold and the bend holds of one run ("forward#1:hold", ...)."""
    if stand not in holds:
        raise ValueError(f"no {stand} in the recording")
    mountings, report = {}, {}
    for region in REGIONS.values():
        bends = {b: holds[f"{b}#{run}:hold"][region] for b in BENDS
                 if f"{b}#{run}:hold" in holds and region in holds[f"{b}#{run}:hold"]}
        if region not in holds[stand] or not bends:
            continue
        mountings[region], report[region] = fit_mounting(holds[stand][region], bends)
    if not mountings:
        raise ValueError("no bend steps (forward/backward/left/right) to fit a mounting from")
    return SegmentModel(mountings, source, report)
