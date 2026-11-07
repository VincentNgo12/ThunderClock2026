"""ThunderSim FallModel that records the canonical FrameBatch stream of a run or shift to .npz,
so detector versions can be replayed offline without re-simulating. It emits no events.

    FALLDET_REC_DIR=/path/to/dir  python -m thundersim shifts --models sim_tools.recorder:RecordingModel ...
File: <FALLDET_REC_DIR>/<profile>_<seed>[_<n>].npz with t, acc, gyro, slot_mask, sat_mask, work_mode.
"""
import os

import numpy as np


class RecordingModel:
    name = "recorder"

    def reset(self, context):
        self.ctx = context
        self.parts = []

    def update(self, batch):
        self.parts.append((np.asarray(batch.t, np.float64), np.asarray(batch.acc, np.float32),
                           np.asarray(batch.gyro, np.float32), np.asarray(batch.slot_mask, bool),
                           np.asarray(batch.sat_mask, bool),
                           np.full(len(batch.t), batch.work_mode == "lying_work")))
        return []

    def finish(self):
        out = os.environ.get("FALLDET_REC_DIR", ".")
        os.makedirs(out, exist_ok=True)
        base = os.path.join(out, f"{self.ctx.profile}_{self.ctx.seed}")
        path, k = base + ".npz", 1
        while os.path.exists(path):
            path, k = f"{base}_{k}.npz", k + 1
        cat = [np.concatenate([p[i] for p in self.parts]) if self.parts else np.zeros(0) for i in range(6)]
        np.savez_compressed(path, t=cat[0], acc=cat[1], gyro=cat[2], slot_mask=cat[3], sat_mask=cat[4],
                            lying_work=cat[5], fs_hz=self.ctx.fs_hz, seed=self.ctx.seed, profile=self.ctx.profile)
        return []
