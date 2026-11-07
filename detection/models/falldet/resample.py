"""Turn real-world 4-IMU frames into the detector's clean uniform 100 Hz stream.

Used by FallDetectorStream.push_batch() and falldet.sim_adapter.SpineFallModel.

Per batch of frames (t, acc, gyro, slot_mask):
 1. A slot is unusable in a frame if slot_mask is False or any of its 6 values is NaN/inf.
    It is replaced by the nearest usable anchor along the spine (lumbar <- pelvis, thoracic,
    upper_thoracic, ...). So a missing lumbar means Stage 1 runs on the pelvis alone
    (svm = mean(pelvis, pelvis)). Frames with no usable slot are dropped like lost packets.
 2. Frames are linearly interpolated onto a 100 Hz grid. Gaps up to MAX_INTERP_GAP_S (0.1 s)
    are bridged. Longer gaps are NOT filled: the grid restarts at the next frame and the
    segment is flagged `gap_before` so the stream resets its Stage-1 timing (no fabricated
    free fall / impact across a gap).
 3. Rate check: if the median of the last 25 frame intervals exceeds 1/MIN_RATE_HZ (25 ms),
    the data are too slow to show a fall (0.5 Hz firmware samples every 2 s) and nothing is
    passed on; `insufficient_rate` is set.
"""
import collections

import numpy as np

from .constants import FS_HZ

MIN_RATE_HZ = 40.0
MAX_INTERP_GAP_S = 0.10
DT = 1.0 / FS_HZ
NEIGHBOURS = {0: (1, 2, 3), 1: (0, 2, 3), 2: (3, 1, 0), 3: (2, 1, 0)}


class Segment:
    __slots__ = ("t", "acc", "gyro", "missing", "gap_before", "gap_s")

    def __init__(self, t, acc, gyro, missing, gap_before, gap_s):
        self.t, self.acc, self.gyro, self.missing = t, acc, gyro, missing
        self.gap_before, self.gap_s = gap_before, gap_s


class Preprocessor:
    def __init__(self, nominal_fs_hz=None):
        self.last = None              # (t, acc(4,3), gyro(4,3), missing(4,))
        self.next_t = None
        self.recent_dt = collections.deque(maxlen=25)
        self.insufficient_rate = nominal_fs_hz is not None and nominal_fs_hz < MIN_RATE_HZ
        self.frames_in = self.frames_dropped = 0

    @staticmethod
    def fill_slots(acc, gyro, ok):
        acc, gyro = acc.copy(), gyro.copy()
        for s in range(4):
            miss = ~ok[:, s]
            if not miss.any():
                continue
            for nb in NEIGHBOURS[s]:
                take = miss & ok[:, nb]
                acc[take, s], gyro[take, s] = acc[take, nb], gyro[take, nb]
                miss &= ~take
            acc[miss, s], gyro[miss, s] = 0.0, 0.0      # no usable slot at all: frame dropped
        return acc, gyro

    def process(self, t, acc, gyro, slot_mask=None):
        """Returns a list of Segment (possibly empty). Never raises on NaN / missing slots."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        n = len(t)
        if n == 0:
            return []
        acc = np.asarray(acc, dtype=np.float64).reshape(n, 4, 3)
        gyro = np.asarray(gyro, dtype=np.float64).reshape(n, 4, 3)
        ok = np.isfinite(acc).all(-1) & np.isfinite(gyro).all(-1) & np.isfinite(t)[:, None]
        if slot_mask is not None:
            ok &= np.asarray(slot_mask, dtype=bool).reshape(n, 4)
        acc, gyro = self.fill_slots(acc, gyro, ok)
        keep = ok.any(axis=1)
        if self.last is not None:
            keep &= t > self.last[0]                     # drop overlapping / backwards frames
        keep &= np.concatenate([[True], np.diff(t) > 0]) if n > 1 else keep
        self.frames_in += n
        self.frames_dropped += int((~keep).sum())
        t, acc, gyro, miss = t[keep], acc[keep], gyro[keep], ~ok[keep]
        if len(t) == 0:
            return []
        tt = t if self.last is None else np.concatenate([[self.last[0]], t])
        self.recent_dt.extend(np.diff(tt).tolist())
        if len(self.recent_dt) >= 2 and np.median(self.recent_dt) > 1.0 / MIN_RATE_HZ:
            self.insufficient_rate = True
        if self.insufficient_rate:
            self.last, self.next_t = (t[-1], acc[-1], gyro[-1], miss[-1]), None
            return []
        if self.last is not None:
            T = np.concatenate([[self.last[0]], t])
            A = np.concatenate([self.last[1][None], acc])
            W = np.concatenate([self.last[2][None], gyro])
            M = np.concatenate([self.last[3][None], miss])
        else:
            T, A, W, M = t, acc, gyro, miss
        self.last = (t[-1], acc[-1], gyro[-1], miss[-1])
        if self.next_t is None:
            self.next_t = T[0]
        out = []
        cuts = np.flatnonzero(np.diff(T) > MAX_INTERP_GAP_S + 1e-9) + 1
        for k, seg in enumerate(np.split(np.arange(len(T)), cuts)):
            ts = T[seg]
            gap_before, gap_s = False, 0.0
            if k > 0:
                gap_before, gap_s = True, float(T[seg[0]] - T[seg[0] - 1])
                self.next_t = ts[0]                      # restart the grid; invent nothing
            if ts[-1] < self.next_t - 1e-9:
                continue
            m = int(np.floor((ts[-1] - self.next_t) / DT + 1e-6)) + 1
            g = self.next_t + DT * np.arange(m)
            g = g[g <= ts[-1] + 1e-9]
            if len(g) == 0:
                continue
            j = np.clip(np.searchsorted(ts, g, side="right") - 1, 0, max(len(ts) - 2, 0))
            if len(ts) == 1:
                Ag, Wg, Mg = (np.repeat(x[seg], len(g), axis=0) for x in (A, W, M))
            else:
                w = ((g - ts[j]) / np.maximum(ts[j + 1] - ts[j], 1e-9)).clip(0, 1)
                Ag = A[seg][j] * (1 - w)[:, None, None] + A[seg][j + 1] * w[:, None, None]
                Wg = W[seg][j] * (1 - w)[:, None, None] + W[seg][j + 1] * w[:, None, None]
                Mg = M[seg][j] | M[seg][j + 1]
            out.append(Segment(g, Ag, Wg, Mg, gap_before, gap_s))
            self.next_t = g[-1] + DT
        return out
