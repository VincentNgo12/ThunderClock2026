# File: rate_watch.py
# Purpose: RateWatch (DETECTION_ARCHITECTURE.md 3.4): rolling 5 s frame rate from device time.
#          falldet's own valid_frame_hz is a session average and its rate latch never clears.
# Contributors: Unassigned
"""Counts frames in one-second bins of (internal) device time. rate_hz() is the number of frames
in the last window_s complete seconds divided by window_s. Fixed memory (window_s + 1 bins)."""

import math


class RateWatch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = int(cfg["window_s"]) + 1
        self.reset()

    def reset(self):
        self.index = [None] * self.size
        self.count = [0] * self.size
        self.first_t = None
        self.last_t = None

    def add(self, t):
        b = int(math.floor(t))
        k = b % self.size
        if self.index[k] != b:
            self.index[k] = b
            self.count[k] = 0
        self.count[k] += 1
        if self.first_t is None:
            self.first_t = t
        self.last_t = t

    def rate_hz(self, t=None, since=None):
        """Frames per second over the last window_s complete seconds before t (default: last frame).
        since: only count if the whole window starts at or after this time, else None."""
        t = self.last_t if t is None else t
        if t is None:
            return None
        now = int(math.floor(t))
        n = int(self.cfg["window_s"])
        lo = now - n
        if since is not None and lo < since:
            return None
        if self.first_t is None or lo < math.floor(self.first_t):
            return None                     # not enough history yet
        total = sum(c for i, c in zip(self.index, self.count) if i is not None and lo <= i < now)
        return total / n
