# File: topple.py
# Purpose: T3 topple trigger (DETECTION_ARCHITECTURE.md 3.5): a fast descent from "not down" to
#          "down" with trunk rotation, for collapses without a hard impact. O(1) state per strip,
#          thresholds in config.json "topple", ESP32-portable like falldet's Stage1Scalar.
# Contributors: Unassigned
"""Fire when, on one sample:
    the strip is "down" now (PostureTrack definition),
    "not down" was seen <= max_descent_s ago,
    max(|w_pelvis|, |w_lumbar|) reached >= rot_rad_s since rot_lead_s before the last
    not-down sample,
    and it has not fired since that not-down sample (once per descent).

Memory: the last not-down time, the running rotation peak since then, and five 0.1 s bins of
rotation peaks to cover the rot_lead_s look-back. No buffers, no FFT.
"""

NEVER = -1e18
LEAD_BINS = 5          # 5 x 0.1 s covers rot_lead_s = 0.5 s
LEAD_BIN_S = 0.1


class ToppleTrigger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_trigger_t = NEVER
        self.fires = 0
        self.reset_timing()

    def reset_timing(self):
        """Forget the descent (after a long gap or a restart); keep the fire counter."""
        self.last_not_down_t = NEVER
        self.peak_since = 0.0
        self.armed = False
        self.lead_index = [None] * LEAD_BINS
        self.lead_peak = [0.0] * LEAD_BINS
        self.last_peak = 0.0       # rotation peak of the last fire (evidence)

    def _lead(self, t, rot):
        b = int(t // LEAD_BIN_S)
        k = b % LEAD_BINS
        if self.lead_index[k] != b:
            self.lead_index[k] = b
            self.lead_peak[k] = rot
        elif rot > self.lead_peak[k]:
            self.lead_peak[k] = rot
        lo = b - int(round(self.cfg["rot_lead_s"] / LEAD_BIN_S))
        return max((p for i, p in zip(self.lead_index, self.lead_peak) if i is not None and i >= lo), default=0.0)

    def step(self, t, rot, down, not_down):
        """t: seconds; rot: rad/s; down / not_down: PostureTrack flags. Returns True on a fire."""
        cfg = self.cfg
        lead_peak = self._lead(t, rot)
        if not_down:
            self.last_not_down_t = t
            self.peak_since = lead_peak
            self.armed = True
            return False
        if rot > self.peak_since:
            self.peak_since = rot
        if (down and self.armed and t - self.last_not_down_t <= cfg["max_descent_s"]
                and self.peak_since >= cfg["rot_rad_s"]):
            self.armed = False
            self.last_trigger_t = t
            self.last_peak = self.peak_since
            self.fires += 1
            return True
        return False


def run_topple(t, rot, down, not_down, cfg):
    """Batch helper (same code path as streaming). Returns the indices of fires."""
    trig = ToppleTrigger(cfg)
    return [i for i in range(len(t)) if trig.step(float(t[i]), float(rot[i]), bool(down[i]), bool(not_down[i]))]
