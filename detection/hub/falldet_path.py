# File: falldet_path.py
# Purpose: Wrap research-95's falldet stream (T1 impact -> C1, T2 perturbation -> C2) with an
#          explicit profile chosen by filter bandwidth (3.3) and a recoverable rate latch (3.4).
#          detection/models is not edited; see proposals/ for the matching falldet diffs.
# Contributors: Unassigned
import os

from falldet.detector import FallDetectorStream, get_detector

PROFILE_ENV = "FALLDET_PROFILE"


def choose_profile(cfg, dlpf_hz=None, accel_range_g=None, explicit=None):
    """Explicit (argument, FALLDET_PROFILE, config) wins; else 'current' when the low-pass is
    <= current_max_dlpf_hz or the range <= current_max_range_g; else 'target'.
    Returns (profile, reason)."""
    for source, value in (("argument", explicit), ("env", os.environ.get(PROFILE_ENV)), ("config", cfg.get("profile"))):
        if value in ("current", "target"):
            return value, f"explicit ({source})"
    dlpf = cfg.get("dlpf_hz") if dlpf_hz is None else dlpf_hz
    if dlpf is not None and dlpf <= cfg["current_max_dlpf_hz"]:
        return "current", f"dlpf {dlpf:g} Hz <= {cfg['current_max_dlpf_hz']:g} Hz"
    if accel_range_g is not None and accel_range_g <= cfg["current_max_range_g"]:
        return "current", f"range {accel_range_g:g} g <= {cfg['current_max_range_g']:g} g"
    return "target", "wide filter and range"


class FalldetPath:
    """One FallDetectorStream; recreated on a proven restart or a rate-latch recovery."""

    def __init__(self, profile, backend="gbt"):
        self.profile = profile
        self.detector = get_detector(profile, backend)   # cached per (profile, backend)
        self.backend = self.detector.backend
        self.error = None
        self.new_stream()

    def new_stream(self):
        # nominal_fs_hz=None: latch only on the measured rate, never on a declared one.
        self.stream = FallDetectorStream(detector=self.detector, nominal_fs_hz=None)

    @property
    def latched(self):
        pre = self.stream.pre
        return pre is not None and pre.insufficient_rate

    def push(self, t, acc, gyro, slot_mask, sat_mask, work_mode):
        """Returns (events, diagnostics): falldet event dicts and diagnostic dicts."""
        events, diags = [], []
        for e in self.stream.push_batch(t, acc, gyro, slot_mask, sat_mask, work_mode):
            if e["type"] == "diagnostic":
                diags.append(e)
                if e.get("code") == "internal_error":
                    self.error = e.get("message", "internal_error")
                continue
            if e["type"] == "stage2_result" and str(e.get("rule", "")).startswith("error"):
                self.error = e["rule"]           # falldet hides Stage-2 errors as rule "error: ..."
            events.append(e)
        return events, diags

    def flush(self):
        return self.stream.flush()
