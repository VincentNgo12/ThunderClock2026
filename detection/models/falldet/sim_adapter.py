"""ThunderSim adapter: FallDetectorStream.push_batch() behind the simulator's FallModel protocol.

    load_model("falldet.sim_adapter:SpineFallModel")      # with .../fall_model on sys.path
    model.reset(context)       # StreamContext: fs_hz, profile, accel_range_g (any may be absent)
    model.update(batch)        # FrameBatch: t, acc, gyro, slot_mask, sat_mask, work_mode
    model.update_arrays(t, acc, gyro, slot_mask=None, sat_mask=None, work_mode="normal")

All data handling (resampling, gaps, NaN, missing slots, rate check) is in
FallDetectorStream.push_batch / falldet/resample.py; see there and HANDOFF.md.

Profile -> model: names starting with "current" (or accel_range_g <= 4) use the model trained on
+-2 g / +-250 deg/s / 10 Hz data; everything else uses the target model.
Events (ThunderSim kinds): Stage-1 trigger -> "suspected_fall"; 8 s later Stage 2 ->
"fall_confirmed" | "near_fall" | "recovered" (Stage 2 said normal). The stream's
"insufficient_rate" diagnostic is not a ThunderSim event kind, so it is reported in
`model.status` / `model.diagnostics` instead. No man-down timers are produced.
"""
from dataclasses import dataclass, field

from .constants import FS_HZ
from .detector import FallDetectorStream, get_detector

try:  # use the simulator's own Event class when it is importable (validates kinds)
    from thundersim.model_api import Event  # type: ignore
except Exception:  # standalone use
    @dataclass
    class Event:  # same fields as thundersim.model_api.Event
        t: float
        kind: str
        confidence: float = 1.0
        detail: dict = field(default_factory=dict)

_KIND = {"normal": "recovered", "near_fall": "near_fall", "fall_confirmed": "fall_confirmed"}
_DETAIL_FEATURES = ("impact_peak_g", "ff_min_g", "rot_peak_rad_s", "n_anchors_lying",
                    "longest_still_s", "incl_post_pelvis", "incl_post_upper_thoracic")


class SpineFallModel:
    name = "falldet-2stage-nfv2"

    def __init__(self, profile=None, backend="auto", nearfall="v2"):
        """profile: force 'target' or 'current' (None = from the stream context).
        backend: 'auto' (GBT if scikit-learn 1.9.1 + joblib load, else numpy-only logistic
        regression), 'gbt' or 'logreg_json'."""
        self.forced_profile = profile
        self.backend = backend
        self.nearfall = nearfall
        self.reset(None)

    def reset(self, context=None):
        fs = float(getattr(context, "fs_hz", FS_HZ) or FS_HZ)
        prof = str(getattr(context, "profile", "") or "")
        rng = getattr(context, "accel_range_g", None)
        if self.forced_profile in ("target", "current"):
            self.model_profile = self.forced_profile
        else:
            self.model_profile = "current" if (prof.startswith("current") or (rng is not None and rng <= 4.0)) \
                else "target"
        det = get_detector(self.model_profile, self.backend)
        self.stream = FallDetectorStream(detector=det, nominal_fs_hz=fs, nearfall=self.nearfall)
        self.status = "ok"
        self.diagnostics = []

    def update(self, batch):
        return self.update_arrays(batch.t, batch.acc, batch.gyro,
                                  getattr(batch, "slot_mask", None), getattr(batch, "sat_mask", None),
                                  getattr(batch, "work_mode", "normal"))

    def update_arrays(self, t, acc, gyro, slot_mask=None, sat_mask=None, work_mode="normal"):
        out = []
        for e in self.stream.push_batch(t, acc, gyro, slot_mask, sat_mask, work_mode):
            if e["type"] == "diagnostic":
                self.diagnostics.append(e)
                self.status = f"{e['code']}: {e.get('message', '')}"
            else:
                out.append(self._to_event(e))
        return out

    def finish(self):
        """Optional (not in the ThunderSim protocol): decide triggers still waiting for 8 s."""
        return [self._to_event(e) for e in self.stream.flush()]

    def _to_event(self, e):
        if e["type"] == "nearfall_v2":
            f = e["nf_features"]
            return Event(float(e["decided_t_s"]), "near_fall", float(e["confidence"]),
                         {"path": "nearfall_v2", "trigger_t": float(e["t_s"]), "deferred": e.get("deferred", False),
                          "model_profile": self.model_profile,
                          **{k: round(f[k], 3) for k in ("dip_min_g", "peak_g", "rot_peak", "incl_exc_ut",
                                                          "incl_post_max", "rot_ratio")}})
        if e["type"] == "stage1_trigger":
            return Event(float(e["t_s"]), "suspected_fall", 0.5,
                         {"svm_g": round(e["svm_g"], 3), "rot_rad_s": round(e["rot_rad_s"], 3),
                          "model_profile": self.model_profile})
        f, aux = e.get("features") or {}, e.get("aux") or {}
        detail = {"trigger_t": float(e["t_s"]), "state": e["state"], "rule": e["rule"],
                  "probabilities": {k: round(v, 4) for k, v in e["probabilities"].items()},
                  "free_fall_s": round(aux.get("free_fall_s", 0.0), 3),
                  "walking_after": aux.get("walking_after"), "model_profile": self.model_profile,
                  "backend": self.stream.det.backend, "slots_missing_frac": e["slots_missing_frac"],
                  "gap_in_window": e["gap_in_window"],
                  **{k: round(f[k], 3) for k in _DETAIL_FEATURES if k in f}}
        if "note" in e:
            detail["note"] = e["note"]
        return Event(float(e["decided_t_s"]), _KIND[e["state"]], float(e["confidence"]), detail)


class SpineFallModelV1(SpineFallModel):
    """Same as SpineFallModel but with the original (v1) near-fall behaviour, for before/after runs."""
    name = "falldet-2stage-v1"

    def __init__(self):
        super().__init__(nearfall="v1")
