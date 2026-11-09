# File: events.py
# Purpose: Event schema of the hub (docs/DETECTION_ARCHITECTURE.md section 5). Kinds come from
#          ThunderSim's EVENT_KINDS; only fall_confirmed and man_down_* are alerts.
# Contributors: Unassigned

from dataclasses import dataclass, field

try:  # use the simulator's Event when importable, so ThunderSim validates kinds itself
    from thundersim.model_api import ALERT_KINDS, EVENT_KINDS, Event  # type: ignore
except Exception:  # backend / standalone use: identical fields and kind list
    EVENT_KINDS = ("suspected_fall", "near_fall", "recovered", "fall_confirmed",
                   "man_down_warning", "man_down_critical", "cleared")
    ALERT_KINDS = ("fall_confirmed", "man_down_warning", "man_down_critical")

    @dataclass
    class Event:
        t: float
        kind: str
        confidence: float = 1.0
        detail: dict = field(default_factory=dict)

        def __post_init__(self):
            if self.kind not in EVENT_KINDS:
                raise ValueError(f"Unknown event kind {self.kind!r}")

PATHS = ("impact", "perturbation", "topple", "man_down")
RULES = ("model", "free_fall_no_gait", "long_free_fall", "lying_work_suppressed",
         "topple_still_down", "topple_mostly_still", "moving_or_up",
         "down_mostly_still", "no_motion_after_trigger", "lying_work_still",
         "upright_and_gait", "stage1", "topple", "nearfall_v2")
SEVERITY = {"suspected_fall": "info", "near_fall": "info", "recovered": "info", "cleared": "info",
            "fall_confirmed": "warning", "man_down_warning": "warning", "man_down_critical": "critical"}
DIAGNOSTIC_CODES = ("insufficient_rate", "rate_recovered", "link_gap", "restart", "sensor_dead",
                    "internal_error")


def make_event(t, kind, confidence, *, episode_id=None, trigger_t=None, path, rule,
               merged_from=None, tags=None, evidence=None, data=None):
    """Build one Event with the section-5 detail block. t is device seconds of the current boot."""
    detail = {"episode_id": episode_id, "trigger_t": None if trigger_t is None else round(float(trigger_t), 3),
              "path": path, "rule": rule, "severity": SEVERITY[kind],
              "merged_from": list(merged_from or [path]), "tags": sorted(set(tags or ())),
              "evidence": evidence or {}, "data": data or {}}
    return Event(float(t), kind, float(confidence), detail)


def diagnostic(code, t=None, **extra):
    """Device diagnostic record (not an Event: ThunderSim has no kind for it)."""
    assert code in DIAGNOSTIC_CODES, code
    return {"type": "device", "code": code, "t": t, **extra}
