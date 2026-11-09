# File: trunk_posture.py
# Purpose: Posture assessment from the heading-free segment model: trunk inclination against gravity
#          (forward / backward / sideways), sustained-bend warnings in ergonomic zones, lying detection and
#          the bend between neighbouring segments. Replaces the heading-dependent joint posture whenever a
#          sensor mounting is known.
# Contributors: Unassigned
"""Trunk = the upper thoracic segment's long axis relative to vertical (the usual ergonomic "trunk
inclination", measured at the upper back); pelvis inclination is reported too.

Zones for forward flexion follow ISO 11226 / EN 1005-4 practice: 0-20 deg acceptable, 20-60 deg
acceptable only for limited time, > 60 deg not recommended. Backward extension is flagged above 10 deg
(ISO 11226 does not recommend unsupported extension; the margin covers standing sway) and sideways
bending above 15 deg. A warning is raised when a zone is held for POSTURE_WARNING_SECONDS
(default 2 s). Lying (trunk and pelvis, or lumbar, both more than 60 deg from vertical) is a state, not a
posture warning: the fall card owns emergencies.
Prototype indicator; thresholds are configurable and not a medical or compliance claim.
"""
FLEXION_MODERATE_DEG = 20.0
FLEXION_HIGH_DEG = 60.0
EXTENSION_DEG = 10.0    # ISO 11226: unsupported extension is not recommended; 10 deg allows standing sway
LATERAL_DEG = 15.0
LYING_DEG = 60.0
JOINTS = (("lumbar", "pelvis", "lumbar"), ("thoracic", "lumbar", "thoracic"), ("upper_thoracic", "thoracic", "upper_thoracic"))


class TrunkPosture:
    def __init__(self, warning_seconds=2.0):
        self.warning_seconds = warning_seconds
        self._since = {}                 # condition key -> monotonic time first seen
        self.state = None
        self.alerts = []

    def reset(self):
        self._since.clear()
        self.state = None
        self.alerts = []

    def _sustained(self, key, active, now):
        if not active:
            self._since.pop(key, None)
            return None
        start = self._since.setdefault(key, now)
        return now - start

    def update(self, segments, now):
        """segments: {region: {"flexion_deg", "lateral_deg", "inclination_deg", ...}} from SegmentModel."""
        trunk = (segments or {}).get("upper_thoracic") or (segments or {}).get("thoracic")
        if trunk is None:
            self.reset()
            return None
        flexion, lateral, inclination = trunk["flexion_deg"], trunk["lateral_deg"], trunk["inclination_deg"]
        pelvis = segments.get("pelvis")
        # A deep stoop also tips the upper back past 60 deg; lying needs the lower back near horizontal too
        # (and is never assumed when neither pelvis nor lumbar is measured).
        lower = pelvis or segments.get("lumbar")
        lying = inclination >= LYING_DEG and lower is not None and lower["inclination_deg"] >= LYING_DEG
        if lying:
            zone = "lying"
        elif flexion > FLEXION_HIGH_DEG:
            zone = "high"
        elif flexion > FLEXION_MODERATE_DEG or flexion < -EXTENSION_DEG or abs(lateral) > LATERAL_DEG:
            zone = "moderate"
        else:
            zone = "neutral"

        conditions = [
            ("flexion_high", not lying and flexion > FLEXION_HIGH_DEG, "high",
             "Deep forward bend (> 60°)"),
            ("flexion", not lying and FLEXION_MODERATE_DEG < flexion <= FLEXION_HIGH_DEG, "warning",
             "Sustained forward bend (> 20°)"),
            ("extension", not lying and flexion < -EXTENSION_DEG, "warning", "Sustained backward lean (> 10°)"),
            ("lateral", not lying and abs(lateral) > LATERAL_DEG, "warning",
             f"Sustained side bend to the {'right' if lateral > 0 else 'left'} (> 15°)"),
        ]
        alerts = []
        for key, active, severity, summary in conditions:
            held = self._sustained(key, active, now)
            if held is not None and held >= self.warning_seconds:
                alerts.append({"severity": severity, "category": "posture", "region": "trunk", "summary": summary,
                               "active": True, "sustained_seconds": round(held, 2)})
        self.alerts = alerts

        joints = {}
        for name, lower, upper in JOINTS:
            if lower in segments and upper in segments:
                joints[name] = {"flexion_deg": round(segments[upper]["flexion_deg"] - segments[lower]["flexion_deg"], 1),
                                "lateral_deg": round(segments[upper]["lateral_deg"] - segments[lower]["lateral_deg"], 1)}
        self.state = {
            "trunk": {"flexion_deg": round(flexion, 1), "lateral_deg": round(lateral, 1),
                      "inclination_deg": round(inclination, 1)},
            "pelvis_inclination_deg": None if pelvis is None else round(pelvis["inclination_deg"], 1),
            "zone": zone, "lying": lying, "joints": joints,
            "thresholds": {"flexion_moderate_deg": FLEXION_MODERATE_DEG, "flexion_high_deg": FLEXION_HIGH_DEG,
                           "extension_deg": EXTENSION_DEG, "lateral_deg": LATERAL_DEG,
                           "warning_seconds": self.warning_seconds},
        }
        return self.state
