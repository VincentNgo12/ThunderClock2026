# File: sim_models.py
# Purpose: ThunderSim entry points (FallModel protocol, no-argument constructors) for the hub and
#          the comparison arms of DETECTION_ARCHITECTURE.md 7. The hub classes are MainModel itself.
# Contributors: Unassigned
"""Load with PYTHONPATH=detection (hub/__init__.py adds detection/models):

    python -m thundersim suite --models hub.sim_models:HubFull --profiles target_8g --out <tmp>

Arms
    FalldetShipped   A  falldet as shipped (SpineFallModel(): profile from context, backend auto)
    FalldetCurrent   B  falldet forced profile 'current', backend 'gbt'
    HubNoT4          C  hub without the man-down timer
    HubFull          D  hub as configured in hub/config.json
    HubPassThrough   D1 hub with T3 and T4 off (must equal arm B on fall_confirmed)
    HubLogreg           hub with falldet backend 'logreg_json'
    (E is ThunderSim's 'baseline'.)
"""
from .main_model import MainModel


class HubFull(MainModel):
    name = "hub-D-full"


class HubNoT4(MainModel):
    name = "hub-C-noT4"

    def __init__(self):
        super().__init__(overrides={"man_down": {"enabled": False}})


class HubPassThrough(MainModel):
    name = "hub-D1-passthrough"

    def __init__(self):
        super().__init__(overrides={"man_down": {"enabled": False}, "topple": {"enabled": False}})


class HubLogreg(MainModel):
    name = "hub-D-logreg"

    def __init__(self):
        super().__init__(backend="logreg_json")


def _spine(**kwargs):
    from falldet.sim_adapter import SpineFallModel
    return SpineFallModel(**kwargs)


class FalldetShipped:
    name = "falldet-A-shipped"

    def __init__(self):
        self.inner = _spine()

    def reset(self, context):
        self.inner.reset(context)

    def update(self, batch):
        return self.inner.update(batch)

    def finish(self):
        return self.inner.finish()


class FalldetCurrent(FalldetShipped):
    name = "falldet-B-current"

    def __init__(self):
        self.inner = _spine(profile="current", backend="gbt")
