# File: scenarios.py
# Purpose: Scenario catalogue (falls, near-falls, construction hard negatives) with ground
#          truth and expected outputs, plus the day-in-the-life shift generator.
# Contributors: Unassigned
"""Scenario catalogue. IDs follow fall_model_sim_spec/SIM_SPEC.md sections 3-4; IDs marked
(ThunderSim) are additions. Expected outputs use SIM_SPEC's three states:
fall_confirmed, near_fall, normal. report_only scenarios are measured but not scored.

Every run starts with a 5 s quiet neutral stand (the product's calibration capture) and
ends with enough observation time for man-down timers and the worker check-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import falls
from .motions import Builder
from .site import (CLOTHING, OSHA_PROTECTION_HEIGHT_M, SLIP_FALL_DISTANCE_M, SOFT_TISSUE_FN_HZ,
                   SOFT_TISSUE_ZETA, Site, available_friction, random_site, slip_distance)
from .spine import PoseTrack, Worker

CALIBRATION_S = 5.0


@dataclass
class ScenarioRun:
    scenario: str
    seed: int
    track: PoseTrack
    worker: Worker
    site: Site
    events: list                # ground truth (t, name, info)
    expected: str               # fall_confirmed | near_fall | normal
    report_only: bool
    ref_time: float | None      # first impact, or the end of the event for SYNC / ARR
    params: dict
    work_modes: list            # (t, mode) changes
    checkin_s: float | None     # how fast the worker cancels a check-in; None = cannot respond
    soft_fn: np.ndarray         # soft-tissue natural frequency per slot (Hz)
    soft_zeta: np.ndarray
    slot_failures: list = field(default_factory=list)  # (slot, t_start, t_end)


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str        # height | level | arrest | collapse | near_fall | negative
    expected: str
    description: str
    build: Callable
    report_only: bool = False


CATALOGUE: dict[str, Scenario] = {}


def scenario(id: str, group: str, expected: str, description: str, report_only: bool = False):
    def register(fn):
        CATALOGUE[id] = Scenario(id, group, expected, description, fn, report_only)
        return fn
    return register


def random_worker(rng: np.random.Generator, clothing: str) -> Worker:
    """SIM_SPEC 4b: mass 55-120 kg, height 1.55-1.95 m; tool belt / harness / PPE 3-15 kg."""
    extra = CLOTHING[clothing].extra_mount_deg
    return Worker(height_m=float(rng.uniform(1.55, 1.95)), mass_kg=float(rng.uniform(55, 120)),
                  load_kg=float(rng.uniform(3, 15)), mount_error_deg=6.0 + extra)


def soft_tissue(rng: np.random.Generator, clothing: str, fn_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    scale = CLOTHING[clothing].fn_scale * fn_scale
    fn = np.array([rng.uniform(*SOFT_TISSUE_FN_HZ[i]) for i in range(4)]) * scale
    zeta = rng.uniform(*SOFT_TISSUE_ZETA, 4)
    return fn, zeta


def post_fall_behaviour(rng: np.random.Generator) -> str:
    """SIM_SPEC 1.3: stillness (with breathing) or attempted movement with p = 0.3."""
    r = rng.random()
    return "unconscious" if r < 0.4 else ("still_then_move" if r < 0.7 else "struggle")


def begin(b: Builder, height: float = 0.0) -> None:
    """Place the worker (optionally on a platform `height` m up) and record the 5 s neutral
    calibration stand. Must be the first thing a scenario does."""
    if b.segs:
        raise RuntimeError("begin() must run before any motion")
    b.ground_y = height
    b.pose.pos[1] = height + b.worker.pelvis_height
    b.hold(CALIBRATION_S, "calibration_stand")


def lead_in(b: Builder, rng: np.random.Generator, walk: bool = True, keep_moving: bool = False) -> None:
    """A little normal activity before the event."""
    b.stand(float(rng.uniform(2, 6)))
    if walk:
        b.walk(float(rng.uniform(4, 10)), stop=not keep_moving)


def first_time(events: list, name: str) -> float | None:
    for t, n, _ in events:
        if n == name:
            return t
    return None


def build_run(scenario_id: str, seed: int, obs_s: float = 120.0, fn_scale: float = 1.0,
              slot_failure: int | None = None) -> ScenarioRun:
    """Generate one randomised run of a catalogue scenario."""
    spec = CATALOGUE[scenario_id]
    rng = np.random.default_rng(seed)
    site_rng = np.random.default_rng(seed + 7_919)
    site = random_site(site_rng)
    worker = random_worker(rng, site.clothing)
    b = Builder(worker, site, rng)
    ctx = {"obs_s": obs_s, "expected": spec.expected, "checkin_s": None, "ref": "impact", "params": {}}
    spec.build(b, rng, ctx)
    if not b.segs or b.segs[0].phase[0] != 1:
        raise RuntimeError(f"{scenario_id}: scenarios must call begin() first (calibration stand)")
    track = b.build()
    fn, zeta = soft_tissue(rng, site.clothing, fn_scale)
    ref = first_time(b.events, ctx["ref"]) if ctx["ref"] else None
    if ref is None and ctx["ref"] == "impact":
        ref = first_time(b.events, "fall_onset")
    if ctx.get("ref_time") is not None:
        ref = ctx["ref_time"]
    failures = []
    if slot_failure is not None:
        failures.append((slot_failure, 0.0, len(track) / track.fs + 1))
    params = {"surface": site.surface, "condition": site.condition, "clothing": site.clothing, "radio": site.radio,
              "height_m": round(worker.height_m, 2), "mass_kg": round(worker.mass_kg, 1), **ctx["params"]}
    return ScenarioRun(scenario_id, seed, track, worker, site, b.events, ctx["expected"], spec.report_only, ref,
                       params, b.work_modes, ctx["checkin_s"], fn, zeta, failures)


def _observe_after_fall(b: Builder, rng: np.random.Generator, ctx: dict, phase: str = "lying") -> None:
    behaviour = post_fall_behaviour(rng)
    ctx["params"]["behaviour"] = behaviour
    falls.post_fall(b, behaviour, ctx["obs_s"], phase)


# --- falls from height (SIM_SPEC H-*) ------------------------------------------------------------
@scenario("H-LAD", "height", "fall_confirmed", "Fall from a ladder, 1-3 m; feet-first 60 % / backward 40 %")
def _h_lad(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    height = float(rng.uniform(1.0, 3.0))
    b.ladder(height, up=True)
    b.hold(float(rng.uniform(2, 8)), sway=0.5)
    start = "ladder_slip" if rng.random() < 0.6 else "ladder_back"
    info = falls.height_fall(b, b.ground_y, start=start, surface=b.site.surface)
    ctx["params"].update(height=round(height, 2), start=start, ends=info.get("ends", "lying"))
    if info.get("ends") == "standing":
        b.walk(float(rng.uniform(3, 8)), phase="recovery")
        ctx["expected"] = "fall_confirmed"  # still a fall from height; SIM_SPEC expects confirmation
    _observe_after_fall(b, rng, ctx)


@scenario("H-SCAF", "height", "fall_confirmed", "Fall from a scaffold edge, 2-6 m, tumbling 4-10 rad/s")
def _h_scaf(b, rng, ctx):
    height = float(rng.uniform(2.0, 6.0))
    begin(b, height)
    lead_in(b, rng, walk=rng.random() < 0.5)
    falls.height_fall(b, height, start="edge", surface=b.site.surface)
    ctx["params"].update(height=round(height, 2))
    _observe_after_fall(b, rng, ctx)


@scenario("H-ROOF", "height", "fall_confirmed", "Slide down a roof slope then fall off the edge, 3-6 m")
def _h_roof(b, rng, ctx):
    height = float(rng.uniform(3.0, 6.0))
    begin(b, height)
    b.stand(float(rng.uniform(3, 6)))
    b.kneel(30, 1.5)
    falls.height_fall(b, height, start="roof", surface="packed_soil" if rng.random() < 0.5 else "concrete")
    ctx["params"].update(height=round(height, 2))
    _observe_after_fall(b, rng, ctx)


@scenario("H-HOLE", "height", "fall_confirmed", "Fall through a floor opening or skylight, 2-4 m, feet-first")
def _h_hole(b, rng, ctx):
    height = float(rng.uniform(2.0, 4.0))
    begin(b, height)
    lead_in(b, rng, keep_moving=True)
    info = falls.height_fall(b, height, start="hole", surface=b.site.surface)
    ctx["params"].update(height=round(height, 2), ends=info.get("ends"))
    _observe_after_fall(b, rng, ctx)


@scenario("H-EXC", "height", "fall_confirmed", "Slide into an excavation, 1.5-3 m, end curled against the wall")
def _h_exc(b, rng, ctx):
    height = float(rng.uniform(1.5, 3.0))
    begin(b, height)
    lead_in(b, rng)
    falls.height_fall(b, height, start="excavation", surface="packed_soil")
    ctx["params"].update(height=round(height, 2))
    _observe_after_fall(b, rng, ctx)


# --- same-level falls -------------------------------------------------------------------------
@scenario("L-SLIP", "level", "fall_confirmed", "Slip on mud, ice or oil: feet out, backward fall")
def _l_slip(b, rng, ctx):
    begin(b)
    b.site = Site(**{**b.site.__dict__, "condition": str(rng.choice(["muddy", "icy", "oily", "wet"]))})
    lead_in(b, rng, keep_moving=True)
    info = falls.same_level_fall(b, "backward", slip=True)
    ctx["params"].update(condition=b.site.condition, hip_speed_ms=info["hip_speed_ms"])
    _observe_after_fall(b, rng, ctx)


@scenario("L-TRIP", "level", "fall_confirmed", "Trip on rebar/debris: forward fall, hands-first in 50 %")
def _l_trip(b, rng, ctx):
    begin(b)
    lead_in(b, rng, keep_moving=True)
    hands = bool(rng.random() < 0.5)
    info = falls.same_level_fall(b, "forward", hands=hands)
    ctx["params"].update(hands_first=hands, hip_speed_ms=info["hip_speed_ms"])
    _observe_after_fall(b, rng, ctx)


@scenario("STAIR", "level", "fall_confirmed", "Fall down 3-12 stairs: several impacts")
def _stair(b, rng, ctx):
    steps = int(rng.integers(3, 13))
    begin(b, 0.18 * steps)
    lead_in(b, rng, keep_moving=True)
    info = falls.stairs_fall(b, steps)
    ctx["params"].update(steps=steps, impacts=info["impacts"])
    _observe_after_fall(b, rng, ctx)


@scenario("STRUCK-KD", "level", "fall_confirmed", "Struck by a swinging load and knocked down")
def _struck_kd(b, rng, ctx):
    begin(b)
    lead_in(b, rng, walk=False)
    info = falls.struck_by(b, knockdown=True)
    ctx["params"].update(jolt_g=round(info["jolt_g"], 2))
    _observe_after_fall(b, rng, ctx)


# --- arrest, nets, collapse -------------------------------------------------------------------
@scenario("ARR", "arrest", "fall_confirmed", "Harness-arrested fall, then suspended near upright (still or struggling)")
def _arr(b, rng, ctx):
    height = float(rng.uniform(max(OSHA_PROTECTION_HEIGHT_M, 3.0), 8.0))
    begin(b, height)
    lead_in(b, rng, walk=rng.random() < 0.5)
    info = falls.height_fall(b, height, start="edge", protection="harness")
    behaviour = "still" if rng.random() < 0.5 else "struggle"
    ctx["params"].update(height=round(height, 2), free_fall_m=info.get("free_fall_m"), behaviour=behaviour)
    ctx["ref"] = "arrest_end"
    if behaviour == "still":
        b.conscious = False
        b.hold(ctx["obs_s"], "suspended", sway=0.0, fidget=0.0)
    else:
        falls.post_fall(b, "struggle", ctx["obs_s"], "suspended")


@scenario("NET", "arrest", "fall_confirmed", "Fall into a safety net 2-6 m below (ThunderSim)")
def _net(b, rng, ctx):
    height = float(rng.uniform(2.0, 6.0))
    begin(b, height)
    lead_in(b, rng, walk=rng.random() < 0.5)
    falls.height_fall(b, height, start="edge", protection="net")
    ctx["params"].update(height=round(height, 2))
    _observe_after_fall(b, rng, ctx)


@scenario("SYNC", "collapse", "fall_confirmed", "Syncope / heat collapse from standing: slow sag, no free fall", report_only=True)
def _sync(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.stand(float(rng.uniform(5, 20)))
    falls.slow_collapse(b, "standing")
    ctx["ref"] = "impact"
    b.conscious = False
    b.hold(ctx["obs_s"], "lying", sway=0.0, fidget=0.0)


@scenario("SYNC-KNEEL", "collapse", "fall_confirmed", "Collapse while kneeling at work (ThunderSim)", report_only=True)
def _sync_kneel(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.kneel(float(rng.uniform(20, 50)), 2.0)
    b.hold(float(rng.uniform(10, 30)), sway=0.3)
    falls.slow_collapse(b, "kneeling")
    b.conscious = False
    b.hold(ctx["obs_s"], "lying", sway=0.0, fidget=0.0)


# --- near falls -------------------------------------------------------------------------------
@scenario("NF-STUMBLE", "near_fall", "near_fall", "Stumble and catch: fast pitch, short dip, recovery steps")
def _nf_stumble(b, rng, ctx):
    begin(b)
    lead_in(b, rng, keep_moving=True)
    falls.near_fall(b, "stumble")
    ctx["ref"] = "near_fall_onset"
    b.walk(float(rng.uniform(5, 15)))
    b.stand(ctx["obs_s"] / 2)


@scenario("NF-RAIL", "near_fall", "near_fall", "Slip and grab a rail: partial fall, arm arrest, upright within 3 s")
def _nf_rail(b, rng, ctx):
    begin(b)
    lead_in(b, rng, keep_moving=True)
    falls.near_fall(b, "rail")
    ctx["ref"] = "near_fall_onset"
    b.walk(float(rng.uniform(5, 15)))
    b.stand(ctx["obs_s"] / 2)


@scenario("STRUCK-NF", "near_fall", "near_fall", "Struck by an object, stumbles, stays up")
def _struck_nf(b, rng, ctx):
    begin(b)
    lead_in(b, rng, walk=False)
    falls.struck_by(b, knockdown=False)
    ctx["ref"] = "struck"
    b.walk(float(rng.uniform(5, 15)))
    b.stand(ctx["obs_s"] / 2)


@scenario("SLIP-FRICTION", "near_fall", "near_fall", "Slip whose outcome follows site friction: >0.15 m slide = fall (ThunderSim)")
def _slip_friction(b, rng, ctx):
    begin(b)
    condition = str(rng.choice(["dry", "dusty", "wet", "muddy", "oily", "icy"]))
    b.site = Site(**{**b.site.__dict__, "condition": condition})
    mu = available_friction(b.site.surface, condition)
    distance = slip_distance(float(rng.uniform(0.5, 1.5)), mu)
    ctx["params"].update(condition=condition, friction=round(mu, 2), slip_m=round(distance, 3))
    lead_in(b, rng, keep_moving=True)
    if distance > SLIP_FALL_DISTANCE_M:
        ctx["expected"] = "fall_confirmed"
        falls.same_level_fall(b, "backward", slip=True)
        _observe_after_fall(b, rng, ctx)
    else:
        falls.near_fall(b, "rail" if rng.random() < 0.3 else "stumble")
        ctx["ref"] = "near_fall_onset"
        b.walk(float(rng.uniform(5, 15)))
        b.stand(ctx["obs_s"] / 2)


# --- construction hard negatives (SIM_SPEC 4) --------------------------------------------------
def _neg_tail(b, rng, ctx):
    ctx["ref"] = None
    ctx["checkin_s"] = float(rng.uniform(2, 8))  # a fine worker cancels a check-in prompt
    b.walk(float(rng.uniform(3, 8)))
    b.stand(float(rng.uniform(3, 10)))


def _neg_minutes(rng, ctx) -> float:
    return float(rng.uniform(60, max(ctx["obs_s"], 61)))


@scenario("A-HAMMER", "negative", "normal", "Hammering, including overhead")
def _a_hammer(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    total = _neg_minutes(rng, ctx)
    while total > 0:
        burst = float(rng.uniform(5, 20))
        b.hammer(burst, overhead=bool(rng.random() < 0.4))
        b.stand(float(rng.uniform(2, 8)))
        total -= burst + 5
    _neg_tail(b, rng, ctx)


@scenario("A-DRILL", "negative", "normal", "Overhead drilling: trunk extended, 20-50 Hz vibration")
def _a_drill(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    total = _neg_minutes(rng, ctx)
    while total > 0:
        bout = float(rng.uniform(10, 30))
        b.drill_overhead(bout)
        b.stand(float(rng.uniform(3, 8)))
        total -= bout + 5
    _neg_tail(b, rng, ctx)


@scenario("A-JACK", "negative", "normal", "Jackhammer / plate compactor bouts")
def _a_jack(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    total = _neg_minutes(rng, ctx)
    while total > 0:
        bout = float(rng.uniform(30, 90))
        b.jackhammer(bout)
        b.stand(float(rng.uniform(5, 15)))
        total -= bout + 10
    _neg_tail(b, rng, ctx)


@scenario("A-EQUIP", "negative", "normal", "Operating equipment over rough ground (seated, jolts)")
def _a_equip(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.cab_ride(_neg_minutes(rng, ctx))
    _neg_tail(b, rng, ctx)


@scenario("A-KNEEL", "negative", "normal", "Kneeling / crawling floor work, trunk 45-90 deg for minutes")
def _a_kneel(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.kneel_work(_neg_minutes(rng, ctx))
    if rng.random() < 0.5:
        b.kneel(10, 1.5)
        b.quadruped(1.5)
        b.crawl(float(rng.uniform(10, 30)))
        b.stand_up(2.0)
    _neg_tail(b, rng, ctx)


@scenario("A-REBAR", "negative", "normal", "Rebar tying: sustained 70-100 deg stoop (ThunderSim)")
def _a_rebar(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.rebar_tie(_neg_minutes(rng, ctx))
    _neg_tail(b, rng, ctx)


@scenario("A-LADDER", "negative", "normal", "Climbing a ladder up and down")
def _a_ladder(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    for _ in range(int(rng.integers(1, 4))):
        height = float(rng.uniform(1.5, 4.0))
        b.ladder(height, up=True)
        b.hammer(float(rng.uniform(5, 20)))
        b.ladder(height, up=False)
        b.walk(float(rng.uniform(3, 8)))
    _neg_tail(b, rng, ctx)


@scenario("A-JUMP", "negative", "normal", "Jumping down 0.3-0.6 m from a step or truck bed (Stage 1 will fire)")
def _a_jump(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    for _ in range(int(rng.integers(1, 4))):
        height = float(rng.uniform(0.3, 0.6))
        b.ladder(height, up=True)  # step up on to the truck bed / platform
        b.stand(float(rng.uniform(2, 5)))
        b.jump_down(b.ground_y)
        b.walk(float(rng.uniform(3, 8)))
    _neg_tail(b, rng, ctx)


@scenario("A-SIT", "negative", "normal", "Sitting down hard on a bucket or bench")
def _a_sit(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    for _ in range(int(rng.integers(1, 4))):
        b.sit(float(rng.uniform(0.35, 0.5)), hard=True)
        b.hold(float(rng.uniform(10, 60)), sway=0.3)
        b.stand_up(1.5)
        b.walk(float(rng.uniform(3, 8)))
    _neg_tail(b, rng, ctx)


@scenario("A-LIE", "negative", "normal", "Lying down on purpose (break / working under things), no mode declared")
def _a_lie(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.lie(str(rng.choice(["supine", "prone", "right", "left"])), float(rng.uniform(2.0, 4.0)))
    b.lying_activity(_neg_minutes(rng, ctx))
    b.get_up("activity")
    _neg_tail(b, rng, ctx)


@scenario("A-LIEWORK", "negative", "normal", "Lying work with work_mode='lying_work' declared (ThunderSim)")
def _a_liework(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.set_work_mode("lying_work")
    b.lie("supine", float(rng.uniform(2.0, 4.0)))
    total = _neg_minutes(rng, ctx)
    while total > 0:
        bout = float(rng.uniform(10, 40))
        b.lying_activity(bout)
        seg = b.seg(2.0, "lying_work", fidget=2.0)  # creeper slide along the body axis
        seg.pos += np.outer(np.sin(np.pi * np.linspace(0, 1, seg.k)) ** 2 * rng.uniform(-0.3, 0.3), [0, 0, 1])
        b.commit(seg)
        total -= bout + 2.0
    b.get_up("activity")
    b.set_work_mode("normal")
    _neg_tail(b, rng, ctx)


@scenario("A-NAP", "negative", "normal", "Lunch nap lying still for minutes (ThunderSim)")
def _a_nap(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.lie("supine", 3.0)
    b.conscious = False
    b.hold(_neg_minutes(rng, ctx), "lying_work", sway=0.0, fidget=0.0)
    b.conscious = True
    b.get_up("activity")
    _neg_tail(b, rng, ctx)


@scenario("A-CARRY", "negative", "normal", "Carrying loads and bending to lift (10-40 kg)")
def _a_carry(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    for _ in range(int(rng.integers(2, 6))):
        b.carry(float(rng.uniform(5, 20)), float(rng.uniform(10, 40)))
        b.stand(float(rng.uniform(2, 6)))
    _neg_tail(b, rng, ctx)


@scenario("A-RUN", "negative", "normal", "Running / jogging across the site")
def _a_run(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    for _ in range(int(rng.integers(1, 3))):
        b.run(float(rng.uniform(5, 15)), speed=float(rng.uniform(2.5, 3.5)))
        b.walk(float(rng.uniform(3, 8)))
    _neg_tail(b, rng, ctx)


@scenario("HAT-OFF", "negative", "normal", "Hard hat knocked off: cable tug on the upper sensor, body stays up")
def _hat_off(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.hat_off()
    b.stand(float(rng.uniform(3, 8)))
    _neg_tail(b, rng, ctx)


@scenario("STRAP-SLIP", "negative", "normal", "Strap slips 5-20 deg on one anchor during work (ThunderSim)")
def _strap_slip(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.stoop(float(rng.uniform(60, 90)), hold_s=1.0)
    b.strap_slip(int(rng.integers(0, 4)), float(rng.uniform(5, 20)))
    b.walk(float(rng.uniform(10, 30)))
    b.stoop(float(rng.uniform(40, 80)), hold_s=2.0)
    _neg_tail(b, rng, ctx)


@scenario("DEVICE-OFF", "negative", "normal", "Strip taken off and left on a table (ThunderSim)")
def _device_off(b, rng, ctx):
    begin(b)
    lead_in(b, rng)
    b.remove_strip(float(rng.uniform(0.7, 0.9)))
    b.hold(ctx["obs_s"], "device_off", sway=0.0, fidget=0.0)
    ctx["ref"] = None
    ctx["checkin_s"] = None  # nobody is wearing it to answer


@scenario("FALL-GETUP", "level", "fall_confirmed", "Trip, then gets up within 3-15 s and walks on (ThunderSim)", report_only=True)
def _fall_getup(b, rng, ctx):
    begin(b)
    lead_in(b, rng, keep_moving=True)
    falls.same_level_fall(b, "forward", hands=bool(rng.random() < 0.5))
    falls.post_fall(b, "get_up", 20.0)
    ctx["checkin_s"] = float(rng.uniform(2, 8))
    b.stand(ctx["obs_s"] / 2)


# --- day-in-the-life shifts (SIM_SPEC 5.2) ------------------------------------------------------
SHIFT_MIX = {  # SIM_SPEC 5.2 suggested proportions [design]
    "walk_stand": 0.30, "kneel_crawl": 0.15, "ladder": 0.10, "carry": 0.10, "tools": 0.10,
    "jackhammer": 0.05, "equipment": 0.05, "breaks": 0.05, "rebar": 0.05, "lying_work": 0.05,
}


def shift_blocks(seed: int, hours: float, block_s: tuple[float, float] = (60, 600)):
    """Yield (activity, PoseTrack, events) blocks for one fall-free shift.

    Every block starts and ends standing still at rest, so blocks can be streamed back to back
    through one device/transport/model chain. 10 jump-downs and 5 runs are spread over 8 h.
    """
    rng = np.random.default_rng(seed)
    site = random_site(np.random.default_rng(seed + 7_919))
    worker = random_worker(rng, site.clothing)
    names = list(SHIFT_MIX)
    probs = np.array(list(SHIFT_MIX.values()))
    probs /= probs.sum()
    total = hours * 3600.0
    elapsed = 0.0
    jumps_left = int(round(10 * hours / 8))
    runs_left = int(round(5 * hours / 8))
    first = True
    while elapsed < total:
        b = Builder(worker, site, rng)
        if first:
            begin(b)
        else:
            b.stand(2.0)
        duration = float(min(rng.uniform(*block_s), max(total - elapsed, 30)))
        activity = str(names[rng.choice(len(names), p=probs)])
        _shift_activity(b, rng, activity, duration)
        blocks_expected = max(total / np.mean(block_s), 1.0)
        if jumps_left > 0 and rng.random() < 10 * hours / 8 / blocks_expected:
            b.ladder(float(rng.uniform(0.3, 0.6)), up=True)
            b.jump_down(b.ground_y)
            jumps_left -= 1
        if runs_left > 0 and rng.random() < 5 * hours / 8 / blocks_expected:
            b.run(float(rng.uniform(5, 12)))
            runs_left -= 1
        b.stand(1.0)
        track = b.build()
        elapsed += len(track) / track.fs
        yield activity, track, b.events, b.work_modes, worker, site
        first = False


def _shift_activity(b: Builder, rng: np.random.Generator, activity: str, duration: float) -> None:
    end = b.t + duration
    while b.t < end:
        if activity == "walk_stand":
            b.walk(float(rng.uniform(10, 60)), turn_rate=float(rng.uniform(-0.1, 0.1)))
            b.stand(float(rng.uniform(5, 40)))
        elif activity == "kneel_crawl":
            b.kneel_work(float(rng.uniform(30, 120)))
        elif activity == "ladder":
            height = float(rng.uniform(1.5, 4.0))
            b.ladder(height)
            b.hammer(float(rng.uniform(10, 40)))
            b.ladder(height, up=False)
        elif activity == "carry":
            b.carry(float(rng.uniform(5, 20)), float(rng.uniform(10, 40)))
        elif activity == "tools":
            if rng.random() < 0.5:
                b.hammer(float(rng.uniform(10, 40)), overhead=bool(rng.random() < 0.3))
            else:
                b.drill_overhead(float(rng.uniform(10, 40)))
            b.stand(float(rng.uniform(3, 10)))
        elif activity == "jackhammer":
            b.jackhammer(float(rng.uniform(30, 120)))
        elif activity == "equipment":
            b.cab_ride(float(rng.uniform(60, 300)))
        elif activity == "breaks":
            b.sit(0.45, hard=bool(rng.random() < 0.3))
            b.hold(float(rng.uniform(60, 300)), sway=0.2)
            b.stand_up(1.5)
        elif activity == "rebar":
            b.rebar_tie(float(rng.uniform(60, 240)))
        elif activity == "lying_work":
            b.set_work_mode("lying_work")
            b.lie("supine", 3.0)
            b.lying_activity(float(rng.uniform(60, 240)))
            b.get_up("activity")
            b.set_work_mode("normal")
