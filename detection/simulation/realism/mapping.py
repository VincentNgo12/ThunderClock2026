# File: realism/mapping.py
# Purpose: Which simulator scenario or activity is compared with which real recordings, and
#          how sure we are that they are the same kind of movement.
# Contributors: Unassigned
"""Category mapping.

Event categories (falls and fall-like ADLs) compare one aligned event per recording/run.
Activity categories compare 5 s windows of ongoing movement.

confidence: high  = same movement, same sensor, same conditions after harmonisation
            medium = same movement class, but the protocol or conditions differ
            low    = closest available counterpart only; read the gap with care
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventCategory:
    name: str
    real_codes: tuple[int, ...]      # Erciyes activity codes
    scenario: str                    # ThunderSim catalogue id
    confidence: str
    note: str
    onset_event: str = "fall_onset"  # ground-truth event that starts the sim search window
    filter: dict = field(default_factory=dict)
    # features fixed by the real protocol (e.g. Erciyes falls start from quiet standing while
    # the sim trips while walking): reported, but given no weight in the impact ranking
    protocol: tuple[str, ...] = ()
    dataset: str = "erciyes"         # erciyes | univr
    drop: tuple[str, ...] = ()       # features not comparable for this dataset (set to NaN both sides)


@dataclass(frozen=True)
class ActivityCategory:
    name: str
    dataset: str                     # erciyes | vtt
    real_codes: tuple[int, ...]
    sim_activity: str                # key in emulate.ACTIVITIES
    confidence: str
    note: str
    drop: tuple[str, ...] = ()       # features not comparable for this dataset (NaN on both sides)
    # Erciyes activity trials are 32-56 % quiet standing (before/after the activity) while sim clips
    # are 0-29 %: compare only windows that are active on BOTH sides (score.ACTIVE_* thresholds).
    active_only: bool = False


# UNIVRFall has ONE lower-back sensor (L1-L2), duplicated into both slots on both sides; the
# duplicate "chest" magnitude/posture features are dropped, the rotation features are kept (they
# describe the same lumbar sensor on both sides). Its accelerometer clips at +/-4 g per axis.
UNIVR_DROP = ("peak_c", "post_incl_c", "dincl_c", "gpk_c")
UNIVR_ACT_DROP = tuple(f"{n}_{s}" for s in ("w", "c") for n in ("astd_x", "astd_z", "gstd_x", "gstd_z")) + tuple(
    f"{n}_c" for n in ("astd_y", "amag_std", "amag_p95", "gstd_y", "gmag_mean", "incl_mean", "incl_std", "fdom",
                       "bp_lo", "bp_mid", "bp_hi", "jerk", "period")) + ("incl_diff",)

EVENTS = [
    EventCategory("fall_forward", (901, 902, 904), "L-TRIP", "medium",
                  "Erciyes forward falls end lying (901 front-lying, 902 protecting with hands, 904 knees then lying); "
                  "volunteers fall from standing on purpose onto a mat, the sim trips while walking. "
                  "Sim run on crash_mat with an Erciyes-like population. 903 (ends on the knees) and 907/908 (oblique) excluded.",
                  protocol=("pre_std_w",)),
    EventCategory("fall_backward", (909, 910, 911, 912), "L-SLIP", "medium",
                  "Erciyes backward falls (909 ends sitting, 910 lying, 911/912 oblique) are deliberate, not slips; the sim "
                  "slips the foot out. Same direction and landing, different initiation.",
                  protocol=("pre_std_w",)),
    EventCategory("fall_side", (913, 915), "STRUCK-KD", "low",
                  "Erciyes sideways falls from standing vs the sim's struck-by knock-down, keeping only left/right blows "
                  "(the knock-down then runs the ordinary same-level side fall). Erciyes has no push.",
                  protocol=("pre_std_w",)),
    EventCategory("syncope", (919,), "SYNC", "medium",
                  "Erciyes 919 syncope: acted faint from standing onto a mat. SYNC: knees buckle, guided descent. "
                  "Acted faints keep some muscle control."),
    EventCategory("fall_recover", (905, 906, 914, 916), "FALL-GETUP", "medium",
                  "Erciyes falls with quick/slow recovery (forward 905/906, sideways 914/916) vs a trip and getting up "
                  "within 3-15 s. Erciyes volunteers are up again within ~2 s, the sim lies 3-15 s by design, so the "
                  "post-fall features are protocol.",
                  protocol=("pre_std_w", "post_incl_w", "post_incl_c", "dincl_w", "dincl_c", "still_acc_w",
                            "still_gyro_c", "settle_s")),
    EventCategory("stumble", (808,), "NF-STUMBLE", "low",
                  "Erciyes 808 stumble with recovery (no fall) vs the sim stumble-and-catch.",
                  onset_event="near_fall_onset"),
    EventCategory("sit_hard", (811, 812), "A-SIT", "medium",
                  "Sitting on a chair (811) or sofa (812) vs A-SIT hard sits on a bucket/bench (first sit of each run). "
                  "A-SIT is always hard, Erciyes sits are normal-speed: a gap here is partly by design.",
                  onset_event="first_sit",
                  protocol=("peak_w", "peak_c", "ffmin_w", "ffdur_w", "jerk_w", "npk_w", "postmin_w")),
    EventCategory("lie_down", (815,), "A-LIE", "medium",
                  "Lying down on a bed (815) vs a controlled lie-down onto the ground.",
                  onset_event="first_lie"),
    # --- UNIVRFall (Zenodo 18346755, CC BY 4.0): lower back L1-L2, 100 Hz, video-labelled onset/impact
    EventCategory("u_trip", (30,), "L-TRIP", "high",
                  "UNIVRFall task 30: forward fall while walking caused by a trip (lab, onto mats [verify]).",
                  dataset="univr", drop=UNIVR_DROP),
    EventCategory("u_slip_back", (34,), "L-SLIP", "high",
                  "UNIVRFall task 34: backward fall while walking caused by a slip.", dataset="univr", drop=UNIVR_DROP),
    EventCategory("u_faint_walk", (28,), "SYNC", "medium",
                  "UNIVRFall task 28: vertical (forward) fall while walking, caused by fainting (acted). The sim "
                  "stands 5-20 s before collapsing.", dataset="univr", drop=UNIVR_DROP, protocol=("pre_std_w",)),
    EventCategory("u_height", (39, 40), "H-LOW", "low",
                  "UNIVRFall tasks 39/40: forward/backward fall from a height. Only 0.17-0.25 s of free fall in "
                  "the data, so a LOW platform; the sim drops from 0.4-0.8 m edges (height_fall, start=edge) "
                  "[verify the platform height].", onset_event="fall_onset", dataset="univr", drop=UNIVR_DROP),
    EventCategory("u_stairs", (41, 42), "STAIR", "low",
                  "UNIVRFall tasks 41/42: backward fall while climbing up/down stairs (few steps, onto mats "
                  "[verify]); STAIR tumbles down 3-12 steps forward or backward.", dataset="univr", drop=UNIVR_DROP),
]

ACTIVITIES = [
    ActivityCategory("walk", "erciyes", (801,), "walk", "high", "Walking forward, waist + chest.", active_only=True),
    ActivityCategory("jog", "erciyes", (803,), "run", "medium", "Jogging vs the sim run (2.5-3.5 m/s).", active_only=True),
    ActivityCategory("bend", "erciyes", (805, 806), "bend", "high", "Bending and picking up vs stoop 60-90 deg.", active_only=True),
    ActivityCategory("squat", "erciyes", (804,), "squat", "high", "Squatting down vs the sim squat.", active_only=True),
    ActivityCategory("walk_site", "vtt", (1013, 1014), "walk", "high",
                     "Construction-site walking (straight, winding); hip + back sensors at +/-2 g."),
    ActivityCategory("cart", "vtt", (1015,), "walk_load", "medium", "Pushing a cart vs walking with a 20 kg load."),
    ActivityCategory("pick", "vtt", (1005,), "pick", "medium", "Picking objects from the floor vs stoop-carry-stoop."),
    ActivityCategory("kneel", "vtt", (1011, 1012), "kneel", "medium", "Crouching/kneeling floor work vs kneel_work."),
    ActivityCategory("overhead", "vtt", (1009,), "overhead", "low",
                     "Hands up high (no tool) vs overhead drilling (adds 20-50 Hz vibration)."),
    ActivityCategory("lie_work", "vtt", (1008,), "lie_work", "medium", "Laying on the back working vs lying_activity."),
    ActivityCategory("jump_down", "vtt", (1007,), "jump", "high", "Jumping down from a step vs jump_down 0.3-0.6 m."),
    ActivityCategory("stairs", "vtt", (1006, 1016), "ladder", "low",
                     "Stairs vs ladder climbing: ThunderSim has no stair walking, the ladder is the nearest primitive."),
    ActivityCategory("u_site", "univr", (), "shift", "medium",
                     "UNIVRFall construction-site recordings (10 workers, 42 h, lower back) vs ThunderSim fall-free "
                     "shift blocks (SIM_SPEC 5.2 activity mix). Horizontal-axis features dropped.",
                     drop=UNIVR_ACT_DROP),
]

UNMAPPED_REAL = {
    903: "front-knees: ends kneeling; no sim scenario ends kneeling after a fall",
    907: "front-right (oblique)", 908: "front-left (oblique)",
    917: "rolling out of bed", 918: "podium fall (from ~0.5 m)",
    920: "syncope sliding down a wall: no sim scenario",
    802: "walking backward", 807: "limp", 809: "trip-over ADL", 810: "coughing", 813: "sit in air",
    814: "sit on bed", 816: "rising from bed",
    1001: "roll painting", 1002: "spraying paint", 1003: "levelling paint", 1004: "vacuum cleaning",
    1010: "hands up low",
}

UNMAPPED_SIM = {
    "H-LAD, H-SCAF, H-ROOF, H-HOLE, H-EXC": "falls from height: no open real recording on disk",
    "ARR, NET": "arrested falls: no real recording",
    "STAIR": "stair falls: no real recording",
    "SYNC-KNEEL": "collapse from kneeling: Erciyes 919 starts standing",
    "NF-RAIL, STRUCK-NF, SLIP-FRICTION": "no labelled real counterpart",
    "A-HAMMER, A-JACK, A-EQUIP, A-REBAR, HAT-OFF, STRAP-SLIP, DEVICE-OFF, A-NAP, A-RUN shifts": "no real counterpart on disk "
    "(VTT has no tools/vehicles; Erciyes has jogging only)",
}
