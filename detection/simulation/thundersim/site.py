# File: site.py
# Purpose: Construction-site physical properties and constraints: surfaces, friction,
#          fall-protection rules, contact mechanics, soft tissue, clothing, temperature, radio.
# Contributors: Unassigned
"""Site physics. Every value is tagged:

[src]    sourced; the reference is next to it.
[design] our starting value. It is randomised over its range per run; check sensitivity to it.
[verify] a sourced claim not yet confirmed from the primary text.

Merged with fall_model_sim_spec/SIM_SPEC.md (research-95). Where both had a value, the
better-sourced one was kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Fall-protection rules -----------------------------------------------------------------
# [src] OSHA 29 CFR 1926.501(b)(1): protect unprotected sides/edges >= 6 ft (1.8 m).
OSHA_PROTECTION_HEIGHT_M = 1.8
# [src] OSHA 1926.502(d)(16)(ii)-(iv): harness arrest force <= 8 kN (1,800 lbf), free fall
# <= 1.8 m, deceleration distance <= 1.07 m (3.5 ft).
HARNESS_MAX_FORCE_N = 8000.0
HARNESS_MAX_FREE_FALL_M = 1.8
HARNESS_MAX_DECEL_DISTANCE_M = 1.07
# [src] OSHA 1926.502(c)(1): safety nets no more than 30 ft (9.1 m) below the work level.
NET_MAX_DEPTH_M = 9.1
# [src] OSHA 1926.502(b)(1): guardrail top rail 42 in +/- 3 in (1.07 m +/- 0.08 m).
GUARDRAIL_TOP_RAIL_M = 1.07
# [src via SIM_SPEC.md] Alberta OHS Code Part 9 s.139: fall protection at 3 m (1.2-3 m in
# some cases). Falls from 1.2-3 m are therefore often unprotected on Alberta sites.
ALBERTA_PROTECTION_HEIGHT_M = 3.0

# --- Impact mechanics (Mode A, body-first) ----------------------------------------------------
# [src via SIM_SPEC.md] hip-protector testing consensus (Robinovitch et al.): effective mass
# 22-33 kg, pelvis stiffness 39-55 kN/m (47 kN/m standard rig), hip impact velocity
# 3.4 +/- 0.5 m/s for same-level falls from standing.
HIP_IMPACT_SPEED_MS = (3.4, 0.5)
BODY_STIFFNESS_N_PER_M = (39e3, 55e3)
BODY_DAMPING_RATIO = (0.2, 0.4)  # [design]; 641-860 Ns/m in the literature gives ~0.3-0.4


def effective_mass(body_mass_kg: float) -> float:
    """[design] Mode-A effective mass: 0.35 x body mass, clipped to 20-40 kg (SIM_SPEC.md 4b)."""
    return float(np.clip(0.35 * body_mass_kg, 20.0, 40.0))


@dataclass(frozen=True)
class Surface:
    """Landing / walking surface. Stiffness and damping are [design] values."""

    name: str
    stiffness_range: tuple[float, float]  # N/m, randomised per run
    damping_ratio: float
    friction_dry: float     # available coefficient of friction for work boots [design]
    heel_strike_g: float    # pelvis heel-strike transient amplitude while walking [design]
    heel_strike_hz: float   # its dominant frequency [design]
    irregularity_m: float   # step-height variability of the walking surface [design]


SURFACES = {
    # Stiffness ranges from SIM_SPEC.md 1.2 [design]; extra surfaces added here [design].
    "concrete": Surface("concrete", (1e7, 1e8), 0.25, 0.65, 0.55, 22.0, 0.002),
    "steel_deck": Surface("steel_deck", (1e7, 1e8), 0.20, 0.45, 0.60, 25.0, 0.002),
    "plywood_deck": Surface("plywood_deck", (3e5, 1e6), 0.25, 0.55, 0.50, 18.0, 0.003),
    "scaffold_plank": Surface("scaffold_plank", (3e5, 8e5), 0.25, 0.50, 0.45, 15.0, 0.005),
    "rebar_mat": Surface("rebar_mat", (5e5, 2e6), 0.25, 0.50, 0.50, 18.0, 0.035),
    "gravel": Surface("gravel", (1e5, 3e5), 0.35, 0.60, 0.35, 12.0, 0.020),
    "packed_soil": Surface("packed_soil", (1e5, 3e5), 0.35, 0.60, 0.35, 12.0, 0.015),
    "mud": Surface("mud", (2e4, 6e4), 0.50, 0.35, 0.20, 8.0, 0.030),
    "crash_mat": Surface("crash_mat", (1.5e4, 3e4), 0.50, 0.70, 0.25, 10.0, 0.005),
    "safety_net": Surface("safety_net", (4e3, 9e3), 0.15, 0.50, 0.0, 0.0, 0.0),
}

# [design] multiplier on available friction by surface condition.
CONDITIONS = {"dry": 1.0, "dusty": 0.8, "wet": 0.6, "muddy": 0.5, "oily": 0.35, "icy": 0.15}
# [verify] required coefficient of friction for level walking is commonly reported ~0.17-0.20.
REQUIRED_FRICTION = 0.20
# [design] slip distance beyond which a slip becomes a fall (shorter slips are recovered).
SLIP_FALL_DISTANCE_M = 0.15


def series_stiffness(k_body: float, k_surface: float) -> float:
    return 1.0 / (1.0 / k_body + 1.0 / k_surface)


def available_friction(surface: str, condition: str) -> float:
    return SURFACES[surface].friction_dry * CONDITIONS[condition]


def slip_distance(heel_speed_ms: float, friction: float) -> float:
    """Sliding distance of a slipping foot decelerated by kinetic friction."""
    return heel_speed_ms**2 / (2 * max(friction, 0.02) * 9.80665)


# --- Soft tissue / strap between skeleton and IMU (SIM_SPEC.md 1.5, all [design]) ---------
# 2nd-order base-excited mount per anchor: natural frequency and damping. Pelvis/lumbar sit
# over more soft tissue (lower f_n); thoracic over scapula/ribs (higher f_n).
SOFT_TISSUE_FN_HZ = {0: (8.0, 16.0), 1: (9.0, 18.0), 2: (14.0, 25.0), 3: (14.0, 25.0)}
SOFT_TISSUE_ZETA = (0.4, 0.9)


@dataclass(frozen=True)
class Clothing:
    """Layers between strap and skin."""

    name: str
    fn_scale: float          # multiplier on soft-tissue f_n
    extra_mount_deg: float   # extra misalignment
    slip_multiplier: float   # strap-slip probability multiplier


CLOTHING = {
    "light": Clothing("light", 1.0, 0.0, 1.0),
    # [design] SIM_SPEC.md 4b: winter layers lower f_n 20-40 %, +3 deg misalignment, slip x2.
    "winter": Clothing("winter", 0.7, 3.0, 2.0),
}

# --- Device temperature (gyro bias drift driver) -------------------------------------------
# [design] the strip sits on the body: device temp ~ 0.6 ambient + 0.4 skin (33 C), plus sun.
SKIN_TEMP_C = 33.0


def device_temperature(ambient_c: np.ndarray, sun: float = 0.0) -> np.ndarray:
    return 0.6 * np.asarray(ambient_c) + 0.4 * SKIN_TEMP_C + 5.0 * sun


# --- Radio (BLE) environments [design] -----------------------------------------------------
@dataclass(frozen=True)
class RadioEnv:
    """Gilbert-Elliott packet loss plus rare multi-packet gaps and link throughput."""

    name: str
    p_good_to_bad: float      # per packet
    p_bad_to_good: float      # per packet
    loss_bad: float           # loss probability while in the bad state
    gap_rate_per_s: float     # rate of 50-300 ms outages (SIM_SPEC.md 2: 0.1 %/s)
    throughput_bytes_s: float  # sustained notification throughput; excess packets are dropped
    latency_s: tuple[float, float]  # connection-interval latency range


RADIO = {
    "open_site": RadioEnv("open_site", 0.002, 0.30, 0.6, 0.001, 30000.0, (0.0075, 0.030)),
    "steel_frame": RadioEnv("steel_frame", 0.004, 0.25, 0.8, 0.003, 20000.0, (0.0075, 0.045)),
    "heavy_equipment": RadioEnv("heavy_equipment", 0.008, 0.20, 0.9, 0.005, 18000.0, (0.0075, 0.060)),
    # SIM_SPEC.md 2 assumes only 2-8 kB/s effective throughput: continuous 100 Hz raw (~15 kB/s)
    # then cannot keep up and the link drops packets; use this to see that failure.
    "constrained": RadioEnv("constrained", 0.005, 0.25, 0.7, 0.001, 6000.0, (0.0075, 0.050)),
}


@dataclass(frozen=True)
class Site:
    """Everything about where the scenario happens."""

    surface: str = "concrete"
    condition: str = "dry"
    clothing: str = "light"
    radio: str = "open_site"
    ambient_c: tuple[float, float] = (18.0, 24.0)  # start, end of the run
    sun: float = 0.0  # 0..1 direct sun on the strip


def random_site(rng: np.random.Generator, surfaces: dict[str, float] | None = None) -> Site:
    """Randomise a site. Default surface mix follows SIM_SPEC.md 3 [design]."""
    mix = surfaces or {"concrete": 0.40, "plywood_deck": 0.25, "packed_soil": 0.15, "gravel": 0.10, "crash_mat": 0.10}
    names = list(mix)
    surface = names[rng.choice(len(names), p=np.array(list(mix.values())) / sum(mix.values()))]
    winter = rng.random() < 0.4  # [design] about 40 % of Alberta runs in winter clothing
    ambient = (rng.uniform(-20, 5), rng.uniform(-15, 8)) if winter else (rng.uniform(10, 28), rng.uniform(15, 35))
    condition = rng.choice(["dry", "dry", "dusty", "wet"] + (["icy"] if winter else []))
    radio = rng.choice(["open_site", "open_site", "steel_frame", "heavy_equipment"])
    return Site(surface, str(condition), "winter" if winter else "light", str(radio), ambient, float(rng.uniform(0, 1) if not winter else 0.0))
