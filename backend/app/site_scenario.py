# File: site_scenario.py
# Purpose: Scripted 90-second construction-site story for simulated environment and vitals.
# Contributors: Suvir Duggal

import math

from .models import Environment

PERIOD_S = 90.0

# Each phase: (start second, label). The values below ramp smoothly between phases.
PHASES = (
    (0.0, "normal"),        # Mild site, resting-ish worker.
    (20.0, "heat"),         # Afternoon heat builds; heart rate climbs with it.
    (45.0, "solvent"),      # Worker near adhesives/paint: TVOC and eCO2 spike.
    (60.0, "loud"),         # Saw or jackhammer nearby: noise peaks.
    (75.0, "recovery"),     # Break in the shade; everything settles.
)


def phase_at(elapsed_s: float) -> str:
    t = elapsed_s % PERIOD_S
    return [label for start, label in PHASES if t >= start][-1]


def _ramp(t: float, start: float, end: float) -> float:
    """0 before start, 1 after end, smooth in between."""
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    x = (t - start) / (end - start)
    return x * x * (3 - 2 * x)


def _wobble(t: float, period: float, size: float) -> float:
    return size * math.sin(2 * math.pi * t / period)


def scenario_at(elapsed_s: float) -> tuple[Environment, float, float]:
    """Environment, heart rate (bpm) and SpO2 (%) for this point in the loop."""
    t = elapsed_s % PERIOD_S
    heat = _ramp(t, 20, 40) * (1 - _ramp(t, 75, 88))
    solvent = _ramp(t, 45, 50) * (1 - _ramp(t, 58, 66))
    loud = _ramp(t, 60, 62) * (1 - _ramp(t, 72, 75))

    temperature_c = 24 + 10 * heat + _wobble(t, 13, 0.2)
    humidity_percent = 40 + 25 * heat + _wobble(t, 17, 0.8)
    tvoc_ppb = int(80 + 1400 * solvent + _wobble(t, 7, 10))
    eco2_ppm = int(450 + 750 * solvent + 150 * heat + _wobble(t, 11, 15))
    air_quality_index = 1 + round(3 * solvent) + (1 if heat > 0.5 else 0)
    noise_db = 62 + 30 * loud + _wobble(t, 3, 2.5)

    bpm = 78 + 45 * heat + 10 * loud + _wobble(t, 9, 2)
    spo2 = 98 - 1.5 * heat + _wobble(t, 23, 0.3)

    environment = Environment(
        temperature_c=round(temperature_c, 1),
        humidity_percent=round(humidity_percent, 1),
        air_quality_index=min(air_quality_index, 5),
        eco2_ppm=eco2_ppm,
        tvoc_ppb=tvoc_ppb,
        noise_relative_db=round(noise_db, 1),
    )
    return environment, round(bpm, 1), round(min(spo2, 100.0), 1)
