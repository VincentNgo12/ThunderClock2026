# File: heat.py
# Purpose: Humidex from AHT21 temperature/humidity, with Environment Canada comfort bands.
# Contributors: Suvir Duggal

import math

# Environment Canada humidex guidance: (lower bound, label).
_BANDS = (
    (45.0, "dangerous"),
    (40.0, "great_discomfort"),
    (30.0, "some_discomfort"),
    (float("-inf"), "comfortable"),
)


def dew_point_c(temperature_c: float, humidity_percent: float) -> float:
    """Magnus approximation; humidity must be above 0%."""
    a, b = 17.62, 243.12
    gamma = math.log(humidity_percent / 100.0) + a * temperature_c / (b + temperature_c)
    return b * gamma / (a - gamma)


def humidex(temperature_c: float | None, humidity_percent: float | None) -> float | None:
    """Humidex, or None when either input is missing or out of range."""
    if temperature_c is None or humidity_percent is None:
        return None
    if not 0.0 < humidity_percent <= 100.0:
        return None
    dew_k = dew_point_c(temperature_c, humidity_percent) + 273.15
    vapour_hpa = 6.11 * math.exp(5417.7530 * (1 / 273.16 - 1 / dew_k))
    return temperature_c + 0.5555 * (vapour_hpa - 10.0)


def humidex_band(value: float | None) -> str | None:
    if value is None:
        return None
    return next(label for bound, label in _BANDS if value >= bound)
