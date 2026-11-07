# File: test_heat.py
# Purpose: Humidex values against published reference points.
# Contributors: Suvir Duggal

from app.heat import humidex, humidex_band


def test_humidex_matches_reference_points():
    # Environment Canada table: 30 °C at 70% RH gives a humidex near 41.
    assert abs(humidex(30.0, 70.0) - 41.0) < 1.0
    # 25 °C at 50% RH gives a humidex near 29.
    assert abs(humidex(25.0, 50.0) - 29.0) < 1.0


def test_missing_or_invalid_inputs_are_unavailable():
    assert humidex(None, 50.0) is None
    assert humidex(25.0, None) is None
    assert humidex(25.0, 0.0) is None


def test_bands():
    assert humidex_band(25.0) == "comfortable"
    assert humidex_band(35.0) == "some_discomfort"
    assert humidex_band(42.0) == "great_discomfort"
    assert humidex_band(47.0) == "dangerous"
    assert humidex_band(None) is None
