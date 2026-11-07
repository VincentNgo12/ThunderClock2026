# File: test_site_scenario.py
# Purpose: The scripted scenario hits each demo moment with plausible values.
# Contributors: Suvir Duggal

from app.heat import humidex, humidex_band
from app.site_scenario import PERIOD_S, phase_at, scenario_at


def test_normal_phase_is_comfortable():
    environment, bpm, spo2 = scenario_at(10)
    assert phase_at(10) == "normal"
    assert humidex_band(humidex(environment.temperature_c, environment.humidity_percent)) == "comfortable"
    assert 70 < bpm < 90 and spo2 >= 97


def test_heat_phase_reaches_great_discomfort_with_raised_heart_rate():
    environment, bpm, _ = scenario_at(42)
    assert humidex(environment.temperature_c, environment.humidity_percent) >= 40
    assert bpm > 110


def test_solvent_and_noise_spikes():
    solvent, _, _ = scenario_at(54)
    loud, _, _ = scenario_at(67)
    assert solvent.tvoc_ppb > 1000 and solvent.air_quality_index >= 4
    assert loud.noise_relative_db > 85


def test_values_stay_in_contract_ranges_and_loop():
    for tenth in range(int(PERIOD_S * 10)):
        environment, bpm, spo2 = scenario_at(tenth / 10)
        assert 0 <= environment.humidity_percent <= 100
        assert 1 <= environment.air_quality_index <= 5
        assert environment.eco2_ppm >= 400 and environment.tvoc_ppb >= 0
        assert 40 < bpm < 200 and 0 < spo2 <= 100
    assert scenario_at(5) == scenario_at(5 + PERIOD_S)
