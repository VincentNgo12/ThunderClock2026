# File: device.py
# Purpose: MPU-6050 strip emulator. Turns true chip-frame motion (1 kHz) into what the
#          firmware would report: DLPF, sample timing, bias/noise/temperature drift,
#          16-bit quantization, full-scale clipping and millis() timestamps.
# Contributors: Unassigned
"""MPU-6050 emulation for the 4-sensor spine strip.

Numbers from the InvenSense MPU-6000/6050 register map rev 4.2 and product spec rev 3.4
(as summarised in research_notes/Sensor signal preprocessing pipeline/imu_signal_cleaning.md):
LSB per g / per deg/s, DLPF bandwidths and delays, 400 ug/sqrt(Hz) accel and
0.005 deg/s/sqrt(Hz) gyro noise density, +/-50 mg (X/Y) / +/-80 mg (Z) zero-g offset,
+/-3 % scale and up to +/-20 deg/s gyro zero-rate offset before calibration.
Calibrated residuals, temperature coefficient and per-sensor read skew are DESIGN
ESTIMATES: measure them on the team's own sensors and update DeviceErrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .model_api import GRAVITY

INTERNAL_HZ = 1000.0  # motion is generated at the MPU-6050's 1 kHz accel output rate

ACCEL_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
GYRO_LSB_PER_DPS = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
DLPF_BANDWIDTHS_HZ = (260, 184, 94, 44, 21, 10, 5)
ACCEL_NOISE_G_PER_RTHZ = 400e-6
GYRO_NOISE_DPS_PER_RTHZ = 0.005


@dataclass(frozen=True)
class DeviceProfile:
    """Firmware acquisition settings. odr_hz is the acquisition (and notification) rate."""

    name: str
    odr_hz: float
    accel_range_g: int
    gyro_range_dps: int
    dlpf_hz: int
    description: str

    def __post_init__(self) -> None:
        if self.accel_range_g not in ACCEL_LSB_PER_G or self.gyro_range_dps not in GYRO_LSB_PER_DPS:
            raise ValueError(f"Unsupported MPU-6050 range in profile {self.name}")
        if self.dlpf_hz not in DLPF_BANDWIDTHS_HZ:
            raise ValueError(f"DLPF bandwidth must be one of {DLPF_BANDWIDTHS_HZ}")


PROFILES = {
    "current": DeviceProfile(
        "current", 0.5, 2, 250, 10,
        "Firmware today (ImuManager.cpp, SystemConfig.h): 2000 ms period, +/-2 g, +/-250 deg/s, 10 Hz DLPF",
    ),
    "current_100hz": DeviceProfile(
        "current_100hz", 100.0, 2, 250, 10,
        "Only the period fixed to 10 ms; range and filter still posture-only",
    ),
    "target": DeviceProfile(
        "target", 100.0, 16, 2000, 44,
        "Report recommendation: 10 ms period, +/-16 g, +/-2000 deg/s, DLPF 44 Hz",
    ),
    "target_8g": DeviceProfile(
        "target_8g", 100.0, 8, 1000, 44,
        "Preprocessing-note option: 10 ms period, +/-8 g, +/-1000 deg/s, DLPF 44 Hz",
    ),
}


@dataclass(frozen=True)
class DeviceErrors:
    """Per-sensor error magnitudes. Each strip draws its own values from these bounds."""

    accel_bias_g: tuple[float, float, float] = (0.010, 0.010, 0.015)  # calibrated residual (design estimate)
    accel_scale: float = 0.005            # +/- fraction, calibrated residual (design estimate)
    gyro_bias_dps: float = 0.3            # residual after startup capture (design estimate)
    gyro_tempco_dps_per_c: float = 0.03   # per axis, design estimate; measure on real units
    read_skew_s: float = 0.0006           # delay between sequential mux reads (design estimate)
    timing_jitter_s: float = 0.0005       # FreeRTOS wake jitter, uniform +/- (design estimate)

    @staticmethod
    def uncalibrated() -> "DeviceErrors":
        """Datasheet worst-case offsets with no calibration at all."""
        return DeviceErrors(accel_bias_g=(0.050, 0.050, 0.080), accel_scale=0.03, gyro_bias_dps=20.0)


@dataclass
class DeviceSamples:
    """Device output for one chunk: K acquisitions of S sensors."""

    t_true: np.ndarray    # (K,) s, true acquisition time in simulation time
    time_ms: np.ndarray   # (K,) int64, firmware millis() at end of the scan (wraps at 2^32 on the wire)
    acc: np.ndarray       # (K,S,3) m/s^2 as the Adafruit driver reports it
    gyro: np.ndarray      # (K,S,3) rad/s
    sat: np.ndarray       # (K,S) bool, any axis at full scale

    @staticmethod
    def empty(sensors: int) -> "DeviceSamples":
        return DeviceSamples(np.zeros(0), np.zeros(0, np.int64), np.zeros((0, sensors, 3)), np.zeros((0, sensors, 3)), np.zeros((0, sensors), bool))


def _dlpf_sos(bandwidth_hz: float) -> np.ndarray:
    # 2nd-order Butterworth approximates the MPU-6050 DLPF magnitude and group delay.
    return signal.butter(2, bandwidth_hz, fs=INTERNAL_HZ, output="sos")


@dataclass
class ImuStrip:
    """Stateful emulator for S sensors; feed consecutive 1 kHz chunks in order."""

    profile: DeviceProfile
    sensors: int = 4
    errors: DeviceErrors = field(default_factory=DeviceErrors)
    seed: int = 0
    boot_offset_ms: int = 5000      # millis() at simulation t=0 (device booted earlier)
    calibration_temp_c: float = 20.0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        s = self.sensors
        self.accel_bias = self.rng.uniform(-1, 1, (s, 3)) * np.array(self.errors.accel_bias_g) * GRAVITY
        self.accel_scale = 1.0 + self.rng.uniform(-1, 1, (s, 3)) * self.errors.accel_scale
        self.gyro_bias = np.deg2rad(self.rng.uniform(-1, 1, (s, 3)) * self.errors.gyro_bias_dps)
        self.gyro_tempco = np.deg2rad(self.rng.uniform(-1, 1, (s, 3)) * self.errors.gyro_tempco_dps_per_c)
        self.sos = _dlpf_sos(self.profile.dlpf_hz)
        self._zi = None                 # filter state, shape (sections, 2, s*6)
        self._prev_t = None             # last filtered sample, kept for interpolation across chunks
        self._prev_y = None
        self._next_acq = 0.0            # next acquisition time (s)
        self._acq_index = 0

    def process(self, t: np.ndarray, acc_true: np.ndarray, gyro_true: np.ndarray, temp_c: np.ndarray | float) -> DeviceSamples:
        """t: (N,) s at 1 kHz; acc_true/gyro_true: (N,S,3) chip-frame specific force and rate."""
        n, s = acc_true.shape[0], self.sensors
        temp = np.broadcast_to(np.asarray(temp_c, float), (n,))
        accel_noise_std = ACCEL_NOISE_G_PER_RTHZ * GRAVITY * np.sqrt(INTERNAL_HZ / 2)
        gyro_noise_std = np.deg2rad(GYRO_NOISE_DPS_PER_RTHZ) * np.sqrt(INTERNAL_HZ / 2)
        acc = acc_true * self.accel_scale + self.accel_bias + self.rng.normal(0, accel_noise_std, acc_true.shape)
        drift = self.gyro_tempco[None] * (temp - self.calibration_temp_c)[:, None, None]
        gyro = gyro_true + self.gyro_bias + drift + self.rng.normal(0, gyro_noise_std, gyro_true.shape)

        stacked = np.concatenate([acc.reshape(n, -1), gyro.reshape(n, -1)], axis=1)
        if self._zi is None:
            # Start the filter in steady state at the first value (no power-on transient).
            self._zi = signal.sosfilt_zi(self.sos)[:, :, None] * stacked[0][None, None, :]
        filtered, self._zi = signal.sosfilt(self.sos, stacked, axis=0, zi=self._zi)

        # Acquisition instants whose last sequential read completes inside this chunk.
        period = 1.0 / self.profile.odr_hz
        if self._prev_t is None:
            self._acq_index = int(np.ceil(t[0] * self.profile.odr_hz))
            self._next_acq = self._acq_index * period
        times = []
        while self._next_acq + (s - 1) * self.errors.read_skew_s <= t[-1]:
            times.append(self._next_acq)
            self._acq_index += 1
            jitter = self.rng.uniform(-1, 1) * self.errors.timing_jitter_s
            self._next_acq = self._acq_index * period + jitter
        # Keep a short tail of the previous chunk so reads straddling the boundary interpolate correctly.
        if self._prev_t is not None:
            grid_t = np.concatenate([self._prev_t, t])
            grid_y = np.concatenate([self._prev_y, filtered])
        else:
            grid_t, grid_y = t, filtered
        self._prev_t, self._prev_y = t[-16:].copy(), filtered[-16:].copy()
        if not times:
            return DeviceSamples.empty(s)

        times_arr = np.asarray(times)
        out = np.empty((len(times_arr), s, 6))
        filtered_cols = grid_y.reshape(len(grid_t), 2, s, 3)
        for sensor in range(s):
            read_t = times_arr + sensor * self.errors.read_skew_s  # sequential mux reads
            for kind in range(2):
                for axis in range(3):
                    out[:, sensor, kind * 3 + axis] = np.interp(read_t, grid_t, filtered_cols[:, kind, sensor, axis])

        accel_lsb = ACCEL_LSB_PER_G[self.profile.accel_range_g] / GRAVITY       # counts per m/s^2
        gyro_lsb = GYRO_LSB_PER_DPS[self.profile.gyro_range_dps] * 180 / np.pi  # counts per rad/s
        acc_counts = np.clip(np.round(out[:, :, :3] * accel_lsb), -32768, 32767)
        gyro_counts = np.clip(np.round(out[:, :, 3:] * gyro_lsb), -32768, 32767)
        sat = (np.abs(acc_counts) >= 32767).any(axis=2) | (np.abs(gyro_counts) >= 32767).any(axis=2)
        scan_end = times_arr + (s - 1) * self.errors.read_skew_s + 0.0002
        time_ms = np.floor(scan_end * 1000).astype(np.int64) + self.boot_offset_ms
        return DeviceSamples(times_arr, time_ms, acc_counts / accel_lsb, gyro_counts / gyro_lsb, sat)


def emulate_recorded(acc: np.ndarray, gyro: np.ndarray, fs_in: float, profile: DeviceProfile | str,
                     seed: int = 0, errors: DeviceErrors | None = None) -> DeviceSamples:
    """Make a public-dataset recording look like a ThunderClock sensor (one direction only).

    acc in m/s^2 (gravity included) and gyro in rad/s, shape (N,3) or (N,S,3), already in
    the canonical segment frame. The signal is upsampled to 1 kHz (no new bandwidth is
    invented), then passes the same DLPF, sampling, noise, quantization and clipping as
    the strip. Use this on training data so models never see impossible values.
    """
    profile = PROFILES[profile] if isinstance(profile, str) else profile
    acc = np.asarray(acc, float)
    gyro = np.asarray(gyro, float)
    if acc.ndim == 2:
        acc, gyro = acc[:, None, :], gyro[:, None, :]
    up = int(round(INTERNAL_HZ))
    down = int(round(fs_in))
    gcd = np.gcd(up, down)
    acc_1k = signal.resample_poly(acc, up // gcd, down // gcd, axis=0)
    gyro_1k = signal.resample_poly(gyro, up // gcd, down // gcd, axis=0)
    t = np.arange(acc_1k.shape[0]) / INTERNAL_HZ
    strip = ImuStrip(profile, sensors=acc.shape[1], errors=errors or DeviceErrors(), seed=seed)
    return strip.process(t, acc_1k, gyro_1k, strip.calibration_temp_c)
