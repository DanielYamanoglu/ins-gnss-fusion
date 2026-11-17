"""Ground-truth vehicle trajectory plus IMU and GNSS measurements with realistic error models.

Planar motion in a local East-North-Up frame; heading psi is measured from East, counter-clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import enu_to_geodetic

ORIGIN = (39.9334, 32.8597, 900.0)  # lat, lon, height of the local frame (Ankara)
IMU_RATE, GNSS_RATE = 100, 1  # Hz
DURATION = 300.0  # s
OUTAGE = (150.0, 180.0)  # s, GNSS unavailable (tunnel, jamming)


@dataclass(frozen=True)
class ImuSpec:
    """Consumer/industrial MEMS grade. Sigmas are per sample at IMU_RATE."""

    accel_noise: float = 0.05  # m/s^2
    gyro_noise: float = 0.002  # rad/s (about 0.11 deg/s)
    accel_bias_sigma: float = 0.05  # m/s^2, turn-on bias
    gyro_bias_sigma: float = 0.002  # rad/s (about 400 deg/h), turn-on bias
    accel_bias_walk: float = 1e-4  # m/s^2 per sqrt(s)
    gyro_bias_walk: float = 1e-5  # rad/s per sqrt(s)


@dataclass(frozen=True)
class GnssSpec:
    position_sigma: float = 2.5  # m, per horizontal axis


@dataclass
class Run:
    t: np.ndarray  # IMU timestamps
    pos: np.ndarray  # true East, North [m]
    vel: np.ndarray  # true velocity in ENU [m/s]
    psi: np.ndarray  # true heading [rad]
    accel: np.ndarray  # measured specific force in body frame (forward, left) [m/s^2]
    gyro: np.ndarray  # measured yaw rate [rad/s]
    accel_bias: np.ndarray  # true bias history
    gyro_bias: np.ndarray
    gnss_t: np.ndarray  # GNSS epochs that produced a fix
    gnss_lla: np.ndarray  # lat, lon, height of each fix
    gnss_enu: np.ndarray  # same fixes in the local frame (for the baseline)


def speed_and_yaw_rate(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Speed profile s(t), its derivative and yaw rate r(t): accelerate, cruise, turns, slalom, brake."""
    # Speed ramps to 15 m/s over 10 s, then oscillates by +-5 m/s (starting from zero so s is continuous).
    phase = 2 * np.pi * (t - 10.0) / 90.0
    s = np.clip(t / 10.0, 0, 1) * 15.0 + 5.0 * np.sin(phase) * (t > 10)
    ds = np.where(t < 10, 1.5, 0.0) + 5.0 * 2 * np.pi / 90.0 * np.cos(phase) * (t > 10)
    r = np.zeros_like(t)
    r += np.where((t > 40) & (t < 55), np.radians(6.0), 0)  # 90 deg left turn
    r += np.where((t > 100) & (t < 130), np.radians(-6.0), 0)  # 180 deg right turn
    r += np.where((t > 200) & (t < 260), np.radians(8.0) * np.sin(2 * np.pi * (t - 200) / 20.0), 0)  # slalom
    return s, ds, r


def simulate(seed: int, imu: ImuSpec = ImuSpec(), gnss: GnssSpec = GnssSpec(),
             duration: float = DURATION, outage: tuple[float, float] | None = OUTAGE) -> Run:
    rng = np.random.default_rng(seed)
    dt = 1.0 / IMU_RATE
    t = np.arange(0, duration, dt)
    s, ds, r = speed_and_yaw_rate(t)

    # Body-frame acceleration of a vehicle moving forward at speed s while turning at rate r.
    accel_true = np.column_stack([ds, s * r])
    # Integrate the true motion with the same second-order scheme the navigation filter uses, so that
    # error-free sensors reproduce the trajectory exactly and every filter error comes from sensor errors.
    n = len(t)
    pos, vel, psi = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n)
    psi[0] = np.radians(30.0)
    for k in range(1, n):
        c, sn = np.cos(psi[k - 1]), np.sin(psi[k - 1])
        a_nav = np.array([c * accel_true[k - 1, 0] - sn * accel_true[k - 1, 1],
                          sn * accel_true[k - 1, 0] + c * accel_true[k - 1, 1]])
        pos[k] = pos[k - 1] + vel[k - 1] * dt + 0.5 * a_nav * dt**2
        vel[k] = vel[k - 1] + a_nav * dt
        psi[k] = psi[k - 1] + r[k - 1] * dt

    accel_bias = rng.normal(0, imu.accel_bias_sigma, 2) + np.cumsum(
        rng.normal(0, imu.accel_bias_walk * np.sqrt(dt), (n, 2)), axis=0)
    gyro_bias = rng.normal(0, imu.gyro_bias_sigma) + np.cumsum(rng.normal(0, imu.gyro_bias_walk * np.sqrt(dt), n))
    accel = accel_true + accel_bias + rng.normal(0, imu.accel_noise, (n, 2))
    gyro = r + gyro_bias + rng.normal(0, imu.gyro_noise, n)

    step = IMU_RATE // GNSS_RATE
    idx = np.arange(step, n, step)
    if outage is not None:
        idx = idx[(t[idx] < outage[0]) | (t[idx] >= outage[1])]
    gnss_enu = pos[idx] + rng.normal(0, gnss.position_sigma, (len(idx), 2))
    lat, lon, h = enu_to_geodetic(gnss_enu[:, 0], gnss_enu[:, 1], np.zeros(len(idx)), ORIGIN)
    return Run(t, pos, vel, psi, accel, gyro, accel_bias, gyro_bias, t[idx], np.column_stack([lat, lon, h]), gnss_enu)
