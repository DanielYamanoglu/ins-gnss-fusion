"""IMU and GNSS measurement models, specified in datasheet units."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import ned_to_geodetic
from .trajectory import TRUTH_RATE, Truth

ORIGIN = (39.9334, 32.8597, 900.0)  # lat, lon, height of the navigation frame origin (Ankara)
IMU_RATE, GNSS_RATE = 100, 1  # Hz
OUTAGE = (180.0, 240.0)  # s without GNSS: tunnel, jamming or masking during the descent
DEG = np.pi / 180
MG = 9.80665e-3


@dataclass(frozen=True)
class ImuSpec:
    """Industrial-grade MEMS IMU class (not a specific product)."""

    gyro_arw: float = 0.3  # angle random walk [deg/sqrt(h)]
    accel_vrw: float = 0.1  # velocity random walk [m/s/sqrt(h)]
    gyro_turn_on_bias: float = 50.0  # 1 sigma, constant per run [deg/h]
    accel_turn_on_bias: float = 2.0  # 1 sigma [mg]
    gyro_bias_walk: float = 2.0  # rate random walk [deg/h/sqrt(h)]
    accel_bias_walk: float = 0.05  # acceleration random walk [mg/sqrt(h)]

    # Conversions to SI densities.
    @property
    def gyro_noise_density(self) -> float:  # rad/s/sqrt(Hz)
        return self.gyro_arw * DEG / 60.0

    @property
    def accel_noise_density(self) -> float:  # m/s^2/sqrt(Hz)
        return self.accel_vrw / 60.0

    @property
    def gyro_bias_sigma(self) -> float:  # rad/s
        return self.gyro_turn_on_bias * DEG / 3600.0

    @property
    def accel_bias_sigma(self) -> float:  # m/s^2
        return self.accel_turn_on_bias * MG

    @property
    def gyro_bias_walk_density(self) -> float:  # rad/s/sqrt(s)
        return self.gyro_bias_walk * DEG / 3600.0 / 60.0

    @property
    def accel_bias_walk_density(self) -> float:  # m/s^2/sqrt(s)
        return self.accel_bias_walk * MG / 60.0


@dataclass(frozen=True)
class GnssSpec:
    horizontal_sigma: float = 2.0  # m per axis
    vertical_sigma: float = 4.0  # m
    velocity_sigma: float = 0.1  # m/s per axis
    outlier_rate: float = 0.03  # share of fixes hit by a multipath or spoofing jump
    outlier_size: float = 30.0  # m


@dataclass
class Measurements:
    t: np.ndarray  # IMU timestamps [s]
    specific_force: np.ndarray  # body frame, averaged over each IMU interval
    angular_rate: np.ndarray
    gyro_bias: np.ndarray  # true bias history, for evaluation
    accel_bias: np.ndarray
    gnss_t: np.ndarray
    gnss_lla: np.ndarray  # lat, lon, height of each fix
    gnss_vel: np.ndarray  # NED velocity [m/s]
    gnss_ned: np.ndarray  # fixes in the navigation frame, for the GNSS-only baseline
    gnss_outlier: np.ndarray  # which fixes carry an injected jump


def measure(truth: Truth, seed: int, imu: ImuSpec = ImuSpec(), gnss: GnssSpec = GnssSpec(),
            outage: tuple[float, float] | None = OUTAGE) -> Measurements:
    rng = np.random.default_rng(seed)
    step = TRUTH_RATE // IMU_RATE
    dt = 1.0 / IMU_RATE
    n = (len(truth.t) - 1) // step
    # A real IMU reports increments over its interval: average the 1 kHz truth over each window.
    f = truth.specific_force[: n * step].reshape(n, step, 3).mean(axis=1)
    w = truth.angular_rate[: n * step].reshape(n, step, 3).mean(axis=1)
    t = truth.t[: n * step : step]

    gyro_bias = rng.normal(0, imu.gyro_bias_sigma, 3) + np.cumsum(
        rng.normal(0, imu.gyro_bias_walk_density * np.sqrt(dt), (n, 3)), axis=0)
    accel_bias = rng.normal(0, imu.accel_bias_sigma, 3) + np.cumsum(
        rng.normal(0, imu.accel_bias_walk_density * np.sqrt(dt), (n, 3)), axis=0)
    gyro = w + gyro_bias + rng.normal(0, imu.gyro_noise_density / np.sqrt(dt), (n, 3))
    accel = f + accel_bias + rng.normal(0, imu.accel_noise_density / np.sqrt(dt), (n, 3))

    gstep = TRUTH_RATE // GNSS_RATE
    idx = np.arange(gstep, len(truth.t), gstep)
    if outage is not None:
        idx = idx[(truth.t[idx] < outage[0]) | (truth.t[idx] >= outage[1])]
    sig = np.array([gnss.horizontal_sigma, gnss.horizontal_sigma, gnss.vertical_sigma])
    ned = truth.pos[idx] + rng.normal(0, 1, (len(idx), 3)) * sig
    outlier = rng.uniform(size=len(idx)) < gnss.outlier_rate
    direction = rng.normal(size=(len(idx), 2))
    ned[outlier, :2] += gnss.outlier_size * direction[outlier] / np.linalg.norm(direction[outlier], axis=1, keepdims=True)
    vel = truth.vel[idx] + rng.normal(0, gnss.velocity_sigma, (len(idx), 3))
    lat, lon, h = ned_to_geodetic(ned[:, 0], ned[:, 1], ned[:, 2], ORIGIN)
    return Measurements(t, accel, gyro, gyro_bias, accel_bias, truth.t[idx], np.column_stack([lat, lon, h]),
                        vel, ned, outlier)
