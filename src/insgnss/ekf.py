"""Loosely coupled INS/GNSS extended Kalman filter.

State x = [pE, pN, vE, vN, psi, ba_f, ba_l, bg]: position, velocity, heading, accelerometer
biases (forward, left) and gyro bias. The IMU drives the prediction (strapdown mechanisation);
each GNSS fix, converted from latitude/longitude to the local frame, corrects it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import geodetic_to_enu
from .simulate import ORIGIN, GnssSpec, ImuSpec

N = 8
NAV = slice(0, 5)  # position, velocity, heading: the states judged by NEES
H = np.zeros((2, N))
H[0, 0] = H[1, 1] = 1.0


def rot(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def propagate(x: np.ndarray, accel: np.ndarray, gyro: float, dt: float) -> np.ndarray:
    """Strapdown mechanisation in the plane: rotate bias-corrected specific force to ENU, integrate."""
    psi = x[4]
    a_nav = rot(psi) @ (accel - x[5:7])
    out = x.copy()
    out[0:2] = x[0:2] + x[2:4] * dt + 0.5 * a_nav * dt**2
    out[2:4] = x[2:4] + a_nav * dt
    out[4] = wrap(psi + (gyro - x[7]) * dt)
    return out


def transition_jacobian(x: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
    psi = x[4]
    f = accel - x[5:7]
    d_rot = np.array([[-np.sin(psi), -np.cos(psi)], [np.cos(psi), -np.sin(psi)]])
    F = np.zeros((N, N))
    F[0:2, 2:4] = np.eye(2)
    F[2:4, 4] = d_rot @ f
    F[2:4, 5:7] = -rot(psi)
    F[4, 7] = -1.0
    return np.eye(N) + F * dt


def process_noise(x: np.ndarray, dt: float, imu: ImuSpec) -> np.ndarray:
    Q = np.zeros((N, N))
    R = rot(x[4])
    Q[2:4, 2:4] = R @ np.diag([(imu.accel_noise * dt) ** 2] * 2) @ R.T
    Q[4, 4] = (imu.gyro_noise * dt) ** 2
    Q[5:7, 5:7] = np.eye(2) * imu.accel_bias_walk**2 * dt
    Q[7, 7] = imu.gyro_bias_walk**2 * dt
    return Q


@dataclass
class InsGnssEkf:
    x: np.ndarray
    P: np.ndarray
    imu: ImuSpec = ImuSpec()
    gnss: GnssSpec = GnssSpec()

    @classmethod
    def initialise(cls, pos, vel, psi, imu: ImuSpec = ImuSpec(), gnss: GnssSpec = GnssSpec(),
                   pos_sigma=2.5, vel_sigma=0.5, psi_sigma=np.radians(2.0)) -> "InsGnssEkf":
        x = np.r_[pos, vel, psi, 0.0, 0.0, 0.0]
        P = np.diag([pos_sigma**2] * 2 + [vel_sigma**2] * 2 + [psi_sigma**2]
                    + [imu.accel_bias_sigma**2] * 2 + [imu.gyro_bias_sigma**2])
        return cls(x, P, imu, gnss)

    def predict(self, accel: np.ndarray, gyro: float, dt: float) -> None:
        Phi = transition_jacobian(self.x, accel, dt)
        self.x = propagate(self.x, accel, gyro, dt)
        self.P = Phi @ self.P @ Phi.T + process_noise(self.x, dt, self.imu)

    def update_gnss(self, lat: float, lon: float, h: float) -> float:
        """Correct with one GNSS fix; returns the normalised innovation squared."""
        z = geodetic_to_enu(lat, lon, h, ORIGIN)[:2]
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * self.gnss.position_sigma**2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[4] = wrap(self.x[4])
        IKH = np.eye(N) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ (np.eye(2) * self.gnss.position_sigma**2) @ K.T  # Joseph form
        return float(innovation @ np.linalg.solve(S, innovation))
