"""15-state error-state extended Kalman filter for loosely coupled INS/GNSS in a local NED frame.

Nominal state: position, velocity (NED), attitude quaternion (body to NED), accelerometer and gyro
biases. Error state (15): position, velocity, attitude error (local, small angle), both biases.
Earth rotation and transport rate are neglected: over 10 km and with a 50 deg/h gyro they are well
below the sensor errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import geodetic_to_ned
from .rotations import quat_exp, quat_mul, quat_to_rot, skew, so3_exp
from .sensors import ORIGIN, GnssSpec, ImuSpec
from .trajectory import G_NED

N = 15
P_, V_, TH, BA, BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
H = np.zeros((6, N))
H[0:3, P_] = np.eye(3)
H[3:6, V_] = np.eye(3)
GATE = 22.46  # chi-square, 6 dof, 99.9%: a valid fix is rejected once in a thousand


@dataclass
class State:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    ba: np.ndarray
    bg: np.ndarray

    def copy(self) -> "State":
        return State(self.p.copy(), self.v.copy(), self.q.copy(), self.ba.copy(), self.bg.copy())


def mechanise(x: State, f_b: np.ndarray, w_b: np.ndarray, dt: float) -> State:
    """Strapdown: rotate bias-corrected specific force to NED, add gravity, integrate; update attitude."""
    R = quat_to_rot(x.q)
    a = R @ (f_b - x.ba) + G_NED
    q = quat_mul(x.q, quat_exp((w_b - x.bg) * dt))
    return State(x.p + x.v * dt + 0.5 * a * dt**2, x.v + a * dt, q / np.linalg.norm(q), x.ba, x.bg)


def error_transition(x: State, f_b: np.ndarray, w_b: np.ndarray, dt: float) -> np.ndarray:
    R = quat_to_rot(x.q)
    F = np.eye(N)
    F[P_, V_] = np.eye(3) * dt
    # Position also sees attitude and accelerometer errors through the 0.5*a*dt^2 term.
    F[P_, TH] = -0.5 * R @ skew(f_b - x.ba) * dt**2
    F[P_, BA] = -0.5 * R * dt**2
    F[V_, TH] = -R @ skew(f_b - x.ba) * dt
    F[V_, BA] = -R * dt
    F[TH, TH] = so3_exp((w_b - x.bg) * dt).T
    F[TH, BG] = -np.eye(3) * dt
    return F


def process_noise(x: State, dt: float, imu: ImuSpec) -> np.ndarray:
    Q = np.zeros((N, N))
    Q[V_, V_] = np.eye(3) * imu.accel_noise_density**2 * dt
    Q[TH, TH] = np.eye(3) * imu.gyro_noise_density**2 * dt
    Q[BA, BA] = np.eye(3) * imu.accel_bias_walk_density**2 * dt
    Q[BG, BG] = np.eye(3) * imu.gyro_bias_walk_density**2 * dt
    return Q


@dataclass
class InsGnssEkf:
    x: State
    P: np.ndarray
    imu: ImuSpec
    gnss: GnssSpec
    gating: bool = True
    rejected: int = 0

    def predict(self, f_b: np.ndarray, w_b: np.ndarray, dt: float) -> None:
        F = error_transition(self.x, f_b, w_b, dt)
        self.x = mechanise(self.x, f_b, w_b, dt)
        self.P = F @ self.P @ F.T + process_noise(self.x, dt, self.imu)

    def update_gnss(self, lla: np.ndarray, vel_ned: np.ndarray) -> float:
        """Position (from lat/lon/height) and velocity update. Returns the NIS; gated fixes are skipped."""
        z = np.r_[geodetic_to_ned(*lla, ORIGIN), vel_ned]
        r = np.r_[self.gnss.horizontal_sigma, self.gnss.horizontal_sigma, self.gnss.vertical_sigma,
                  [self.gnss.velocity_sigma] * 3] ** 2
        Rm = np.diag(r)
        y = z - np.r_[self.x.p, self.x.v]
        S = H @ self.P @ H.T + Rm
        nis = float(y @ np.linalg.solve(S, y))
        if self.gating and nis > GATE:
            self.rejected += 1
            return nis
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        IKH = np.eye(N) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T  # Joseph form
        self.x.p = self.x.p + dx[P_]
        self.x.v = self.x.v + dx[V_]
        q = quat_mul(self.x.q, quat_exp(dx[TH]))
        self.x.q = q / np.linalg.norm(q)
        self.x.ba = self.x.ba + dx[BA]
        self.x.bg = self.x.bg + dx[BG]
        return nis
