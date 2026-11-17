"""Ground-truth flight of a small fixed-wing UAV, generated at 1 kHz independently of the filter.

The flight is defined by smooth speed, heading and flight-path-angle profiles; attitude follows from
coordinated-turn kinematics. Specific force and angular rate are derived from the 1 kHz truth by
differentiation, so the filter's 100 Hz mechanisation meets real discretisation error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import euler_to_rot, so3_log

TRUTH_RATE = 1000  # Hz
DURATION = 330.0  # s
STATIC = 10.0  # s at rest on the ground for coarse alignment
GRAVITY = 9.80665
G_NED = np.array([0.0, 0.0, GRAVITY])


def smoothstep(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)  # C2-continuous


@dataclass
class Truth:
    t: np.ndarray
    pos: np.ndarray  # NED [m]
    vel: np.ndarray  # NED [m/s]
    R: np.ndarray  # body to NED rotation matrices, shape (n, 3, 3)
    euler: np.ndarray  # roll, pitch, yaw [rad]
    specific_force: np.ndarray  # body frame [m/s^2]
    angular_rate: np.ndarray  # body frame [rad/s]


def profiles(t: np.ndarray):
    speed = 35.0 * smoothstep(t, STATIC, STATIC + 25)
    # Heading: start north-east, 90 deg right turn, 180 deg left turn, then S-turns.
    yaw = (np.radians(45.0) + np.radians(90.0) * smoothstep(t, 80, 100) - np.radians(180.0) * smoothstep(t, 140, 175)
           + np.radians(40.0) * (smoothstep(t, 250, 262) - 2 * smoothstep(t, 268, 286) + smoothstep(t, 292, 304)))
    # Flight path angle: climb, level, descend, level.
    gamma = np.radians(6.0) * (smoothstep(t, 40, 50) - smoothstep(t, 70, 80)) \
        - np.radians(4.0) * (smoothstep(t, 200, 210) - smoothstep(t, 225, 235))
    return speed, yaw, gamma


def generate(duration: float = DURATION) -> Truth:
    dt = 1.0 / TRUTH_RATE
    t = np.arange(0, duration + dt / 2, dt)
    speed, yaw, gamma = profiles(t)
    vel = speed[:, None] * np.column_stack([np.cos(gamma) * np.cos(yaw), np.cos(gamma) * np.sin(yaw), -np.sin(gamma)])
    pos = np.vstack([np.zeros(3), np.cumsum((vel[1:] + vel[:-1]) / 2 * dt, axis=0)])  # trapezoid at 1 kHz
    yaw_rate = np.gradient(yaw, dt)
    roll = np.arctan2(speed * np.cos(gamma) * yaw_rate, GRAVITY)  # coordinated turn
    pitch = gamma  # zero angle of attack
    R = np.stack([euler_to_rot(r, p, y) for r, p, y in zip(roll, pitch, yaw)])
    accel_ned = np.gradient(vel, dt, axis=0)
    specific_force = np.einsum("nji,nj->ni", R, accel_ned - G_NED)
    angular_rate = np.zeros_like(vel)
    for k in range(len(t) - 1):
        angular_rate[k] = so3_log(R[k].T @ R[k + 1]) / dt
    angular_rate[-1] = angular_rate[-2]
    return Truth(t, pos, vel, R, np.column_stack([roll, pitch, yaw]), specific_force, angular_rate)
