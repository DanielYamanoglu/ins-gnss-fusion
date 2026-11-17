"""SO(3) and quaternion helpers. Quaternions are [w, x, y, z], Hamilton convention, body to navigation."""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.eye(3) + skew(phi)
    k = skew(phi / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def so3_log(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-9:
        return 0.5 * w
    return angle / (2 * np.sin(angle)) * w


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_exp(phi: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.r_[1.0, 0.5 * phi]
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * phi / angle]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aerospace ZYX convention: body to NED."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])


def rot_to_euler(R: np.ndarray) -> np.ndarray:
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    return q / np.linalg.norm(q)
