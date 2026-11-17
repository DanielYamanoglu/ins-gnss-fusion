"""Static coarse alignment: roll and pitch from the gravity vector measured by the accelerometers.

Heading is not observable at rest without a magnetometer or gyrocompassing (a MEMS gyro cannot sense
Earth rate), so it comes from a launch azimuth with a few degrees of uncertainty and is refined by
the filter once the vehicle accelerates.
"""
from __future__ import annotations

import numpy as np


def level(specific_force: np.ndarray) -> tuple[float, float]:
    """Mean specific force at rest is -g expressed in the body frame."""
    fx, fy, fz = specific_force.mean(axis=0)
    return float(np.arctan2(-fy, -fz)), float(np.arctan2(fx, np.hypot(fy, fz)))
