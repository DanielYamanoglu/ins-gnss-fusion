"""WGS-84 conversions between geodetic coordinates, ECEF and a local East-North-Up frame."""
from __future__ import annotations

import numpy as np

A = 6_378_137.0  # semi-major axis [m]
F = 1 / 298.257223563
E2 = F * (2 - F)  # first eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)  # prime vertical radius of curvature
    return np.array([
        (n + h) * np.cos(lat) * np.cos(lon),
        (n + h) * np.cos(lat) * np.sin(lon),
        (n * (1 - E2) + h) * np.sin(lat),
    ])


def ecef_to_geodetic(x, y, z, iterations: int = 6):
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(iterations):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


def _ecef_to_enu_rotation(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])


def geodetic_to_enu(lat, lon, h, origin):
    """origin = (lat0, lon0, h0). Returns East, North, Up in metres."""
    d = geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(*origin)[:, None] if np.ndim(lat) else \
        geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(*origin)
    return _ecef_to_enu_rotation(origin[0], origin[1]) @ d


def enu_to_geodetic(e, n, u, origin):
    ecef = geodetic_to_ecef(*origin)[:, None] + _ecef_to_enu_rotation(origin[0], origin[1]).T @ np.vstack([e, n, u])
    lat, lon, h = ecef_to_geodetic(*ecef)
    return lat, lon, h
