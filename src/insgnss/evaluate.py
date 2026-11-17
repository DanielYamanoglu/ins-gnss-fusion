"""Run the filter against dead reckoning and raw GNSS, over many independent Monte Carlo runs."""
from __future__ import annotations

import numpy as np

from .ekf import NAV, InsGnssEkf, propagate, wrap
from .simulate import IMU_RATE, OUTAGE, Run, simulate

INIT_SEED_OFFSET = 10_000


def run_filter(run: Run, seed: int, dead_reckoning: bool = False) -> dict:
    """Filter one run. Initial position, velocity and heading carry realistic errors."""
    rng = np.random.default_rng(seed + INIT_SEED_OFFSET)
    filt = InsGnssEkf.initialise(run.pos[0] + rng.normal(0, 2.5, 2), run.vel[0] + rng.normal(0, 0.5, 2),
                                 run.psi[0] + rng.normal(0, np.radians(2.0)))
    dt = 1.0 / IMU_RATE
    n = len(run.t)
    est = np.zeros((n, 8))
    sigma = np.zeros((n, 8))
    nees = np.zeros(n)
    fix = {round(t * IMU_RATE): i for i, t in enumerate(run.gnss_t)}
    for k in range(n):
        if k:
            if dead_reckoning:
                filt.x = propagate(filt.x, run.accel[k - 1], run.gyro[k - 1], dt)
            else:
                filt.predict(run.accel[k - 1], run.gyro[k - 1], dt)
        if not dead_reckoning and k in fix:
            filt.update_gnss(*run.gnss_lla[fix[k]])
        est[k] = filt.x
        sigma[k] = np.sqrt(np.diag(filt.P))
        err = errors(run, k, filt.x)
        nees[k] = err @ np.linalg.solve(filt.P[NAV, NAV], err)
    return {"est": est, "sigma": sigma, "nees": nees}


def errors(run: Run, k: int, x: np.ndarray) -> np.ndarray:
    return np.r_[x[0:2] - run.pos[k], x[2:4] - run.vel[k], wrap(x[4] - run.psi[k])]


def monte_carlo(runs: int = 50) -> dict:
    pos_err, vel_err, psi_err, dr_err, gnss_err, nees, gyro_bias_err = [], [], [], [], [], [], []
    for seed in range(runs):
        run = simulate(seed)
        fused = run_filter(run, seed)
        dr = run_filter(run, seed, dead_reckoning=True)
        pos_err.append(np.linalg.norm(fused["est"][:, 0:2] - run.pos, axis=1))
        vel_err.append(np.linalg.norm(fused["est"][:, 2:4] - run.vel, axis=1))
        psi_err.append(np.abs(wrap(fused["est"][:, 4] - run.psi)))
        dr_err.append(np.linalg.norm(dr["est"][:, 0:2] - run.pos, axis=1))
        gnss_idx = np.round(run.gnss_t * IMU_RATE).astype(int)
        gnss_err.append(np.linalg.norm(run.gnss_enu - run.pos[gnss_idx], axis=1))
        nees.append(fused["nees"])
        gyro_bias_err.append(np.abs(fused["est"][:, 7] - run.gyro_bias))
    return {"t": run.t, "pos_err": np.array(pos_err), "vel_err": np.array(vel_err), "psi_err": np.array(psi_err),
            "dr_err": np.array(dr_err), "gnss_err": np.concatenate(gnss_err), "nees": np.array(nees),
            "gyro_bias_err": np.array(gyro_bias_err), "runs": runs, "outage": OUTAGE}
