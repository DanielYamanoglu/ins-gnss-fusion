"""Run the filter and its baselines on one simulated flight, and over many Monte Carlo runs."""
from __future__ import annotations

import numpy as np

from .alignment import level
from .ekf import BA, BG, GATE, N, P_, TH, V_, InsGnssEkf, State, mechanise
from .rotations import euler_to_rot, quat_to_rot, rot_to_quat, so3_log
from .sensors import IMU_RATE, OUTAGE, GnssSpec, ImuSpec, Measurements, measure
from .trajectory import STATIC, TRUTH_RATE, Truth, generate

ALIGN_TIME = 5.0  # s of static data used for levelling; navigation starts afterwards
YAW_INIT_SIGMA = np.radians(3.0)
INIT_SEED_OFFSET = 10_000
NAV_STATES = 9  # position, velocity, attitude: the states judged by NEES


def initialise(truth: Truth, meas: Measurements, seed: int, imu: ImuSpec, gnss: GnssSpec) -> tuple[InsGnssEkf, int]:
    rng = np.random.default_rng(seed + INIT_SEED_OFFSET)
    k0 = int(ALIGN_TIME * IMU_RATE)
    roll, pitch = level(meas.specific_force[:k0])
    yaw = truth.euler[0, 2] + rng.normal(0, YAW_INIT_SIGMA)
    x = State(truth.pos[k0 * (TRUTH_RATE // IMU_RATE)] + rng.normal(0, 1, 3) * [2, 2, 4], np.zeros(3),
              rot_to_quat(euler_to_rot(roll, pitch, yaw)), np.zeros(3), np.zeros(3))
    P = np.zeros((N, N))
    P[P_, P_] = np.diag([2.0, 2.0, 4.0]) ** 2
    P[V_, V_] = np.eye(3) * 0.05**2  # at rest
    P[TH, TH] = np.diag([np.radians(0.3), np.radians(0.3), YAW_INIT_SIGMA]) ** 2
    P[BA, BA] = np.eye(3) * imu.accel_bias_sigma**2
    P[BG, BG] = np.eye(3) * imu.gyro_bias_sigma**2
    return InsGnssEkf(x, P, imu, gnss), k0


def nav_error(truth: Truth, k_truth: int, x: State) -> np.ndarray:
    """True minus estimated, with the attitude error in the filter's local convention."""
    att = so3_log(quat_to_rot(x.q).T @ truth.R[k_truth])
    return np.r_[truth.pos[k_truth] - x.p, truth.vel[k_truth] - x.v, att]


def run_filter(truth: Truth, meas: Measurements, seed: int, imu: ImuSpec = ImuSpec(), gnss: GnssSpec = GnssSpec(),
               mode: str = "fused", gating: bool = True) -> dict:
    filt, k0 = initialise(truth, meas, seed, imu, gnss)
    filt.gating = gating
    dt = 1.0 / IMU_RATE
    step = TRUTH_RATE // IMU_RATE
    fixes = {int(round(t * IMU_RATE)): i for i, t in enumerate(meas.gnss_t)}
    n = len(meas.t)
    err = np.full((n, NAV_STATES), np.nan)
    sigma = np.full((n, N), np.nan)
    nees = np.full(n, np.nan)
    bias = np.full((n, 6), np.nan)
    rejected = np.zeros(len(meas.gnss_t), dtype=bool)
    for k in range(k0, n):
        if k > k0:
            if mode == "dead_reckoning":
                filt.x = mechanise(filt.x, meas.specific_force[k - 1], meas.angular_rate[k - 1], dt)
            else:
                filt.predict(meas.specific_force[k - 1], meas.angular_rate[k - 1], dt)
        if mode == "fused" and k in fixes:
            i = fixes[k]
            rejected[i] = filt.update_gnss(meas.gnss_lla[i], meas.gnss_vel[i]) > GATE and gating
        e = nav_error(truth, k * step, filt.x)
        err[k] = e
        sigma[k] = np.sqrt(np.diag(filt.P))
        nees[k] = e @ np.linalg.solve(filt.P[:NAV_STATES, :NAV_STATES], e)
        bias[k] = np.r_[filt.x.ba, filt.x.bg]
    return {"t": meas.t, "err": err, "sigma": sigma, "nees": nees, "bias": bias, "rejected": rejected}


def monte_carlo(runs: int = 50, compare_ungated: int = 20) -> dict:
    truth = generate()
    keys = ["pos", "horiz", "vel", "roll_pitch", "yaw", "nees", "gyro_bias_err", "dr_horiz"]
    out = {k: [] for k in keys}
    gnss_err, gnss_err_clean, ungated_horiz = [], [], []
    gate = {"outliers": 0, "caught": 0, "valid": 0, "valid_rejected": 0}
    for seed in range(runs):
        meas = measure(truth, seed)
        fused = run_filter(truth, meas, seed)
        dr = run_filter(truth, meas, seed, mode="dead_reckoning")
        e = fused["err"]
        out["horiz"].append(np.linalg.norm(e[:, 0:2], axis=1))
        out["pos"].append(np.linalg.norm(e[:, 0:3], axis=1))
        out["vel"].append(np.linalg.norm(e[:, 3:6], axis=1))
        out["roll_pitch"].append(np.linalg.norm(e[:, 6:8], axis=1))
        out["yaw"].append(np.abs(e[:, 8]))
        out["nees"].append(fused["nees"])
        out["gyro_bias_err"].append(np.linalg.norm(fused["bias"][:, 3:6] - meas.gyro_bias, axis=1))
        out["dr_horiz"].append(np.linalg.norm(dr["err"][:, 0:2], axis=1))
        idx = np.round(meas.gnss_t * TRUTH_RATE).astype(int)
        horiz = np.linalg.norm(meas.gnss_ned[:, :2] - truth.pos[idx, :2], axis=1)
        gnss_err.append(horiz)
        gnss_err_clean.append(horiz[~meas.gnss_outlier])
        # Only fixes inside the navigation window (after alignment, up to the last IMU sample) reach the filter.
        used = (meas.gnss_t >= ALIGN_TIME) & (meas.gnss_t <= meas.t[-1])
        out_mask, rej = meas.gnss_outlier & used, fused["rejected"]
        gate["outliers"] += int(out_mask.sum())
        gate["caught"] += int((out_mask & rej).sum())
        gate["valid"] += int((~meas.gnss_outlier & used).sum())
        gate["valid_rejected"] += int((~meas.gnss_outlier & used & rej).sum())
        if seed < compare_ungated:
            ungated = run_filter(truth, meas, seed, gating=False)
            ungated_horiz.append(np.linalg.norm(ungated["err"][:, 0:2], axis=1))
    result = {k: np.array(v) for k, v in out.items()}
    result.update({"t": meas.t, "gnss_horiz_err": np.concatenate(gnss_err), "runs": runs, "outage": OUTAGE,
                   "gnss_horiz_err_clean": np.concatenate(gnss_err_clean), "gate": gate,
                   "ungated_horiz": np.array(ungated_horiz), "gated_horiz_subset": result_subset(out["horiz"], len(ungated_horiz)),
                   "static": STATIC, "align_time": ALIGN_TIME})
    return result


def result_subset(rows: list, n: int) -> np.ndarray:
    return np.array(rows[:n])
