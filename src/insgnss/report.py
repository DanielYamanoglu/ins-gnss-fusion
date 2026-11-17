"""Monte Carlo summary numbers and figures for the README."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ekf import NAV
from .evaluate import monte_carlo, run_filter
from .simulate import OUTAGE, simulate

ROOT = Path(__file__).resolve().parents[2]
SETTLE = 20.0  # s, initial convergence excluded from the steady-state numbers
NEES_DOF = NAV.stop - NAV.start


def chi2_band(runs: int, dof: int) -> tuple[float, float]:
    """95% band for the Monte Carlo average of NEES (Wilson-Hilferty approximation, no scipy)."""
    k = runs * dof
    z = 1.959964

    def q(sign):
        return k * (1 - 2 / (9 * k) + sign * z * np.sqrt(2 / (9 * k))) ** 3 / runs

    return q(-1), q(+1)


def summarise(mc: dict) -> dict:
    t = mc["t"]
    o0, o1 = mc["outage"]
    steady = (t > SETTLE) & ((t < o0) | (t > o1 + SETTLE))
    end_outage = np.searchsorted(t, o1) - 1
    lo, hi = chi2_band(mc["runs"], NEES_DOF)
    mean_nees = mc["nees"].mean(axis=0)
    rms = lambda a: float(np.sqrt(np.mean(a[:, steady] ** 2)))  # noqa: E731
    gap = int((o1 - o0) * 100)
    return {
        "runs": mc["runs"],
        "position_rmse_m": {"ins_gnss_ekf": rms(mc["pos_err"]), "gnss_only": float(np.sqrt(np.mean(mc["gnss_err"] ** 2)))},
        "velocity_rmse_mps": rms(mc["vel_err"]),
        "heading_rmse_deg": float(np.degrees(rms(mc["psi_err"]))),
        "outage_s": o1 - o0,
        "error_at_end_of_outage_m": {"mean": float(mc["pos_err"][:, end_outage].mean()),
                                     "p95": float(np.percentile(mc["pos_err"][:, end_outage], 95))},
        "uncalibrated_dead_reckoning_after_same_gap_m": float(mc["dr_err"][:, gap].mean()),
        "nees": {"mean": float(mc["nees"][:, steady].mean()), "dof": NEES_DOF, "band95": [lo, hi],
                 "share_of_time_in_band": float(np.mean((mean_nees[steady] > lo) & (mean_nees[steady] < hi)))},
        "gyro_bias_error_deg_per_h": {"start": float(np.degrees(mc["gyro_bias_err"][:, 0].mean()) * 3600),
                                      "end": float(np.degrees(mc["gyro_bias_err"][:, -1].mean()) * 3600)},
    }


def figures(mc: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t = mc["t"]
    o0, o1 = mc["outage"]

    run = simulate(0)
    fused = run_filter(run, 0)
    dr = run_filter(run, 0, dead_reckoning=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(run.pos[:, 0], run.pos[:, 1], color="#0f172a", lw=2, label="truth")
    ax.scatter(run.gnss_enu[:, 0], run.gnss_enu[:, 1], s=6, color="#94a3b8", label="GNSS fixes (1 Hz)")
    ax.plot(fused["est"][:, 0], fused["est"][:, 1], color="#0284c7", lw=1.2, label="INS/GNSS EKF")
    keep = dr["est"][:, :2]
    keep = keep[np.linalg.norm(keep - run.pos, axis=1) < 400]
    ax.plot(keep[:, 0], keep[:, 1], color="#dc2626", lw=1, ls="--", label="IMU dead reckoning (until 400 m off)")
    gap = (t >= o0) & (t < o1)
    ax.plot(run.pos[gap, 0], run.pos[gap, 1], color="#d97706", lw=6, alpha=0.35, label=f"GNSS outage ({o1 - o0:.0f} s)")
    ax.set(title="One run: 300 s drive, 4.5 km", xlabel="East [m]", ylabel="North [m]")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.axvspan(o0, o1, color="#d97706", alpha=0.15, label="GNSS outage")
    ax.plot(t, mc["pos_err"].mean(axis=0), color="#0284c7", label="EKF, mean of runs")
    ax.fill_between(t, np.percentile(mc["pos_err"], 5, axis=0), np.percentile(mc["pos_err"], 95, axis=0),
                    color="#0284c7", alpha=0.2, label="EKF, 5th to 95th percentile")
    ax.axhline(np.sqrt(np.mean(mc["gnss_err"] ** 2)), color="#475569", ls=":", label="GNSS only, RMSE")
    ax.set(title="Horizontal position error", xlabel="time [s]", ylabel="m", ylim=(0, 10))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    lo, hi = chi2_band(mc["runs"], NEES_DOF)
    ax.axvspan(o0, o1, color="#d97706", alpha=0.15)
    ax.plot(t, mc["nees"].mean(axis=0), color="#0284c7", lw=0.8, label="average NEES")
    ax.axhspan(lo, hi, color="#16a34a", alpha=0.2, label="95% chi-square band")
    ax.set(title=f"Filter consistency ({mc['runs']} runs)", xlabel="time [s]", ylabel="NEES (5 states)", ylim=(0, 12))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    est, sig = fused["est"][:, 7], fused["sigma"][:, 7]
    to_dph = np.degrees(1) * 3600
    ax.plot(t, run.gyro_bias * to_dph, color="#0f172a", label="true gyro bias")
    ax.plot(t, est * to_dph, color="#0284c7", label="estimate")
    ax.fill_between(t, (est - 3 * sig) * to_dph, (est + 3 * sig) * to_dph, color="#0284c7", alpha=0.2, label="3 sigma")
    ax.axvspan(o0, o1, color="#d97706", alpha=0.15)
    ax.set(title="Gyro bias calibration in flight", xlabel="time [s]", ylabel="deg/h")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "monte_carlo.png", dpi=130)
    plt.close(fig)


def run(runs: int = 50) -> dict:
    mc = monte_carlo(runs)
    summary = summarise(mc)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2))
    figures(mc, ROOT / "reports" / "figures")
    return summary
