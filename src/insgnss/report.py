"""Monte Carlo summary numbers and figures for the README."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import NAV_STATES, monte_carlo, run_filter
from .sensors import ImuSpec, measure
from .trajectory import generate

ROOT = Path(__file__).resolve().parents[2]
SETTLE = 30.0  # s after navigation starts and after the outage, excluded from steady state


def chi2_band(runs: int, dof: int) -> tuple[float, float]:
    """95% band for the Monte Carlo average of NEES (Wilson-Hilferty approximation)."""
    k, z = runs * dof, 1.959964
    return tuple(k * (1 - 2 / (9 * k) + s * z * np.sqrt(2 / (9 * k))) ** 3 / runs for s in (-1, 1))


def summarise(mc: dict) -> dict:
    t = mc["t"]
    o0, o1 = mc["outage"]
    start = mc["align_time"] + SETTLE
    steady = (t > start) & ((t < o0) | (t > o1 + SETTLE))
    end = np.searchsorted(t, o1) - 1
    rms = lambda a: float(np.sqrt(np.nanmean(a[:, steady] ** 2)))  # noqa: E731
    lo, hi = chi2_band(mc["runs"], NAV_STATES)
    mean_nees = np.nanmean(mc["nees"], axis=0)
    gap_end = np.searchsorted(t, mc["align_time"] + (o1 - o0)) - 1
    spec = ImuSpec()
    return {
        "runs": mc["runs"],
        "imu": {"gyro_arw_deg_sqrt_h": spec.gyro_arw, "accel_vrw_mps_sqrt_h": spec.accel_vrw,
                "gyro_turn_on_bias_deg_h": spec.gyro_turn_on_bias, "accel_turn_on_bias_mg": spec.accel_turn_on_bias},
        "horizontal_rmse_m": {"ins_gnss": rms(mc["horiz"]),
                              "gnss_only_all_fixes": float(np.sqrt(np.mean(mc["gnss_horiz_err"] ** 2))),
                              "gnss_only_without_outliers": float(np.sqrt(np.mean(mc["gnss_horiz_err_clean"] ** 2)))},
        "velocity_rmse_mps": rms(mc["vel"]),
        "roll_pitch_rmse_deg": float(np.degrees(rms(mc["roll_pitch"]))),
        "yaw_rmse_deg": float(np.degrees(rms(mc["yaw"]))),
        "outage_s": o1 - o0,
        "horizontal_error_end_of_outage_m": {"mean": float(np.mean(mc["horiz"][:, end])),
                                             "p95": float(np.percentile(mc["horiz"][:, end], 95))},
        "dead_reckoning_after_same_time_from_alignment_m": float(np.nanmean(mc["dr_horiz"][:, gap_end])),
        "gyro_bias_error_deg_h": {"after_alignment": float(np.degrees(np.nanmean(mc["gyro_bias_err"][:, int(mc["align_time"] * 100)])) * 3600),
                                  "end": float(np.degrees(np.mean(mc["gyro_bias_err"][:, -1])) * 3600)},
        "nees": {"mean": float(np.nanmean(mc["nees"][:, steady])), "dof": NAV_STATES, "band95": [lo, hi],
                 "share_of_time_in_band": float(np.mean((mean_nees[steady] > lo) & (mean_nees[steady] < hi)))},
        "outlier_gate": mc["gate"],
        "outlier_gate_effect_horizontal_rmse_m": {"gated": rms(mc["gated_horiz_subset"]), "ungated": rms(mc["ungated_horiz"]),
                                                  "runs": int(len(mc["ungated_horiz"]))},
    }


def figures(mc: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t = mc["t"]
    o0, o1 = mc["outage"]
    truth = generate()
    meas = measure(truth, 0)
    fused = run_filter(truth, meas, 0)
    k_truth = np.arange(len(t)) * 10

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    est = truth.pos[k_truth] - np.nan_to_num(fused["err"][:, 0:3])
    ax.plot(truth.pos[:, 1], truth.pos[:, 0], color="#0f172a", lw=2, label="truth")
    good = ~meas.gnss_outlier
    ax.scatter(meas.gnss_ned[good, 1], meas.gnss_ned[good, 0], s=5, color="#94a3b8", label="GNSS fixes (1 Hz)")
    ax.scatter(meas.gnss_ned[~good, 1], meas.gnss_ned[~good, 0], s=30, marker="x", color="#dc2626", label="GNSS outliers (injected)")
    nav = t > 5
    ax.plot(est[nav, 1], est[nav, 0], color="#0284c7", lw=1.2, label="INS/GNSS EKF")
    gap = (truth.t >= o0) & (truth.t < o1)
    ax.plot(truth.pos[gap, 1], truth.pos[gap, 0], color="#d97706", lw=7, alpha=0.35, label=f"GNSS outage ({o1 - o0:.0f} s)")
    ax.set(title="One flight: 330 s, 10.8 km", xlabel="East [m]", ylabel="North [m]")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    ax.plot(truth.t, -truth.pos[:, 2], color="#0f172a", label="altitude [m]")
    ax.plot(truth.t, np.degrees(truth.euler[:, 0]), color="#16a34a", label="roll [deg]")
    ax.axvspan(o0, o1, color="#d97706", alpha=0.15)
    ax.set(title="Flight profile", xlabel="time [s]")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    shade = dict(color="#d97706", alpha=0.15)
    ax = axes[0, 0]
    ax.axvspan(o0, o1, **shade, label="GNSS outage")
    ax.plot(t, np.nanmean(mc["horiz"], axis=0), color="#0284c7", label="EKF, mean")
    ax.fill_between(t, np.nanpercentile(mc["horiz"], 5, axis=0), np.nanpercentile(mc["horiz"], 95, axis=0),
                    color="#0284c7", alpha=0.2, label="EKF, 5th to 95th percentile")
    ax.axhline(np.sqrt(np.mean(mc["gnss_horiz_err_clean"] ** 2)), color="#475569", ls=":", label="GNSS only, RMSE without outliers")
    ax.set(title="Horizontal position error", xlabel="time [s]", ylabel="m", ylim=(0, 25))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    to_deg = np.degrees(1)
    ax.axvspan(o0, o1, **shade)
    ax.plot(t, np.nanmean(mc["roll_pitch"], axis=0) * to_deg, color="#16a34a", label="roll / pitch error, mean")
    ax.plot(t, np.nanmean(mc["yaw"], axis=0) * to_deg, color="#0284c7", label="yaw error, mean")
    ax.set(title="Attitude error", xlabel="time [s]", ylabel="deg", yscale="log")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    lo, hi = chi2_band(mc["runs"], NAV_STATES)
    ax.axvspan(o0, o1, **shade)
    ax.plot(t, np.nanmean(mc["nees"], axis=0), color="#0284c7", lw=0.8, label="average NEES")
    ax.axhspan(lo, hi, color="#16a34a", alpha=0.2, label="95% chi-square band")
    ax.set(title=f"Filter consistency ({mc['runs']} runs, 9 states)", xlabel="time [s]", ylabel="NEES", ylim=(0, 20))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    dph = to_deg * 3600
    ax.axvspan(o0, o1, **shade)
    for i, c in zip(range(3), ("#0284c7", "#16a34a", "#d97706")):
        ax.plot(t, meas.gyro_bias[:, i] * dph, color=c, ls="--", lw=1)
        ax.plot(t, fused["bias"][:, 3 + i] * dph, color=c, lw=1.2, label="xyz"[i] + " axis: estimate (solid), truth (dashed)")
    ax.set(title="Gyro bias calibration in flight (one run)", xlabel="time [s]", ylabel="deg/h")
    ax.legend(frameon=False, fontsize=8)
    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "monte_carlo.png", dpi=130)
    plt.close(fig)


def run(runs: int = 50) -> dict:
    warnings.filterwarnings("ignore", message="(Mean of empty slice|All-NaN slice)", category=RuntimeWarning)
    mc = monte_carlo(runs, compare_ungated=min(20, runs))
    summary = summarise(mc)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2))
    figures(mc, ROOT / "reports" / "figures")
    return summary
