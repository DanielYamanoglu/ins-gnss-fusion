import numpy as np

from insgnss import geodesy
from insgnss.ekf import InsGnssEkf, propagate, transition_jacobian, wrap
from insgnss.evaluate import run_filter
from insgnss.simulate import IMU_RATE, ORIGIN, ImuSpec, simulate


def test_ecef_of_the_equator_at_the_prime_meridian_is_the_semi_major_axis():
    assert np.allclose(geodesy.geodetic_to_ecef(0.0, 0.0, 0.0), [geodesy.A, 0.0, 0.0])


def test_geodetic_enu_round_trip_and_one_degree_of_latitude():
    e, n, u = np.array([120.0, -3000.0]), np.array([45.0, 2500.0]), np.array([0.0, 10.0])
    lat, lon, h = geodesy.enu_to_geodetic(e, n, u, ORIGIN)
    back = geodesy.geodetic_to_enu(lat, lon, h, ORIGIN)
    assert np.allclose(back, np.vstack([e, n, u]), atol=1e-6)
    # One degree of latitude near 40 N is about 111 km (straight line, so a little under the arc).
    north = geodesy.geodetic_to_enu(ORIGIN[0] + 1, ORIGIN[1], ORIGIN[2], ORIGIN)[1]
    assert 110_900 < north < 111_200


def test_error_free_sensors_reproduce_the_trajectory():
    run = simulate(0, imu=ImuSpec(0, 0, 0, 0, 0, 0), duration=120.0)
    x = np.r_[run.pos[0], run.vel[0], run.psi[0], 0, 0, 0]
    for k in range(1, len(run.t)):
        x = propagate(x, run.accel[k - 1], run.gyro[k - 1], 1 / IMU_RATE)
    assert np.linalg.norm(x[:2] - run.pos[-1]) < 1e-6 and abs(wrap(x[4] - run.psi[-1])) < 1e-9


def test_transition_jacobian_matches_finite_differences():
    x = np.array([10.0, -5.0, 12.0, 3.0, 0.7, 0.03, -0.02, 0.001])
    accel, gyro, dt, eps = np.array([0.8, 1.9]), 0.05, 0.01, 1e-6
    numeric = np.column_stack([
        (propagate(x + eps * np.eye(8)[i], accel, gyro, dt) - propagate(x - eps * np.eye(8)[i], accel, gyro, dt)) / (2 * eps)
        for i in range(8)
    ])
    # The analytic Jacobian is first order in dt; the position rows differ by the 0.5*a*dt^2 term only.
    assert np.allclose(transition_jacobian(x, accel, dt), numeric, atol=dt**2)


def test_gnss_update_shrinks_uncertainty_and_keeps_covariance_valid():
    f = InsGnssEkf.initialise(np.zeros(2), np.array([10.0, 0.0]), 0.0)
    before = np.trace(f.P[:2, :2])
    lat, lon, h = geodesy.enu_to_geodetic(np.array([1.0]), np.array([-2.0]), np.array([0.0]), ORIGIN)
    f.update_gnss(lat[0], lon[0], h[0])
    assert np.trace(f.P[:2, :2]) < before
    assert np.allclose(f.P, f.P.T) and np.all(np.linalg.eigvalsh(f.P) > 0)


def test_filter_beats_raw_gnss_and_bridges_the_outage():
    run = simulate(3)
    out = run_filter(run, 3)
    err = np.linalg.norm(out["est"][:, :2] - run.pos, axis=1)
    gnss_idx = np.round(run.gnss_t * IMU_RATE).astype(int)
    gnss_rmse = np.sqrt(np.mean(np.sum((run.gnss_enu - run.pos[gnss_idx]) ** 2, axis=1)))
    steady = run.t > 60
    assert np.sqrt(np.mean(err[steady & (run.t < 150)] ** 2)) < 0.7 * gnss_rmse
    # Error grows while GNSS is missing and collapses again once fixes return.
    assert err[np.searchsorted(run.t, 179.9)] > err[np.searchsorted(run.t, 150)]
    assert err[np.searchsorted(run.t, 200)] < err[np.searchsorted(run.t, 179.9)]


def test_filter_is_statistically_consistent_over_monte_carlo_runs():
    nees = [run_filter(simulate(s, duration=120.0, outage=None), s)["nees"] for s in range(8)]
    mean = float(np.mean(np.array(nees)[:, 20 * IMU_RATE:]))
    assert 4.0 < mean < 6.5  # expected value 5 for five navigation states
