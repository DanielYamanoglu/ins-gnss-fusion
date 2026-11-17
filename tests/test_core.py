import numpy as np
import pytest

from insgnss import geodesy
from insgnss.alignment import level
from insgnss.ekf import GATE, InsGnssEkf, State, error_transition, mechanise
from insgnss.evaluate import run_filter
from insgnss.rotations import euler_to_rot, quat_exp, quat_mul, quat_to_rot, rot_to_euler, rot_to_quat, so3_exp, so3_log
from insgnss.sensors import ORIGIN, GnssSpec, ImuSpec, measure
from insgnss.trajectory import GRAVITY, generate

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def truth():
    return generate()


def test_ecef_and_ned_round_trips():
    assert np.allclose(geodesy.geodetic_to_ecef(0.0, 0.0, 0.0), [geodesy.A, 0.0, 0.0])
    n, e, d = np.array([120.0, -3000.0]), np.array([45.0, 2500.0]), np.array([-30.0, 10.0])
    lat, lon, h = geodesy.ned_to_geodetic(n, e, d, ORIGIN)
    assert np.allclose(geodesy.geodetic_to_ned(lat, lon, h, ORIGIN), np.vstack([n, e, d]), atol=1e-6)
    north = geodesy.geodetic_to_ned(ORIGIN[0] + 1, ORIGIN[1], ORIGIN[2], ORIGIN)[0]
    assert 110_900 < north < 111_200  # one degree of latitude near 40 N, straight line


def test_rotation_helpers_agree():
    roll, pitch, yaw = 0.3, -0.2, 2.5
    R = euler_to_rot(roll, pitch, yaw)
    assert np.allclose(rot_to_euler(R), [roll, pitch, yaw])
    assert np.allclose(quat_to_rot(rot_to_quat(R)), R)
    phi = np.array([0.1, -0.4, 0.25])
    assert np.allclose(so3_log(so3_exp(phi)), phi)
    assert np.allclose(quat_to_rot(quat_exp(phi)), so3_exp(phi))


def test_truth_has_gravity_at_rest_and_coordinated_turns(truth):
    assert np.allclose(truth.specific_force[100], [0, 0, -GRAVITY], atol=1e-9)
    # A coordinated turn needs no side force: the lateral specific force stays near zero.
    assert np.abs(truth.specific_force[:, 1]).max() < 0.05


def test_error_free_imu_follows_the_independent_truth(truth):
    """The truth comes from a 1 kHz model, not the filter: only discretisation error remains."""
    meas = measure(truth, 0, imu=ImuSpec(0, 0, 0, 0, 0, 0))
    x = State(truth.pos[0].copy(), truth.vel[0].copy(), rot_to_quat(truth.R[0]), np.zeros(3), np.zeros(3))
    for k in range(1, len(meas.t)):
        x = mechanise(x, meas.specific_force[k - 1], meas.angular_rate[k - 1], 0.01)
    k_truth = (len(meas.t) - 1) * 10
    assert 0 < np.linalg.norm(x.p - truth.pos[k_truth]) < 3.0  # metres after 10.8 km and 330 s
    assert np.linalg.norm(so3_log(quat_to_rot(x.q).T @ truth.R[k_truth])) < np.radians(0.01)


def _boxplus(x: State, dx: np.ndarray) -> State:
    return State(x.p + dx[0:3], x.v + dx[3:6], quat_mul(x.q, quat_exp(dx[6:9])), x.ba + dx[9:12], x.bg + dx[12:15])


def _boxminus(a: State, b: State) -> np.ndarray:
    return np.r_[a.p - b.p, a.v - b.v, so3_log(quat_to_rot(b.q).T @ quat_to_rot(a.q)), a.ba - b.ba, a.bg - b.bg]


def test_error_state_jacobian_predicts_how_a_small_error_propagates():
    x = State(np.array([10.0, -5.0, -100.0]), np.array([30.0, 5.0, -1.0]), rot_to_quat(euler_to_rot(0.2, 0.05, 1.0)),
              np.array([0.01, -0.02, 0.005]), np.radians([0.01, -0.02, 0.015]))
    f, w, dt = np.array([0.5, 0.1, -9.9]), np.array([0.01, -0.02, 0.1]), 0.01
    dx = RNG.normal(0, 1e-4, 15)
    actual = _boxminus(mechanise(_boxplus(x, dx), f, w, dt), mechanise(x, f, w, dt))
    # Exact to first order in the error; the remaining gap is the second-order attitude-bias cross term.
    assert np.allclose(error_transition(x, f, w, dt) @ dx, actual, atol=2e-9)


def test_static_levelling_recovers_roll_and_pitch():
    R = euler_to_rot(np.radians(3.0), np.radians(-2.0), 1.2)
    f = np.tile(R.T @ np.array([0, 0, -GRAVITY]), (500, 1)) + RNG.normal(0, 0.01, (500, 3))
    roll, pitch = level(f)
    assert abs(roll - np.radians(3.0)) < 1e-3 and abs(pitch + np.radians(2.0)) < 1e-3


def test_gnss_update_shrinks_uncertainty_and_the_gate_rejects_a_jump():
    imu, gnss = ImuSpec(), GnssSpec()
    x = State(np.zeros(3), np.array([30.0, 0, 0]), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    filt = InsGnssEkf(x, np.eye(15) * 4.0, imu, gnss)
    lla = np.array(geodesy.ned_to_geodetic(np.array([1.0]), np.array([-1.0]), np.array([0.0]), ORIGIN)).ravel()
    before = np.trace(filt.P[0:3, 0:3])
    filt.update_gnss(lla, np.array([30.0, 0, 0]))
    assert np.trace(filt.P[0:3, 0:3]) < before
    assert np.allclose(filt.P, filt.P.T) and np.all(np.linalg.eigvalsh(filt.P) > 0)
    jump = np.array(geodesy.ned_to_geodetic(np.array([31.0]), np.array([-1.0]), np.array([0.0]), ORIGIN)).ravel()
    assert filt.update_gnss(jump, np.array([30.0, 0, 0])) > GATE and filt.rejected == 1


def test_filter_bridges_the_outage_and_recovers(truth):
    meas = measure(truth, 3)
    err = np.linalg.norm(run_filter(truth, meas, 3)["err"][:, 0:2], axis=1)
    at = lambda s: err[int(s * 100)]  # noqa: E731
    assert at(170) < 2.0  # GNSS available
    assert at(239.9) > at(181)  # error grows while GNSS is missing
    assert at(250) < 3.0  # and collapses once fixes return


def test_filter_is_statistically_consistent():
    short = generate(duration=120.0)
    nees = [run_filter(short, measure(short, s, outage=None), s)["nees"] for s in range(6)]
    mean = float(np.nanmean(np.array(nees)[:, 3000:]))
    assert 7.0 < mean < 11.5  # expected value 9 for nine navigation states
