"""Correctness proofs for the from-scratch algorithms.

Each test checks a *theoretical* property (CRLB, chi-square consistency,
convergence to the Wiener solution) or agreement with a reference library.
"""
import numpy as np
import pytest

from ssp.core import adaptive, estimation, kalman, probability


# ---------------------------------------------------------------- probability
def test_beta_update_and_bayes_rule():
    a, b = probability.beta_bernoulli_update(2, 2, 7, 3)
    assert (a, b) == (9, 5)
    # textbook: 1% prior, 99% sensitivity, 95% specificity -> ~16.7%
    p = probability.posterior_given_alarm(0.01, 0.99, 0.95)
    assert p == pytest.approx(0.0099 / (0.0099 + 0.0495))


def test_markov_stationary_matches_long_run():
    P = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    pi = probability.stationary_distribution(P)
    assert pi @ P == pytest.approx(pi, abs=1e-12)
    path = probability.sample_markov_chain(P, 0, 200_000, np.random.default_rng(0))
    emp = np.bincount(path, minlength=3) / path.size
    assert emp == pytest.approx(pi, abs=0.01)


# ---------------------------------------------------------------- estimation
def test_dc_mle_is_efficient_and_map_beats_crlb_with_right_prior():
    rng = np.random.default_rng(1)
    r = estimation.monte_carlo_dc([5, 20, 80], sigma=1.0, true_A=0.0, prior_mean=0.0,
                                  prior_std=0.5, trials=20_000, rng=rng, draw_A_from_prior=True)
    # MLE variance reaches the CRLB (it is efficient)
    assert r["MLE"] == pytest.approx(r["CRLB"], rel=0.05)
    # Bayesian MSE of MAP is below the CRLB, because MAP is biased
    assert np.all(r["MAP"] < r["CRLB"])


def test_map_posterior_matches_closed_form():
    x = np.array([1.0, 2.0, 3.0])
    m, s = estimation.dc_map(x, sigma=1.0, prior_mean=0.0, prior_std=1.0)
    assert m == pytest.approx(6.0 / 4.0)
    assert s == pytest.approx(np.sqrt(1 / 4.0))


def test_bernoulli_estimators_order():
    r = estimation.bernoulli_estimators(k=3, n=4, alpha=2, beta=2)
    assert r["MLE"] == pytest.approx(0.75)
    assert r["MAP"] == pytest.approx(4 / 6)
    assert r["MMSE"] == pytest.approx(5 / 8)


def test_frequency_mle_reaches_crlb_at_high_snr():
    rng = np.random.default_rng(2)
    r = estimation.monte_carlo_freq([20.0], N=64, f0=0.1234, trials=300, rng=rng)
    assert r["MLE"][0] == pytest.approx(r["CRLB"][0], rel=0.35)


def test_median_beats_mean_with_outliers():
    rng = np.random.default_rng(3)
    r = estimation.monte_carlo_outliers(50, 1.0, [0.0, 0.2], 20.0, 3000, rng)
    assert r["mean"][0] < r["median"][0]      # Gaussian noise: mean wins
    assert r["median"][1] < r["mean"][1] / 5  # impulsive noise: median wins big


# ---------------------------------------------------------------- adaptive
@pytest.fixture
def sysid():
    rng = np.random.default_rng(4)
    h = adaptive.room_like_ir(8, 0.3, rng)
    p = adaptive.make_system_id(4000, h, None, None, 0.01, 0.0, rng)
    return h, p


def test_wiener_recovers_system(sysid):
    h, p = sysid
    assert adaptive.wiener_solution(p["x"], p["d"], 8) == pytest.approx(h, abs=0.01)


@pytest.mark.parametrize("algo,kw", [("lms", {"mu": 0.01}), ("nlms", {"mu": 0.5}),
                                     ("rls", {"lam": 0.999})])
def test_adaptive_filters_converge(sysid, algo, kw):
    h, p = sysid
    r = getattr(adaptive, algo)(p["x"], p["d"], 8, **kw)
    assert adaptive.misalignment_db(r["W"], h)[-1] < -25


def test_lms_diverges_with_large_step(sysid):
    h, p = sysid
    r = adaptive.lms(p["x"], p["d"], 8, mu=0.5)
    assert np.isnan(r["W"][-1]).all()


# ---------------------------------------------------------------- kalman
def test_kalman_matches_filterpy():
    filterpy = pytest.importorskip("filterpy.kalman")
    rng = np.random.default_rng(5)
    F, Q = kalman.constant_velocity(0.1, 0.5, dims=2)
    H = kalman.position_observation(2)
    R = np.eye(2) * 0.3
    X, Z = kalman.simulate_cv_2d(200, 0.1, 0.5, 0.3, rng)
    ours = kalman.run_kalman(kalman.KalmanFilter(F, H, Q, R, np.zeros(4), np.eye(4) * 10), list(Z))
    ref = filterpy.KalmanFilter(dim_x=4, dim_z=2)
    ref.F, ref.H, ref.Q, ref.R, ref.x, ref.P = F, H, Q, R, np.zeros(4), np.eye(4) * 10
    xs = []
    for z in Z:
        ref.predict()
        ref.update(z)
        xs.append(ref.x.copy())
    assert ours["x"] == pytest.approx(np.array(xs), abs=1e-9)
    # RTS smoother against filterpy's
    xs_s, _ = kalman.rts_smoother(ours["x"], ours["P"], ours["x_pred"], ours["P_pred"], F)
    ref_s, _, _, _ = ref.rts_smoother(ours["x"], ours["P"], Fs=[F] * 200, Qs=[Q] * 200)
    assert xs_s == pytest.approx(ref_s, abs=1e-8)


def test_kalman_is_consistent_nees_in_chi2_band():
    """With the right model, average NEES over runs sits inside the chi2 band."""
    runs, n, dt, q, r = 50, 150, 0.1, 0.5, 0.3
    F, Q = kalman.constant_velocity(dt, q, dims=2)
    H = kalman.position_observation(2)
    total = np.zeros(n)
    for i in range(runs):
        rng = np.random.default_rng(100 + i)
        X, Z = kalman.simulate_cv_2d(n, dt, q, r, rng, x0=(0, 0, 0, 0))
        kf = kalman.KalmanFilter(F, H, Q, np.eye(2) * r, np.zeros(4), np.eye(4) * 1e-6)
        out = kalman.run_kalman(kf, list(Z))
        total += kalman.nees(X, out["x"], out["P"])
    avg = total / runs
    lo, hi = kalman.chi2_band(4, runs)
    inside = np.mean((avg > lo) & (avg < hi))
    assert inside > 0.85
    assert avg.mean() == pytest.approx(4.0, rel=0.15)


def test_missing_measurements_grow_uncertainty():
    F, Q = kalman.constant_velocity(0.1, 1.0, dims=1)
    kf = kalman.KalmanFilter(F, kalman.position_observation(1), Q, [[0.1]], [0, 0], np.eye(2))
    out = kalman.run_kalman(kf, [0.0] * 20 + [None] * 20)
    var = out["P"][:, 0, 0]
    assert np.all(np.diff(var[20:]) > 0)


# ---------------------------------------------------------------- domain sims
def test_vehicle_ekf_estimates_gyro_bias_and_beats_dead_reckoning():
    from ssp.sims.vehicle import VehicleConfig, simulate_vehicle
    r = simulate_vehicle(VehicleConfig())
    assert r["ekf"][-1, 4] == pytest.approx(r["true_bias"], abs=0.003)
    assert np.sqrt(np.mean(r["err_ekf"] ** 2)) < np.sqrt(np.mean(r["gps_err"] ** 2))
    assert np.mean(r["err_ekf"]) < 0.05 * np.mean(r["err_dr"])
    assert np.mean(r["err_ekf"] < r["bound95"]) > 0.9


def test_netcode_kalman_beats_naive_methods():
    from ssp.sims.netcode import NetConfig, simulate_netcode, summarize
    s = summarize(simulate_netcode(NetConfig())["err"])
    assert s["kalman"]["mean"] < s["snap"]["mean"]
    assert s["kalman"]["mean"] < s["extrapolate"]["mean"]
    assert s["kalman"]["mean"] < s["interpolate"]["mean"]
