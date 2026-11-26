"""Sprint 3 correctness proofs: particle filter, IMM, BOCPD, enhancement, feedback, ratings."""
import numpy as np
import pytest

from ssp.core import changepoint as cp
from ssp.core import imm, kalman
from ssp.core import particle as pf


def test_resamplers_preserve_weights_in_expectation():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(20))
    for name, f in pf.RESAMPLERS.items():
        counts = np.zeros(20)
        for _ in range(3000):
            counts += np.bincount(f(w, rng), minlength=20)
        assert counts / counts.sum() == pytest.approx(w, abs=0.01), name


def test_particle_filter_equals_kalman_on_linear_gaussian_model():
    """On a linear-Gaussian model the KF is exact, so a big PF must agree with it."""
    q, r, T = 0.5, 1.0, 60
    x, z = kalman.simulate_random_walk(T, q, r, 0.0, np.random.default_rng(1))
    f = kalman.KalmanFilter([[1.0]], [[1.0]], [[q]], [[r]], [0.0], [[1.0]])
    ref = kalman.run_kalman(f, list(z))
    res = pf.bootstrap_pf(list(z), 20000, lambda N, g: g.standard_normal(N),
                          lambda p, t, g: p + np.sqrt(q) * g.standard_normal(p.size),
                          lambda zz, p, t: -0.5 * (zz - p) ** 2 / r, np.random.default_rng(2))
    assert res["mean"] == pytest.approx(ref["x"][:, 0], abs=0.05)
    assert res["std"] == pytest.approx(np.sqrt(ref["P"][:, 0, 0]), abs=0.05)


def test_particle_filter_matches_grid_and_beats_ekf_on_benchmark():
    x, z = pf.ungm_simulate(100, np.random.default_rng(0))
    res = pf.bootstrap_pf(list(z), 2000, lambda N, g: 2 * g.standard_normal(N),
                          lambda p, t, g: pf.ungm_f(p, t) + np.sqrt(10) * g.standard_normal(p.size),
                          lambda zz, p, t: -0.5 * (zz - pf.ungm_h(p)) ** 2, np.random.default_rng(1))
    grid = np.linspace(-40, 40, 801)
    gr = pf.grid_filter(list(z), grid, np.exp(-0.5 * grid**2 / 4), pf.ungm_f, 10.0,
                        lambda zz, g, t: -0.5 * (zz - pf.ungm_h(g)) ** 2)
    ekf = pf.ungm_ekf(z, 10.0, 1.0)
    rmse = lambda e: np.sqrt(np.mean(e**2))
    assert rmse(res["mean"] - gr["mean"]) < 1.0           # PF ≈ exact Bayes
    assert rmse(res["mean"] - x) < 0.6 * rmse(ekf["mean"] - x)


def test_road_localisation_converges_from_unknown_start():
    d = pf.road_simulate(400, 0.1, 1200, np.random.default_rng(2), 1.0, 0.3)
    res = pf.bootstrap_pf(list(d["z"]), 3000, lambda N, g: g.uniform(0, 6000, N),
                          lambda s, t, g: s + (d["v_meas"][t] + 0.3 * g.standard_normal(s.size)) * 0.1 if t else s,
                          lambda zz, s, t: -0.5 * (zz - pf.road_profile(s)) ** 2, np.random.default_rng(3))
    assert res["std"][0] > 1000                            # starts lost
    assert abs(res["mean"][-1] - d["s"][-1]) < 5.0         # ends within a few metres


def test_imm_beats_low_noise_model_and_matches_high_on_turns():
    X = imm.coordinated_turn_truth([(30, 0), (20, 9), (30, 0), (15, -12), (25, 0)], 1.0, 100)
    H = kalman.position_observation(2)
    R = np.eye(2) * 400
    F, Q1 = kalman.constant_velocity(1.0, 0.1, 2)
    _, Q2 = kalman.constant_velocity(1.0, 200.0, 2)
    straight = np.ones(len(X), bool)
    straight[30:55] = straight[80:100] = False
    e_imm, e_high = [], []
    for i in range(20):
        Z = X[:, [0, 2]] + 20 * np.random.default_rng(i).standard_normal((len(X), 2))
        x0 = np.array([Z[0, 0], 100, Z[0, 1], 0.0])
        P0 = np.diag([400, 100, 400, 100])
        o = imm.imm_filter(Z, [F, F], [Q1, Q2], H, R, np.array([[0.95, 0.05], [0.05, 0.95]]), x0, P0, [0.5, 0.5])
        k = kalman.run_kalman(kalman.KalmanFilter(F, H, Q2, R, x0, P0), list(Z))
        e_imm.append(np.linalg.norm(o["x"][:, [0, 2]] - X[:, [0, 2]], axis=1))
        e_high.append(np.linalg.norm(k["x"][:, [0, 2]] - X[:, [0, 2]], axis=1))
        assert np.allclose(o["mu"].sum(axis=1), 1.0)
    e_imm, e_high = np.array(e_imm), np.array(e_high)
    assert np.sqrt(np.mean(e_imm[:, straight] ** 2)) < 0.9 * np.sqrt(np.mean(e_high[:, straight] ** 2))


def test_bocpd_run_length_posterior_and_detection():
    x, cps, _, _ = cp.piecewise_signal(500, 4, 2.5, 1.0, np.random.default_rng(3))
    b = cp.bocpd(x, 100, 0, 0.1, 1, 1)
    assert np.allclose(b["R"].sum(axis=1), 1.0)
    found = cp.detect_changes(b["map"])
    hits = sum(any(abs(c - f) <= 10 for f in found) for c in cps)
    assert hits >= len(cps) - 1
    assert len(found) <= len(cps) + 2


def test_enhancement_improves_snr_and_tracks_changing_noise():
    from ssp.sims import enhance as en
    sc = en.make_scene(8, 5, "car", 10, 1)
    spp = en.enhance(sc["noisy"], "MMSE-LSA (log-spectral)", "spp")
    init = en.enhance(sc["noisy"], "MMSE-LSA (log-spectral)", "initial")
    base = en.snr_db(sc["clean"], sc["noisy"])
    assert en.snr_db(sc["clean"], spp["out"]) > base + 6
    assert en.snr_db(sc["clean"], spp["out"]) > en.snr_db(sc["clean"], init["out"]) + 3
    ml = en.enhance(sc["noisy"], "MMSE-LSA (log-spectral)", "spp", dd_alpha=0.0)
    assert en.noise_attenuation_db(sc["noisy"], ml["out"], sc["mask"]) < \
        en.noise_attenuation_db(sc["noisy"], spp["out"], sc["mask"])


def test_mmse_stsa_gain_limits():
    from ssp.sims.enhance import gain_mmse_stsa, gain_wiener
    xi = np.array([1e3])
    gamma = np.array([1e3])
    assert gain_mmse_stsa(xi, gamma)[0] == pytest.approx(gain_wiener(xi, gamma)[0], abs=0.01)


def test_feedback_canceller_tracks_path_change():
    from ssp.sims import feedback as fb
    cfg = fb.HAConfig(source="speech", gain_margin_db=10, mu=0.002)
    none, fixed, nlms = (fb.simulate(cfg, m) for m in ("none", "fixed", "nlms"))
    assert fb.howling_fraction(none) > 0.9
    assert fb.howling_fraction(fixed) > 0.45                # fine until the path changes
    assert fb.howling_fraction(nlms) < 0.25                 # adapts and recovers
    assert nlms["misalignment"][-1] < -10


def test_kalman_ratings_beat_elo_on_average():
    from ssp.sims import sports as sp
    ll_e, ll_k = [], []
    for seed in range(6):
        lg = sp.simulate_league(sp.LeagueConfig(seed=seed))
        o = lg["outcome"]
        ll_e.append(sp.log_loss(sp.elo(lg, 16, 0.2, 0.3)["pred"], o).mean())
        ll_k.append(sp.log_loss(sp.kalman_ratings(lg, 16, 0.002, 0.3)["pred"], o).mean())
    assert np.mean(ll_k) < np.mean(ll_e)


def test_live_win_probabilities():
    from ssp.sims import sports as sp
    p = sp.win_probabilities(0, 0, 0, 1.5, 1.5)
    assert sum(p) == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[2])                      # symmetric teams
    assert sp.win_probabilities(1, 0, 90, 1.5, 1.5) == (1.0, 0.0, 0.0)
    preds, outcomes = sp.calibration_experiment(1.6, 1.1, 60, 3000, np.random.default_rng(0))
    mp, fr, _ = sp.calibration_bins(preds, outcomes, 5)
    assert np.max(np.abs(mp - fr)) < 0.06                   # calibrated


def test_blind_levels_without_clean_reference():
    from ssp.sims import enhance as en
    sc = en.make_scene(8, 5, "car", 0, 2)
    r = en.enhance(sc["noisy"])
    bl = en.blind_levels(sc["noisy"], r["out"])
    assert bl["noise reduction in pauses (dB)"] > 8
    assert abs(bl["level change on loud speech (dB)"]) < 3
