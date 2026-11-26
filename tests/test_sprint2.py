"""Sprint 2 correctness proofs: HMMs, GARCH, Kalman regression, HMM VAD."""
import itertools

import numpy as np
import pytest

from ssp.core import finance, hmm

PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.2, 0.8]])
E = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
X = np.array([0, 2, 1, 2, 2, 0])


def _brute_force():
    """Enumerate all 2^T state paths: exact likelihood, MAP path, smoothed marginals."""
    T = len(X)
    total, best, marg = 0.0, (-1.0, None), np.zeros((T, 2))
    for seq in itertools.product(range(2), repeat=T):
        p = PI[seq[0]] * E[seq[0], X[0]]
        for t in range(1, T):
            p *= A[seq[t - 1], seq[t]] * E[seq[t], X[t]]
        total += p
        best = max(best, (p, seq))
        for t, s in enumerate(seq):
            marg[t, s] += p
    return total, np.array(best[1]), marg / total


def test_forward_viterbi_and_smoothing_match_brute_force():
    total, best, marg = _brute_force()
    logB = hmm.categorical_loglik(X, E)
    assert hmm.forward(PI, A, logB)[2] == pytest.approx(np.log(total), abs=1e-12)
    assert np.array_equal(hmm.viterbi(PI, A, logB)[0], best)
    assert hmm.posteriors(PI, A, logB)["gamma"] == pytest.approx(marg, abs=1e-12)


def test_long_sequence_does_not_underflow():
    rng = np.random.default_rng(0)
    s, rolls, true = hmm.sample_casino(20_000, 0.02, 0.1, 0.5, rng)
    ll = hmm.forward(true.pi, true.A, true.loglik_matrix(rolls))[2]
    assert np.isfinite(ll) and ll < 0


def test_casino_decoding_and_baum_welch():
    rng = np.random.default_rng(1)
    s, rolls, true = hmm.sample_casino(5000, 0.02, 0.1, 0.5, rng)
    logB = true.loglik_matrix(rolls)
    path, _ = hmm.viterbi(true.pi, true.A, logB)
    assert np.mean(path == s) > 0.85
    fit = hmm.baum_welch(rolls, hmm.init_categorical(2, 6, np.random.default_rng(3)), n_iter=300)
    ll = fit["loglik"]
    assert np.all(np.diff(ll) > -1e-6)                    # EM never decreases the likelihood
    E_hat = fit["params"].E
    loaded = int(np.argmax(E_hat[:, 5]))
    assert E_hat[loaded, 5] == pytest.approx(0.5, abs=0.06)
    assert E_hat[1 - loaded] == pytest.approx(np.full(6, 1 / 6), abs=0.03)


def test_gaussian_hmm_recovers_regimes():
    from ssp.sims.market import regime_returns
    m = regime_returns(3000, 2, 0.985, np.random.default_rng(2))
    fit = hmm.baum_welch(m["r"], hmm.init_gaussian(m["r"], 2), n_iter=200)
    p = fit["params"]
    sd = np.sqrt(p.variances.ravel())
    assert np.sort(sd) == pytest.approx(np.sort(m["sigma"]), rel=0.1)
    order = np.argsort(sd)                                  # calm = low vol
    path, _ = hmm.viterbi(p.pi, p.A, p.loglik_matrix(m["r"]))
    assert np.mean(np.argsort(order)[path] == m["states"]) > 0.95


def test_garch_mle_recovers_parameters():
    r, _ = finance.garch11_simulate(4000, 0.03, 0.05, 0.08, 0.90, np.random.default_rng(4))
    f = finance.garch11_fit(r)
    assert f["alpha"] == pytest.approx(0.08, abs=0.03)
    assert f["beta"] == pytest.approx(0.90, abs=0.04)
    fc = finance.garch11_forecast(f, 500)
    assert fc[-1] == pytest.approx(f["long_run_var"], rel=0.05)


def test_kalman_regression_tracks_changing_beta_better_than_rolling_ols():
    from ssp.sims.market import pairs
    p = pairs(1500, 1.5, 0.5, 0.6, 0.5, np.random.default_rng(5))
    k = finance.kalman_regression(p["y"], p["x"], delta=1e-4)
    ols = finance.rolling_ols_beta(p["y"], p["x"], 60)
    sl = slice(200, None)
    err_k = np.sqrt(np.mean((k["beta"][sl] - p["beta"][sl]) ** 2))
    err_o = np.sqrt(np.nanmean((ols[sl] - p["beta"][sl]) ** 2))
    assert err_k < err_o


def test_hmm_vad_beats_energy_vad_in_car_noise():
    from ssp.sims import speech as sp
    sc = sp.make_scene(sp.VadScene(snr_db=5, noise="car", seed=1))
    f = sp.frame_features(sc["noisy"])
    y = sp.frame_labels(sc["mask"], f)
    h = sp.hmm_vad(f, 0.99, 0.99, evidence_scale=0.1)
    acc_hmm = sp.scores(h["viterbi"], y, 0.01)
    best_energy = max(sp.scores(sp.energy_vad(f["energy"], m), y, 0.01)["accuracy"] for m in (3, 6, 9, 12))
    assert acc_hmm["accuracy"] > best_energy
    assert acc_hmm["accuracy"] > 0.9
    assert acc_hmm["switches / s"] < sp.scores(sp.energy_vad(f["energy"], 6), y, 0.01)["switches / s"]


def test_wav_roundtrip(tmp_path):
    import io

    from scipy.io import wavfile

    from ssp.sims import speech as sp
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, 8000, x)
    y = sp.load_wav(buf.getvalue())
    assert len(y) == 16000 and np.abs(y).max() <= 1.0


def test_returns_from_csv():
    from ssp.sims.market import returns_from_csv
    csv = b"Date,Open,Close,Volume\n2024-01-02,10,10,5\n2024-01-03,10,11,5\n2024-01-04,11,9.9,5\n"
    r, col = returns_from_csv(csv)
    assert col == "Close" and len(r) == 2
    assert r.iloc[0] == pytest.approx(100 * np.log(1.1))


def test_hedge_mle_gives_calibrated_innovations():
    from ssp.sims.market import pairs
    p = pairs(1200, 1.5, 0.5, 0.6, 0.5, np.random.default_rng(6))
    m = finance.kalman_regression_mle(p["y"], p["x"])
    k = finance.kalman_regression(p["y"], p["x"], m["delta"], m["r"])
    z = k["innov"][50:] / k["innov_sd"][50:]
    assert z.std() == pytest.approx(1.0, abs=0.1)
    assert np.sqrt(np.mean((k["beta"][100:] - p["beta"][100:]) ** 2)) < 0.15
