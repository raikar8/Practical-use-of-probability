"""Correctness proofs for Part II (graphical models, EM, MCMC/VI, GPs, bandits) and Part III (neural nets,
flows, VAE, GAN, diffusion)."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ssp.core import approx as ap
from ssp.core import bandits as bd
from ssp.core import generative as gen
from ssp.core import gp, graphical as gm, hmm, kalman, mixtures as mx, nn


# ----------------------------------------------------------------------------- neural-network core
@pytest.mark.parametrize("act", list(nn.ACTS))
def test_mlp_backprop_matches_finite_differences(act):
    rng = np.random.default_rng(0)
    net = nn.MLP([3, 5, 4, 2], act=act, rng=rng)
    x, t = rng.standard_normal((7, 3)), rng.standard_normal((7, 2))

    def loss():
        return float(0.5 * np.sum((net(x) - t) ** 2))
    y = net(x)
    dx = net.backward(y - t)
    for p, g in zip(net.params(), net.grads()):
        assert nn.numerical_grad(loss, p) == pytest.approx(g, abs=1e-6)
    assert nn.numerical_grad(loss, x) == pytest.approx(dx, abs=1e-6)


def test_adam_fits_a_curve_to_the_noise_floor():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(-3, 3, 200))
    y = np.sin(2 * x) + 0.3 * x + 0.1 * rng.standard_normal(200)
    r = nn.fit_regression(x, y, [16, 16], 1000, lr=0.01, seed=0)
    assert np.mean(r["loss"][-20:]) < 0.015            # noise variance is 0.01


# ----------------------------------------------------------------------------- graphical models
def test_message_passing_equals_forward_backward_viterbi_and_kalman():
    s, rolls, true = hmm.sample_casino(300, 0.02, 0.1, 0.5, np.random.default_rng(1))
    logB = true.loglik_matrix(rolls)
    post = hmm.posteriors(true.pi, true.A, logB)
    sp = gm.chain_sum_product(true.pi, true.A, np.exp(logB))
    assert sp["marginal"] == pytest.approx(post["gamma"], abs=1e-12)
    path, _ = hmm.viterbi(true.pi, true.A, logB)
    assert np.array_equal(gm.chain_max_product(true.pi, true.A, np.exp(logB))["path"], path)

    x, z = kalman.simulate_random_walk(60, 0.3, 1.0, 0.0, np.random.default_rng(2))
    kf = kalman.KalmanFilter([[1.0]], [[1.0]], [[0.3]], [[1.0]], [0.0], [[2.0]])
    ref = kalman.run_kalman(kf, list(z))
    xs, Ps = kalman.rts_smoother(ref["x"], ref["P"], ref["x_pred"], ref["P_pred"], np.eye(1))
    b = gm.gaussian_chain_bp(z, 0.3, 1.0, 0.0, 2.3)
    assert b["filt_mean"] == pytest.approx(ref["x"][:, 0], abs=1e-10)
    assert b["smooth_mean"] == pytest.approx(xs[:, 0], abs=1e-10)
    assert b["smooth_var"] == pytest.approx(Ps[:, 0, 0], abs=1e-10)


def test_explaining_away_in_the_alarm_network():
    bn = gm.alarm_network()
    m, _ = bn.marginals({"Alarm": 1})
    m2, _ = bn.marginals({"Alarm": 1, "Earthquake": 1})
    assert m["Burglary"] == pytest.approx(0.583, abs=0.005)
    assert m2["Burglary"] < 0.05
    # given the alarm, John and Mary are independent
    assert bn.marginals({"Alarm": 1, "JohnCalls": 1})[0]["MaryCalls"] == pytest.approx(0.70)


# ----------------------------------------------------------------------------- EM
def test_gmm_em_never_decreases_likelihood_and_bic_finds_k():
    X, _ = mx.blobs("five clusters", 600, np.random.default_rng(0))
    r = mx.gmm_em(X, 5, 60, seed=0)
    ll = np.array([h["loglik"] for h in r["hist"]])
    assert np.all(np.diff(ll) > -1e-8)
    bics = [mx.gmm_bic(max(mx.gmm_em(X, K, 150, seed=s, init="k-means++")["loglik"] for s in range(5)), K, 2, 600)
            for K in range(1, 8)]
    assert int(np.argmin(bics)) + 1 == 5


def test_mixture_of_experts_recovers_the_three_laws():
    x, y = mx.piecewise_data(400, np.random.default_rng(1), 0.15)
    r = mx.moe_em(x, y, 3, 60, seed=0)
    assert sorted(r["coef"][:, 0]) == pytest.approx([-0.8, 0.4, 1.6], abs=0.1)


# ----------------------------------------------------------------------------- approximate inference
def test_hmc_and_vi_on_a_gaussian_match_the_exact_answer():
    t = ap.target("correlated Gaussian")
    ch = ap.hmc(t, 3000, 0.2, 15, np.random.default_rng(0))["chain"][300:]
    assert ch.mean(0) == pytest.approx([0, 0], abs=0.08)
    assert np.cov(ch.T)[0, 1] == pytest.approx(0.95, abs=0.08)
    v = ap.vi_gaussian(t, 1500, seed=0)
    assert v["cov"] == pytest.approx(np.array([[1, 0.95], [0.95, 1]]), abs=0.08)
    la = ap.laplace(t)
    assert la["cov"] == pytest.approx(np.array([[1, 0.95], [0.95, 1]]), abs=1e-3)


def test_reverse_kl_vi_is_mode_seeking_and_metropolis_targets_the_posterior():
    t = ap.target("two modes")
    g = ap.grid_density(t)
    v = ap.vi_gaussian(t, 1500, seed=0, init=(-3.0, 0.0))
    assert v["mean"] == pytest.approx([-2.0, -0.5], abs=0.15)          # one mode, not the overall mean
    assert np.sqrt(v["cov"][0, 0]) < np.sqrt(g["cov"][0, 0]) / 2
    ch = ap.metropolis(t, 40000, 2.5, np.random.default_rng(0))["chain"][2000:]
    assert ch.mean(0) == pytest.approx(g["mean"], abs=0.25)


def test_ess_of_independent_samples_is_close_to_n():
    x = np.random.default_rng(0).standard_normal((4000, 2))
    assert ap.ess(x) == pytest.approx(4000, rel=0.25)


# ----------------------------------------------------------------------------- GPs and BLR
def test_blr_evidence_and_gp_marginal_likelihood_match_direct_formulas():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 15)
    y = np.sin(x) + 0.2 * rng.standard_normal(15)
    P = gp.poly_features(x, 3)
    alpha, beta = 2.0, 25.0
    direct = multivariate_normal(np.zeros(15), P @ P.T / alpha + np.eye(15) / beta).logpdf(y)
    assert gp.blr(P, y, alpha, beta)["log_evidence"] == pytest.approx(direct, abs=1e-8)
    K = gp.kernel("RBF (very smooth)", x, x, 1.2, 0.9) + 0.2**2 * np.eye(15)
    assert gp.gp_posterior(x, y, x, "RBF (very smooth)", 1.2, 0.9, 0.2)["lml"] == pytest.approx(
        multivariate_normal(np.zeros(15), K).logpdf(y), abs=1e-6)          # tiny jitter on the diagonal


def test_bayesian_optimisation_beats_random_search_on_the_tuning_curve():
    found = []
    for seed in range(10):
        r = gp.bayes_opt(gp.tuning_objective, 0, 10, 3, 10, 0.05, np.random.default_rng(seed), ell=1.0)
        s = r["steps"][-1]
        found.append(abs(s["X"][np.argmax(s["Y"])] - 7.6) < 0.5)
    assert np.mean(found) >= 0.9                         # random search with 13 tries: about 0.73


# ----------------------------------------------------------------------------- bandits
def test_thompson_sampling_has_lowest_regret():
    p = [0.04, 0.05, 0.045]
    reg = {pol: bd.mean_regret(pol, p, 10000, 20, seed=0)[-1] for pol in bd.POLICIES}
    assert reg["Thompson sampling"] == min(reg.values())
    pb = bd.prob_best(np.array([30.0, 60, 40]), np.array([700.0, 700, 700]), np.random.default_rng(0))
    assert pb.sum() == pytest.approx(1.0) and pb.argmax() == 1


# ----------------------------------------------------------------------------- generative models
def test_flow_gradients_and_normalisation():
    rng = np.random.default_rng(0)
    f = gen.Flow1D(4, 0)
    x = rng.standard_normal(30) * 2
    _, g = f.loss_and_grads(x)
    for p, gg in zip(f.params(), g):
        assert nn.numerical_grad(lambda: f.loss_and_grads(x)[0], p) == pytest.approx(gg, abs=1e-6)
    X = gen.data_1d("three peaks", 2000, rng)
    r = gen.train_flow1d(X, 8, 1500, 0.05)
    xs = np.linspace(-10, 10, 8001)
    assert np.sum(np.exp(r["flow"].log_prob(xs))) * (xs[1] - xs[0]) == pytest.approx(1.0, abs=0.01)
    S = r["flow"].sample(2000, rng)
    assert gen.two_sample_classifier(X[:, None], S[:, None])["accuracy"] < 0.56


def test_vae_gradients_match_finite_differences():
    X, _ = gen.load_digits()
    vae = gen.VAE(64, 2, [16], "bernoulli", seed=0)
    x = X[:5]

    def loss():
        return vae.loss_and_grads(x, np.random.default_rng(3), beta=0.7)["loss"]
    r = vae.loss_and_grads(x, np.random.default_rng(3), beta=0.7)
    for p, g in zip(vae.params(), r["grads"]):
        assert nn.numerical_grad(loss, p) == pytest.approx(g, abs=1e-5)


def test_toy_elbo_identity_and_bound():
    x = gen.toy_decoder(np.array(1.5))
    for m, s in [(0.0, 1.0), (1.4, 0.2), (-1.0, 0.5)]:
        r = gen.toy_elbo(x, m, s, 0.45)
        assert r["elbo"] <= r["log_px"] + 1e-9
        assert r["rec"] - r["kl_prior"] == pytest.approx(r["elbo"], abs=1e-6)


def test_gan_generator_gradient_flows_through_the_discriminator():
    """The generator update uses dLoss/dx from D's backward pass; check it against finite differences."""
    rng = np.random.default_rng(0)
    G = nn.MLP([2, 8, 2], act="relu", rng=rng)
    D = nn.MLP([2, 8, 1], act="leaky", rng=rng)
    z = rng.standard_normal((6, 2))

    def loss():
        return float(np.mean(nn.softplus(-D(G(z))[:, 0])))     # non-saturating generator loss
    lg = D(G(z))[:, 0]
    G.backward(D.backward(((nn.sigmoid(lg) - 1.0) / len(z))[:, None]))
    for p, g in zip(G.params(), G.grads()):
        assert nn.numerical_grad(loss, p) == pytest.approx(g, abs=1e-6)


def test_gan_learns_the_ring_without_collapse():
    r = gen.train_gan(lambda n, rng: gen.toy_2d("ring", n, rng), 2, steps=2000, seed=0)
    modes, good = gen.ring_coverage(r["G"](np.random.default_rng(9).standard_normal((2000, 2))))
    assert modes == 8 and good > 0.8


def test_diffusion_with_the_exact_denoiser_reproduces_the_data():
    M = np.array([[-1.5, 1.0], [1.5, 1.0], [0.0, -1.5]])
    SD = np.array([0.3, 0.3, 0.45])
    W = np.array([0.3, 0.3, 0.4])
    sch = gen.schedule(100)

    def eps(x, t):
        return gen.mixture_eps(x, t, sch, M, SD, W)
    S = gen.sample_ddpm(eps, sch, 4000, 2, np.random.default_rng(0))["x"]
    k = np.argmin(((S[:, None] - M[None]) ** 2).sum(-1), 1)
    assert np.bincount(k, minlength=3) / len(S) == pytest.approx(W, abs=0.03)
    assert S[k == 2].std(0) == pytest.approx([0.45, 0.45], abs=0.05)
    # Tweedie at tiny noise returns (almost) the input
    x = np.array([[0.1, -1.4]])
    assert gen.tweedie_x0(x, np.array([0]), eps(x, np.array([0])), sch) == pytest.approx(x, abs=0.02)


def test_pretrained_digit_diffusion_loads_and_denoises():
    d = np.load(gen.Path(gen.__file__).resolve().parents[1] / "data" / "digits_diffusion.npz")
    m = gen.Denoiser(64, int(d["T"]), tuple(int(h) for h in d["hidden"]))
    m.net.load({k: d[k].astype(float) for k in d.files if k[0] in "Wb"})
    sch = gen.schedule(int(d["T"]))
    X, _ = gen.load_digits()
    x0 = X[:50] * 2 - 1
    t = np.full(50, 80)
    xt = gen.q_sample(x0, t, sch, np.random.default_rng(0).standard_normal(x0.shape))
    x0h = np.clip(gen.tweedie_x0(xt, t, m(xt, t), sch), -1, 1)
    noisy_err = np.mean((xt / np.sqrt(sch["abar"][80]) - x0) ** 2)
    assert np.mean((x0h - x0) ** 2) < 0.25 * noisy_err
