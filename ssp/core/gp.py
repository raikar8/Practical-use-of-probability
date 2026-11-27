"""Bayesian linear regression, Gaussian processes and Bayesian optimisation (1-D inputs, from scratch)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ============================================================================== Bayesian linear regression
def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(np.asarray(x) / 1.5, degree + 1, increasing=True)      # inputs live in about [-3, 3]


def blr(Phi: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> dict:
    """Prior w ~ N(0, α⁻¹ I), noise precision β. Posterior w ~ N(m, S) with S⁻¹ = αI + βΦᵀΦ, m = βSΦᵀy.

    Also returns the log evidence log p(y | α, β), which automatically trades fit against complexity.
    """
    N, M = Phi.shape
    A = alpha * np.eye(M) + beta * Phi.T @ Phi
    S = np.linalg.inv(A)
    m = beta * S @ Phi.T @ y
    E = 0.5 * beta * np.sum((y - Phi @ m) ** 2) + 0.5 * alpha * m @ m
    sign, logdetA = np.linalg.slogdet(A)
    log_ev = 0.5 * M * np.log(alpha) + 0.5 * N * np.log(beta) - E - 0.5 * logdetA - 0.5 * N * np.log(2 * np.pi)
    return {"m": m, "S": S, "log_evidence": float(log_ev), "alpha": alpha, "beta": beta}


def blr_predict(Phi_star: np.ndarray, post: dict) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and variance (noise included): mean = φᵀm, var = 1/β + φᵀSφ."""
    mean = Phi_star @ post["m"]
    var = 1.0 / post["beta"] + np.einsum("ij,jk,ik->i", Phi_star, post["S"], Phi_star)
    return mean, var


# ============================================================================== kernels and GP regression
def kernel(name: str, x1: np.ndarray, x2: np.ndarray, ell: float, sf: float, period: float = 2.0) -> np.ndarray:
    d = np.abs(np.asarray(x1)[:, None] - np.asarray(x2)[None, :])
    r = d / ell
    if name == "RBF (very smooth)":
        k = np.exp(-0.5 * r**2)
    elif name == "Matérn 3/2 (rougher)":
        k = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif name == "Matérn 1/2 (random walk-like)":
        k = np.exp(-r)
    elif name == "Periodic":
        k = np.exp(-2 * np.sin(np.pi * d / period) ** 2 / ell**2)
    else:
        raise ValueError(name)
    return sf**2 * k


KERNELS = ["RBF (very smooth)", "Matérn 3/2 (rougher)", "Matérn 1/2 (random walk-like)", "Periodic"]


def gp_posterior(xtr, ytr, xte, kname: str, ell: float, sf: float, sn: float, period: float = 2.0) -> dict:
    """GP regression: mean = K*ᵀ(K + σ²I)⁻¹y, var = k** − K*ᵀ(K + σ²I)⁻¹K*. Also the log marginal likelihood."""
    xtr, ytr = np.asarray(xtr, float), np.asarray(ytr, float)
    K = kernel(kname, xtr, xtr, ell, sf, period) + (sn**2 + 1e-9) * np.eye(len(xtr))
    Ks = kernel(kname, xtr, xte, ell, sf, period)
    kss = sf**2 * np.ones(len(xte))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    v = np.linalg.solve(L, Ks)
    mean = Ks.T @ a
    var = np.maximum(kss - np.sum(v**2, 0), 1e-12)
    lml = -0.5 * ytr @ a - np.sum(np.log(np.diag(L))) - 0.5 * len(xtr) * np.log(2 * np.pi)
    return {"mean": mean, "var": var, "lml": float(lml)}


def gp_prior_samples(xg: np.ndarray, kname: str, ell: float, sf: float, n: int, rng: np.random.Generator,
                     period: float = 2.0) -> np.ndarray:
    K = kernel(kname, xg, xg, ell, sf, period) + 1e-8 * np.eye(len(xg))
    return (np.linalg.cholesky(K) @ rng.standard_normal((len(xg), n))).T


def gp_fit_hypers(xtr, ytr, kname: str, init=(1.0, 1.0, 0.3), period: float = 2.0,
                  ell_bounds: tuple[float, float] = (0.05, 20.0)) -> dict:
    """Choose (ℓ, σ_f, σ_n) by maximising the log marginal likelihood (type-II maximum likelihood)."""
    def nlml(lp):
        ell, sf, sn = np.exp(lp)
        try:
            return -gp_posterior(xtr, ytr, np.array([0.0]), kname, ell, sf, sn, period)["lml"]
        except np.linalg.LinAlgError:
            return 1e10
    best = None
    for s in (1.0, 0.3, 3.0):
        r = minimize(nlml, np.log(np.array(init) * [s, 1, 1]), method="L-BFGS-B",
                     bounds=[tuple(np.log(ell_bounds)), (np.log(0.05), np.log(20)), (np.log(1e-3), np.log(5))])
        if best is None or r.fun < best.fun:
            best = r
    ell, sf, sn = np.exp(best.x)
    return {"ell": float(ell), "sf": float(sf), "sn": float(sn), "lml": float(-best.fun)}


# ============================================================================== Bayesian optimisation
def tuning_objective(x: np.ndarray) -> np.ndarray:
    """A made-up 'listening score' vs one tuning parameter in [0, 10]: a broad hill, a narrow better peak, and
    a dip. Expensive to evaluate in real life (every point is a listening test), so we want few evaluations."""
    x = np.asarray(x, float)
    return (2.0 * np.exp(-0.5 * ((x - 2.5) / 1.4) ** 2) + 2.8 * np.exp(-0.5 * ((x - 7.6) / 0.6) ** 2)
            - 0.8 * np.exp(-0.5 * ((x - 5.2) / 0.6) ** 2) + 0.15 * np.sin(2.3 * x))


def expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI = E[max(f − best − ξ, 0)] under the GP: high where the mean is good OR the uncertainty is large."""
    z = (mean - best - xi) / sd
    return (mean - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def bayes_opt(f, lo: float, hi: float, n_init: int, n_iter: int, noise: float, rng: np.random.Generator,
              ell: float = 0.8, sf: float = 1.5, kname: str = "Matérn 3/2 (rougher)",
              acq: str = "Expected improvement", kappa: float = 2.0, grid_n: int = 400) -> dict:
    """Bayesian optimisation loop: fit a GP to the evaluations so far, evaluate where the acquisition is highest.

    The GP prior (zero mean, length-scale ℓ, amplitude σ_f, known measurement noise) is our belief about the
    unknown score curve *before* testing; the acquisition turns the posterior into 'where to test next'.
    """
    xg = np.linspace(lo, hi, grid_n)
    X = list(rng.uniform(lo, hi, n_init))
    Y = list(f(np.array(X)) + noise * rng.standard_normal(n_init))
    steps = []
    for i in range(n_iter + 1):
        post = gp_posterior(np.array(X), np.array(Y), xg, kname, ell, sf, max(noise, 1e-3))
        mean, sd = post["mean"], np.sqrt(post["var"])
        if acq == "Expected improvement":
            best = float(np.max(np.interp(X, xg, mean)))        # best *believed* value among evaluated points
            a = expected_improvement(mean, sd, best)
        else:
            a = mean + kappa * sd
        xn = float(xg[np.argmax(a)])
        steps.append({"X": np.array(X), "Y": np.array(Y), "mean": mean, "sd": sd, "acq": a, "next": xn})
        if i < n_iter:
            X.append(xn)
            Y.append(float(f(np.array([xn]))[0] + noise * rng.standard_normal()))
    return {"grid": xg, "steps": steps}
