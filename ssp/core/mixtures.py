"""Latent-variable models fitted by expectation–maximisation: Gaussian mixtures and a mixture of experts."""
from __future__ import annotations

import numpy as np


# ============================================================================== Gaussian mixture model
def gaussian_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mean).T)
    return -0.5 * np.sum(sol**2, 0) - np.sum(np.log(np.diag(L))) - 0.5 * d * np.log(2 * np.pi)


def gmm_log_resp(X: np.ndarray, w: np.ndarray, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, float]:
    """E-step: log responsibilities log P(component k | x_n) and the total log-likelihood."""
    lp = np.stack([np.log(w[k]) + gaussian_logpdf(X, means[k], covs[k]) for k in range(len(w))], 1)
    mx = lp.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
    return lp - lse[:, None], float(lse.sum())


def gmm_em(X: np.ndarray, K: int, n_iter: int = 50, reg: float = 1e-6, seed: int = 0,
           init: str = "random points", tol: float = 0.0) -> dict:
    """Fit a K-component Gaussian mixture by EM. Records every iteration so the page can animate it.

    E-step: r_nk = P(k | x_n) with the current parameters.
    M-step: weighted counts: N_k = Σ r_nk, w_k = N_k / N, μ_k = Σ r_nk x_n / N_k, Σ_k = weighted covariance + reg I.
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    if init == "k-means++":
        means = [X[rng.integers(N)]]
        for _ in range(K - 1):
            d2 = np.min([np.sum((X - m) ** 2, 1) for m in means], 0)
            means.append(X[rng.choice(N, p=d2 / d2.sum())])
        means = np.array(means)
    else:
        means = X[rng.choice(N, K, replace=False)].copy()
    covs = np.array([np.cov(X.T) + reg * np.eye(d)] * K)
    w = np.full(K, 1.0 / K)
    hist = []
    prev = -np.inf
    for it in range(n_iter + 1):
        logr, ll = gmm_log_resp(X, w, means, covs)
        hist.append({"w": w.copy(), "means": means.copy(), "covs": covs.copy(), "loglik": ll})
        if it == n_iter or (tol and ll - prev < tol):
            break
        prev = ll
        r = np.exp(logr)
        Nk = r.sum(0) + 1e-12
        w = Nk / N
        means = (r.T @ X) / Nk[:, None]
        covs = np.array([((r[:, k, None] * (X - means[k])).T @ (X - means[k])) / Nk[k] + reg * np.eye(d)
                         for k in range(K)])
    return {"hist": hist, "resp": np.exp(logr), "w": w, "means": means, "covs": covs, "loglik": ll}


def gmm_bic(loglik: float, K: int, d: int, N: int) -> float:
    """Bayesian information criterion: −2 log L + (number of parameters) log N. Lower is better."""
    n_par = (K - 1) + K * d + K * d * (d + 1) / 2
    return float(-2 * loglik + n_par * np.log(N))


def sample_gmm(n: int, w, means, covs, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    k = rng.choice(len(w), n, p=np.asarray(w) / np.sum(w))
    X = np.array([rng.multivariate_normal(means[j], covs[j]) for j in k])
    return X, k


def blobs(name: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Example 2-D clusterings with known labels."""
    if name == "three round clusters":
        return sample_gmm(n, [0.3, 0.4, 0.3], np.array([[-2, 0], [1.5, 1.5], [1.5, -1.5]]),
                          np.array([np.eye(2) * 0.5] * 3), rng)
    if name == "stretched & overlapping":
        c1 = np.array([[2.0, 1.6], [1.6, 1.6]])
        c2 = np.array([[1.5, -1.2], [-1.2, 1.5]])
        return sample_gmm(n, [0.5, 0.3, 0.2], np.array([[-1, 0], [1.5, 0.5], [0, -2.5]]),
                          np.array([c1, c2, np.eye(2) * 0.2]), rng)
    if name == "five clusters":
        ang = np.arange(5) * 2 * np.pi / 5
        m = np.c_[np.cos(ang), np.sin(ang)] * 2.5
        return sample_gmm(n, np.full(5, 0.2), m, np.array([np.eye(2) * 0.25] * 5), rng)
    raise ValueError(name)


# ============================================================================== mixture of experts
def softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def moe_em(x: np.ndarray, y: np.ndarray, K: int, n_iter: int = 60, seed: int = 0, gate_steps: int = 20,
           gate_lr: float = 0.5) -> dict:
    """Mixture of linear experts:  p(y | x) = Σ_k g_k(x) N(y; a_k x + b_k, s_k²),  g = softmax(v_k x + c_k).

    E-step: r_nk ∝ g_k(x_n) N(y_n; a_k x_n + b_k, s_k²)   (which expert explains this point?)
    M-step: each expert = weighted least squares with weights r_nk; the gate = a soft classifier trained to
            predict r (a few gradient steps on Σ r_nk log g_k(x_n), i.e. generalised EM).
    """
    rng = np.random.default_rng(seed)
    N = len(x)
    Xf = np.c_[x, np.ones(N)]
    coef = rng.standard_normal((K, 2)) * np.array([1.0, np.std(y)]) + np.array([0.0, np.mean(y)])
    s2 = np.full(K, np.var(y))
    G = np.c_[rng.standard_normal(K) * 0.1, np.zeros(K)]          # gate weights (v_k, c_k)
    hist = []
    for it in range(n_iter + 1):
        g = softmax(Xf @ G.T)
        mu = Xf @ coef.T
        logl = -0.5 * (y[:, None] - mu) ** 2 / s2 - 0.5 * np.log(2 * np.pi * s2)
        lp = np.log(g + 1e-300) + logl
        mx = lp.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
        r = np.exp(lp - lse[:, None])
        hist.append({"coef": coef.copy(), "s2": s2.copy(), "G": G.copy(), "loglik": float(lse.sum())})
        if it == n_iter:
            break
        for k in range(K):
            W = r[:, k] + 1e-9
            A = Xf.T @ (W[:, None] * Xf) + 1e-6 * np.eye(2)
            coef[k] = np.linalg.solve(A, Xf.T @ (W * y))
            s2[k] = max(np.sum(W * (y - Xf @ coef[k]) ** 2) / W.sum(), 1e-4)
        for _ in range(gate_steps):
            g = softmax(Xf @ G.T)
            G += gate_lr * ((r - g).T @ Xf) / N
    return {"hist": hist, "resp": r, "coef": coef, "s2": s2, "G": G}


def moe_predict(xg: np.ndarray, res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gate probabilities, each expert's line, and the mixture's mean prediction on a grid."""
    Xf = np.c_[xg, np.ones(len(xg))]
    g = softmax(Xf @ res["G"].T)
    lines = Xf @ res["coef"].T
    return g, lines, np.sum(g * lines, 1)


def piecewise_data(n: int, rng: np.random.Generator, noise: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """A response that follows different straight-line laws in different input regions (think: an amplifier
    that's linear, then compresses, then clips)."""
    x = rng.uniform(-3, 3, n)
    y = np.where(x < -1, 0.4 * x - 0.6, np.where(x < 1, 1.6 * x + 0.6, -0.8 * x + 3.0))
    return x, y + noise * rng.standard_normal(n)
