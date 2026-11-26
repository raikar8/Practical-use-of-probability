"""L4 — Hidden Markov models.

A hidden state s_t in {0..K-1} follows a Markov chain (pi, A) and emits an
observation x_t with likelihood b_k(x_t) = p(x_t | s_t = k).

    forward        P(s_t | x_1..t)          filtering  (causal, real time)
    backward       p(x_t+1..T | s_t)        used for smoothing
    posteriors     P(s_t | x_1..T)          smoothing  (offline)
    viterbi        argmax P(s_1..T | x)     decoding   (best whole path)
    baum_welch     learn pi, A, emissions   EM          (no labels needed)

All algorithms work on a (T, K) matrix of emission *log*-likelihoods, so the
same code serves discrete symbols (dice), Gaussian features (returns, audio
frames) or anything else you can score.

Numerical care: per-step scaling (Rabiner) for forward/backward, logs for
Viterbi — a 10,000-step sequence would otherwise underflow to 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Emission log-likelihoods
# ---------------------------------------------------------------------------
def categorical_loglik(x: np.ndarray, E: np.ndarray) -> np.ndarray:
    """x: int symbols (T,), E: (K, M) emission probabilities -> (T, K)."""
    return np.log(np.asarray(E)[:, np.asarray(x)].T + 1e-300)


def gaussian_loglik(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Diagonal-covariance Gaussian emissions. X: (T, D) or (T,), means/vars: (K, D)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mu = np.atleast_2d(means)
    var = np.atleast_2d(variances)
    diff = X[:, None, :] - mu[None, :, :]
    return -0.5 * np.sum(np.log(2 * np.pi * var)[None] + diff**2 / var[None], axis=2)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def forward(pi: np.ndarray, A: np.ndarray, logB: np.ndarray):
    """Scaled forward pass.

    Returns alpha (T, K) with alpha[t] = P(s_t | x_1..t)  (each row sums to 1),
    the scaling factors c (T,), and the total log-likelihood log p(x_1..T).
    """
    T, K = logB.shape
    m = logB.max(axis=1, keepdims=True)
    B = np.exp(logB - m)                     # per-step shift avoids underflow
    alpha = np.zeros((T, K))
    c = np.zeros(T)
    a = pi * B[0]
    for t in range(T):
        if t:
            a = (alpha[t - 1] @ A) * B[t]    # predict with A, then weigh by evidence
        c[t] = a.sum()
        alpha[t] = a / c[t]
    loglik = float(np.sum(np.log(c)) + m.sum())
    return alpha, c, loglik


def backward(A: np.ndarray, logB: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scaled backward pass using the forward scaling factors."""
    T, K = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1]) / c[t + 1]
    return beta


def posteriors(pi: np.ndarray, A: np.ndarray, logB: np.ndarray) -> dict:
    """Forward–backward smoothing.

    gamma[t, k] = P(s_t = k | all data); xi_sum[i, j] = expected number of
    i -> j transitions. These are the sufficient statistics for EM.
    """
    alpha, c, ll = forward(pi, A, logB)
    beta = backward(A, logB, c)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    xi_sum = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
    return {"alpha": alpha, "gamma": gamma, "xi_sum": xi_sum, "loglik": ll}


def viterbi(pi: np.ndarray, A: np.ndarray, logB: np.ndarray):
    """Most probable state *sequence* (dynamic programming in the log domain)."""
    T, K = logB.shape
    logA = np.log(A + 1e-300)
    delta = np.log(pi + 1e-300) + logB[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + logA          # (from, to)
        back[t] = np.argmax(scores, axis=0)
        delta = scores[back[t], np.arange(K)] + logB[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(delta.max())


# ---------------------------------------------------------------------------
# Learning: Baum–Welch (EM)
# ---------------------------------------------------------------------------
@dataclass
class HMMParams:
    pi: np.ndarray
    A: np.ndarray
    E: np.ndarray | None = None          # categorical emissions (K, M)
    means: np.ndarray | None = None      # Gaussian emissions (K, D)
    variances: np.ndarray | None = None

    def loglik_matrix(self, x) -> np.ndarray:
        if self.E is not None:
            return categorical_loglik(x, self.E)
        return gaussian_loglik(x, self.means, self.variances)

    def copy(self) -> "HMMParams":
        return HMMParams(*(None if v is None else np.array(v, dtype=float, copy=True)
                           for v in (self.pi, self.A, self.E, self.means, self.variances)))


def init_gaussian(X: np.ndarray, K: int, stay: float = 0.9, rng: np.random.Generator | None = None) -> HMMParams:
    """Means at data quantiles (or random data points if rng given), shared variance."""
    X = np.asarray(X, dtype=float)
    X2 = X[:, None] if X.ndim == 1 else X
    if rng is None:
        order = np.argsort(X2[:, 0])
        idx = order[((np.arange(K) + 0.5) / K * len(order)).astype(int)]
    else:
        idx = rng.choice(len(X2), K, replace=False)
    means = X2[idx].copy()
    variances = np.tile(X2.var(axis=0), (K, 1))
    A = np.full((K, K), (1 - stay) / max(K - 1, 1))
    np.fill_diagonal(A, stay if K > 1 else 1.0)
    return HMMParams(np.full(K, 1 / K), A, means=means, variances=variances)


def init_categorical(K: int, M: int, rng: np.random.Generator, stay: float = 0.9) -> HMMParams:
    E = rng.dirichlet(np.ones(M) * 5, size=K)
    A = np.full((K, K), (1 - stay) / max(K - 1, 1))
    np.fill_diagonal(A, stay if K > 1 else 1.0)
    return HMMParams(np.full(K, 1 / K), A, E=E)


def baum_welch(x, params: HMMParams, n_iter: int = 50, tol: float = 1e-6,
               var_floor: float = 1e-6, learn_A: bool = True) -> dict:
    """Expectation–maximisation for an HMM. Log-likelihood never decreases.

    E-step: gamma, xi from forward–backward with current parameters.
    M-step: re-estimate
        pi    = gamma[0]
        A_ij  = E[# i->j] / E[# leaving i]
        E_km  = E[# in k emitting m] / E[# in k]                 (categorical)
        mu_k  = sum_t gamma_tk x_t / sum_t gamma_tk                (Gaussian)
        var_k = sum_t gamma_tk (x_t - mu_k)^2 / sum_t gamma_tk
    """
    p = params.copy()
    X = np.asarray(x)
    history = []
    for _ in range(n_iter):
        post = posteriors(p.pi, p.A, p.loglik_matrix(X))
        history.append(post["loglik"])
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * abs(history[-2]):
            break
        g = post["gamma"]
        p.pi = g[0] / g[0].sum()
        if learn_A:
            p.A = post["xi_sum"] / post["xi_sum"].sum(axis=1, keepdims=True)
        w = g.sum(axis=0)
        if p.E is not None:
            M = p.E.shape[1]
            onehot = np.eye(M)[X.astype(int)]
            p.E = (g.T @ onehot) / w[:, None]
            p.E = np.clip(p.E, 1e-12, None)
            p.E /= p.E.sum(axis=1, keepdims=True)
        else:
            X2 = X[:, None] if X.ndim == 1 else X
            p.means = (g.T @ X2) / w[:, None]
            p.variances = (g.T @ X2**2) / w[:, None] - p.means**2
            p.variances = np.maximum(p.variances, var_floor)
    post = posteriors(p.pi, p.A, p.loglik_matrix(X))
    history.append(post["loglik"])
    return {"params": p, "loglik": np.array(history), "gamma": post["gamma"]}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sample_states(pi: np.ndarray, A: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros(T, dtype=int)
    s[0] = rng.choice(len(pi), p=pi)
    cdf = np.cumsum(A, axis=1)
    u = rng.random(T)
    for t in range(1, T):
        s[t] = min(int(np.searchsorted(cdf[s[t - 1]], u[t])), len(pi) - 1)
    return s


def sample_casino(T: int, p_switch_to_loaded: float, p_switch_to_fair: float, loaded_six: float,
                  rng: np.random.Generator):
    """The classic 'dishonest casino': a fair die and a loaded die that favours 6.

    Returns states (0 = fair, 1 = loaded), rolls (0..5 meaning faces 1..6) and
    the true parameters.
    """
    A = np.array([[1 - p_switch_to_loaded, p_switch_to_loaded], [p_switch_to_fair, 1 - p_switch_to_fair]])
    E = np.vstack([np.full(6, 1 / 6), np.r_[np.full(5, (1 - loaded_six) / 5), loaded_six]])
    pi = np.array([1.0, 0.0])
    s = sample_states(pi, A, T, rng)
    rolls = np.array([rng.choice(6, p=E[k]) for k in s])
    return s, rolls, HMMParams(pi, A, E=E)


def count_switches(path: np.ndarray) -> int:
    return int(np.sum(path[1:] != path[:-1]))
