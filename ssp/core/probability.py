"""L0 — probability building blocks.

* Beta–Bernoulli updating: the simplest example of "posterior becomes the next
  prior", which is exactly what a Kalman filter or HMM forward pass does.
* Bayes' rule for a binary hypothesis (alarm / fault).
* Markov chains: n-step prediction and the stationary distribution — the
  hidden-state engine behind HMMs.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Beta–Bernoulli
# ---------------------------------------------------------------------------
def beta_bernoulli_update(alpha: float, beta: float, successes: int, failures: int):
    """Conjugate update: Beta(a, b) prior + Bernoulli data -> Beta(a+k, b+n-k)."""
    return alpha + successes, beta + failures


def beta_summary(alpha: float, beta: float, level: float = 0.95) -> dict:
    """Mean, mode and equal-tailed credible interval of a Beta distribution."""
    dist = stats.beta(alpha, beta)
    lo, hi = dist.ppf([(1 - level) / 2, 1 - (1 - level) / 2])
    mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan
    return {"mean": alpha / (alpha + beta), "mode": mode, "lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# Bayes' rule for a binary event
# ---------------------------------------------------------------------------
def posterior_given_alarm(prior: float, sensitivity: float, specificity: float) -> float:
    """P(fault | alarm) with P(alarm|fault)=sensitivity, P(no alarm|ok)=specificity."""
    p_alarm = sensitivity * prior + (1 - specificity) * (1 - prior)
    return sensitivity * prior / p_alarm


# ---------------------------------------------------------------------------
# Markov chains
# ---------------------------------------------------------------------------
def sample_markov_chain(P: np.ndarray, x0: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a length-n state path from transition matrix P (rows sum to 1)."""
    P = np.asarray(P, dtype=float)
    states = np.empty(n, dtype=int)
    states[0] = x0
    cdf = np.cumsum(P, axis=1)
    u = rng.random(n)
    for t in range(1, n):
        states[t] = int(np.searchsorted(cdf[states[t - 1]], u[t]))
    return states


def n_step_distribution(P: np.ndarray, p0: np.ndarray, n: int) -> np.ndarray:
    """Distribution after n steps: p_n = p_0 P^n. Returns array of shape (n+1, K)."""
    out = [np.asarray(p0, dtype=float)]
    for _ in range(n):
        out.append(out[-1] @ P)
    return np.array(out)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalised to sum to 1."""
    w, v = np.linalg.eig(np.asarray(P, dtype=float).T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, k])
    return pi / pi.sum()
