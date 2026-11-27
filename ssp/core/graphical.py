"""Graphical models and message passing.

* A small discrete Bayesian network, solved exactly by enumeration (fine for a handful of variables).
* Sum-product and max-product on a chain of discrete variables: these *are* HMM forward–backward and Viterbi.
* Gaussian belief propagation on a chain: this *is* the Kalman filter + RTS smoother.
"""
from __future__ import annotations

import itertools

import numpy as np


# ============================================================================== discrete Bayesian network
class BayesNet:
    """Binary variables. cpt[name] maps a tuple of parent values to P(name = 1 | parents)."""

    def __init__(self, parents: dict[str, list[str]], cpt: dict[str, dict[tuple, float]]):
        self.parents, self.cpt = parents, cpt
        self.names = list(parents)

    def joint(self, assign: dict[str, int]) -> float:
        p = 1.0
        for v in self.names:
            p1 = self.cpt[v][tuple(assign[u] for u in self.parents[v])]
            p *= p1 if assign[v] == 1 else 1.0 - p1
        return p

    def marginals(self, evidence: dict[str, int] | None = None) -> tuple[dict[str, float], float]:
        """P(v = 1 | evidence) for every variable, and P(evidence). Sums the joint over all 2^n assignments."""
        evidence = evidence or {}
        tot, ones = 0.0, dict.fromkeys(self.names, 0.0)
        for vals in itertools.product([0, 1], repeat=len(self.names)):
            a = dict(zip(self.names, vals))
            if any(a[k] != v for k, v in evidence.items()):
                continue
            p = self.joint(a)
            tot += p
            for v in self.names:
                if a[v]:
                    ones[v] += p
        return {v: ones[v] / tot for v in self.names}, tot


def alarm_network(p_burglary: float = 0.01, p_quake: float = 0.02, p_alarm_given_both: float = 0.95,
                  p_alarm_burglary: float = 0.94, p_alarm_quake: float = 0.29, p_false_alarm: float = 0.001,
                  p_john: float = 0.90, p_john_false: float = 0.05, p_mary: float = 0.70,
                  p_mary_false: float = 0.01) -> BayesNet:
    """Pearl's burglar alarm: Burglary and Earthquake both cause the Alarm; John and Mary call when they hear it."""
    return BayesNet(
        {"Burglary": [], "Earthquake": [], "Alarm": ["Burglary", "Earthquake"], "JohnCalls": ["Alarm"],
         "MaryCalls": ["Alarm"]},
        {"Burglary": {(): p_burglary}, "Earthquake": {(): p_quake},
         "Alarm": {(1, 1): p_alarm_given_both, (1, 0): p_alarm_burglary, (0, 1): p_alarm_quake,
                   (0, 0): p_false_alarm},
         "JohnCalls": {(1,): p_john, (0,): p_john_false}, "MaryCalls": {(1,): p_mary, (0,): p_mary_false}},
    )


# ============================================================================== discrete chain: sum / max product
def chain_sum_product(pi: np.ndarray, A: np.ndarray, lik: np.ndarray) -> dict:
    """Messages on the chain s₁ − s₂ − … − s_T with local evidence lik[t, k] = p(x_t | s_t = k).

    forward  m→_t(k) ∝ Σ_j m→_{t−1}(j) lik[t−1, j] A[j, k]      (message arriving at t from the left)
    backward m←_t(k) ∝ Σ_j A[k, j] lik[t+1, j] m←_{t+1}(j)      (message arriving at t from the right)
    marginal P(s_t | x_{1:T}) ∝ m→_t · lik[t] · m←_t
    Every message is normalised (only its shape matters).
    """
    T, K = lik.shape
    fwd = np.zeros((T, K))
    bwd = np.ones((T, K)) / K
    fwd[0] = pi / pi.sum()
    for t in range(1, T):
        m = (fwd[t - 1] * lik[t - 1]) @ A
        fwd[t] = m / m.sum()
    for t in range(T - 2, -1, -1):
        m = A @ (lik[t + 1] * bwd[t + 1])
        bwd[t] = m / m.sum()
    marg = fwd * lik * bwd
    marg /= marg.sum(1, keepdims=True)
    return {"forward": fwd, "backward": bwd, "marginal": marg}


def chain_max_product(pi: np.ndarray, A: np.ndarray, lik: np.ndarray) -> dict:
    """Max-product: replace every sum by a max (in logs). The best joint assignment = Viterbi."""
    T, K = lik.shape
    logA, loglik = np.log(A + 1e-300), np.log(lik + 1e-300)
    fwd = np.zeros((T, K))
    back = np.zeros((T, K), int)
    fwd[0] = np.log(pi + 1e-300)
    for t in range(1, T):
        sc = (fwd[t - 1] + loglik[t - 1])[:, None] + logA
        back[t] = sc.argmax(0)
        fwd[t] = sc.max(0)
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(fwd[-1] + loglik[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    # max-marginals: best score of any full assignment with s_t = k (forward max × local × backward max)
    bwd = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        bwd[t] = np.max(logA + (loglik[t + 1] + bwd[t + 1])[None, :], 1)
    maxmarg = fwd + loglik + bwd
    return {"path": path, "max_marginal": maxmarg - maxmarg.max(1, keepdims=True)}


# ============================================================================== Gaussian chain = Kalman smoother
def gaussian_chain_bp(z: np.ndarray, q: float, r: float, m0: float = 0.0, p0: float = 1.0) -> dict:
    """Belief propagation on x_t = x_{t−1} + w (var q), z_t = x_t + v (var r), x₁ ~ N(m0, p0).

    Messages are Gaussians, carried as (mean, variance). The forward message into t (before z_t) is the Kalman
    prediction, the backward message is what z_{t+1:T} say about x_t, and their product with the local
    evidence N(z_t; x_t, r) is the smoothed marginal.
    """
    T = len(z)
    fm, fv = np.zeros(T), np.zeros(T)           # forward message into x_t (prediction)
    bm, bv = np.zeros(T), np.full(T, np.inf)    # backward message into x_t
    fm[0], fv[0] = m0, p0
    for t in range(1, T):
        # combine message with local evidence (a Kalman update), then pass through the transition (add q)
        v = 1.0 / (1.0 / fv[t - 1] + 1.0 / r)
        m = v * (fm[t - 1] / fv[t - 1] + z[t - 1] / r)
        fm[t], fv[t] = m, v + q
    for t in range(T - 2, -1, -1):
        prec = 1.0 / r + (0.0 if np.isinf(bv[t + 1]) else 1.0 / bv[t + 1])
        v = 1.0 / prec
        m = v * (z[t + 1] / r + (0.0 if np.isinf(bv[t + 1]) else bm[t + 1] / bv[t + 1]))
        bm[t], bv[t] = m, v + q
    # filtered = forward × local; smoothed = forward × local × backward (precisions add)
    filt_v = 1.0 / (1.0 / fv + 1.0 / r)
    filt_m = filt_v * (fm / fv + z / r)
    inv_b = np.where(np.isinf(bv), 0.0, 1.0 / np.where(np.isinf(bv), 1.0, bv))
    sm_v = 1.0 / (1.0 / fv + 1.0 / r + inv_b)
    sm_m = sm_v * (fm / fv + z / r + np.where(np.isinf(bv), 0.0, bm * inv_b))
    return {"fwd_mean": fm, "fwd_var": fv, "bwd_mean": bm, "bwd_var": bv, "filt_mean": filt_m, "filt_var": filt_v,
            "smooth_mean": sm_m, "smooth_var": sm_v}
