"""L5 — Interacting Multiple Model (IMM) filter.

A target that cruises *and* manoeuvres doesn't fit one motion model: a low
process noise is accurate on straights but lags in turns, and a high one copes
with turns but is noisy everywhere. IMM runs one Kalman filter per model and
treats "which model is active" as a hidden Markov chain. It is the HMM of L4
and the Kalman filter of L3 working together.

One cycle (Blom & Bar-Shalom):
  1. mixing      each filter starts from a blend of all filters, weighted by
                 how likely each model is to have switched into it
  2. filtering   ordinary predict/update in every model
  3. model prob  mu_j ∝ likelihood of model j's innovation × prior mu_j
  4. combine     output = probability-weighted mixture of the model estimates
"""
from __future__ import annotations

import numpy as np


def _gauss_lik(nu: np.ndarray, S: np.ndarray) -> float:
    d = nu.size
    return float(np.exp(-0.5 * nu @ np.linalg.solve(S, nu)) / np.sqrt((2 * np.pi) ** d * np.linalg.det(S)))


def imm_filter(zs: np.ndarray, Fs: list, Qs: list, H: np.ndarray, R: np.ndarray, Pt: np.ndarray,
               x0: np.ndarray, P0: np.ndarray, mu0: np.ndarray) -> dict:
    """IMM for linear-Gaussian models that share the same state vector.

    Fs, Qs : per-model transition and process-noise matrices
    Pt     : model transition matrix, Pt[i, j] = P(model j at k | model i at k-1)
    Returns combined estimates, per-model probabilities and covariances.
    """
    M = len(Fs)
    n = x0.size
    xs = [x0.copy() for _ in range(M)]
    Ps = [P0.copy() for _ in range(M)]
    mu = np.asarray(mu0, dtype=float)
    T = len(zs)
    X, P_out, MU = np.zeros((T, n)), np.zeros((T, n, n)), np.zeros((T, M))
    for k, z in enumerate(zs):
        # 1. mixing
        c = Pt.T @ mu                                    # predicted model probabilities
        mix = (Pt * mu[:, None]) / c[None, :]            # mix[i, j] = P(model i at k-1 | model j at k)
        x0s, P0s = [], []
        for j in range(M):
            xm = sum(mix[i, j] * xs[i] for i in range(M))
            Pm = sum(mix[i, j] * (Ps[i] + np.outer(xs[i] - xm, xs[i] - xm)) for i in range(M))
            x0s.append(xm)
            P0s.append(Pm)
        # 2. model-matched Kalman filters
        lik = np.zeros(M)
        for j in range(M):
            x = Fs[j] @ x0s[j]
            P = Fs[j] @ P0s[j] @ Fs[j].T + Qs[j]
            nu = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            xs[j] = x + K @ nu
            I_KH = np.eye(n) - K @ H
            Ps[j] = I_KH @ P @ I_KH.T + K @ R @ K.T
            lik[j] = _gauss_lik(nu, S)
        # 3. model probabilities
        mu = lik * c
        mu = mu / mu.sum() if mu.sum() > 0 else c
        # 4. combination (for output only)
        xc = sum(mu[j] * xs[j] for j in range(M))
        Pc = sum(mu[j] * (Ps[j] + np.outer(xs[j] - xc, xs[j] - xc)) for j in range(M))
        X[k], P_out[k], MU[k] = xc, Pc, mu
    return {"x": X, "P": P_out, "mu": MU}


def coordinated_turn_truth(segments, dt: float, speed: float, x0=(0.0, 0.0), heading: float = 0.0):
    """Ground truth from (duration_s, turn_rate_deg_per_s) segments. State [px, vx, py, vy]."""
    X = []
    px, py = x0
    th = np.deg2rad(heading)
    for dur, rate in segments:
        w = np.deg2rad(rate)
        for _ in range(int(round(dur / dt))):
            th += w * dt
            px += speed * np.cos(th) * dt
            py += speed * np.sin(th) * dt
            X.append([px, speed * np.cos(th), py, speed * np.sin(th)])
    return np.array(X)
