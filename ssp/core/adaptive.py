"""L2 — linear and adaptive estimation: Wiener, LMS, NLMS, RLS.

Convention: the tap-input vector at time n is
    u[n] = [x[n], x[n-1], ..., x[n-M+1]]
and the filter output is y[n] = w^T u[n]. The error e[n] = d[n] - y[n] drives
adaptation. In echo cancellation, x is the far-end (loudspeaker) signal, d is
the microphone signal and w converges to the echo path.
"""
from __future__ import annotations

import numpy as np


def tap_matrix(x: np.ndarray, M: int) -> np.ndarray:
    """Rows are u[n] for n = 0..N-1 (zeros before the signal starts)."""
    x = np.asarray(x, dtype=float)
    N = x.size
    U = np.zeros((N, M))
    for k in range(M):
        U[k:, k] = x[: N - k]
    return U


def wiener_solution(x: np.ndarray, d: np.ndarray, M: int) -> np.ndarray:
    """Batch optimum w = R^-1 p from sample statistics (the target LMS/RLS chase)."""
    U = tap_matrix(x, M)
    R = U.T @ U / len(d)
    p = U.T @ np.asarray(d, dtype=float) / len(d)
    return np.linalg.solve(R + 1e-10 * np.eye(M), p)


def lms(x: np.ndarray, d: np.ndarray, M: int, mu: float) -> dict:
    """Least-mean-squares: w <- w + mu * e[n] * u[n]. Stochastic gradient on E[e^2]."""
    U = tap_matrix(x, M)
    N = len(d)
    w = np.zeros(M)
    W = np.zeros((N, M))
    e = np.zeros(N)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(N):
            u = U[n]
            e[n] = d[n] - w @ u
            w = w + mu * e[n] * u
            if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:  # diverged
                W[n:] = np.nan
                e[n:] = np.nan
                break
            W[n] = w
    return {"w": w, "W": W, "e": e}


def nlms(x: np.ndarray, d: np.ndarray, M: int, mu: float, eps: float = 1e-6) -> dict:
    """Normalised LMS: step scaled by input power, so mu in (0, 2) is stable."""
    U = tap_matrix(x, M)
    N = len(d)
    w = np.zeros(M)
    W = np.zeros((N, M))
    e = np.zeros(N)
    for n in range(N):
        u = U[n]
        e[n] = d[n] - w @ u
        w = w + mu * e[n] * u / (eps + u @ u)
        W[n] = w
    return {"w": w, "W": W, "e": e}


def rls(x: np.ndarray, d: np.ndarray, M: int, lam: float = 0.999, delta: float = 100.0) -> dict:
    """Recursive least squares with forgetting factor lam.

    Minimises sum_k lam^(n-k) e[k]^2 exactly at every step. Equivalent to a
    Kalman filter whose state (the weights) is constant, with lam < 1 acting
    like process noise that lets the weights track a changing system.
    """
    U = tap_matrix(x, M)
    N = len(d)
    w = np.zeros(M)
    P = delta * np.eye(M)
    W = np.zeros((N, M))
    e = np.zeros(N)
    for n in range(N):
        u = U[n]
        Pu = P @ u
        k = Pu / (lam + u @ Pu)          # gain vector
        e[n] = d[n] - w @ u              # a-priori error
        w = w + k * e[n]
        P = (P - np.outer(k, Pu)) / lam
        P = 0.5 * (P + P.T)              # keep symmetric numerically
        W[n] = w
    return {"w": w, "W": W, "e": e}


def misalignment_db(W: np.ndarray, h_true: np.ndarray) -> np.ndarray:
    """10 log10(||h - w[n]||^2 / ||h||^2) per step. h_true may be (M,) or (N, M)."""
    h = np.asarray(h_true, dtype=float)
    if h.ndim == 1:
        h = np.broadcast_to(h, W.shape)
    num = np.sum((h - W) ** 2, axis=1)
    den = np.sum(h**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(num / den)


def make_system_id(N: int, h1: np.ndarray, h2: np.ndarray | None, change_at: int | None,
                   noise_std: float, ar_coef: float, rng: np.random.Generator) -> dict:
    """Unknown FIR system identification problem, optionally with an abrupt change.

    The input is AR(1): x[n] = ar_coef * x[n-1] + v[n]. ar_coef = 0 is white;
    ar_coef -> 1 is strongly coloured (speech-like), which slows LMS down.
    """
    v = rng.standard_normal(N)
    x = np.zeros(N)
    for n in range(N):
        x[n] = (ar_coef * x[n - 1] if n else 0.0) + v[n]
    x /= np.std(x)
    M = len(h1)
    U = tap_matrix(x, M)
    H = np.tile(h1, (N, 1))
    if h2 is not None and change_at is not None:
        H[change_at:] = h2
    d = np.sum(U * H, axis=1) + noise_std * rng.standard_normal(N)
    return {"x": x, "d": d, "H": H}


def room_like_ir(M: int, decay: float, rng: np.random.Generator) -> np.ndarray:
    """Short exponentially-decaying random FIR — a toy echo path."""
    h = rng.standard_normal(M) * np.exp(-decay * np.arange(M))
    return h / np.linalg.norm(h)
