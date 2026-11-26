"""L3 — state-space models: Kalman filter, EKF, RTS smoother, consistency tests.

State-space model
-----------------
    x[k]   = F x[k-1] + B u[k] + w[k],   w ~ N(0, Q)     (how the world moves)
    z[k]   = H x[k] + v[k],              v ~ N(0, R)     (what the sensor sees)

The Kalman filter keeps a Gaussian belief N(x_hat, P) and alternates
    predict : push the belief through the motion model (uncertainty grows)
    update  : fuse a measurement, weighted by the Kalman gain (uncertainty shrinks)

Everything here is written out explicitly; no hidden library calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Linear Kalman filter
# ---------------------------------------------------------------------------
@dataclass
class KalmanFilter:
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray
    B: np.ndarray | None = None
    # diagnostics from the most recent update
    innovation: np.ndarray | None = field(default=None, init=False)
    S: np.ndarray | None = field(default=None, init=False)
    K: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        for name in ("F", "H", "Q", "R", "P"):
            setattr(self, name, np.atleast_2d(np.asarray(getattr(self, name), dtype=float)))
        self.x = np.asarray(self.x, dtype=float).reshape(-1)

    def predict(self, u: np.ndarray | None = None, F: np.ndarray | None = None,
                Q: np.ndarray | None = None) -> None:
        """x <- F x + B u ;  P <- F P F^T + Q  (F, Q may be overridden per step)."""
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q
        self.x = F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.atleast_1d(u)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray | None) -> float | None:
        """Fuse measurement z. Returns the NIS (normalised innovation squared).

        Pass z=None for a missed measurement: the belief stays as predicted.
        Uses the Joseph form for P, which stays symmetric positive-definite even
        with a slightly wrong gain.
        """
        if z is None:
            self.innovation, self.S, self.K = None, None, None
            return None
        z = np.atleast_1d(np.asarray(z, dtype=float))
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T  # K = P H^T S^-1
        self.x = self.x + K @ y
        I_KH = np.eye(self.P.shape[0]) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.innovation, self.S, self.K = y, S, K
        return float(y @ np.linalg.solve(S, y))


def run_kalman(kf: KalmanFilter, zs, us=None) -> dict:
    """Run predict/update over a sequence. zs[k] may be None (missing).

    Returns filtered states/covariances, the one-step predictions (needed by
    the RTS smoother), NIS values and Kalman gains.
    """
    n = len(zs)
    d = kf.x.size
    xs, Ps = np.zeros((n, d)), np.zeros((n, d, d))
    xps, Pps = np.zeros((n, d)), np.zeros((n, d, d))
    nis = np.full(n, np.nan)
    gains = []
    for k in range(n):
        kf.predict(None if us is None else us[k])
        xps[k], Pps[k] = kf.x, kf.P
        r = kf.update(zs[k])
        if r is not None:
            nis[k] = r
        gains.append(None if kf.K is None else kf.K.copy())
        xs[k], Ps[k] = kf.x, kf.P
    return {"x": xs, "P": Ps, "x_pred": xps, "P_pred": Pps, "nis": nis, "K": gains}


def rts_smoother(xs: np.ndarray, Ps: np.ndarray, xps: np.ndarray, Pps: np.ndarray,
                 F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rauch–Tung–Striebel smoother: uses *future* measurements too (offline).

    C_k = P_k F^T P_pred[k+1]^-1
    x_s[k] = x_k + C_k (x_s[k+1] - x_pred[k+1])
    P_s[k] = P_k + C_k (P_s[k+1] - P_pred[k+1]) C_k^T
    """
    n = xs.shape[0]
    xs_s, Ps_s = xs.copy(), Ps.copy()
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ F.T @ np.linalg.inv(Pps[k + 1])
        xs_s[k] = xs[k] + C @ (xs_s[k + 1] - xps[k + 1])
        Ps_s[k] = Ps[k] + C @ (Ps_s[k + 1] - Pps[k + 1]) @ C.T
    return xs_s, Ps_s


# ---------------------------------------------------------------------------
# Extended Kalman filter (nonlinear f and h, linearised by Jacobians)
# ---------------------------------------------------------------------------
@dataclass
class ExtendedKalmanFilter:
    f: Callable          # f(x, u, dt) -> x_next
    F_jac: Callable      # F_jac(x, u, dt) -> df/dx
    h: Callable          # h(x) -> z_pred
    H_jac: Callable      # H_jac(x) -> dh/dx
    Q: Callable | np.ndarray   # Q or Q(x, u, dt)
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray

    def predict(self, u, dt: float) -> None:
        F = self.F_jac(self.x, u, dt)
        Q = self.Q(self.x, u, dt) if callable(self.Q) else self.Q
        self.x = self.f(self.x, u, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z) -> float | None:
        if z is None:
            return None
        z = np.atleast_1d(np.asarray(z, dtype=float))
        H = self.H_jac(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ y
        I_KH = np.eye(self.P.shape[0]) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return float(y @ np.linalg.solve(S, y))


# ---------------------------------------------------------------------------
# Motion models
# ---------------------------------------------------------------------------
def constant_velocity(dt: float, q: float, dims: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Discrete white-noise-acceleration model, state [p, v] per axis.

    q is the continuous acceleration noise spectral density (units^2 / s^3).
    Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]. For dims=2 the state is
    [px, vx, py, vy].
    """
    F1 = np.array([[1.0, dt], [0.0, 1.0]])
    Q1 = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    F = np.kron(np.eye(dims), F1)
    Q = np.kron(np.eye(dims), Q1)
    return F, Q


def position_observation(dims: int = 1) -> np.ndarray:
    """H that picks out positions from a [p, v, p, v, ...] state."""
    return np.kron(np.eye(dims), np.array([[1.0, 0.0]]))


# ---------------------------------------------------------------------------
# Consistency tests — "is my filter honest about its uncertainty?"
# ---------------------------------------------------------------------------
def nees(x_true: np.ndarray, x_est: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared e^T P^-1 e per time step."""
    e = x_true - x_est
    return np.einsum("ki,kij,kj->k", e, np.linalg.inv(P), e)


def chi2_band(dof: int, runs: int = 1, level: float = 0.95) -> tuple[float, float]:
    """Two-sided band for the *average* of `runs` chi-square(dof) variables."""
    lo = stats.chi2.ppf((1 - level) / 2, dof * runs) / runs
    hi = stats.chi2.ppf(1 - (1 - level) / 2, dof * runs) / runs
    return float(lo), float(hi)


def cov_ellipse(mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, n_pts: int = 48):
    """Points of the n-sigma ellipse of a 2D Gaussian (for plotting)."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(t), np.sin(t)])
    pts = vecs @ (n_std * np.sqrt(vals)[:, None] * circle)
    return mean[0] + pts[0], mean[1] + pts[1]


# ---------------------------------------------------------------------------
# Simulators for the L3 playgrounds
# ---------------------------------------------------------------------------
def simulate_random_walk(n: int, q: float, r: float, x0: float, rng: np.random.Generator):
    """x[k] = x[k-1] + w, z[k] = x[k] + v with var(w)=q, var(v)=r."""
    x = x0 + np.cumsum(np.sqrt(q) * rng.standard_normal(n))
    z = x + np.sqrt(r) * rng.standard_normal(n)
    return x, z


def simulate_cv_2d(n: int, dt: float, q: float, r: float, rng: np.random.Generator,
                   x0=(0.0, 1.0, 0.0, 0.5)):
    """Truth and position measurements from the 2D constant-velocity model itself."""
    F, Q = constant_velocity(dt, q, dims=2)
    H = position_observation(2)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    X = np.zeros((n, 4))
    x = np.asarray(x0, dtype=float)
    for k in range(n):
        x = F @ x + L @ rng.standard_normal(4)
        X[k] = x
    Z = X @ H.T + np.sqrt(r) * rng.standard_normal((n, 2))
    return X, Z
