"""L5 — particle filters (sequential Monte Carlo).

When the model is nonlinear or the posterior has several peaks, a single
Gaussian (Kalman / EKF) cannot represent the belief. A particle filter keeps N
samples ("particles") with weights instead:

    propagate  each particle through the motion model (with its own random noise)
    weight     multiply each weight by the measurement likelihood p(z | particle)
    resample   when the weights degenerate, clone heavy particles, drop light ones

Also here: a grid ("point-mass") filter for 1-D problems. It computes the exact
Bayesian posterior on a fine grid, so we can check what the particle filter is
approximating.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Resampling schemes
# ---------------------------------------------------------------------------
def multinomial_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N independent draws from the weights. Simple, but the noisiest scheme."""
    return rng.choice(w.size, size=w.size, p=w)


def stratified_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw inside each of N equal strata of [0, 1)."""
    N = w.size
    positions = (rng.random(N) + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(w), positions), N - 1)


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random offset, then N evenly spaced pointers. Low variance, O(N)."""
    N = w.size
    positions = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(w), positions), N - 1)


RESAMPLERS = {"systematic": systematic_resample, "stratified": stratified_resample,
              "multinomial": multinomial_resample}


def effective_sample_size(w: np.ndarray) -> float:
    """ESS = 1 / sum w^2: N for equal weights, 1 when one particle has all the weight."""
    return float(1.0 / np.sum(w**2))


# ---------------------------------------------------------------------------
# Bootstrap (SIR) particle filter
# ---------------------------------------------------------------------------
def bootstrap_pf(zs, N: int, init: Callable, propagate: Callable, loglik: Callable,
                 rng: np.random.Generator, resample: str = "systematic", ess_frac: float = 0.5,
                 snapshot_every: int | None = None) -> dict:
    """Sequential importance resampling with the motion model as proposal.

    init(N, rng)              -> particles at t = 0 (shape (N,) or (N, d))
    propagate(x, t, rng)      -> particles moved from t-1 to t
    loglik(z, x, t)           -> log p(z | x) for every particle
    zs[t] may be None (no measurement: only propagate).
    resample: "systematic" | "stratified" | "multinomial" | "none"
    Resampling happens only when ESS < ess_frac * N.
    """
    x = init(N, rng)
    logw = np.full(N, -np.log(N))
    T = len(zs)
    d = 1 if x.ndim == 1 else x.shape[1]
    mean = np.zeros((T, d))
    std = np.zeros((T, d))
    ess = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)
    snaps = []
    for t, z in enumerate(zs):
        x = propagate(x, t, rng)
        if z is not None:
            logw = logw + loglik(z, x, t)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        X = x.reshape(N, d)
        mean[t] = w @ X
        std[t] = np.sqrt(np.maximum(w @ (X - mean[t]) ** 2, 0))
        ess[t] = effective_sample_size(w)
        if snapshot_every and t % snapshot_every == 0:
            snaps.append((t, X.copy(), w.copy()))
        if resample != "none" and ess[t] < ess_frac * N:
            idx = RESAMPLERS[resample](w, rng)
            x = x[idx]
            logw = np.full(N, -np.log(N))
            resampled[t] = True
        else:
            logw = np.log(w + 1e-300)
    return {"mean": mean.squeeze(), "std": std.squeeze(), "ess": ess, "resampled": resampled, "snapshots": snaps}


# ---------------------------------------------------------------------------
# Exact 1-D Bayesian filter on a grid (reference for the particle filter)
# ---------------------------------------------------------------------------
def grid_filter(zs, grid: np.ndarray, prior: np.ndarray, trans_mean: Callable, trans_var: float,
                loglik: Callable) -> dict:
    """Point-mass filter: p(x_t | z_1:t) evaluated on `grid`.

    trans_mean(x, t) gives the mean of x_t given x_{t-1} = x (Gaussian noise
    with variance trans_var). Cost is O(G^2) per step, which is fine in 1-D and
    impossible in high dimensions. That curse of dimensionality is why
    particle filters exist.
    """
    G = grid.size
    dx = grid[1] - grid[0]
    p = prior / (prior.sum() * dx)
    post = np.zeros((len(zs), G))
    for t, z in enumerate(zs):
        m = trans_mean(grid, t)                                     # (G,) from each source point
        K = np.exp(-0.5 * (grid[:, None] - m[None, :]) ** 2 / trans_var)
        K /= np.sqrt(2 * np.pi * trans_var)
        p = K @ p * dx                                               # Chapman–Kolmogorov
        if z is not None:
            ll = loglik(z, grid, t)
            p = p * np.exp(ll - ll.max())
        p /= p.sum() * dx
        post[t] = p
    return {"post": post, "mean": post @ grid * dx}


# ---------------------------------------------------------------------------
# Benchmark: the univariate non-stationary growth model (Gordon et al. 1993)
# ---------------------------------------------------------------------------
UNGM = {"q": 10.0, "r": 1.0}


def ungm_f(x, t):
    return 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * t)


def ungm_h(x):
    return x**2 / 20.0


def ungm_simulate(T: int, rng: np.random.Generator, q: float = 10.0, r: float = 1.0, x0_std: float = 2.0):
    x = np.zeros(T)
    z = np.zeros(T)
    xp = x0_std * rng.standard_normal()
    for t in range(T):
        xp = ungm_f(xp, t) + np.sqrt(q) * rng.standard_normal()
        x[t] = xp
        z[t] = ungm_h(xp) + np.sqrt(r) * rng.standard_normal()
    return x, z


def ungm_ekf(z: np.ndarray, q: float, r: float, x0: float = 0.0, P0: float = 4.0) -> dict:
    """EKF on the benchmark. h(x) = x^2/20 loses the sign of x, so a single
    Gaussian has to pick one side and often picks the wrong one."""
    x, P = x0, P0
    xs, Ps = np.zeros(len(z)), np.zeros(len(z))
    for t, zt in enumerate(z):
        F = 0.5 + 25 * (1 - x**2) / (1 + x**2) ** 2
        x = ungm_f(x, t)
        P = F * P * F + q
        H = x / 10.0
        S = H * P * H + r
        K = P * H / S
        x = x + K * (zt - ungm_h(x))
        P = (1 - K * H) * P
        xs[t], Ps[t] = x, P
    return {"mean": xs, "std": np.sqrt(Ps)}


# ---------------------------------------------------------------------------
# Demo: "where am I on this road?" from a barometric altimeter + odometer
# ---------------------------------------------------------------------------
def road_profile(s: np.ndarray, seed: int = 0) -> np.ndarray:
    """Elevation (m) along a 6 km road: rolling hills, repeating patterns and two big climbs."""
    rng = np.random.default_rng(seed)
    h = 40 + 8 * np.sin(2 * np.pi * s / 900) + 5 * np.sin(2 * np.pi * s / 370 + 1.0)
    for c, a, w in zip(rng.uniform(500, 5500, 6), rng.uniform(-10, 15, 6), rng.uniform(80, 300, 6)):
        h = h + a * np.exp(-0.5 * ((s - c) / w) ** 2)
    return h


def road_simulate(T: int, dt: float, start: float, rng: np.random.Generator, baro_std: float, odo_std: float,
                  seed: int = 0) -> dict:
    v = 15 + 3 * np.sin(np.arange(T) * dt / 8.0)
    s = start + np.cumsum(v * dt)
    z_alt = road_profile(s, seed) + baro_std * rng.standard_normal(T)
    v_meas = v + odo_std * rng.standard_normal(T)
    return {"s": s, "v": v, "z": z_alt, "v_meas": v_meas}
