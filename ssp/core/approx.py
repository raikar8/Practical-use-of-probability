"""Approximate inference on 2-D posteriors you can see: MCMC (Metropolis–Hastings, Gibbs, Hamiltonian Monte Carlo),
the Laplace approximation, and variational inference (reverse KL), compared with the exact answer on a grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ssp.core.nn import Adam


@dataclass
class Target:
    """An unnormalised log-density on R² with its gradient (both vectorised over rows)."""
    name: str
    logp: Callable[[np.ndarray], np.ndarray]
    grad: Callable[[np.ndarray], np.ndarray]
    box: tuple[float, float, float, float]          # x0, x1, y0, y1 for plotting
    start: tuple[float, float]


def _banana() -> Target:
    # x ~ N(0, 2²); y | x ~ N(x²/4 − 1, 0.5²): a curved ridge, like a posterior with a weakly identified product
    def logp(X):
        x, y = X[:, 0], X[:, 1]
        return -x**2 / 8 - (y - x**2 / 4 + 1) ** 2 / (2 * 0.25)

    def grad(X):
        x, y = X[:, 0], X[:, 1]
        r = (y - x**2 / 4 + 1) / 0.25
        return np.c_[-x / 4 + r * x / 2, -r]
    return Target("banana", logp, grad, (-6, 6, -2.5, 8), (-4.0, 6.0))


def _correlated(rho: float = 0.95) -> Target:
    P = np.linalg.inv(np.array([[1, rho], [rho, 1]]))

    def logp(X):
        return -0.5 * np.einsum("ni,ij,nj->n", X, P, X)

    def grad(X):
        return -X @ P
    t = Target("correlated Gaussian", logp, grad, (-3.5, 3.5, -3.5, 3.5), (-3.0, 3.0))
    t.rho = rho
    return t


def _bimodal() -> Target:
    m = np.array([[-2.0, -0.5], [2.0, 1.0]])
    w = np.array([0.65, 0.35])
    s2 = np.array([0.5, 0.35])

    def comps(X):
        return np.log(w) - np.sum((X[:, None, :] - m[None]) ** 2, -1) / (2 * s2) - np.log(2 * np.pi * s2)

    def logp(X):
        c = comps(X)
        mx = c.max(1, keepdims=True)
        return mx[:, 0] + np.log(np.exp(c - mx).sum(1))

    def grad(X):
        c = comps(X)
        r = np.exp(c - c.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        return np.sum(r[:, :, None] * (-(X[:, None, :] - m[None]) / s2[None, :, None]), 1)
    return Target("two modes", logp, grad, (-4.5, 4.5, -3.5, 3.5), (0.0, 3.0))


TARGETS = {"banana": _banana, "correlated Gaussian": _correlated, "two modes": _bimodal}


def target(name: str) -> Target:
    return TARGETS[name]()


# ============================================================================== exact answer on a grid
def grid_density(t: Target, n: int = 160) -> dict:
    xs = np.linspace(t.box[0], t.box[1], n)
    ys = np.linspace(t.box[2], t.box[3], n)
    XX, YY = np.meshgrid(xs, ys)
    lp = t.logp(np.c_[XX.ravel(), YY.ravel()]).reshape(n, n)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    mean = np.array([np.sum(p * XX), np.sum(p * YY)])
    dx, dy = XX - mean[0], YY - mean[1]
    cov = np.array([[np.sum(p * dx * dx), np.sum(p * dx * dy)], [np.sum(p * dx * dy), np.sum(p * dy * dy)]])
    return {"x": xs, "y": ys, "p": p, "mean": mean, "cov": cov}


# ============================================================================== MCMC
def metropolis(t: Target, n: int, step: float, rng: np.random.Generator, x0=None) -> dict:
    """Random-walk Metropolis: propose x' = x + step·ε; accept with probability min(1, p(x')/p(x))."""
    x = np.array(t.start if x0 is None else x0, float)
    lp = t.logp(x[None])[0]
    chain = np.zeros((n, 2))
    acc = 0
    for i in range(n):
        prop = x + step * rng.standard_normal(2)
        lp2 = t.logp(prop[None])[0]
        if np.log(rng.random()) < lp2 - lp:
            x, lp = prop, lp2
            acc += 1
        chain[i] = x
    return {"chain": chain, "accept": acc / n}


def hmc(t: Target, n: int, eps: float, L: int, rng: np.random.Generator, x0=None, keep_paths: int = 0) -> dict:
    """Hamiltonian Monte Carlo: give the point a random momentum, slide along the log-density for L leapfrog
    steps of size eps (like a frictionless puck on the surface −log p), then accept/reject on the energy error."""
    x = np.array(t.start if x0 is None else x0, float)
    chain = np.zeros((n, 2))
    paths = []
    acc = 0
    with np.errstate(all="ignore"):          # a too-large step can blow the simulation up; it's then rejected
        for i in range(n):
            p0 = rng.standard_normal(2)
            xn, p = x.copy(), p0.copy()
            path = [xn.copy()]
            p = p + 0.5 * eps * t.grad(xn[None])[0]              # half step in momentum
            for j in range(L):
                xn = xn + eps * p                                  # full step in position
                path.append(xn.copy())
                if j < L - 1:
                    p = p + eps * t.grad(xn[None])[0]
            p = p + 0.5 * eps * t.grad(xn[None])[0]              # final half step in momentum
            H0 = -t.logp(x[None])[0] + 0.5 * p0 @ p0
            H1 = -t.logp(xn[None])[0] + 0.5 * p @ p
            ok = bool(np.isfinite(H1) and np.log(rng.random()) < H0 - H1)
            if ok:
                x = xn
                acc += 1
            if i < keep_paths:
                paths.append((np.array(path), ok))
            chain[i] = x
    return {"chain": chain, "accept": acc / n, "paths": paths}


def gibbs_gaussian(rho: float, n: int, rng: np.random.Generator, x0=(-3.0, 3.0)) -> dict:
    """Gibbs sampling for a standard bivariate Gaussian with correlation rho: alternate exact 1-D conditionals
    x | y ~ N(ρy, 1 − ρ²) and y | x ~ N(ρx, 1 − ρ²). Returns the chain and the zig-zag path."""
    x, y = x0
    s = np.sqrt(1 - rho**2)
    chain, zig = np.zeros((n, 2)), [(x, y)]
    for i in range(n):
        x = rho * y + s * rng.standard_normal()
        zig.append((x, y))
        y = rho * x + s * rng.standard_normal()
        zig.append((x, y))
        chain[i] = (x, y)
    return {"chain": chain, "zigzag": np.array(zig), "accept": 1.0}


def ess(chain: np.ndarray) -> float:
    """Effective sample size (smallest over dimensions): n / τ, with the integrated autocorrelation time τ from
    Geyer's initial monotone sequence: pairs P_m = ρ_{2m} + ρ_{2m+1}, summed while positive and non-increasing.
    (Anti-correlated chains, e.g. HMC on a Gaussian, can have τ < 1; we cap the ESS at n.)"""
    out = []
    for d in range(chain.shape[1]):
        x = chain[:, d] - chain[:, d].mean()
        n = len(x)
        if np.allclose(x, 0):
            out.append(1.0)
            continue
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        ac /= ac[0]
        total, prev, m = 0.0, np.inf, 0
        while 2 * m + 1 < n:
            pair = ac[2 * m] + ac[2 * m + 1]
            if pair <= 0:
                break
            pair = min(pair, prev)
            total += pair
            prev = pair
            m += 1
        out.append(min(float(n), n / max(-1.0 + 2.0 * total, 1e-12)))    # capped at n for display
    return float(min(out))


# ============================================================================== Laplace and VI
def laplace(t: Target, x0=None) -> dict:
    """Laplace: a Gaussian at the mode, with covariance = inverse curvature (negative Hessian) there."""
    x0 = np.array(t.start if x0 is None else x0, float)
    res = minimize(lambda v: -t.logp(v[None])[0], x0, jac=lambda v: -t.grad(v[None])[0], method="BFGS")
    m = res.x
    h = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        H[i] = (t.grad((m + e)[None])[0] - t.grad((m - e)[None])[0]) / (2 * h)
    H = 0.5 * (H + H.T)
    return {"mean": m, "cov": np.linalg.inv(-H)}


def vi_gaussian(t: Target, steps: int = 1500, lr: float = 0.05, n_mc: int = 64, seed: int = 0, init=None,
                snapshots: int = 0) -> dict:
    """Variational inference: fit q = N(m, LLᵀ) by maximising the ELBO = E_q[log p] + entropy(q).

    Reparameterise z = m + Lε, so ∇_m ELBO = E[∇log p(z)], ∇_L ELBO = E[∇log p(z) εᵀ] + diag(1/L_ii).
    This minimises KL(q || p) (reverse KL): q avoids putting mass where p has none, so it locks onto one mode.
    """
    rng = np.random.default_rng(seed)
    m = np.array(t.start if init is None else init, float)
    ld = np.zeros(2)                         # log of the diagonal of L
    off = np.zeros(1)                        # L[1, 0]
    params = [m, ld, off]
    opt = Adam(params, lr=lr)
    hist, elbos = [], []
    every = max(1, steps // snapshots) if snapshots else None
    for k in range(steps):
        L = np.array([[np.exp(ld[0]), 0.0], [off[0], np.exp(ld[1])]])
        eps = rng.standard_normal((n_mc, 2))
        z = m + eps @ L.T
        g = t.grad(z)
        elbos.append(float(np.mean(t.logp(z)) + np.sum(ld)))
        gm = g.mean(0)
        gL = (g[:, :, None] * eps[:, None, :]).mean(0)
        g_ld = gL[[0, 1], [0, 1]] * np.exp(ld) + 1.0          # chain rule through exp, + entropy term
        g_off = np.array([gL[1, 0]])
        opt.step(params, [-gm, -g_ld, -g_off])                # Adam minimises, so negate the ELBO gradient
        if every and (k % every == 0 or k == steps - 1):
            hist.append((m.copy(), (L @ L.T).copy(), k + 1))
    L = np.array([[np.exp(ld[0]), 0.0], [off[0], np.exp(ld[1])]])
    return {"mean": m.copy(), "cov": L @ L.T, "elbo": np.array(elbos), "hist": hist}


def gauss_kl_to_grid(mean: np.ndarray, cov: np.ndarray, grid: dict) -> tuple[float, float]:
    """(KL(q || p), KL(p || q)) between a Gaussian q and the gridded exact p (on the grid's support)."""
    XX, YY = np.meshgrid(grid["x"], grid["y"])
    P = np.c_[XX.ravel(), YY.ravel()]
    Ci = np.linalg.inv(cov)
    d = P - mean
    lq = -0.5 * np.einsum("ni,ij,nj->n", d, Ci, d)
    q = np.exp(lq - lq.max())
    q /= q.sum()
    p = grid["p"].ravel()
    m1 = q > 1e-300
    m2 = p > 1e-300
    kl_qp = float(np.sum(q[m1] * (np.log(q[m1]) - np.log(np.maximum(p[m1], 1e-300)))))
    kl_pq = float(np.sum(p[m2] * (np.log(p[m2]) - np.log(np.maximum(q[m2], 1e-300)))))
    return kl_qp, kl_pq


def ellipse(mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, n: int = 60) -> np.ndarray:
    w, V = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, n)
    return mean + (np.c_[np.cos(a), np.sin(a)] * n_std * np.sqrt(np.maximum(w, 0))) @ V.T
