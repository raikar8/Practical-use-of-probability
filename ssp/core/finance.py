"""Estimation tools used in quantitative finance (for learning only, not advice).

* GARCH(1,1) fitted by maximum likelihood: volatility clusters, so today's
  variance depends on yesterday's shock and yesterday's variance.
* A Kalman filter for a time-varying regression (dynamic hedge ratio between
  two related assets): the regression coefficients are the hidden state.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# GARCH(1,1):  r_t = mu + e_t,  e_t = sigma_t z_t,
#              sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2
# ---------------------------------------------------------------------------
def garch11_variance(r: np.ndarray, mu: float, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional variance path; sigma_0^2 starts at the sample variance."""
    e = np.asarray(r, dtype=float) - mu
    s2 = np.empty(e.size + 1)              # s2[t] = variance for r[t]; s2[T] = 1-step forecast
    s2[0] = e.var()
    for t in range(e.size):
        s2[t + 1] = omega + alpha * e[t] ** 2 + beta * s2[t]
    return s2


def garch11_negloglik(theta: np.ndarray, r: np.ndarray) -> float:
    mu, omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10
    s2 = garch11_variance(r, mu, omega, alpha, beta)[:-1]
    e = r - mu
    return float(0.5 * np.sum(np.log(2 * np.pi * s2) + e**2 / s2))


def garch11_fit(r: np.ndarray) -> dict:
    """Maximum-likelihood fit (Gaussian innovations). Returns params and paths."""
    r = np.asarray(r, dtype=float)
    v = r.var()
    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.80), (0.20, 0.70)):
        x0 = np.array([r.mean(), v * (1 - a0 - b0), a0, b0])
        res = minimize(garch11_negloglik, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    mu, omega, alpha, beta = best.x
    s2 = garch11_variance(r, mu, omega, alpha, beta)
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "sigma2": s2,
            "loglik": -best.fun, "persistence": alpha + beta,
            "long_run_var": omega / (1 - alpha - beta)}


def garch11_forecast(fit: dict, horizon: int) -> np.ndarray:
    """h-step variance forecasts: mean-revert to the long-run variance at rate alpha+beta."""
    p = fit["persistence"]
    lr = fit["long_run_var"]
    s1 = fit["sigma2"][-1]
    h = np.arange(horizon)
    return lr + p**h * (s1 - lr)


def garch11_simulate(T: int, mu: float, omega: float, alpha: float, beta: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    s2 = np.empty(T)
    r = np.empty(T)
    prev_s2 = omega / (1 - alpha - beta)
    prev_e = 0.0
    for t in range(T):
        s2[t] = omega + alpha * prev_e**2 + beta * prev_s2
        prev_e = np.sqrt(s2[t]) * rng.standard_normal()
        r[t] = mu + prev_e
        prev_s2 = s2[t]
    return r, s2


# ---------------------------------------------------------------------------
# Kalman filter for a time-varying regression y_t = beta_t x_t + alpha_t + v_t
# ---------------------------------------------------------------------------
def kalman_regression(y: np.ndarray, x: np.ndarray, delta: float = 1e-4, r: float | None = None) -> dict:
    """State theta_t = [beta_t, alpha_t] follows a random walk.

    delta sets how fast the coefficients may drift: Q = delta / (1 - delta) * I.
    Observation matrix changes every step: H_t = [x_t, 1].
    Returns coefficient paths, their std devs, innovations and their std devs
    (the 'spread' and its z-score used in pairs trading).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    T = y.size
    if r is None:
        r = np.var(y - np.polyval(np.polyfit(x[:50], y[:50], 1), x)) if T > 50 else 1.0
    Q = delta / (1 - delta) * np.eye(2)
    theta = np.zeros(2)
    P = np.eye(2) * 1e3
    betas, alphas = np.zeros(T), np.zeros(T)
    sd = np.zeros((T, 2))
    innov, innov_sd = np.zeros(T), np.zeros(T)
    for t in range(T):
        P = P + Q                                    # predict (random walk)
        H = np.array([x[t], 1.0])
        e = y[t] - H @ theta                         # innovation = spread
        S = H @ P @ H + r
        K = P @ H / S
        theta = theta + K * e
        P = P - np.outer(K, H @ P)
        betas[t], alphas[t] = theta
        sd[t] = np.sqrt(np.diag(P))
        innov[t], innov_sd[t] = e, np.sqrt(S)
    return {"beta": betas, "alpha": alphas, "sd": sd, "innov": innov, "innov_sd": innov_sd}


def rolling_ols_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """The common heuristic the Kalman filter replaces: OLS over a sliding window."""
    T = len(y)
    out = np.full(T, np.nan)
    for t in range(window, T + 1):
        xs, ys = x[t - window:t], y[t - window:t]
        xm, ym = xs.mean(), ys.mean()
        out[t - 1] = np.sum((xs - xm) * (ys - ym)) / np.sum((xs - xm) ** 2)
    return out


def kalman_regression_loglik(y: np.ndarray, x: np.ndarray, delta: float, r: float, burn: int = 20) -> float:
    """Prediction-error-decomposition likelihood: sum of log N(innovation; 0, S_t)."""
    k = kalman_regression(y, x, delta=delta, r=r)
    e, s = k["innov"][burn:], k["innov_sd"][burn:]
    return float(-0.5 * np.sum(np.log(2 * np.pi * s**2) + (e / s) ** 2))


def kalman_regression_mle(y: np.ndarray, x: np.ndarray) -> dict:
    """Choose delta and r by maximum likelihood of the innovations (L1's MLE, applied to a filter)."""
    def nll(theta):
        return -kalman_regression_loglik(y, x, 10.0 ** theta[0], 10.0 ** theta[1])
    v0 = np.log10(np.var(np.diff(y)) + 1e-9)
    best = None
    for d0 in (-7.0, -5.0, -3.0):
        res = minimize(nll, np.array([d0, v0]), method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    return {"delta": float(10 ** best.x[0]), "r": float(10 ** best.x[1]), "loglik": float(-best.fun)}
