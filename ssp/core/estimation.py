"""L1 — point estimation: MLE, MAP, MMSE and the Cramér–Rao lower bound.

Signal models used in the app
-----------------------------
1. DC level in white Gaussian noise          x[n] = A + w[n],  w ~ N(0, s^2)
2. Bernoulli rate (click-through, defects)   x[n] ~ Bernoulli(theta)
3. Sinusoid frequency in noise               x[n] = a cos(2 pi f0 n + phi) + w[n]

For each we give the estimators in closed form (or as a short search) and the
CRLB, so Monte Carlo runs can be checked against theory.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# 1. DC level in WGN
# ---------------------------------------------------------------------------
def dc_log_likelihood(A_grid: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """log p(x | A) evaluated on a grid of candidate A values."""
    x = np.asarray(x, dtype=float)
    N = x.size
    sq = ((x[None, :] - np.asarray(A_grid)[:, None]) ** 2).sum(axis=1)
    return -0.5 * N * np.log(2 * np.pi * sigma**2) - sq / (2 * sigma**2)


def dc_mle(x: np.ndarray) -> float:
    """MLE of A under Gaussian noise is the sample mean."""
    return float(np.mean(x))


def dc_map(x: np.ndarray, sigma: float, prior_mean: float, prior_std: float) -> tuple[float, float]:
    """MAP (= MMSE here, posterior is Gaussian) with prior A ~ N(mu0, s0^2).

    Posterior precision = prior precision + N / sigma^2, and the posterior mean
    is a precision-weighted average of the prior mean and the sample mean.
    Returns (posterior mean, posterior std).
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    prec = 1.0 / prior_std**2 + N / sigma**2
    mean = (prior_mean / prior_std**2 + x.sum() / sigma**2) / prec
    return float(mean), float(np.sqrt(1.0 / prec))


def dc_crlb(sigma: float, N: int) -> float:
    """CRLB for any unbiased estimator of A: var >= sigma^2 / N."""
    return sigma**2 / N


def dc_median(x: np.ndarray) -> float:
    """MLE of A if the noise were Laplacian — robust to impulsive outliers."""
    return float(np.median(x))


# ---------------------------------------------------------------------------
# 2. Bernoulli rate with a Beta prior — where MLE, MAP and MMSE differ
# ---------------------------------------------------------------------------
def bernoulli_estimators(k: int, n: int, alpha: float, beta: float) -> dict:
    """Three answers to "what is theta?" after k successes in n trials.

    MLE  = k / n                         (ignores the prior)
    MAP  = (k + a - 1) / (n + a + b - 2) (posterior mode)
    MMSE = (k + a) / (n + a + b)         (posterior mean, minimises squared error)
    """
    mle = k / n if n > 0 else np.nan
    a, b = alpha + k, beta + n - k
    map_ = (a - 1) / (a + b - 2) if (a > 1 and b > 1) else np.nan
    mmse = a / (a + b)
    return {"MLE": mle, "MAP": map_, "MMSE": mmse, "post_alpha": a, "post_beta": b}


# ---------------------------------------------------------------------------
# 3. Sinusoid frequency — MLE via periodogram peak, CRLB and threshold effect
# ---------------------------------------------------------------------------
def _tone_energy(x: np.ndarray, f: float) -> float:
    """Energy of x explained by the best real tone at frequency f.

    With H = [cos(2 pi f n), sin(2 pi f n)], amplitude and phase are linear
    parameters, so the exact MLE of f maximises x^T H (H^T H)^-1 H^T x.
    (The plain periodogram ignores the -f image and is slightly biased.)
    """
    n = np.arange(x.size)
    H = np.stack([np.cos(2 * np.pi * f * n), np.sin(2 * np.pi * f * n)], axis=1)
    Hx = H.T @ x
    return float(Hx @ np.linalg.solve(H.T @ H, Hx))


def freq_mle(x: np.ndarray, zero_pad: int = 8) -> float:
    """MLE of a single real sinusoid's frequency (cycles/sample).

    Coarse search: zero-padded FFT peak (finds the right neighbourhood).
    Fine search: maximise the exact tone-fit energy within +/- one coarse bin.
    If the coarse peak lands on a noise spike (low SNR), the estimate is far
    off: that is the "threshold effect" seen in the Monte Carlo plot.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    N = x.size
    nfft = int(2 ** np.ceil(np.log2(N * zero_pad)))
    P = np.abs(np.fft.rfft(x, nfft)) ** 2
    k = int(np.argmax(P[1:-1])) + 1
    df = 1.0 / nfft
    res = minimize_scalar(lambda f: -_tone_energy(x, f),
                          bounds=(max(k * df - df, 1e-6), min(k * df + df, 0.5 - 1e-6)),
                          method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def freq_crlb(amplitude: float, sigma: float, N: int) -> float:
    """CRLB on var(f_hat) for a real sinusoid (Kay, Fundamentals Vol. I, eq. 3.41).

    var(f) >= 12 / ((2 pi)^2 * eta * N (N^2 - 1)),  eta = a^2 / (2 sigma^2).
    """
    eta = amplitude**2 / (2 * sigma**2)
    return 12.0 / ((2 * np.pi) ** 2 * eta * N * (N**2 - 1))


def sinusoid(N: int, f0: float, amplitude: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0, 2 * np.pi)
    n = np.arange(N)
    return amplitude * np.cos(2 * np.pi * f0 * n + phi) + sigma * rng.standard_normal(N)


# ---------------------------------------------------------------------------
# Monte Carlo helpers used by the app (and checked in tests)
# ---------------------------------------------------------------------------
def monte_carlo_dc(N_values, sigma: float, true_A: float, prior_mean: float, prior_std: float,
                   trials: int, rng: np.random.Generator, draw_A_from_prior: bool = False) -> dict:
    """Empirical MSE of MLE and MAP versus N.

    draw_A_from_prior=True  -> A ~ prior each trial (the prior is "right"):
                               the Bayesian MSE of MAP falls *below* the CRLB,
                               which only bounds *unbiased* estimators.
    draw_A_from_prior=False -> A is fixed at true_A (the prior may be wrong).
    """
    mse_mle, mse_map = [], []
    for N in N_values:
        if draw_A_from_prior:
            A = prior_mean + prior_std * rng.standard_normal(trials)
        else:
            A = np.full(trials, true_A)
        X = A[:, None] + sigma * rng.standard_normal((trials, N))
        mle = X.mean(axis=1)
        prec = 1 / prior_std**2 + N / sigma**2
        mapv = (prior_mean / prior_std**2 + X.sum(axis=1) / sigma**2) / prec
        mse_mle.append(np.mean((mle - A) ** 2))
        mse_map.append(np.mean((mapv - A) ** 2))
    return {"N": np.asarray(N_values), "MLE": np.array(mse_mle), "MAP": np.array(mse_map),
            "CRLB": sigma**2 / np.asarray(N_values, dtype=float)}


def monte_carlo_freq(snr_db_values, N: int, f0: float, trials: int, rng: np.random.Generator) -> dict:
    """Empirical MSE of the frequency MLE vs SNR, alongside the CRLB."""
    amplitude = 1.0
    mse, crlb = [], []
    for snr_db in snr_db_values:
        # SNR defined as a^2 / (2 sigma^2)
        sigma = np.sqrt(amplitude**2 / (2 * 10 ** (snr_db / 10)))
        est = np.array([freq_mle(sinusoid(N, f0, amplitude, sigma, rng)) for _ in range(trials)])
        mse.append(np.mean((est - f0) ** 2))
        crlb.append(freq_crlb(amplitude, sigma, N))
    return {"snr_db": np.asarray(snr_db_values), "MLE": np.array(mse), "CRLB": np.array(crlb)}


def monte_carlo_outliers(N: int, sigma: float, outlier_frac_values, outlier_scale: float,
                         trials: int, rng: np.random.Generator, A: float = 1.0) -> dict:
    """Mean (Gaussian MLE) vs median (Laplacian MLE) under impulsive noise."""
    mse_mean, mse_med = [], []
    for p in outlier_frac_values:
        noise = sigma * rng.standard_normal((trials, N))
        mask = rng.random((trials, N)) < p
        noise[mask] = outlier_scale * sigma * rng.standard_normal(mask.sum())
        X = A + noise
        mse_mean.append(np.mean((X.mean(axis=1) - A) ** 2))
        mse_med.append(np.mean((np.median(X, axis=1) - A) ** 2))
    return {"p": np.asarray(outlier_frac_values), "mean": np.array(mse_mean), "median": np.array(mse_med)}
