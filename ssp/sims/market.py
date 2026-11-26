"""Synthetic market data with a known hidden truth (for learning only, not advice)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ssp.core.hmm import sample_states

# Daily return regimes: mean and volatility (in % per day)
REGIMES = {
    2: {"names": ["calm", "turbulent"], "mu": [0.06, -0.12], "sigma": [0.7, 2.2]},
    3: {"names": ["calm bull", "choppy", "crash"], "mu": [0.07, 0.0, -0.25], "sigma": [0.6, 1.3, 3.0]},
}


def regime_returns(T: int, K: int, stay: float, rng: np.random.Generator) -> dict:
    """Markov-switching Gaussian returns. stay = probability of keeping the regime each day."""
    cfg = REGIMES[K]
    A = np.full((K, K), (1 - stay) / (K - 1))
    np.fill_diagonal(A, stay)
    s = sample_states(np.full(K, 1 / K), A, T, rng)
    mu = np.array(cfg["mu"])[s]
    sig = np.array(cfg["sigma"])[s]
    r = mu + sig * rng.standard_normal(T)                    # % per day
    price = 100 * np.exp(np.cumsum(r / 100))
    return {"r": r, "price": price, "states": s, "A": A, "names": cfg["names"],
            "mu": np.array(cfg["mu"]), "sigma": np.array(cfg["sigma"])}


def pairs(T: int, beta0: float, beta_shift: float, shift_at: float, spread_vol: float,
          rng: np.random.Generator) -> dict:
    """Two related prices: y = beta_t * x + alpha + mean-reverting spread.

    beta_t drifts slowly and jumps by `beta_shift` at fraction `shift_at` of the
    sample (e.g. a business change altering how the two assets co-move).
    """
    x = 50 * np.exp(np.cumsum(0.012 * rng.standard_normal(T)))
    beta = beta0 + np.cumsum(0.002 * rng.standard_normal(T))
    beta[int(shift_at * T):] += beta_shift
    spread = np.zeros(T)
    for t in range(1, T):                                   # Ornstein–Uhlenbeck
        spread[t] = 0.95 * spread[t - 1] + spread_vol * rng.standard_normal()
    alpha = 5.0
    y = beta * x + alpha + spread
    return {"x": x, "y": y, "beta": beta, "alpha": alpha, "spread": spread}


def returns_from_csv(raw: bytes) -> tuple[pd.Series, str]:
    """Daily % log-returns from an uploaded CSV of prices.

    Uses a column called Adj Close / Close / Price if present, else the last
    numeric column. A Date column, if present, becomes the index.
    """
    df = pd.read_csv(io.BytesIO(raw))
    date_col = next((c for c in df.columns if str(c).lower() in ("date", "datetime", "time", "timestamp")), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    preferred = [c for c in df.columns if str(c).lower() in ("adj close", "adj_close", "close", "price")]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    col = preferred[0] if preferred else numeric[-1]
    px = pd.to_numeric(df[col], errors="coerce").dropna()
    px = px[px > 0]
    r = 100 * np.log(px).diff().dropna()
    return r, str(col)
