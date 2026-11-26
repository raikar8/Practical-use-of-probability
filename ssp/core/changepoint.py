"""L5 — Bayesian online change-point detection (Adams & MacKay, 2007).

Hidden quantity: the *run length* r_t, the number of steps since the last
change. Each step either continues the run (r -> r+1) or resets it (r -> 0)
with hazard H = 1/lambda. For every possible run length we keep conjugate
posterior parameters of the data model, so the predictive probability of the
next sample is available in closed form.

Data model: Gaussian with unknown mean AND variance, Normal–Inverse-Gamma
prior. The predictive is a Student-t, which handles shifts in level *and* in
volatility.

This is exactly an HMM forward pass (L4) over an ever-growing state space,
truncated at r_max to stay O(r_max) per step.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bocpd(x: np.ndarray, hazard_lambda: float, mu0: float = 0.0, kappa0: float = 1.0,
          alpha0: float = 1.0, beta0: float = 1.0, r_max: int = 400) -> dict:
    """Run-length posterior P(r_t | x_1..t) for every t (truncated at r_max)."""
    x = np.asarray(x, dtype=float)
    T = x.size
    H = 1.0 / hazard_lambda
    R = np.zeros((T, r_max))
    # sufficient statistics per run length (index = run length)
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    p = np.array([1.0])                                  # P(r_{t-1} | x_1..t-1)
    log_evidence = 0.0
    for t in range(T):
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = stats.t.pdf(x[t], df=2 * alpha, loc=mu, scale=scale)      # predictive per run length
        growth = p * pred * (1 - H)
        cp = np.sum(p * pred * H)
        new = np.r_[cp, growth]
        ev = new.sum()
        log_evidence += np.log(ev)
        new /= ev
        # update the sufficient statistics (r=0 restarts from the prior)
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.r_[mu0, mu_n]
        kappa = np.r_[kappa0, kappa + 1]
        alpha = np.r_[alpha0, alpha + 0.5]
        beta = np.r_[beta0, beta_n]
        if new.size > r_max:                             # truncate the oldest run lengths
            new, mu, kappa, alpha, beta = new[:r_max], mu[:r_max], kappa[:r_max], alpha[:r_max], beta[:r_max]
            new /= new.sum()
        p = new
        R[t, : p.size] = p
    map_rl = R.argmax(axis=1)
    return {"R": R, "map": map_rl, "log_evidence": log_evidence}


def detect_changes_detail(map_rl: np.ndarray, min_drop: int = 10, min_gap: int = 10,
                          confirm: int = 5) -> list[tuple[int, int]]:
    """Report a change when the most probable run length collapses *and stays* collapsed.

    A single outlier makes the MAP run length blip to ~0 for one sample, then
    the old run wins again. Requiring the new run to keep growing for `confirm`
    samples filters those blips, at the price of `confirm` samples of delay.
    Returns (start, declared) pairs: the change is dated at start = t - r_t, and
    it is declared at time `declared` = t + confirm, when an online system would act.
    """
    out = []
    T = map_rl.size
    for t in range(1, T - confirm):
        if map_rl[t - 1] - map_rl[t] >= min_drop:
            if all(map_rl[t + j] <= map_rl[t] + j + 1 for j in range(1, confirm + 1)):
                start = int(t - map_rl[t])
                if not out or start - out[-1][0] >= min_gap:
                    out.append((start, t + confirm))
    return out


def detect_changes(map_rl: np.ndarray, min_drop: int = 10, min_gap: int = 10, confirm: int = 5) -> list[int]:
    """Start times of detected changes (see detect_changes_detail)."""
    return [s for s, _ in detect_changes_detail(map_rl, min_drop, min_gap, confirm)]


def piecewise_signal(T: int, n_changes: int, mean_jump: float, var_ratio: float, rng: np.random.Generator):
    """Segments with random levels and noise levels; returns signal, true change times, levels, stds."""
    cps = np.sort(rng.choice(np.arange(30, T - 30), n_changes, replace=False)) if n_changes else np.array([], int)
    # enforce spacing
    cps = np.array([c for i, c in enumerate(cps) if i == 0 or c - cps[i - 1] >= 25], dtype=int)
    bounds = np.r_[0, cps, T]
    x = np.zeros(T)
    level, sd = 0.0, 1.0
    levels, sds = np.zeros(T), np.zeros(T)
    for a, b in zip(bounds[:-1], bounds[1:]):
        x[a:b] = level + sd * rng.standard_normal(b - a)
        levels[a:b], sds[a:b] = level, sd
        level = level + rng.choice([-1, 1]) * mean_jump * rng.uniform(0.6, 1.4)
        sd = sd * var_ratio ** rng.choice([-1, 1]) if var_ratio > 1 else sd
        sd = float(np.clip(sd, 0.3, 4.0))
    return x, cps.tolist(), levels, sds
