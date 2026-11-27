"""Multi-armed bandits with Bernoulli rewards: A/B-then-commit, ε-greedy, UCB1 and Thompson sampling."""
from __future__ import annotations

import numpy as np

POLICIES = ["A/B test, then commit", "ε-greedy", "UCB1", "Thompson sampling"]


def run(policy: str, p: np.ndarray, T: int, rng: np.random.Generator, eps: float = 0.1, ab_n: int = 500,
        prior: tuple[float, float] = (1.0, 1.0), record_every: int = 0) -> dict:
    """Show one of K variants to each of T visitors. Reward 1 = click/conversion with probability p[arm].

    Returns which arm each visitor got, the cumulative regret (clicks lost vs always showing the best arm),
    and (for Thompson) snapshots of the Beta posteriors.
    """
    p = np.asarray(p, float)
    K = len(p)
    succ, fail = np.zeros(K), np.zeros(K)
    arms = np.zeros(T, int)
    rewards = np.zeros(T)
    snaps = []
    committed = None
    a0, b0 = prior
    for t in range(T):
        n = succ + fail
        if policy == "A/B test, then commit":
            if t < ab_n * K:
                a = t % K                                     # round-robin during the test
            else:
                if committed is None:
                    committed = int(np.argmax(succ / np.maximum(n, 1)))
                a = committed
        elif policy == "ε-greedy":
            if rng.random() < eps or t < K:
                a = int(rng.integers(K))
            else:
                a = int(np.argmax(succ / np.maximum(n, 1)))
        elif policy == "UCB1":
            if t < K:
                a = t
            else:
                a = int(np.argmax(succ / n + np.sqrt(2 * np.log(t + 1) / n)))
        elif policy == "Thompson sampling":
            a = int(np.argmax(rng.beta(a0 + succ, b0 + fail)))   # sample a plausible CTR per arm, play the best
        else:
            raise ValueError(policy)
        r = float(rng.random() < p[a])
        succ[a] += r
        fail[a] += 1 - r
        arms[t], rewards[t] = a, r
        if record_every and (t % record_every == 0 or t == T - 1):
            snaps.append((t + 1, a0 + succ.copy(), b0 + fail.copy()))
    regret = np.cumsum(p.max() - p[arms])
    return {"arms": arms, "rewards": rewards, "regret": regret, "succ": succ, "fail": fail, "snaps": snaps,
            "committed": committed}


def prob_best(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 20000) -> np.ndarray:
    """P(arm k has the highest true rate | data), by sampling the Beta posteriors."""
    draws = rng.beta(a[None, :], b[None, :], size=(n, len(a)))
    return np.bincount(draws.argmax(1), minlength=len(a)) / n


def mean_regret(policy: str, p, T: int, runs: int, seed: int = 0, **kw) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.mean([run(policy, p, T, rng, **kw)["regret"] for _ in range(runs)], 0)
