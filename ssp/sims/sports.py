"""Domain demo — skill ratings and live win probability.

Ratings
-------
Every team has a hidden skill theta_i that drifts over time (form, injuries,
transfers). A game is a noisy binary measurement of a skill *difference*:

    P(home wins) = sigmoid(theta_h - theta_a + home_adv)

Elo         theta += K (outcome - expected)            a fixed-gain filter
Kalman      Gaussian belief N(mu_i, v_i) per team: v grows between games
            (process noise) and the logistic measurement is linearised (EKF).
            The effective K is large when a team is uncertain (new, or
            unseen for a while) and small when it is well known.
            This is essentially the idea behind Glicko and TrueSkill.

Live win probability (football)
-------------------------------
Goals arrive as Poisson processes. With score (gh, ga) at minute m, the
remaining goals for each side are Poisson(lambda * (90 - m) / 90), and the
win/draw/loss probabilities follow from the difference. Optionally the
scoring rates are *learned during the match* with a Gamma prior (conjugate
to Poisson), which makes the future-goal counts negative binomial.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# League simulation
# ---------------------------------------------------------------------------
@dataclass
class LeagueConfig:
    n_teams: int = 16
    seasons: int = 3
    drift: float = 0.04            # skill random-walk std per round
    home_adv: float = 0.3
    new_team_skill: float = 1.5    # a strong newcomer replaces team 0 at the start of season 2
    seed: int = 1


def round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Circle method: n-1 rounds of n/2 games, then the return legs."""
    teams = list(range(n))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(teams[i], teams[n - 1 - i]) for i in range(n // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds + [[(a, h) for h, a in r] for r in rounds]


def simulate_league(cfg: LeagueConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    theta = rng.standard_normal(cfg.n_teams)
    games = []                       # (round_idx, home, away, outcome, true_p)
    skills = []
    sched = round_robin(cfg.n_teams)
    r_idx = 0
    new_team_round = len(sched) if cfg.seasons > 1 else None
    for season in range(cfg.seasons):
        for rnd in sched:
            if r_idx == new_team_round:
                theta[0] = cfg.new_team_skill
            theta = theta + cfg.drift * rng.standard_normal(cfg.n_teams)
            skills.append(theta.copy())
            for h, a in rnd:
                p = sigmoid(theta[h] - theta[a] + cfg.home_adv)
                games.append((r_idx, h, a, int(rng.random() < p), p))
            r_idx += 1
    g = np.array(games, dtype=float)
    return {"round": g[:, 0].astype(int), "home": g[:, 1].astype(int), "away": g[:, 2].astype(int),
            "outcome": g[:, 3].astype(int), "true_p": g[:, 4], "skills": np.array(skills),
            "new_team_round": new_team_round, "rounds_per_season": len(sched)}


# ---------------------------------------------------------------------------
# Rating filters (predict before update, so every prediction is honest)
# ---------------------------------------------------------------------------
def elo(lg: dict, n_teams: int, K: float, home_adv: float) -> dict:
    R = np.zeros(n_teams)
    preds = np.zeros(lg["outcome"].size)
    hist = []
    last_round = -1
    for i, (r, h, a, o) in enumerate(zip(lg["round"], lg["home"], lg["away"], lg["outcome"])):
        if r != last_round:
            if r == lg["new_team_round"]:
                R[0] = 0.0                                 # newcomer starts at the average
            hist.append(R.copy())
            last_round = r
        p = sigmoid(R[h] - R[a] + home_adv)
        preds[i] = p
        R[h] += K * (o - p)
        R[a] -= K * (o - p)
    return {"pred": preds, "ratings": np.array(hist)}


def kalman_ratings(lg: dict, n_teams: int, q: float, home_adv: float, v0: float = 1.0) -> dict:
    mu = np.zeros(n_teams)
    v = np.full(n_teams, v0)
    preds = np.zeros(lg["outcome"].size)
    hist_mu, hist_sd, gains_h, gains_a = [], [], [], []
    last_round = -1
    for i, (r, h, a, o) in enumerate(zip(lg["round"], lg["home"], lg["away"], lg["outcome"])):
        if r != last_round:
            if r == lg["new_team_round"]:
                mu[0], v[0] = 0.0, v0                      # newcomer: average, but very uncertain
            v = v + q                                      # predict: skills may have drifted
            hist_mu.append(mu.copy())
            hist_sd.append(np.sqrt(v))
            last_round = r
        p = sigmoid(mu[h] - mu[a] + home_adv)
        preds[i] = p
        g = p * (1 - p)                                    # d sigmoid / d (skill difference)
        S = g * g * (v[h] + v[a]) + g                      # innovation variance (Bernoulli noise ≈ g)
        kh, ka = v[h] * g / S, v[a] * g / S                # per-team gains
        mu[h] += kh * (o - p)
        mu[a] -= ka * (o - p)
        v[h] -= kh * g * v[h]
        v[a] -= ka * g * v[a]
        gains_h.append(kh)
        gains_a.append(ka)
    return {"pred": preds, "ratings": np.array(hist_mu), "sd": np.array(hist_sd),
            "gain_h": np.array(gains_h), "gain_a": np.array(gains_a)}


def log_loss(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def calibration_bins(p: np.ndarray, o: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mean_p, freq, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            mean_p.append(p[m].mean())
            freq.append(o[m].mean())
            count.append(int(m.sum()))
    return np.array(mean_p), np.array(freq), np.array(count)


# ---------------------------------------------------------------------------
# Live win probability for a football match
# ---------------------------------------------------------------------------
MAX_GOALS = 15


def _future_goal_pmf(rate_per_90: float, minutes_left: float, learn: bool, goals_so_far: int,
                     minutes_played: float, prior_minutes: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1)
    if minutes_left <= 0:
        return (k == 0).astype(float)
    if not learn:
        return stats.poisson.pmf(k, rate_per_90 * minutes_left / 90)
    # Gamma(a, b) prior on the per-minute rate, worth `prior_minutes` of evidence
    a = rate_per_90 / 90 * prior_minutes + goals_so_far
    b = prior_minutes + minutes_played
    # rate ~ Gamma(a, b); goals in the next tau minutes ~ NegBin(a, b / (b + tau))
    return stats.nbinom.pmf(k, a, b / (b + minutes_left))


def win_probabilities(gh: int, ga: int, minute: float, lam_h: float, lam_a: float, learn: bool = False,
                      prior_minutes: float = 270.0) -> tuple[float, float, float]:
    """P(home win), P(draw), P(away win) given the score at `minute`."""
    left = max(90.0 - minute, 0.0)
    ph = _future_goal_pmf(lam_h, left, learn, gh, minute, prior_minutes)
    pa = _future_goal_pmf(lam_a, left, learn, ga, minute, prior_minutes)
    joint = np.outer(ph, pa)                       # independent future goals
    diff = (gh + np.arange(MAX_GOALS + 1))[:, None] - (ga + np.arange(MAX_GOALS + 1))[None, :]
    tot = joint.sum()
    return (float(joint[diff > 0].sum() / tot), float(joint[diff == 0].sum() / tot),
            float(joint[diff < 0].sum() / tot))


def simulate_match(lam_h: float, lam_a: float, rng: np.random.Generator, red_card_min: float | None = None,
                   red_card_side: str = "away"):
    """Goal times (minutes) per side. A red card scales the rates from that minute on
    (x0.65 for the side down to ten men, x1.3 for the opponent). Poisson processes are
    memoryless, so each constant-rate stretch can be simulated on its own."""
    goals = {"home": [], "away": []}
    for side, lam in (("home", lam_h), ("away", lam_a)):
        pieces = [(0.0, 90.0, lam / 90)]
        if red_card_min is not None:
            f = 0.65 if side == red_card_side else 1.3
            pieces = [(0.0, red_card_min, lam / 90), (red_card_min, 90.0, f * lam / 90)]
        for t0, t1, r in pieces:
            k = rng.poisson(r * (t1 - t0))
            goals[side].extend(np.sort(rng.uniform(t0, t1, k)).tolist())
        goals[side].sort()
    return goals


def live_curve(goals: dict, lam_h: float, lam_a: float, learn: bool, prior_minutes: float,
               red_card_min: float | None = None, red_card_side: str = "away"):
    minutes = np.arange(0, 90.5, 0.5)
    out = np.zeros((minutes.size, 3))
    for i, m in enumerate(minutes):
        gh = sum(g <= m for g in goals["home"])
        ga = sum(g <= m for g in goals["away"])
        lh, la = lam_h, lam_a
        if red_card_min is not None and m >= red_card_min:     # the model is told about the red card
            lh, la = (lam_h * 1.3, lam_a * 0.65) if red_card_side == "away" else (lam_h * 0.65, lam_a * 1.3)
        out[i] = win_probabilities(gh, ga, m, lh, la, learn, prior_minutes)
    return minutes, out


def calibration_experiment(lam_h: float, lam_a: float, minute: float, n: int, rng: np.random.Generator,
                           learn: bool = False, true_rates_spread: float = 0.0):
    """Predicted P(home win) at `minute` vs actual results over n simulated matches.

    true_rates_spread > 0: each match's true rates differ from the pre-match
    estimate (log-normal), which is the realistic case where learning helps.
    """
    preds, outcomes = np.zeros(n), np.zeros(n)
    for i in range(n):
        f = np.exp(true_rates_spread * rng.standard_normal(2))
        g = simulate_match(lam_h * f[0], lam_a * f[1], rng)
        gh = sum(t <= minute for t in g["home"])
        ga = sum(t <= minute for t in g["away"])
        preds[i] = win_probabilities(gh, ga, minute, lam_h, lam_a, learn)[0]
        outcomes[i] = len(g["home"]) > len(g["away"])
    return preds, outcomes
