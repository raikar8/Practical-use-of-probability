import streamlit as st

from ssp import ui

ui.theory_header("sports", """
**Prerequisites:** Level 1 (MLE, priors), Level 3 (Kalman filter). The live model uses Level 0's conjugate priors.

A match result is a **noisy measurement of a skill difference**. Rating systems are filters that estimate hidden,
drifting skills from those measurements. Live win probabilities are predictions that update as events happen. Both
are estimation problems you can check against reality.
""")

ui.section("1. A measurement model for a game")
st.markdown(r"""Give each team a hidden skill $\theta_i$. In the **Bradley–Terry / logistic** model, the probability that home team $h$ beats away team $a$ is""")
ui.eq(r"p = P(\text{home wins}) = \sigma(\theta_h - \theta_a + \eta), \qquad \sigma(u) = \frac{1}{1+e^{-u}}", 1)
st.markdown(
    r"""
where $\eta$ is home advantage. A skill gap of 1 gives the stronger team about 73% (before home advantage).
Skills **drift** over a season (form, injuries, transfers): $\theta_i \leftarrow \theta_i + \text{noise}$ every
round. The result $o \in \{0, 1\}$ is a very noisy, one-bit measurement of $\theta_h - \theta_a$.
"""
)

ui.section("2. Elo is a gradient step on the log-likelihood")
st.markdown(r"""The log-likelihood of one result under (1) is $\ell = o\log p + (1-o)\log(1-p)$. Its derivative with respect to the home skill is simply $o - p$. A gradient-ascent step with step size $K$ gives exactly the **Elo update**:""")
ui.eq(r"\theta_h \leftarrow \theta_h + K\,(o - p), \qquad \theta_a \leftarrow \theta_a - K\,(o - p)", 2)
st.markdown(
    r"""
"Actual result minus expected result, times K." It's LMS from Level 2: a **fixed-gain** filter. The dilemma is
the same too: a large K reacts fast but makes every rating noisy, and a small K is stable but takes a season to
believe a newcomer is strong.
"""
)

ui.section("3. A Kalman (Glicko-style) rating: the gain comes from uncertainty")
st.markdown(
    r"""
Give each team a **Gaussian belief** $\mathcal N(\mu_i, v_i)$ instead of a single number.

* **Predict** (a round passes): skills may have drifted, so $v_i \leftarrow v_i + q$.
* **Update** after a game: linearise (1) around the current means (an EKF). The slope of $\sigma$ is
  $g = p(1-p)$, and the result's own randomness has variance $p(1-p)$. The Kalman gain for the home team is
"""
)
ui.eq(r"K_h = \frac{v_h\,g}{g^2\,(v_h + v_a) + g}, \qquad \mu_h \leftarrow \mu_h + K_h(o-p), \qquad "
      r"v_h \leftarrow v_h - K_h\,g\,v_h", 3)
st.markdown(
    r"""
A team with large uncertainty $v_h$ (new to the league, or unseen for a while) gets a **large gain**; a
well-known team gets a small one. Glicko (with its "rating deviation") and TrueSkill (Xbox matchmaking) are built
on this idea.

**How we judge ratings:** by their **predictions**, made *before* each game.
**Log-loss** $-[o\log p + (1-o)\log(1-p)]$ (lower is better; a coin flip scores $\ln 2 \approx 0.693$), the
**Brier score** $(p-o)^2$, and **calibration**: of all games given 70%, did about 70% happen?
"""
)

ui.section("4. Live win probability in football")
st.markdown(r"""Goals arrive at random, like a **Poisson process**: the home team scores at rate $\lambda_H$ per 90 minutes. At minute $m$ with score $(g_H, g_A)$, the goals still to come in the remaining $\tau = 90 - m$ minutes are""")
ui.eq(r"X \sim \text{Poisson}\big(\lambda_H\,\tau/90\big),\quad Y \sim \text{Poisson}\big(\lambda_A\,\tau/90\big),"
      r"\qquad P(\text{home win}) = P(g_H + X > g_A + Y)", 4)
st.markdown(
    r"""
(and similarly for the draw and the away win). Every scoreless minute favours the team that's ahead, because
there's less time to come back, so the curve **drifts** between goals and **jumps** at goals.

**Learning the rates during the match.** Pre-match rates are estimates. Put a **Gamma prior** on each rate, worth
$m_0$ minutes of evidence. The Gamma is conjugate to the Poisson (like Beta–Bernoulli in Level 0), so after $k$
goals in $m$ minutes
"""
)
ui.eq(r"\lambda \sim \text{Gamma}\big(a_0 + k,\; m_0 + m\big) \quad\Rightarrow\quad "
      r"\text{goals to come} \sim \text{Negative Binomial}", 5)
st.markdown(
    r"""
A weak prior (small $m_0$) means one early goal convinces the model that the team is unstoppable: the small-sample
overreaction from Level 1. **Red cards** in this app scale the rates (×0.65 for the team down to ten, ×1.3 for the
opponent). That's a modelling assumption you could estimate from data.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("sports", [
    ("Ratings tab: *Skill drift per round*", r"the true random walk of $\theta_i$"),
    ("*Newcomer's true skill*", r"a team replaced at the start of season 2, with a hidden skill both systems must discover"),
    ("*Elo K*", r"$K$ in (2)"),
    ("*Kalman q*", r"$q$ in the predict step of section 3"),
    ("Rolling log-loss plot", r"prediction quality over time (lower is better)"),
    ("Team plot with band", r"true skill vs Elo vs Kalman $\mu \pm 2\sqrt v$"),
    ("Gain plot", r"Kalman $K_h$ from (3) vs Elo's constant $K$"),
    ("Live tab: *expected goals*, *Learn the scoring rates*, *Prior strength*", r"$\lambda_H, \lambda_A$ in (4); the Gamma prior of (5)"),
    ("Calibration check", r"many simulated matches: predicted vs observed frequency"),
])

ui.section("Check yourself")
ui.check([
    ("Two equal teams, home advantage η = 0.3. What is P(home wins)?", "σ(0.3) ≈ **0.574**."),
    ("Elo with K = 0.2: the home team (p = 0.574) wins. How much does its rating rise?",
     "0.2 · (1 − 0.574) ≈ **+0.085**."),
    ("Why does the Kalman rating learn the newcomer faster?",
     "Its uncertainty v is reset to a large value, so its gain (3) is large for the first games, then shrinks as "
     "evidence accumulates."),
    ("At 0–0 with 10 minutes left, why is P(draw) high even between unequal teams?",
     "Expected remaining goals are small (λ·10/90). The most likely outcome is no more goals at all."),
])
