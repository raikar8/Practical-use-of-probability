import streamlit as st

from ssp import ui

ui.theory_header("markets", """
**Prerequisites:** Level 1 (MLE), Level 3 (Kalman filter), Level 4 (HMM).

Financial prices are a good testing ground for estimation: lots of data, lots of noise, and hidden quantities
that genuinely change: the market's *mood*, its *risk level*, and how two assets *move together*. **This is for
learning estimation, not investment advice.** None of these models predicts prices, and that is part of the
lesson.
""")

ui.section("1. Returns, and three facts about them")
st.markdown(r"""We work with **daily log-returns** in percent rather than prices:""")
ui.eq(r"r_t = 100\,\ln\frac{P_t}{P_{t-1}}", 1)
st.markdown(
    r"""
Returns from real markets show three robust patterns:

1. **Direction is nearly unpredictable.** Today's return is almost uncorrelated with yesterday's.
2. **Size is predictable** (*volatility clustering*). Big moves follow big moves and calm follows calm, so
   $r_t^2$ *is* correlated with $r_{t-1}^2$.
3. **Regimes.** Long calm periods alternate with turbulent ones, with different average returns and volatility.

Each of the three tools below targets one hidden quantity.
"""
)

ui.section("2. Regimes with a Gaussian HMM")
st.markdown(r"""Hidden state $s_t \in \{\text{calm}, \text{turbulent}, \dots\}$ follows a Markov chain; each regime emits returns from its own Gaussian:""")
ui.eq(r"r_t \mid s_t = k \;\sim\; \mathcal N(\mu_k, \sigma_k^2)", 2)
st.markdown(
    r"""
Baum–Welch (Level 4) learns $\mu_k$, $\sigma_k$ and the transition matrix from returns alone; we then name the
states by their volatility. The expected regime length is $1/(1-A_{kk})$ days.

Two posteriors answer two different questions:

* **Filtered** $P(s_t \mid r_{1:t})$: what you could have known **on day t**.
* **Smoothed** $P(s_t \mid r_{1:T})$: what is known **with hindsight**. It switches *earlier* and more cleanly.

A backtest that acts on smoothed (or Viterbi) regimes uses information from the future: **look-ahead bias**,
the classic way to fool yourself in quantitative finance. Volatility regimes are easy to detect because variance
shows up in $\lvert r_t\rvert$ immediately; differences in mean return are tiny next to daily noise.
"""
)

ui.section("3. Volatility with GARCH(1,1), fitted by maximum likelihood")
st.markdown(r"""To capture volatility clustering, let today's variance depend on yesterday's shock and yesterday's variance:""")
ui.eq(r"r_t = \mu + \varepsilon_t,\quad \varepsilon_t = \sigma_t z_t,\; z_t\sim\mathcal N(0,1), \qquad "
      r"\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2", 3)
st.markdown(
    r"""
* $\alpha$: how strongly a shock raises tomorrow's variance (*reaction*).
* $\beta$: how much of yesterday's variance carries over (*memory*).
* $\alpha + \beta < 1$: **persistence**. The variance reverts to the long-run level $\omega/(1-\alpha-\beta)$,
  with half-life $\ln 0.5 / \ln(\alpha+\beta)$ days.

**Fitting is pure Level 1.** Given parameters, (3) gives $\sigma_t^2$ for every day, and each return is Gaussian
with that variance, so the log-likelihood is
"""
)
ui.eq(r"\log L(\mu,\omega,\alpha,\beta) = -\frac12\sum_t\Big[\log(2\pi\sigma_t^2) + \frac{(r_t-\mu)^2}{\sigma_t^2}\Big]", 4)
st.markdown(
    r"""
We maximise (4) numerically. **Forecasts** $h$ days ahead decay geometrically toward the long-run variance:
$\sigma^2_{T+h} = \bar\sigma^2 + (\alpha+\beta)^{h-1}(\sigma^2_{T+1} - \bar\sigma^2)$.
"""
)

ui.section("4. A drifting relationship: the Kalman hedge ratio")
st.markdown(r"""Two related assets (two companies in the same industry) roughly satisfy $y_t \approx \beta_t x_t + \alpha_t$. The **hedge ratio** $\beta_t$ (units of $x$ that offset one unit of $y$) drifts over time. Make the coefficients the **hidden state** of a Kalman filter:""")
ui.eq(r"\boldsymbol\theta_t = \begin{bmatrix}\beta_t\\ \alpha_t\end{bmatrix} = \boldsymbol\theta_{t-1} + \mathbf w_t,"
      r"\;\; \mathbf w_t \sim \mathcal N\!\Big(\mathbf 0, \tfrac{\delta}{1-\delta}I\Big), \qquad "
      r"y_t = \underbrace{[\,x_t\;\;1\,]}_{H_t}\,\boldsymbol\theta_t + v_t", 5)
st.markdown(
    r"""
This is the Level 3 random-walk filter with an observation matrix $H_t$ that **changes every day**. RLS from
Level 2 is the special case without process noise.

* **δ** says how fast β may drift (it plays the role of $Q$).
* The **innovation** $y_t - H_t\hat{\boldsymbol\theta}^-_t$ is the **spread**, and dividing by $\sqrt{S_t}$
  gives a **z-score**. $\lvert z\rvert > 2$ marks an unusual spread, and a structural break shows up as one huge z.
* **Choosing δ and R by maximum likelihood.** The innovations of a correct filter are independent Gaussians
  with variance $S_t$, so $\log L = -\tfrac12\sum_t[\log(2\pi S_t) + \nu_t^2/S_t]$. Maximising it over (δ, R) is
  Level 1 applied to a filter's own outputs, and makes the z-scores honest.

The common heuristic it replaces is **rolling OLS**: a regression over the last $W$ days. A short window is noisy,
a long one is slow; the filter makes the same trade-off with one interpretable knob, plus an error bar.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("markets", [
    ("Regimes tab: *True number of regimes*, *Regime persistence*", r"the simulated chain; persistence = $A_{kk}$"),
    ("*Regimes in the fitted model*", r"$K$ used by Baum–Welch for (2)"),
    ("Orange vs violet probability lines", r"filtered vs smoothed $P(\text{high vol})$: the look-ahead bias lesson"),
    ("Volatility tab: *True α*, *True β*", r"the parameters in (3) used to simulate; the table shows the MLE (4)"),
    ("Autocorrelation bars", r"fact 1 (returns) vs fact 2 (squared returns)"),
    ("Hedge tab: *Sudden change in β*, *Spread noise*", r"the hidden $\beta_t$ jump, and the size of $v_t$ in (5)"),
    ("*Fit δ and R by maximum likelihood* / *Kalman δ*", r"MLE of the noise levels, or a hand-set δ"),
    ("*Rolling-OLS window*", r"the heuristic baseline's $W$"),
])

ui.section("Check yourself")
ui.check([
    ("α = 0.08, β = 0.90. What is the volatility half-life?", "ln 0.5 / ln 0.98 ≈ **34 days**."),
    ("A regime model says P(turbulent) = 0.9 *smoothed* on 3 March. Could a trader have acted on it on 3 March?",
     "No. The smoothed value uses returns after 3 March. Only the filtered probability was available that day."),
    ("Why are z-scores from a hand-tuned hedge filter often too small or too large?",
     "If δ or R is wrong, the claimed innovation variance S is wrong. MLE chooses them so the innovations really "
     "have variance S."),
])
