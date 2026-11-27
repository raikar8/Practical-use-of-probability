import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist

from ssp import ui

ui.theory_header("bandits", """
A website has three versions of a headline (or a shop has three layouts, or a game has three tutorial designs).
Which one gets the most clicks? The classic answer is an **A/B test**: split traffic evenly for a while, then
pick the winner. But every visitor sent to a worse version during the test is a lost click. **Multi-armed
bandits** learn and earn at the same time: they shift traffic toward the better versions **as the evidence comes
in**. Thompson sampling, the star of this demo, is just Level 0's Beta–Bernoulli posterior used to make
decisions.
""")
st.markdown("**Prerequisites:** Level 0 (Beta–Bernoulli updating) and Level 1 (estimation with few samples).")

ui.section("1. The problem: explore or exploit?")
st.markdown(
    r"""
There are $K$ variants ("arms", after slot machines). Variant $k$ has an unknown click-through rate $p_k$. Each
visitor is shown one variant and clicks ($r = 1$) or not ($r = 0$). We want as many clicks as possible over $T$
visitors. The yardstick is **regret**: clicks lost compared with always showing the best variant.
"""
)
ui.eq(r"\text{Regret}(T) = \sum_{t=1}^{T} \big(p_{\max} - p_{a_t}\big)", 1)
st.markdown(
    r"""
* **Exploit** too early, and you may lock in a variant that only looked best by luck.
* **Explore** too long, and you keep paying for variants you already know are worse.

Real click-rate differences are small (4.0% vs 5.0%), so luck matters a lot. With 500 visitors per variant, the
standard error of each rate is about $\sqrt{0.045 \times 0.955 / 500} \approx 0.9$ percentage points, as big as
the differences we're trying to detect.
"""
)

ui.section("2. What we believe about each variant: Beta posteriors")
st.markdown(r"Exactly as in Level 0: start each variant with a $\text{Beta}(1, 1)$ (uniform) prior. After $s_k$ clicks and $f_k$ non-clicks:")
ui.eq(r"p_k \mid \text{data} \sim \text{Beta}(1 + s_k,\; 1 + f_k)", 2)
x = np.linspace(0, 0.12, 400)
fig = go.Figure()
for (s, f, nm, c) in [(20, 480, "A: 20 clicks / 500", ui.C[1]), (25, 475, "B: 25 / 500", ui.C[2]),
                      (3, 47, "C: 3 / 50", ui.C[3])]:
    fig.add_scatter(x=x * 100, y=beta_dist.pdf(x, 1 + s, 1 + f) / 100, name=nm, line=dict(color=c))
ui.show(ui.style(fig, "Posterior beliefs about three click rates", "click-through rate (%)", "density", height=260))
st.markdown(
    r"""
B looks better than A, but their curves overlap a lot. C has hardly been tried, so its curve is wide: it could
be the best of all. A good policy should keep giving C some traffic **because** it's uncertain.
"""
)

ui.section("3. Four policies")
st.markdown(
    r"""
| Policy | Rule | Weakness |
|---|---|---|
| **A/B test, then commit** | equal split for $n$ visitors per variant, then show the best-looking one forever | pays full price during the test; commits to the wrong one surprisingly often |
| **ε-greedy** | show the best-looking variant, but a random one with probability ε | explores forever at the same rate, even when the answer is clear |
| **UCB1** | show the variant with the highest optimistic estimate $\hat p_k + \sqrt{2 \ln t / n_k}$ | the bonus is designed for the worst case, so it's slow on small differences |
| **Thompson sampling** | draw one plausible rate from each posterior (2), show the variant whose draw is highest | needs a posterior (here it's free) |
"""
)
st.markdown(
    r"""
**Why Thompson sampling works.** It shows each variant with probability equal to the **probability that it is
the best**, given the data so far:
"""
)
ui.eq(r"P(\text{show } k) = P\big(p_k = \max_j p_j \mid \text{data}\big)", 3)
st.markdown(
    r"""
Clearly worse variants get almost no traffic; uncertain ones still get some. As the posteriors narrow, traffic
concentrates on the winner automatically, with no test length to choose. Its regret grows like $\log T$, the best
possible rate.
"""
)

ui.section("4. The A/B test's hidden cost")
st.markdown(
    r"""
A fixed-length test has two costs. During the test, two-thirds of the traffic (with three variants) goes to
worse versions. After it, there's a real chance of having committed to a worse one: in this demo's setting
(rates 4.0%, 5.0%, 4.5%), a test with 500 visitors per variant picks the wrong winner about **40%** of the time,
and 2,000 per variant still about **25%**. **Peeking** (stopping as soon as the difference looks significant)
makes it worse, because random fluctuations cross the threshold early. Bandits are designed to be checked
continuously.

A/B tests still have their place: when you need an unbiased estimate of the difference itself, or when the
decision is made once and changing it later is costly.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("bandits", [
    ("*True click rate* sliders", r"the unknown $p_k$"),
    ("Bell curves moving as you press ▶", r"the Beta posteriors of eq. (2)"),
    ("Traffic share plot", "how often each policy shows each variant"),
    ("Regret curves", "eq. (1): lower is better"),
    ("*P(best)* metric", "eq. (3): Thompson sampling's traffic split"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Assuming rates don't change.** Click rates drift (novelty effects, seasons). Use a discounted posterior or
  a sliding window, much as a Kalman filter assumes a drifting state.
* **Ignoring delayed feedback.** Conversions that arrive days later make the posterior lag behind the traffic.
* **Optimising the wrong reward.** Clicks are easy to measure; long-term satisfaction isn't. Bandits optimise
  exactly what you reward.
"""
)

ui.section("Check yourself")
ui.check([
    ("Variant C has 3 clicks out of 50 (6%), B has 25 out of 500 (5%). Which should a greedy policy show? Thompson sampling?",
     "Greedy shows **C** (6% > 5%). Thompson samples both posteriors: C's is wide, so it wins often but not always, "
     "and both keep getting traffic until the evidence is clearer."),
    ("Why does ε-greedy's regret keep growing linearly?",
     "It spends a fixed fraction ε of traffic on random variants forever, even after the best is obvious."),
    ("What does Thompson sampling do if two variants have exactly the same rate?",
     "It keeps splitting traffic between them, which costs nothing: neither is worse."),
])
