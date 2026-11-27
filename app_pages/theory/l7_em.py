import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui

ui.theory_header("l7", """
Data often comes from **several sources mixed together**: speech from several talkers, noise of several types,
customers of several kinds, a device that behaves differently in different operating regions. If we knew which
source produced each data point, fitting would be easy: fit each source to its own points. We don't. The source
is a **hidden (latent) variable**. **Expectation–maximisation (EM)** handles this by guessing the hidden labels
softly, fitting, and repeating. You already used EM once: Baum–Welch in Level 4.
""")

ui.section("1. The Gaussian mixture model")
st.markdown(
    r"""
A **mixture of $K$ Gaussians** generates each point in two steps:

1. pick a component $c \in \{1..K\}$ with probabilities $w_1..w_K$ (the **mixing weights**, summing to 1)
2. draw $x$ from that component's Gaussian $\mathcal{N}(\mu_c, \Sigma_c)$

The component $c$ is the hidden variable. Summing it out gives the density of $x$:
"""
)
ui.eq(r"p(x) = \sum_{k=1}^{K} w_k\,\mathcal{N}(x;\,\mu_k, \Sigma_k)", 1)
xs = np.linspace(-5, 6, 400)


def _n(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


fig = go.Figure()
fig.add_scatter(x=xs, y=0.6 * _n(xs, -1.5, 1.0), name="0.6 × N(−1.5, 1²)", line=dict(color=ui.C[1], dash="dash"))
fig.add_scatter(x=xs, y=0.4 * _n(xs, 2.5, 0.7), name="0.4 × N(2.5, 0.7²)", line=dict(color=ui.C[2], dash="dash"))
fig.add_scatter(x=xs, y=0.6 * _n(xs, -1.5, 1.0) + 0.4 * _n(xs, 2.5, 0.7), name="mixture p(x)",
                line=dict(color=ui.C["truth"], width=3))
ui.show(ui.style(fig, "A 1-D mixture of two Gaussians is the sum of weighted bells", "x", "density", height=260))
st.markdown(
    r"""
With enough components a mixture can approximate almost any density. That makes GMMs a workhorse:
speaker models in classic speaker verification, background models in video, noise-type models in speech
enhancement, and the "soft clustering" step in many pipelines.
"""
)

ui.section("2. Why we can't just maximise the likelihood directly")
st.markdown(r"For $N$ points the log-likelihood is")
ui.eq(r"\log p(x_{1:N}) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} w_k\,\mathcal{N}(x_n;\,\mu_k, \Sigma_k)", 2)
st.markdown(
    r"""
The **log of a sum** couples all components together, and setting the derivative to zero has no closed-form
solution. Compare with the case where the labels $c_n$ are known: then the log-likelihood is a sum of separate
terms, one per component, and each $\mu_k$ is just **the average of its own points**. EM turns the hard problem
into a sequence of these easy ones.
"""
)

ui.section("3. The EM algorithm")
st.markdown("Start from any guess of $(w, \\mu, \\Sigma)$ and repeat two steps.")
st.markdown("**E-step: soft labels.** For each point, the probability that each component produced it (Bayes' rule, Level 0):")
ui.eq(r"r_{nk} = P(c_n = k \mid x_n) = \frac{w_k\,\mathcal{N}(x_n;\,\mu_k,\Sigma_k)}{\sum_j w_j\,\mathcal{N}(x_n;\,\mu_j,\Sigma_j)}", 3)
st.markdown("**M-step: fit each component to its soft share of the data** (weighted counts and averages):")
ui.eq(r"N_k = \sum_n r_{nk},\quad w_k = \frac{N_k}{N},\quad \mu_k = \frac{1}{N_k}\sum_n r_{nk}\,x_n,\quad "
      r"\Sigma_k = \frac{1}{N_k}\sum_n r_{nk}\,(x_n-\mu_k)(x_n-\mu_k)^\top", 4)
st.markdown(
    r"""
**Worked example (1-D).** Two components with $\mu_1 = 0$, $\mu_2 = 4$, both $\sigma = 1$, $w = (0.5, 0.5)$. A
point at $x = 1$: $\mathcal{N}(1; 0, 1) = 0.242$ and $\mathcal{N}(1; 4, 1) = 0.0044$, so $r = (0.98, 0.02)$.
A point at $x = 2$ (halfway) gets $r = (0.5, 0.5)$, and it pulls both means equally in the M-step.
"""
)

ui.section("4. Why EM works: a lower bound you push up")
st.markdown(
    r"""
For any distribution $q(c)$ over the hidden labels, Jensen's inequality gives a lower bound on the
log-likelihood of one point. This bound reappears in Level 8 (variational inference) and Level 11 (VAEs), so it's
worth meeting now:
"""
)
ui.eq(r"\log p(x) = \log \sum_c q(c)\,\frac{p(x, c)}{q(c)} \;\ge\; \sum_c q(c)\,\log\frac{p(x,c)}{q(c)} \;=\; "
      r"\mathcal{L}(q, \theta)", 5)
st.markdown(
    r"""
The gap between the two sides is $\mathrm{KL}\big(q(c)\,\|\,p(c \mid x)\big)$, which is zero exactly when $q$ is
the true posterior.

* The **E-step** sets $q = p(c \mid x)$, which closes the gap: the bound now touches $\log p(x)$.
* The **M-step** maximises the bound over the parameters $\theta$. That can only raise $\log p(x)$.

So **every iteration increases the likelihood (or leaves it unchanged)**. It converges, but to a *local*
maximum, which depends on the starting guess.
"""
)

ui.section("5. Practical issues")
st.markdown(
    r"""
* **Initialisation.** Random starting means can put two components on one cluster and leave another cluster
  uncovered. **k-means++** spreads the starting means out (each new one is drawn far from the existing ones),
  and running several restarts and keeping the best likelihood is standard practice.
* **Collapse.** If a component sits on a single point and its variance shrinks to zero, the likelihood goes to
  infinity. A tiny floor on the covariance (a prior, really) prevents it.
* **How many components?** The likelihood always increases with $K$ (more components = more flexibility), so it
  can't choose $K$. The **Bayesian information criterion** penalises the parameter count $P$:
"""
)
ui.eq(r"\mathrm{BIC} = -2\log p(x_{1:N} \mid \hat\theta) + P\,\log N \qquad \text{(lower is better)}", 6)
st.markdown("It approximates the Bayesian **evidence**, which Level 9 meets in exact form.")

ui.section("6. Mixture of experts: mixtures that depend on the input")
st.markdown(
    r"""
Suppose $y$ depends on $x$ differently in different regions: an amplifier that's linear, then compresses, then
clips. A **mixture of experts** has $K$ simple predictors ("experts") and a **gate** that decides, for each
input, how much to trust each expert:
"""
)
ui.eq(r"p(y \mid x) = \sum_{k=1}^{K} g_k(x)\;\mathcal{N}\big(y;\; a_k x + b_k,\; s_k^2\big), \qquad "
      r"g_k(x) = \frac{e^{v_k x + c_k}}{\sum_j e^{v_j x + c_j}}", 7)
st.markdown(
    r"""
The hidden variable is "which expert produced this point". EM works exactly as before:

* **E-step:** $r_{nk} \propto g_k(x_n)\,\mathcal{N}(y_n; a_k x_n + b_k, s_k^2)$
* **M-step:** each expert is a **weighted least-squares** fit with weights $r_{nk}$; the gate is a soft
  classifier trained to predict $r_{nk}$ from $x$.

The gate's job is a posterior over experts. Large modern **mixture-of-experts neural networks** use the same
structure with neural experts and a learned router, so that only a few experts run for each input.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l7", [
    ("Tab **EM step by step**: point colours", r"the responsibilities $r_{nk}$ of eq. (3) (colour = most likely component)"),
    ("Ellipses", r"each component's $\mu_k$ and $\Sigma_k$ after the M-step, eq. (4)"),
    ("Log-likelihood curve", "eq. (2), never decreasing (§4)"),
    ("*Initialisation* and *Random seed*", "local optima (§5)"),
    ("Tab **How many clusters?**", "likelihood vs BIC, eq. (6)"),
    ("Tab **Mixture of experts**", "eq. (7): expert lines, the gate, and EM fitting them"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Treating a local optimum as the answer.** Always use several restarts or a smart initialisation.
* **Choosing $K$ by likelihood.** It always prefers more components. Use BIC, held-out likelihood or the
  evidence.
* **Label switching.** Component "1" in one run may be component "3" in another. Name components by what they
  represent, not by index.
"""
)

ui.section("Check yourself")
ui.check([
    ("A point sits exactly halfway between two equal-weight, equal-variance components. What are its responsibilities?",
     "**(0.5, 0.5)**: both Gaussians give it the same density."),
    ("Why can't EM decrease the log-likelihood?",
     "The E-step makes the lower bound touch log p(x); the M-step raises the bound; log p(x) ≥ bound, so it can only "
     "go up (§4)."),
    ("Going from K = 3 to K = 4 raised the log-likelihood by 2 nats with 600 points in 2-D. Does BIC accept it?",
     "A 2-D full-covariance component adds 6 parameters (1 weight, 2 mean, 3 covariance), costing 6·log(600) ≈ 38 in "
     "BIC, while −2·Δlog L = −4. BIC gets worse: **no**."),
    ("In a mixture of experts, what does the gate output for a given x?",
     "A probability over experts: its estimate of which expert explains points at that x."),
])
