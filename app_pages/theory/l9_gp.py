import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import gp

ui.theory_header("l9", """
Fitting a curve to data usually produces **one** curve. But with ten noisy points, many curves are plausible,
and how much we trust the fit should depend on **where** we're asking: near the data or far from it. This level
puts probability distributions **over functions**. That gives honest error bars, a principled way to choose model
complexity (the **evidence**), and a smart way to optimise things that are expensive to measure (**Bayesian
optimisation**).
""")

ui.section("1. Bayesian linear regression")
st.markdown(
    r"""
Model $y = \mathbf{w}^\top \boldsymbol{\phi}(x) + \text{noise}$, where $\boldsymbol\phi(x)$ are chosen **features**
(for a polynomial: $1, x, x^2, \dots$). Instead of one best $\mathbf w$, put a Gaussian prior on it and use
Gaussian noise with precision $\beta$ (variance $1/\beta$). Everything stays Gaussian (Level 0's conditioning
rule again), so the posterior is exact:
"""
)
ui.eq(r"\mathbf w \sim \mathcal N(0, \alpha^{-1} I) \;\;\Rightarrow\;\; \mathbf w \mid \mathbf y \sim "
      r"\mathcal N(\mathbf m, S),\qquad S^{-1} = \alpha I + \beta\,\Phi^\top\Phi,\qquad \mathbf m = \beta\,S\,\Phi^\top \mathbf y", 1)
ui.eq(r"p(y_* \mid x_*, \mathbf y) = \mathcal N\big(\mathbf m^\top\boldsymbol\phi_*,\;\; "
      r"\underbrace{1/\beta}_{\text{noise}} + \underbrace{\boldsymbol\phi_*^\top S\,\boldsymbol\phi_*}_{\text{uncertainty about }\mathbf w}\big)", 2)
st.markdown(
    r"""
* $\mathbf m$ is the ridge-regression solution (the prior acts as a penalty $\alpha\lVert\mathbf w\rVert^2$, which
  is MAP from Level 1).
* The predictive variance (2) has two parts: noise you can never remove, and **uncertainty about the curve**,
  which is small near the data and grows away from it.
"""
)

ui.section("2. The evidence: Occam's razor, automatically")
st.markdown(r"How do we choose the polynomial degree, or any model $\mathcal M$? Compute how probable the data is under the model, **averaging over all its parameter values** rather than picking the best:")
ui.eq(r"p(\mathbf y \mid \mathcal M) = \int p(\mathbf y \mid \mathbf w, \mathcal M)\;p(\mathbf w \mid \mathcal M)\;d\mathbf w", 3)
st.markdown(
    r"""
This **marginal likelihood** (or evidence) has a closed form for model (1). Why does it prefer simple models?
A probability distribution over datasets must sum to 1. A simple model can only produce a few kinds of dataset,
so it gives each of them high probability. A complex model can produce almost any dataset, so it spreads its
probability thinly. If the simple model **can** explain your data, it wins. If it can't, a more complex one
does. The training error, by contrast, always improves with more parameters.
"""
)
rng = np.random.default_rng(3)
x = rng.uniform(-3, 3, 20)
y = np.sin(1.3 * x) + 0.3 * x + 0.25 * rng.standard_normal(20)
degs = np.arange(0, 11)
ev = [gp.blr(gp.poly_features(x, d), y, 1.0, 1 / 0.25**2)["log_evidence"] for d in degs]
tr = []
for d in degs:
    P = gp.poly_features(x, d)
    w = np.linalg.lstsq(P, y, rcond=None)[0]
    tr.append(np.sqrt(np.mean((P @ w - y) ** 2)))
fig = go.Figure()
fig.add_scatter(x=degs, y=ev, name="log evidence (higher = better)", mode="lines+markers", line=dict(color=ui.C[1]))
fig.add_scatter(x=degs, y=-40 * np.array(tr), name="−40 × training RMS error (always improves)", mode="lines+markers",
                line=dict(color=ui.C[2], dash="dash"))
ui.show(ui.style(fig, "20 noisy points: the evidence picks a moderate degree; training error keeps improving",
                 "polynomial degree", None, height=280))

ui.section("3. Gaussian processes: a prior over functions")
st.markdown(
    r"""
Now skip the features. A **Gaussian process** says: the values of the unknown function at **any** set of inputs
are jointly Gaussian, with covariance given by a **kernel** $k(x, x')$ that measures how similar the outputs at
$x$ and $x'$ should be:
"""
)
ui.eq(r"f \sim \mathcal{GP}\big(0,\; k(x, x')\big), \qquad k_{\text{RBF}}(x, x') = \sigma_f^2\,\exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)", 4)
st.markdown(
    r"""
Nearby inputs ($|x - x'| \ll \ell$) get strongly correlated outputs: the function is smooth on the scale
$\ell$. With noisy observations $\mathbf y$ at inputs $X$, conditioning a Gaussian (Level 0) gives the posterior
at new inputs $X_*$:
"""
)
ui.eq(r"\boldsymbol\mu_* = K_*^\top (K + \sigma_n^2 I)^{-1}\mathbf y, \qquad "
      r"\Sigma_* = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*", 5)
st.markdown(
    r"""
where $K = k(X, X)$, $K_* = k(X, X_*)$, $K_{**} = k(X_*, X_*)$. A GP is Bayesian linear regression with
**infinitely many features**, handled through the kernel instead of explicitly.
"""
)
xg = np.linspace(-4, 4, 200)
fig = go.Figure()
for i, (kn, ell, c) in enumerate([("RBF (very smooth)", 1.0, ui.C[1]), ("Matérn 1/2 (random walk-like)", 1.0, ui.C[2]),
                                  ("RBF (very smooth)", 0.3, ui.C[3])]):
    s = gp.gp_prior_samples(xg, kn, ell, 1.0, 2, np.random.default_rng(i))
    for j in range(2):
        fig.add_scatter(x=xg, y=s[j], line=dict(color=c, width=1.5), showlegend=j == 0,
                        name=f"{kn.split(' (')[0]}, ℓ = {ell}")
ui.show(ui.style(fig, "Random functions drawn from three GP priors (two each)", "x", "f(x)", height=280))

ui.section("4. Learning the kernel: the marginal likelihood again")
st.markdown(r"The hyperparameters $\ell, \sigma_f, \sigma_n$ are chosen by maximising the GP's evidence:")
ui.eq(r"\log p(\mathbf y \mid X, \ell, \sigma_f, \sigma_n) = \underbrace{-\tfrac12 \mathbf y^\top (K + \sigma_n^2 I)^{-1}\mathbf y}_{\text{data fit}} "
      r"\;\underbrace{-\;\tfrac12\log\lvert K + \sigma_n^2 I\rvert}_{\text{complexity penalty}} \;-\; \tfrac{N}{2}\log 2\pi", 6)
st.markdown(
    r"""
A tiny $\ell$ fits every wiggle (good data fit) but can explain any dataset (large determinant penalty); a huge
$\ell$ is simple but fits badly. The maximum balances them, as in §2.

**Cost.** Equation (5) solves an $N \times N$ system: $O(N^3)$. For time series there's a shortcut: Matérn kernels
are exactly **state-space models**. Matérn-1/2 is the Ornstein–Uhlenbeck process (a 1-state Kalman model), and
Matérn-3/2 is a 2-state one. So GP regression over time can be done by a Kalman filter + RTS smoother (Level 3)
in $O(N)$.
"""
)

ui.section("5. Bayesian optimisation: where to measure next")
st.markdown(
    r"""
Some functions are **expensive** to evaluate: one listening test of a tuning setting, one training run of a
neural network with given hyperparameters, one chemistry experiment. We want the best setting with as few
evaluations as possible. Bayesian optimisation repeats:

1. fit a GP to all evaluations so far
2. choose the next point by maximising an **acquisition function** that balances "the mean looks good here"
   (**exploitation**) and "we know little here" (**exploration**)
3. evaluate there and add the result
"""
)
ui.eq(r"\text{UCB}(x) = \mu(x) + \kappa\,\sigma(x), \qquad "
      r"\text{EI}(x) = \mathbb E\big[\max(f(x) - f^{+}, 0)\big] = (\mu - f^{+})\,\Phi(z) + \sigma\,\varphi(z),\;\; z = \frac{\mu - f^{+}}{\sigma}", 7)
st.markdown(
    r"""
$f^{+}$ is the best value found so far; $\Phi$ and $\varphi$ are the standard normal CDF and density. Expected
improvement (EI) is large where the mean beats the best so far **or** where the uncertainty is big enough that it
might.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l9", [
    ("Tab **Bayesian regression**: band around the curve", r"the predictive std of eq. (2)"),
    ("*Prior precision α*", r"$\alpha$ in eq. (1): larger = stronger pull of weights toward 0"),
    ("Evidence plot vs degree", "eq. (3), with the training error for comparison"),
    ("Tab **Gaussian processes**: *Kernel*, *ℓ*, *σ_f*, *σ_n*", r"eq. (4); the posterior is eq. (5)"),
    ("*Fit hyperparameters by maximising evidence*", "eq. (6)"),
    ("Tab **Bayesian optimisation**: bottom curve, star", "the acquisition function (7) and its maximum = next test"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Extrapolating with a GP.** Far from the data the mean returns to the prior mean (zero here). The error bar
  says so honestly, but the mean alone looks like a confident prediction.
* **Overconfident length-scales.** With very few points, a long $\ell$ makes the GP sure that unexplored regions
  are boring, and Bayesian optimisation stops exploring them.
* **Model choice by training error.** It always prefers the most complex model. Use the evidence or held-out data.
"""
)

ui.section("Check yourself")
ui.check([
    ("Why is the predictive band of Bayesian linear regression wider away from the data?",
     "The term φ*ᵀSφ* grows where the features are large and poorly constrained by the data; the noise term is "
     "the same everywhere."),
    ("A GP with RBF kernel and ℓ = 0.1 is fitted to points spaced 1 apart. What does the mean look like between points?",
     "It drops back toward the prior mean (0) between the points: with such a short length-scale, the points "
     "don't inform each other's neighbourhoods."),
    ("Why can the evidence prefer a degree-3 polynomial over degree 9 even though degree 9 fits the data better?",
     "Degree 9 spreads its prior probability over far more possible datasets, so it assigns less probability to "
     "the one we saw (Occam's razor, §2)."),
    ("In UCB, what happens with κ = 0?",
     "Pure exploitation: always test where the mean is highest. It can get stuck on the first decent hill."),
])
