import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui

ui.theory_header("l0", """
Every algorithm in this app answers one question: **given noisy data, what should I now believe?** Probability is
the language for "believe, and how strongly". This page builds the four tools the rest of the app uses:
**random variables**, **Bayes' rule**, **Markov chains** and the **Gaussian**.
""")

# ---------------------------------------------------------------------------
ui.section("1. Random variables: numbers that come with uncertainty")
st.markdown(
    r"""
A **random variable** is a quantity whose value we don't know exactly. We describe it by how likely each value is.

* **Discrete** (a countable set of values, like die faces or "fault / no fault"): a **probability mass function**
  $P(X = x)$. The probabilities are between 0 and 1 and add up to 1.
* **Continuous** (any real number, like a voltage): a **probability density** $p(x)$. The probability of landing
  in an interval is the **area** under $p(x)$ over that interval, and the total area is 1.

Two summaries describe most of what we need:
"""
)
ui.eq(r"E[X] = \sum_x x\,P(X=x) \quad\text{or}\quad \int x\,p(x)\,dx", 1)
ui.eq(r"\operatorname{var}(X) = E\big[(X - E[X])^2\big],\qquad \sigma = \sqrt{\operatorname{var}(X)}", 2)
st.markdown(
    r"""
The **mean** (1) is the centre of mass; the **variance** (2) is the average squared distance from it, and the
**standard deviation** $\sigma$ is in the same units as $X$.

**Example.** A thermometer in a room at 21.0 °C reads $Z = 21.0 + W$, where the noise $W$ has mean 0 and
$\sigma = 0.5$ °C. Then $E[Z] = 21.0$ and the readings typically scatter about ±0.5 °C around it.
"""
)

# ---------------------------------------------------------------------------
ui.section("2. Joint and conditional probability")
st.markdown(
    r"""
With two random quantities $A$ and $B$ we can ask three different things:

* **joint** $P(A, B)$: both happen
* **marginal** $P(A)$: $A$ happens, whatever $B$ does (sum the joint over $B$)
* **conditional** $P(A \mid B)$: $A$ happens, **given that we already know** $B$ happened

They are tied together by the **product rule**:
"""
)
ui.eq(r"P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)", 3)
st.markdown(
    r"""
**Example: 10,000 machines with a vibration alarm.** 1% are faulty. The alarm fires for 99% of faulty
machines and, wrongly, for 5% of healthy ones:

| | alarm | no alarm | total |
|---|---|---|---|
| **faulty** | 99 | 1 | 100 |
| **healthy** | 495 | 9,405 | 9,900 |
| **total** | 594 | 9,406 | 10,000 |

Reading the table: $P(\text{alarm} \mid \text{faulty}) = 99/100$, but
$P(\text{faulty} \mid \text{alarm}) = 99/594 \approx 16.7\%$. **These two conditionals are very different
numbers.** Confusing them is the most common mistake in probability.

**Independence.** $A$ and $B$ are independent if knowing one tells you nothing about the other:
$P(A \mid B) = P(A)$, equivalently $P(A,B) = P(A)P(B)$.
"""
)

# ---------------------------------------------------------------------------
ui.section("3. Bayes' rule: turning 'data given cause' into 'cause given data'")
st.markdown(
    r"""
Divide both sides of (3) by $P(B)$, and rename $A$ as the hidden **cause** $\theta$ and $B$ as the observed
**data** $z$:
"""
)
ui.eq(r"\underbrace{p(\theta \mid z)}_{\text{posterior}} \;=\; "
      r"\frac{\overbrace{p(z \mid \theta)}^{\text{likelihood}}\;\overbrace{p(\theta)}^{\text{prior}}}"
      r"{\underbrace{p(z)}_{\text{evidence}}}", 4)
st.markdown(
    r"""
* **Prior** $p(\theta)$: what we believed before the data.
* **Likelihood** $p(z \mid \theta)$: for each possible cause, how well it explains the data. Sensor models live
  here.
* **Evidence** $p(z) = \sum_\theta p(z\mid\theta)p(\theta)$: just the number that makes the posterior add up to 1.
* **Posterior** $p(\theta \mid z)$: what we should believe now.

Because the evidence doesn't depend on $\theta$, we often write $p(\theta\mid z) \propto p(z\mid\theta)\,p(\theta)$
(multiply, then normalise).

**Worked example (the alarm).** With prior $P(\text{fault}) = 0.01$, sensitivity $0.99$ and specificity $0.95$:
"""
)
ui.eq(r"P(\text{fault}\mid\text{alarm}) = \frac{0.99 \times 0.01}{0.99\times0.01 + 0.05\times0.99} "
      r"= \frac{0.0099}{0.0594} \approx 0.167")
st.markdown(
    r"""
**Sequential updating.** If a second, independent alarm fires, use $0.167$ as the new prior:
$\frac{0.99\times0.167}{0.99\times0.167 + 0.05\times0.833} \approx 0.80$. **Yesterday's posterior is today's
prior.** That one sentence is the Kalman filter, the HMM forward pass and the particle filter.
"""
)

# ---------------------------------------------------------------------------
ui.section("4. Bayes for an unknown rate: the Beta–Bernoulli model")
st.markdown(
    r"""
Now the unknown is a **number**: the defect rate $\theta \in [0,1]$ of a production line. Each part is defective
with probability $\theta$, independently.

**Likelihood.** Inspect $n$ parts and find $k$ defects. The probability of that outcome is
"""
)
ui.eq(r"p(k \mid \theta) = \binom{n}{k}\,\theta^{k}\,(1-\theta)^{n-k}", 5)
st.markdown(
    r"""
**Prior.** A convenient prior for a rate is the **Beta distribution**:
"""
)
ui.eq(r"p(\theta) = \text{Beta}(\theta;\alpha,\beta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}", 6)
st.markdown(
    r"""
Think of $\alpha$ and $\beta$ as **pseudo-counts**: "before inspecting anything, I believe as if I'd already
seen $\alpha$ defective and $\beta$ good parts". Its mean is $\alpha/(\alpha+\beta)$, and a larger $\alpha+\beta$
means a narrower, more confident prior.

**Posterior.** Multiply (5) and (6). The powers of $\theta$ and $(1-\theta)$ simply add:
"""
)
ui.eq(r"p(\theta \mid k) \propto \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1} "
      r"\;\Rightarrow\; \theta \mid k \sim \text{Beta}(\alpha+k,\;\beta+n-k)", 7)
st.markdown(
    r"""
The update is just **add the defects to $\alpha$ and the good parts to $\beta$**. When the posterior has the same
form as the prior like this, the prior is called **conjugate**. It's what makes recursive filtering cheap.

**Worked example.** Prior Beta(2, 20): about a 9% defect rate, worth 22 parts of experience. Inspect 20 parts and
find 3 defects. The posterior is Beta(5, 37), with mean $5/42 \approx 11.9\%$.
"""
)
g = np.linspace(0.001, 0.5, 400)
fig = go.Figure()
fig.add_scatter(x=g, y=stats.beta(2, 20).pdf(g), name="prior Beta(2, 20)", line=dict(color=ui.C["truth"], dash="dash"))
fig.add_scatter(x=g, y=stats.beta(5, 37).pdf(g), name="after 3/20 defects: Beta(5, 37)", line=dict(color=ui.C[1]))
fig.add_scatter(x=g, y=stats.beta(5 + 12, 37 + 88).pdf(g), name="after 12/100 more: Beta(17, 125)",
                line=dict(color=ui.C[2]))
ui.show(ui.style(fig, "More data → narrower belief", "defect rate θ", "density", height=280))
st.markdown(
    r"""
The 95% **credible interval** is the range holding 95% of the posterior's area: "given the data, θ is in here
with 95% probability".
"""
)

# ---------------------------------------------------------------------------
ui.section("5. Markov chains: predicting the next state")
st.markdown(
    r"""
A system jumps between a few **states** (weather: Sunny / Cloudy / Rainy; a machine: OK / Worn / Failed). The
**Markov property** says the next state depends only on the current state:
"""
)
ui.eq(r"P(s_{t+1}=j \mid s_t=i, s_{t-1}, \dots) = P(s_{t+1}=j \mid s_t=i) = A_{ij}", 8)
st.markdown(
    r"""
$A$ is the **transition matrix**: row $i$ lists where you go from state $i$, so every row adds up to 1. If
$\mathbf p_t$ is the row vector of state probabilities today, then tomorrow
"""
)
ui.eq(r"\mathbf p_{t+1} = \mathbf p_t\,A, \qquad \mathbf p_{t+n} = \mathbf p_t\,A^{n}", 9)
st.markdown(
    r"""
**Worked example.** With $A$ = [[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]] (Sunny, Cloudy, Rainy) and
today Sunny, $\mathbf p_0 = [1, 0, 0]$:

* tomorrow: $\mathbf p_1 = $ first row $= [0.80, 0.15, 0.05]$
* the day after: $P(\text{Rainy}) = 0.8\times0.05 + 0.15\times0.2 + 0.05\times0.6 = 0.10$

As $n$ grows, $\mathbf p_n$ forgets where it started and settles at the **stationary distribution**
$\boldsymbol\pi$, which satisfies $\boldsymbol\pi = \boldsymbol\pi A$. That's the long-run fraction of time in
each state. Large diagonal entries ("sticky" states) make the chain remember for longer.
"""
)

# ---------------------------------------------------------------------------
ui.section("6. The Gaussian, and why conditioning on a measurement shrinks uncertainty")
st.markdown(r"""The **Gaussian (normal)** density with mean $\mu$ and variance $\sigma^2$:""")
ui.eq(r"\mathcal N(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)", 10)
st.markdown(
    r"""
It appears everywhere because sums of many small independent effects become Gaussian (the **central limit
theorem**). Sensor noise is usually like that. About 68% of values fall within $\pm1\sigma$ and 95% within
$\pm2\sigma$.

**Two Gaussian quantities together.** Let $x$ be a hidden state and $y$ a sensor reading that is correlated with
it, with standard deviations $\sigma_x, \sigma_y$ and **correlation coefficient** $\rho \in [-1, 1]$. Their joint
density looks like a tilted ellipse. The **1σ and 2σ ellipses** contain about 39% and 86% of the samples in 2-D.

Observing $y$ gives a new Gaussian belief about $x$ (this is Bayes' rule (4) with Gaussians):
"""
)
ui.eq(r"x \mid y \;\sim\; \mathcal N\!\Big(\mu_x + \rho\,\frac{\sigma_x}{\sigma_y}\,(y-\mu_y),\;\;"
      r"\sigma_x^2\,(1-\rho^2)\Big)", 11)
st.markdown(
    r"""
Read (11) as two rules:

* **The mean moves toward the evidence**, by an amount proportional to how surprising $y$ was ($y-\mu_y$) and
  how related the two are ($\rho$).
* **The variance shrinks** by the factor $(1-\rho^2)$. A perfectly correlated sensor ($|\rho| = 1$) pins $x$
  down exactly; an unrelated one ($\rho = 0$) changes nothing.

In Level 3 the same formula, in matrix form and applied once per time step, becomes the **Kalman filter**.
"""
)

# ---------------------------------------------------------------------------
ui.section("How this maps to the Lab")
ui.lab_map("l0", [
    ("Tab **Bayesian updating**: sliders *Prior α*, *Prior β*", r"$\alpha, \beta$ in (6): pseudo-defects and pseudo-good parts"),
    ("*True defect rate θ*", r"the hidden $\theta$. The lab knows it only because it simulates the data"),
    ("*Parts per batch*, *Batches inspected*", r"$n$ per update; each batch applies (7) once, posterior → next prior"),
    ("Curves getting narrower", r"the posterior Beta($\alpha+k$, $\beta+n-k$) after each batch"),
    ("Tab **Bayes' rule & base rates**", r"equation (4) with the alarm numbers; *alarms in a row* = sequential updating"),
    ("Tab **Markov chains**: the editable matrix", r"$A$ in (8); the forecast lines are $\mathbf p_0 A^n$ from (9); dotted lines are $\boldsymbol\pi$"),
    ("Tab **Gaussians & conditioning**: *ρ*, *σ of state*, *σ of sensor*, *observed y*", r"the symbols in (11); the orange bar is the mean ± 2σ of $x \mid y$"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Swapping the conditional.** $P(\text{alarm}\mid\text{fault})$ high does *not* mean $P(\text{fault}\mid\text{alarm})$
  is high. The prior (base rate) decides.
* **Probability vs likelihood.** $p(z\mid\theta)$ is a probability *of the data*. As a function of $\theta$ it
  is a likelihood, and it does **not** integrate to 1 over $\theta$.
* **Overconfident priors.** A prior worth 1,000 pseudo-observations needs a lot of data to overturn, even when
  it's wrong.
"""
)

ui.section("Check yourself")
ui.check([
    ("Prior Beta(2, 20). You inspect 20 parts and find 3 defects. What is the posterior, and its mean?",
     "Beta(2+3, 20+17) = **Beta(5, 37)**; mean = 5/42 ≈ **0.119**."),
    ("Same alarm (99% sensitivity, 95% specificity, 1% prevalence). Two independent alarms fire. P(fault)?",
     "After one: 0.167. After two: 0.99·0.167 / (0.99·0.167 + 0.05·0.833) ≈ **0.80**."),
    ("Using the weather matrix above, starting Sunny, what is P(Rainy) the day after tomorrow?",
     "0.8·0.05 + 0.15·0.2 + 0.05·0.6 = **0.10**."),
    (r"ρ = 0.8, σx = 1, σy = 1.5, means 0, and you observe y = 2. What are E[x | y] and the new σ?",
     r"E[x|y] = 0.8 · (1/1.5) · 2 ≈ **1.07**; σ = 1 · √(1 − 0.64) = **0.6**."),
])
