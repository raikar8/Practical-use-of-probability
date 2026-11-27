import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui

ui.theory_header("l5", """
The Kalman filter assumes a **linear** world with **Gaussian** noise and **one** motion model. Real problems break
all three: a measurement that can't tell $+x$ from $-x$, a car that could be on any of several hills with the same
altitude, a target that cruises and then turns hard, a sensor whose behaviour suddenly changes. This page covers
the three tools for those cases: **particle filters**, **IMM** and **Bayesian change-point detection**.
""")

# ---------------------------------------------------------------------------
ui.section("A. Particle filters: a belief made of samples")
st.markdown(
    r"""
**Idea.** Instead of a mean and covariance, represent the belief $p(\mathbf x_t \mid z_{1:t})$ by $N$ samples
("particles") $\mathbf x^{(i)}$ with weights $w^{(i)}$ that add up to 1. Any summary is a weighted average:
"""
)
ui.eq(r"E[f(\mathbf x_t)\mid z_{1:t}] \;\approx\; \sum_{i=1}^{N} w^{(i)} f\big(\mathbf x^{(i)}_t\big)", 1)
st.markdown(
    r"""
A cloud of particles can take **any shape**: two peaks, a ring, a long banana. The price is computation, since
you need enough particles to cover the region where the truth might be.

**The bootstrap filter (SIR)**, once per time step:

1. **Propagate:** move every particle through the motion model, *each with its own random noise*:
   $\mathbf x^{(i)}_t = f(\mathbf x^{(i)}_{t-1}) + \mathbf w^{(i)}$. This is the "predict" step: the cloud spreads.
2. **Weight:** multiply each weight by how well that particle explains the measurement:
   $w^{(i)} \leftarrow w^{(i)}\,p(z_t \mid \mathbf x^{(i)}_t)$, then normalise. This is Bayes' rule, particle by particle.
3. **Estimate:** use (1).
4. **Resample** when the weights get too uneven: draw $N$ new particles from the old ones with probabilities
   $w^{(i)}$ (heavy particles get cloned, light ones die), then reset all weights to $1/N$.
"""
)
st.markdown(r"""**Why resample?** After a few surprising measurements almost all the weight sits on a handful of particles and the rest are wasted (**degeneracy**). The **effective sample size** measures this:""")
ui.eq(r"\text{ESS} = \frac{1}{\sum_i \big(w^{(i)}\big)^2}\qquad(N \text{ for equal weights, } 1 \text{ if one particle has everything})", 2)
st.markdown(
    r"""
A common rule is to resample when ESS drops below $N/2$. **Systematic** resampling (one random offset, then
evenly spaced pointers) adds the least extra randomness. **Multinomial** (independent draws) is simplest but
noisiest.

**The benchmark.** The lab uses the classic test model
$x_t = \tfrac{x_{t-1}}{2} + \tfrac{25x_{t-1}}{1+x_{t-1}^2} + 8\cos(1.2t) + v_t$ with $z_t = x_t^2/20 + w_t$.
Because $z$ depends on $x^2$, the measurement can't tell $+x$ from $-x$, so the true posterior often has
**two peaks**:
"""
)
xg = np.linspace(-15, 15, 600)
post = 0.55 * stats.norm(-7, 1.2).pdf(xg) + 0.45 * stats.norm(7.5, 1.4).pdf(xg)
fig = go.Figure()
fig.add_scatter(x=xg, y=post, name="true posterior (two peaks)", line=dict(color=ui.C["truth"], width=3),
                fill="tozeroy", fillcolor=ui.rgba(ui.C["truth"], 0.15))
fig.add_scatter(x=xg, y=stats.norm(6.8, 1.6).pdf(xg) * 0.5, name="EKF: one Gaussian, may pick the wrong peak",
                line=dict(color=ui.C[8]))
rng = np.random.default_rng(3)
parts = np.r_[rng.normal(-7, 1.2, 55), rng.normal(7.5, 1.4, 45)]
fig.add_scatter(x=parts, y=np.full(parts.size, -0.01), mode="markers", name="particles (cover both)",
                marker=dict(color=ui.C[1], size=6, symbol="line-ns-open"))
ui.show(ui.style(fig, "Why a single Gaussian fails and particles don't", "state x", "density", height=270))
st.markdown(
    r"""
In 1-D we can also compute the **exact** posterior on a fine grid (the *grid filter*) and check that the particle
filter matches it. In higher dimensions a grid becomes impossibly large (the **curse of dimensionality**), which
is why particle filters exist.

**Localisation on a known map.** In the road example the state is the distance $s$ along a 6 km road. The car's
odometer moves the particles ($s \leftarrow s + v_{\text{odo}}\Delta t + \text{noise}$), and a barometric
altimeter weights them: $p(z \mid s) = \mathcal N(z;\, h(s), \sigma^2)$, where $h(s)$ is the road's elevation
map. At the start the car could be anywhere, so the particles start **uniformly** over the road. A Kalman
filter can't even represent that belief. While several hills match the altitude, the belief has several peaks
and **the mean lies between them, where the car certainly isn't**. Report the peaks, not the mean, until one
wins.
"""
)

# ---------------------------------------------------------------------------
ui.section("B. IMM: several motion models, and a belief over which one is active")
st.markdown(
    r"""
A target **cruises** (nearly constant velocity) and sometimes **manoeuvres** (turns hard). One Kalman filter
can't serve both: a small $q$ is precise on straights but lags in turns, and a large $q$ follows turns but is
noisy everywhere.

The **Interacting Multiple Model** filter runs one Kalman filter per model $j$ (here two constant-velocity
filters with small and large $q$), and treats *which model is active* as a hidden **Markov chain** with switching
matrix $\Pi$. That's Level 4's HMM on top of Level 3's Kalman filters. Each cycle:

1. **Mixing.** Each filter starts from a blend of all filters' estimates, weighted by the probability that the
   system just switched into its model.
2. **Model-matched filtering.** Ordinary predict/update in every filter.
3. **Model probabilities.** Each filter's innovation $\boldsymbol\nu_j$ is scored by its own predicted
   covariance $S_j$:
"""
)
ui.eq(r"\Lambda_j = \mathcal N(\boldsymbol\nu_j;\,\mathbf 0, S_j), \qquad "
      r"\mu_j \;\propto\; \Lambda_j \sum_i \Pi_{ij}\,\mu_i", 3)
st.markdown(
    r"""
4. **Combination.** Output $\hat{\mathbf x} = \sum_j \mu_j\hat{\mathbf x}_j$ (and the matching covariance).

When the target starts to turn, the cruise filter's innovations become too large for its small $S$, its
likelihood collapses, and $\mu_{\text{manoeuvre}}$ rises within a few scans. The **switching probability** sets
how readily it changes its mind. That's the same duration prior as in the HMM.
"""
)

# ---------------------------------------------------------------------------
ui.section("C. Bayesian online change-point detection (BOCPD)")
st.markdown(
    r"""
Some signals are **piecewise stable**: a machine's vibration level, a process temperature, a market's volatility.
They sit at one level for a while, then *jump*. We want to detect jumps **online** and know how sure we are.

BOCPD (Adams & MacKay, 2007) tracks a hidden **run length** $r_t$, the number of samples since the last change.
Each step the run either **grows** ($r_t = r_{t-1}+1$) or **resets** ($r_t = 0$) with **hazard**
$H = 1/\lambda$, where $\lambda$ is the expected segment length:
"""
)
ui.eq(r"P(r_t = r_{t-1}+1,\, x_{1:t}) = P(r_{t-1}, x_{1:t-1})\;\pi^{(r)}_t\,(1-H)", 4)
ui.eq(r"P(r_t = 0,\, x_{1:t}) = \sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1})\;\pi^{(r)}_t\,H", 5)
st.markdown(
    r"""
Here $\pi^{(r)}_t$ is the **predictive probability** of the new sample $x_t$ if the current segment has lasted
$r$ samples. For each candidate run length we keep a running posterior of that segment's mean *and* variance (a
Normal–Inverse-Gamma conjugate prior), so the predictive is a Student-t in closed form. It catches jumps in
**level** and in **noise**.

Normalising (4)–(5) gives the **run-length posterior**. Plotted over time it's a set of rising diagonal ridges,
and a change appears as a new ridge starting at zero. This is an HMM forward pass (Level 4) whose state is "how
long has this lasted?".

**Declaring an alarm.** A single outlier makes the most probable run length blip to zero for one sample. Waiting
a few samples to confirm the new run removes those false alarms, at the cost of that much **delay**. Trading false
alarms against delay is the core design decision in any detector.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l5", [
    ("Tab **Particle filter vs EKF**: *Particles*, *Resampling*, *Resample when ESS below*", r"$N$, the resampling scheme, and the ESS threshold of (2)"),
    ("Grey heat map / blue line / red line", r"exact grid posterior / particle estimate (1) / EKF"),
    ("ESS plot with orange dots", r"ESS (2) over time; dots are steps where resampling happened"),
    ("Tab **Where am I on this road?**: *Barometer noise σ*, *Odometer noise σ*, *True start position*", r"$\sigma$ in $p(z\mid s)$, motion noise, and the hidden start"),
    ("Tab **IMM**: *Cruise model q*, *Manoeuvre model q*, *P(stay in the same model)*", r"the two filters' $q$ and the diagonal of $\Pi$ in (3)"),
    ("*IMM P(manoeuvre)* plot", r"$\mu_{\text{manoeuvre}}$ from (3) against the true turning periods"),
    ("Tab **Change-point detection**: *Expected run length λ*, *Confirmation delay*", r"hazard $H=1/\lambda$ in (4)–(5), and the alarm rule"),
    ("Blue heat map", r"the run-length posterior; the orange line is its most probable value"),
])

ui.section("Check yourself")
ui.check([
    ("Weights are [0.7, 0.1, 0.1, 0.1]. What is the ESS?",
     "1/(0.49 + 3·0.01) = 1/0.52 ≈ **1.9** out of 4: time to resample."),
    ("Why not always use a particle filter instead of a Kalman filter?",
     "Cost and noise. For linear-Gaussian problems the Kalman filter is **exact** and costs microseconds; a particle "
     "filter only approximates it and needs many particles, which grow rapidly with the state dimension."),
    ("In the IMM, what happens if P(stay) is set to 0.8?",
     "The models switch too readily, so the output mixes in the noisy manoeuvre filter even on straights."),
    ("λ = 100 in BOCPD. What does that encode?",
     "A prior expectation that segments last about **100 samples** (hazard 1% per sample). A larger λ makes the "
     "detector more reluctant to declare small changes."),
])
