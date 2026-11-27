import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui

ui.theory_header("l8", """
Bayes' rule is easy to write and usually impossible to compute. The posterior's normalising constant is an
integral over every unknown, and outside a few conjugate cases (Levels 0–3) it has no formula. This level covers
the three standard ways out: **approximate it with a Gaussian** (Laplace), **draw samples from it** (Markov chain
Monte Carlo), or **fit the best simple distribution to it** (variational inference). The last one is the engine
inside variational autoencoders (Level 11).
""")

ui.section("1. Where exact Bayes breaks")
st.markdown(r"For parameters $\theta$ and data $x$:")
ui.eq(r"p(\theta \mid x) = \frac{p(x \mid \theta)\,p(\theta)}{\int p(x \mid \theta')\,p(\theta')\,d\theta'}", 1)
st.markdown(
    r"""
The numerator is easy to evaluate at any $\theta$. The denominator needs an integral over all $\theta$. On a
grid with 100 points per dimension that's $100^d$ evaluations: fine for $d = 2$ (which is why the Lab can show
you the exact answer), hopeless for $d = 20$, and neural networks have millions of parameters.

Usually we don't even want the posterior itself, but **expectations** under it: the posterior mean, an error bar,
a predictive probability. All three methods below are ways to compute those expectations approximately.
"""
)

ui.section("2. The Laplace approximation")
st.markdown(
    r"""
Find the posterior's peak $\hat\theta$ (the MAP estimate, Level 1). Near the peak, the log-posterior is roughly a
parabola, and a parabola in the log is a Gaussian:
"""
)
ui.eq(r"p(\theta \mid x) \approx \mathcal{N}\big(\theta;\;\hat\theta,\; H^{-1}\big), \qquad "
      r"H = -\nabla^2 \log p(\theta \mid x)\big|_{\hat\theta}", 2)
st.markdown(
    r"""
The curvature $H$ says how sharply the peak falls off; sharp peak = small uncertainty. It's fast (one
optimisation and one Hessian) and often good when there's lots of data. It fails when the posterior is skewed,
curved (a banana) or has several peaks: it only ever sees one peak, and only its immediate neighbourhood.
"""
)

ui.section("3. Monte Carlo: averages from samples")
st.markdown(r"If we could draw samples $\theta^{(1)}, \dots, \theta^{(S)}$ from the posterior, any expectation becomes an average:")
ui.eq(r"\mathbb{E}[f(\theta) \mid x] \approx \frac{1}{S}\sum_{s=1}^{S} f(\theta^{(s)}), \qquad "
      r"\text{error} \approx \frac{\mathrm{sd}(f)}{\sqrt{S_{\text{eff}}}}", 3)
st.markdown(
    r"""
The error shrinks like $1/\sqrt{S}$ **whatever the dimension**, which is why sampling scales where grids don't.
The catch is $S_{\text{eff}}$, the **effective sample size**: MCMC samples are correlated (each one is a small
step from the last), and 10,000 highly correlated samples may carry as much information as 50 independent ones.
"""
)

ui.section("4. Metropolis–Hastings: a random walk that knows where to linger")
st.markdown(
    r"""
Build a Markov chain (Level 0) whose long-run distribution is the posterior. From the current point $\theta$:

1. **Propose** a nearby point $\theta' = \theta + \text{step}\cdot\varepsilon$, with $\varepsilon \sim \mathcal{N}(0, I)$.
2. **Accept** it with probability
"""
)
ui.eq(r"\alpha = \min\!\left(1,\; \frac{p(\theta' \mid x)}{p(\theta \mid x)}\right) = "
      r"\min\!\left(1,\; \frac{p(x \mid \theta')\,p(\theta')}{p(x \mid \theta)\,p(\theta)}\right)", 4)
st.markdown(
    r"""
Otherwise **stay** (and record $\theta$ again).

The ratio in (4) cancels the impossible denominator of (1): we only ever need the posterior **up to a constant**.
Uphill moves are always accepted; downhill moves sometimes. The chain spends time in each region in proportion
to its probability.

**The step size is everything.** Tiny steps: almost always accepted, but the chain crawls (high correlation, low
ESS). Huge steps: almost always rejected, so the chain sits still. A rule of thumb aims for 20–50% acceptance.
Early samples depend on where you started (**burn-in**) and are discarded.
"""
)

ui.section("5. Gibbs sampling")
st.markdown(
    r"""
If you can sample each variable **given all the others** exactly, alternate: draw $\theta_1 \mid \theta_2$, then
$\theta_2 \mid \theta_1$, and so on. Every draw is accepted. It's the natural sampler for conjugate models and for
many classic latent-variable models. Its weakness: when variables are strongly correlated, each conditional is
narrow and the chain zig-zags in tiny axis-aligned steps along a diagonal ridge.
"""
)

ui.section("6. Hamiltonian Monte Carlo: use the gradient")
st.markdown(
    r"""
Treat $-\log p(\theta \mid x)$ as a landscape and the current point as a frictionless puck. Give it a random
push (momentum $\rho$), let it slide for a while following Newton's laws, and stop. It glides **along** ridges
and curves instead of stumbling randomly. The total energy
"""
)
ui.eq(r"H(\theta, \rho) = -\log p(\theta \mid x) + \tfrac{1}{2}\rho^\top\rho", 5)
st.markdown(
    r"""
is conserved by exact dynamics. The simulation uses $L$ **leapfrog** steps of size $\epsilon$ (half step in
momentum, full step in position, half step in momentum). Small numerical errors are corrected by a Metropolis
accept/reject on the change in $H$. The result: long, nearly independent moves with high acceptance. HMC (with
automatic tuning, "NUTS") is the default sampler in Stan and PyMC. It can't jump between well-separated peaks,
though: the puck doesn't have the energy to climb out of a deep valley.
"""
)

ui.section("7. Variational inference: optimise instead of sample")
st.markdown(
    r"""
Pick a simple family, here Gaussians $q(\theta) = \mathcal{N}(m, LL^\top)$, and find the member closest to the
posterior, measured by the Kullback–Leibler divergence $\mathrm{KL}(q\,\|\,p)$. We can't compute that KL directly (it
contains the unknown denominator), but the same bound as in EM (Level 7) appears:
"""
)
ui.eq(r"\log p(x) = \underbrace{\mathbb{E}_{q}\big[\log p(x, \theta) - \log q(\theta)\big]}_{\text{ELBO}(q)} "
      r"+ \mathrm{KL}\big(q(\theta)\,\|\,p(\theta \mid x)\big)", 6)
st.markdown(
    r"""
$\log p(x)$ doesn't depend on $q$, so **maximising the ELBO (evidence lower bound) minimises the KL**. The ELBO
only needs the unnormalised joint, and its gradient can be estimated from a few samples using the
**reparameterisation trick**: write $\theta = m + L\varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$, so that
randomness is in $\varepsilon$ and $m, L$ can be differentiated through:
"""
)
ui.eq(r"\nabla_m \text{ELBO} = \mathbb{E}_\varepsilon\big[\nabla_\theta \log p(x, \theta)\big], \qquad "
      r"\nabla_L \text{ELBO} = \mathbb{E}_\varepsilon\big[\nabla_\theta \log p(x, \theta)\,\varepsilon^\top\big] + "
      r"\mathrm{diag}(1/L_{ii})", 7)
st.markdown(
    r"""
That is plain gradient ascent, the same machinery used to train neural networks. VI is fast and scales to huge
models; the price is that the answer is only as good as the chosen family.

**Which KL?** $\mathrm{KL}(q\,\|\,p)$ heavily penalises $q$ putting mass where $p$ has almost none. So with two
peaks, the best Gaussian **locks onto one peak** and underestimates uncertainty ("mode-seeking"). The other
direction, $\mathrm{KL}(p\,\|\,q)$, penalises $q$ missing mass where $p$ has some, so it **covers both** peaks
with one wide Gaussian ("mass-covering"), placing much of its mass in the empty valley between them.
"""
)
x = np.linspace(-6, 6, 1201)
dx = x[1] - x[0]


def _n(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


p = 0.5 * _n(x, -2, 0.6) + 0.5 * _n(x, 2, 0.6)
best = (np.inf, 0, 1)
for m in np.linspace(-3, 3, 61):
    for s in np.linspace(0.3, 3, 28):
        q = _n(x, m, s)
        kl = np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx
        best = min(best, (kl, m, s))
mf, sf = np.sum(p * x) * dx, np.sqrt(np.sum(p * x**2) * dx)
fig = go.Figure()
fig.add_scatter(x=x, y=p, name="posterior p (two peaks)", line=dict(color=ui.C["truth"], width=3))
fig.add_scatter(x=x, y=_n(x, best[1], best[2]), name=f"min KL(q‖p): N({best[1]:.1f}, {best[2]:.2f}²), one peak",
                line=dict(color=ui.C[1]))
fig.add_scatter(x=x, y=_n(x, mf, sf), name=f"min KL(p‖q): N({mf:.1f}, {sf:.2f}²), covers both",
                line=dict(color=ui.C[2], dash="dash"))
ui.show(ui.style(fig, "The best Gaussian depends on which KL you minimise", "θ", "density", height=280))

ui.section("8. Which method when?")
st.markdown(
    r"""
| Method | Cost | Accuracy | Typical use |
|---|---|---|---|
| **Laplace** | one optimisation + Hessian | good for near-Gaussian posteriors | quick error bars, large-data limits, model evidence |
| **Metropolis–Hastings** | many cheap steps | exact in the long run; slow when correlated | simple models, few parameters |
| **Gibbs** | many cheap steps | exact in the long run; slow when correlated | conjugate and latent-variable models |
| **HMC / NUTS** | gradient evaluations | exact in the long run; efficient in high dimension | Stan, PyMC: scientific and statistical modelling |
| **Variational inference** | an optimisation | biased (family-limited), often too confident | huge models and datasets; VAEs; Bayesian deep learning |
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l8", [
    ("Grey contours", r"the exact posterior from eq. (1), computed on a grid (possible only because it's 2-D)"),
    ("Tab **MCMC samplers**: *Step size*", r"the proposal scale in eq. (4), or HMC's leapfrog $\epsilon$"),
    ("*Acceptance rate*, *Effective sample size*", r"§4 and $S_{\text{eff}}$ in eq. (3)"),
    ("*Leapfrog steps L*, the curved HMC paths", "§6"),
    ("Tab **Variational vs Laplace**: blue ellipse moving", r"gradient ascent on the ELBO, eq. (6)–(7)"),
    ("Orange dashed ellipse", "Laplace, eq. (2)"),
    ("Aqua ellipse", r"the best Gaussian under $\mathrm{KL}(p\,\|\,q)$ (moment matching), for comparison"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Trusting a chain that hasn't mixed.** Run several chains from different starts; if they disagree, none is
  right yet. Report the effective sample size, not the raw count.
* **Judging MCMC by acceptance rate alone.** 99% acceptance with tiny steps is a chain that barely moves.
* **Reading VI error bars literally.** Reverse-KL VI typically **underestimates** uncertainty.
* **Forgetting that every method here sees only what the model says.** A wrong likelihood gives a confidently
  wrong posterior, however good the sampler.
"""
)

ui.section("Check yourself")
ui.check([
    ("Why does Metropolis–Hastings never need the normalising constant of the posterior?",
     "It only uses the **ratio** p(θ′|x)/p(θ|x), in which the constant cancels."),
    ("A chain has 95% acceptance and an ESS of 12 out of 5,000 samples. What should you change?",
     "**Increase the step size.** Nearly everything is accepted because the steps are tiny, so successive samples "
     "are almost identical."),
    ("The posterior has two well-separated peaks. What does reverse-KL VI with one Gaussian give you?",
     "A Gaussian on **one** of the peaks (which one depends on the starting point), with too little uncertainty."),
    ("Why does HMC need the gradient of the log-posterior?",
     "To simulate the 'puck' sliding on the landscape: the gradient is the force in the leapfrog updates."),
])
