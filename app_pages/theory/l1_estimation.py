import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui

ui.theory_header("l1", """
Level 0 was about beliefs. Level 1 asks the engineering question: **from N noisy samples, what single number
should I report, and how accurate can it possibly be?** That number could be a sensor offset, a click-through
rate or the pitch of a voice. We meet the three standard answers (**MLE, MAP, MMSE**) and the speed limit on
accuracy (the **Cramér–Rao lower bound**).
""")

ui.section("1. What an estimator is, and how we judge one")
st.markdown(
    r"""
**Model.** A sensor measures a constant level $A$ (a DC offset, a battery voltage) with additive noise:
"""
)
ui.eq(r"x[n] = A + w[n], \qquad w[n] \sim \mathcal N(0, \sigma^2) \text{ independent}, \quad n = 0,\dots,N-1", 1)
st.markdown(
    r"""
An **estimator** $\hat A$ is any recipe that turns the data $x[0..N-1]$ into a number, e.g. "take the average".
Because the data is random, **$\hat A$ is itself a random variable**. Repeat the experiment and you get a
different value. We judge an estimator by the distribution of its errors:
"""
)
ui.eq(r"\text{bias} = E[\hat A] - A,\qquad \text{variance} = \operatorname{var}(\hat A),\qquad "
      r"\text{MSE} = E[(\hat A - A)^2] = \text{bias}^2 + \text{variance}", 2)
rng = np.random.default_rng(0)
fig = go.Figure()
for N, col in ((5, ui.C[2]), (50, ui.C[1])):
    est = (1.0 + 2.0 * rng.standard_normal((4000, N))).mean(axis=1)
    fig.add_histogram(x=est, name=f"sample mean, N = {N}", marker_color=col, opacity=0.6, nbinsx=60,
                      histnorm="probability density")
fig.add_vline(x=1.0, line=dict(color=ui.C["truth"], width=2), annotation_text="true A = 1")
fig.update_layout(barmode="overlay")
ui.show(ui.style(fig, "4,000 repeats of the experiment (σ = 2): the estimate is random, and N makes it tighter",
                 "estimate of A", "density", height=270))
st.markdown("Both histograms are centred on the truth (unbiased). More data shrinks the spread like $1/\\sqrt N$.")

ui.section("2. Maximum likelihood (MLE): the value that best explains the data")
st.markdown(
    r"""
For fixed data, the **likelihood** $p(\mathbf x \mid A)$ tells us how probable the observed samples would be for
each candidate value of $A$. The MLE picks the candidate under which the data is most probable. Because the
samples are independent, the likelihood is a product, and its log is a sum:
"""
)
ui.eq(r"\log p(\mathbf x \mid A) = -\frac N2\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n}(x[n]-A)^2", 3)
st.markdown(
    r"""
Maximising (3) means minimising the sum of squared errors. Set the derivative with respect to $A$ to zero:
$\sum_n (x[n]-A) = 0$, so
"""
)
ui.eq(r"\hat A_{\text{ML}} = \frac1N\sum_n x[n] = \bar x", 4)
st.markdown(
    r"""
**Worked example.** Readings 1.2, 0.7, 1.9 give $\hat A_{\text{ML}} = 3.8/3 \approx 1.27$.

The log-likelihood (3) is a downward parabola in $A$. Its **curvature** $N/\sigma^2$ says how sharply the data
pins $A$ down; we'll meet that number again in section 5.
"""
)

ui.section("3. MAP: add what you knew beforehand")
st.markdown(
    r"""
If we have prior knowledge, e.g. "offsets on this production line are usually near $\mu_0$, give or take
$\sigma_0$", we write it as a prior $A \sim \mathcal N(\mu_0, \sigma_0^2)$ and maximise the **posterior**
(Bayes' rule from Level 0) instead:
"""
)
ui.eq(r"\hat A_{\text{MAP}} = \arg\max_A \big[\log p(\mathbf x\mid A) + \log p(A)\big]", 5)
st.markdown(r"Both terms are quadratic in $A$, so the posterior is Gaussian and the answer is a **precision-weighted average** (precision = 1/variance):")
ui.eq(r"\hat A_{\text{MAP}} = \frac{\frac{N}{\sigma^2}\,\bar x + \frac{1}{\sigma_0^2}\,\mu_0}{\frac{N}{\sigma^2} + \frac{1}{\sigma_0^2}},"
      r"\qquad \sigma_{\text{post}}^2 = \frac{1}{\frac{N}{\sigma^2} + \frac{1}{\sigma_0^2}}", 6)
st.markdown(
    r"""
* With little or very noisy data ($N/\sigma^2$ small), MAP stays near the prior mean $\mu_0$.
* With lots of data ($N/\sigma^2 \gg 1/\sigma_0^2$), the prior is outvoted and MAP → MLE.
* The fraction $\frac{N/\sigma^2}{N/\sigma^2 + 1/\sigma_0^2}$ is the **weight on the data**. You'll see it again
  as the **Kalman gain** in Level 3.

**Worked example.** $\bar x = 1.27$ from $N=3$, $\sigma = 2$, prior $\mu_0 = 0$, $\sigma_0 = 0.5$. The data
precision is $3/4 = 0.75$ and the prior precision is $1/0.25 = 4$, so the weight on the data is
$0.75/4.75 \approx 16\%$ and $\hat A_{\text{MAP}} \approx 0.16 \times 1.27 = 0.20$. Three very noisy samples
don't move a confident prior much.
"""
)

ui.section("4. Three answers from one posterior: MLE, MAP and MMSE")
st.markdown(
    r"""
Once you have a posterior $p(\theta \mid \text{data})$, "the" estimate depends on **what error you care about**:

| Estimator | Definition | Best when you're penalised by… |
|---|---|---|
| MLE | peak of the likelihood (ignores the prior) | (no prior available) |
| MAP | peak (mode) of the posterior | being wrong at all (0–1 loss) |
| MMSE | **mean** of the posterior | squared error |

For a symmetric posterior (Gaussian), MAP = MMSE. For a **skewed** posterior they differ.

**Example: click-through rate.** An ad got $k = 1$ click in $n = 5$ impressions. Past campaigns suggest a
Beta(2, 30) prior (around 6%). The posterior is Beta(3, 34), so
"""
)
ui.eq(r"\hat\theta_{\text{ML}} = \tfrac{k}{n} = 0.20,\quad "
      r"\hat\theta_{\text{MAP}} = \tfrac{k+\alpha-1}{n+\alpha+\beta-2} = \tfrac{2}{35} \approx 0.057,\quad "
      r"\hat\theta_{\text{MMSE}} = \tfrac{k+\alpha}{n+\alpha+\beta} = \tfrac{3}{37} \approx 0.081", 7)
st.markdown("A 20% click rate from 5 impressions is not believable, and the prior protects you from it.")

ui.section("5. The speed limit: Fisher information and the Cramér–Rao bound")
st.markdown(
    r"""
How precise can **any** unbiased estimator be? The answer depends only on how sharply the likelihood peaks.
The **Fisher information** is the expected curvature of the log-likelihood:
"""
)
ui.eq(r"I(A) = -E\!\left[\frac{\partial^2 \log p(\mathbf x\mid A)}{\partial A^2}\right], \qquad "
      r"\operatorname{var}(\hat A) \;\ge\; \frac{1}{I(A)} \quad (\text{CRLB})", 8)
st.markdown(
    r"""
For the DC level, differentiating (3) twice gives $I(A) = N/\sigma^2$, so **CRLB = $\sigma^2/N$**. The sample
mean's variance is exactly $\sigma^2/N$: it **reaches** the bound, so we call it **efficient**. No amount of
cleverness can do better without extra information.

*Extra information* is exactly what a prior is. MAP is **biased** (it's pulled toward $\mu_0$), so the CRLB for
unbiased estimators doesn't apply to it. When the prior is right, its average error is *below* the CRLB.
"""
)

ui.section("6. A harder case: estimating a frequency, and the threshold effect")
st.markdown(r"""**Model.** A tone in noise: pitch of a voice, engine RPM from vibration, Doppler shift in radar.""")
ui.eq(r"x[n] = a\cos(2\pi f_0 n + \phi) + w[n], \qquad \text{SNR } \eta = \frac{a^2}{2\sigma^2}", 9)
st.markdown(
    r"""
The MLE of $f_0$ is (approximately) the frequency where the **periodogram** $\lvert\sum_n x[n]e^{-j2\pi fn}\rvert^2$
peaks. Find it coarsely with a zero-padded FFT, then refine locally. Its CRLB is
"""
)
ui.eq(r"\operatorname{var}(\hat f_0) \;\ge\; \frac{12}{(2\pi)^2\,\eta\,N(N^2-1)}", 10)
st.markdown(
    r"""
The bound falls like $N^{-3}$: doubling the window length cuts the variance by 8. Above some SNR the MLE
follows (10). **Below it, a noise peak sometimes beats the true peak** and the estimate jumps somewhere random.
The error then explodes by orders of magnitude. This is the **threshold effect**, and it's why pitch trackers
make octave errors in noise.
"""
)

ui.section("7. The noise model matters: robustness")
st.markdown(
    r"""
The sample mean is the MLE **if the noise is Gaussian**. If the noise has heavy tails (occasional big
impulses: clicks, packet glitches, multipath), the MLE changes. For Laplacian noise
$p(w) \propto e^{-\lvert w\rvert/b}$, maximising the likelihood means minimising $\sum_n \lvert x[n]-A\rvert$,
whose solution is the **median**. With clean Gaussian noise the median is slightly worse (its variance is about
$\pi/2 \approx 1.57$ times larger), but with even a few percent of outliers it's far better.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l1", [
    ("Tab **Likelihood → MLE & MAP**: *True A*, *Noise σ*, *Samples N*", r"$A$, $\sigma$, $N$ in (1)"),
    ("Blue curve / orange curve / dashed grey curve", r"likelihood (3), posterior, prior, each scaled to peak at 1"),
    ("*Prior mean μ₀*, *Prior σ₀*; metric *Weight on the data*", r"the prior in section 3 and the weight in (6)"),
    ("Tab **MLE vs MAP vs MMSE**: *Impressions n*, *Clicks k*, *Prior α, β*", r"the click-through example, equation (7)"),
    ("Tab **Best possible accuracy (CRLB)**: the MSE vs N plot", r"Monte Carlo MSE of (4) and (6) against $\sigma^2/N$ from (8)"),
    ("Radio *Where does the true A come from?*", r"prior right (A drawn from the prior) vs a fixed, possibly wrong prior"),
    ("Tab **Frequency & the threshold effect**: *Window length N*, *SNR*", r"$N$ and $\eta$ in (9)–(10); the dashed line is (10)"),
    ("Tab **Break it: outliers**", r"mean (Gaussian MLE) vs median (Laplacian MLE), section 7"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **"Unbiased" ≠ "best".** A slightly biased estimator (MAP) can have a much smaller total error when data is scarce.
* **The CRLB is a bound for a model.** If the real noise isn't the noise you assumed, the bound, and the MLE's
  optimality, no longer apply.
* **Priors are assumptions.** A confident wrong prior costs you, and the lab shows how many samples it takes to
  recover.
"""
)

ui.section("Check yourself")
ui.check([
    ("Readings 2.0, 2.4, 1.6 with σ = 1. What is the MLE of A, and the CRLB?",
     "MLE = mean = **2.0**. CRLB = σ²/N = **1/3 ≈ 0.33**, i.e. a standard deviation of about 0.58."),
    ("Same data, prior N(0, 1²). What is the MAP estimate?",
     "Data precision N/σ² = 3, prior precision 1. MAP = (3·2.0 + 1·0)/(3+1) = **1.5**."),
    ("Why can MAP have a lower MSE than the CRLB?",
     "The CRLB only bounds **unbiased** estimators. MAP is biased toward the prior, and when the prior is right that "
     "bias costs less than the variance it removes."),
    ("You double the window length N in frequency estimation. By how much does the CRLB drop?",
     "≈ N³ dependence: about **8×** smaller variance (≈ 9 dB)."),
])
