import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui
from ssp.core import estimation as est

ui.lab_intro(
    "l1",
    goal="Watch estimators at work on simulated sensor data: where the MLE and MAP land, how a prior pulls the "
         "answer, and how close any estimator can get to the truth (the CRLB).",
    on_screen=[
        "**Likelihood → MLE & MAP:** the three curves of Bayes' rule for a constant level in noise (Theory §2–3)",
        "**MLE vs MAP vs MMSE:** three different 'best guesses' from one skewed posterior (Theory §4)",
        "**Best possible accuracy:** thousands of repeated experiments compared with the CRLB (Theory §5)",
        "**Frequency & the threshold effect:** estimating a tone's pitch, and where that breaks down (Theory §6)",
        "**Outliers:** what happens when the noise isn't Gaussian (Theory §7)",
    ],
    start_here="In the first tab, drag *Samples N* from 1 to 200 and watch the blue likelihood narrow and the "
               "orange MAP marker slide from the prior toward the MLE.",
)

t_like, t_three, t_crlb, t_freq, t_out, t_code = st.tabs(
    ["Likelihood → MLE & MAP", "MLE vs MAP vs MMSE", "Best possible accuracy (CRLB)",
     "Frequency & the threshold effect", "Outliers & robustness", "Code"])


@st.cache_data(show_spinner=False)
def _mc_dc(N_values, sigma, true_A, mu0, s0, trials, from_prior, seed):
    return est.monte_carlo_dc(list(N_values), sigma, true_A, mu0, s0, trials,
                              np.random.default_rng(seed), draw_A_from_prior=from_prior)


@st.cache_data(show_spinner=False)
def _mc_freq(snrs, N, f0, trials, seed):
    return est.monte_carlo_freq(list(snrs), N, f0, trials, np.random.default_rng(seed))


@st.cache_data(show_spinner=False)
def _mc_out(N, ps, scale, trials, seed):
    return est.monte_carlo_outliers(N, 1.0, list(ps), scale, trials, np.random.default_rng(seed))


# --------------------------------------------------------------------------- likelihood
with t_like:
    st.markdown(
        r"""
**The situation.** A sensor measures a constant level $A$ (a DC offset, a battery voltage) $N$ times, each with
Gaussian noise of size $\sigma$: $x[n] = A + w[n]$. We plot three curves over every *candidate* value of $A$:

* **likelihood** (blue): how well each candidate explains the samples. Its peak is the **MLE** = the sample mean.
* **prior** (dashed): what we believed before measuring, $\mathcal N(\mu_0, \sigma_0^2)$.
* **posterior** (orange) = likelihood × prior. Its peak is the **MAP** estimate.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        A_true = st.slider("True A (hidden)", -3.0, 3.0, 1.0, 0.1,
                           help="The real level used to simulate the readings. Estimators never see it.")
        sigma = st.slider("Noise σ", 0.1, 5.0, 2.0, 0.1, help="Standard deviation of each reading's noise.")
        N = st.slider("Samples N", 1, 200, 5, help="Number of readings averaged.")
        mu0 = st.slider("Prior mean μ₀", -3.0, 3.0, 0.0, 0.1, help="Your best guess before measuring.")
        s0 = st.slider("Prior σ₀ (smaller = more confident)", 0.1, 5.0, 0.5, 0.1,
                       help="How sure you are of μ₀. Small σ₀ = a strong prior that needs lots of data to move.")
        seed = st.number_input("Random seed", 0, 9999, 0, key="l_seed",
                               help="Draws a different set of noisy readings.")
    x = A_true + sigma * np.random.default_rng(int(seed)).standard_normal(N)
    grid = np.linspace(-5, 5, 600)
    ll = est.dc_log_likelihood(grid, x, sigma)
    lik = np.exp(ll - ll.max())
    prior = stats.norm(mu0, s0).pdf(grid)
    post = lik * prior
    mle = est.dc_mle(x)
    mapv, post_sd = est.dc_map(x, sigma, mu0, s0)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=grid, y=lik / lik.max(), name="likelihood p(x|A)", line=dict(color=ui.C[1]))
        fig.add_scatter(x=grid, y=prior / prior.max(), name="prior p(A)",
                        line=dict(color=ui.C["truth"], dash="dash"))
        fig.add_scatter(x=grid, y=post / post.max(), name="posterior p(A|x)", line=dict(color=ui.C[2]))
        fig.add_vline(x=A_true, line=dict(color=ui.C["truth"], width=1), annotation_text="true A")
        fig.add_scatter(x=[mle], y=[1.0], mode="markers", name=f"MLE = {mle:.2f}",
                        marker=dict(size=11, color=ui.C[1], symbol="diamond"))
        fig.add_scatter(x=[mapv], y=[1.0], mode="markers", name=f"MAP = {mapv:.2f}",
                        marker=dict(size=11, color=ui.C[2], symbol="diamond"))
        ui.show(ui.style(fig, "Each curve scaled to peak at 1", "candidate value of A", "relative density"))
        m1, m2, m3 = st.columns(3)
        m1.metric("MLE error", f"{mle - A_true:+.2f}")
        m2.metric("MAP error", f"{mapv - A_true:+.2f}")
        m3.metric("Weight on the data", f"{(N / sigma**2) / (N / sigma**2 + 1 / s0**2):.0%}",
                  help="(N/σ²) / (N/σ² + 1/σ₀²): Theory eq. 6. The rest of the weight goes to the prior.")
    ui.controls([
        ("True A", r"$A$", "the hidden level", "all readings shift"),
        ("Noise σ", r"$\sigma$", "reading noise", "a wider likelihood, so the prior matters more"),
        ("Samples N", r"$N$", "number of readings", r"the likelihood narrows like $\sigma/\sqrt N$"),
        ("Prior mean μ₀", r"$\mu_0$", "where the prior is centred", "MAP is pulled toward it"),
        ("Prior σ₀", r"$\sigma_0$", "prior uncertainty", "a weaker prior, so MAP → MLE"),
    ])
    ui.experiments([
        ("Keep the defaults (N = 5, σ = 2, σ₀ = 0.5).",
         "The orange MAP sits much closer to the prior (0) than to the MLE. *Weight on the data* is only ~24%.",
         "Five very noisy readings carry less information than a confident prior."),
        ("Drag N to 200.", "The blue curve becomes a spike; MAP and MLE meet; the weight on the data is ~93%.",
         "Data precision N/σ² grows with N while the prior's stays fixed. Data eventually wins."),
        ("Back to N = 5; set σ₀ = 5.", "The prior curve is almost flat and MAP ≈ MLE.",
         "A vague prior adds almost no information. MAP with a flat prior *is* the MLE."),
        ("Change the random seed a few times at N = 5.", "The MLE jumps around; MAP jumps less.",
         "The prior reduces variance, at the cost of a pull (bias) toward μ₀. This trade-off is the whole of Level 1."),
    ])
    ui.takeaway("The posterior is the product of likelihood and prior. With few or very noisy samples, MAP leans "
                "on the prior. As N grows the likelihood narrows like σ/√N and MAP converges to the MLE.")

# --------------------------------------------------------------------------- MLE vs MAP vs MMSE
with t_three:
    st.markdown(
        r"""
**The situation.** An ad was shown $n$ times and clicked $k$ times. What is its click-through rate $\theta$? The
prior Beta($\alpha$, $\beta$) encodes past campaigns. The posterior is Beta($\alpha+k$, $\beta+n-k$), which is
**skewed**, so its peak (MAP) and its mean (MMSE) are different numbers, and the MLE $k/n$ ignores the prior.
The formulas are in Theory eq. 7.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Impressions n", 1, 100, 5, help="Times the ad was shown.")
        k = st.slider("Clicks k", 0, 100, 1, help="Times it was clicked (capped at n).")
        k = min(k, n)
        a = st.slider("Prior α", 1.0, 20.0, 2.0, 0.5, key="bb_a", help="Pseudo-clicks from past campaigns.")
        b = st.slider("Prior β", 1.0, 100.0, 30.0, 0.5, key="bb_b", help="Pseudo-non-clicks. Prior mean = α/(α+β).")
    r = est.bernoulli_estimators(k, n, a, b)
    g = np.linspace(0.0005, 0.9995, 1000)
    pdf = stats.beta(r["post_alpha"], r["post_beta"]).pdf(g)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=g, y=pdf, name="posterior", line=dict(color=ui.C["truth"]), fill="tozeroy",
                        fillcolor=ui.rgba(ui.C["truth"], 0.12))
        for i, nm in enumerate(("MLE", "MAP", "MMSE")):
            if np.isfinite(r[nm]):
                fig.add_vline(x=r[nm], line=dict(color=ui.C[i + 1], width=2))
                fig.add_scatter(x=[r[nm]], y=[0], mode="markers", name=f"{nm} = {r[nm]:.3f}",
                                marker=dict(color=ui.C[i + 1], size=11))
        fig.update_xaxes(range=[0, min(1, max(0.3, (r["MLE"] if np.isfinite(r["MLE"]) else 0) * 1.4))])
        ui.show(ui.style(fig, f"Posterior Beta({r['post_alpha']:.1f}, {r['post_beta']:.1f}) after {k}/{n} clicks",
                         "click-through rate θ", "density"))
    ui.controls([
        ("Impressions n, Clicks k", r"$n, k$", "the data", "the posterior narrows; all three estimates converge"),
        ("Prior α, β", r"$\alpha, \beta$", "past experience as pseudo-counts", "the prior's pull grows"),
    ])
    ui.experiments([
        ("Defaults: 1 click in 5 impressions.", "MLE = 20%; MAP ≈ 5.7%; MMSE ≈ 8.1%.",
         "Five impressions say almost nothing; the prior (≈6%) dominates the Bayesian estimates."),
        ("Set n = 100, k = 20.", "All three estimates crowd near 15–20%.",
         "With plenty of data, the choice of estimator (and prior) matters less and less."),
        ("Set k = 0, n = 5.", "MLE = 0% (never clicked, so 'never will'); MMSE stays positive.",
         "MLE overreacts to small samples; Bayesian estimates never claim impossibility from 5 trials."),
    ])
    ui.takeaway("With 1 click in 5 impressions, the MLE says 20%. A prior built from past campaigns (≈6%) pulls "
                "MAP and MMSE toward something believable. Ad platforms and recommenders shrink small-sample "
                "rates toward a prior like this so they don't overreact to new items.")

# --------------------------------------------------------------------------- CRLB
with t_crlb:
    st.markdown(
        r"""
**The situation.** Repeat the constant-level experiment thousands of times for each N, and measure each
estimator's **mean squared error** (MSE). The dashed line is the **Cramér–Rao lower bound** $\sigma^2/N$: no
*unbiased* estimator can go below it (Theory §5). Both axes are logarithmic, so a straight line means a power law.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        sig = st.slider("Noise σ", 0.2, 5.0, 1.0, 0.1, key="c_sig", help="Noise standard deviation per reading.")
        pm = st.slider("Prior mean μ₀", -3.0, 3.0, 0.0, 0.1, key="c_mu", help="Prior used by MAP.")
        ps = st.slider("Prior σ₀", 0.05, 3.0, 0.3, 0.05, key="c_s0", help="Prior uncertainty used by MAP.")
        mode = st.radio("Where does the true A come from?",
                        ["Drawn from the prior (prior is right)", "Fixed value (prior may be wrong)"],
                        help="'Drawn': each repeat picks a new true A from the prior, so the prior is correct on "
                             "average. 'Fixed': the true A is the value below, which may disagree with the prior.")
        At = st.slider("Fixed true A", -3.0, 3.0, 1.0, 0.1, disabled=mode.startswith("Drawn"),
                       help="Only used in 'Fixed' mode.")
        trials = st.select_slider("Monte Carlo trials", [500, 2000, 5000, 20000], 5000,
                                  help="Repeats per N. More = smoother curves, slower.")
    Ns = tuple(int(v) for v in np.unique(np.logspace(0, 3, 16).astype(int)))
    r = _mc_dc(Ns, sig, At, pm, ps, trials, mode.startswith("Drawn"), 11)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=r["N"], y=r["CRLB"], name="CRLB σ²/N", line=dict(color=ui.C["truth"], dash="dash"))
        fig.add_scatter(x=r["N"], y=r["MLE"], name="MLE (sample mean)", mode="lines+markers",
                        line=dict(color=ui.C[1]))
        fig.add_scatter(x=r["N"], y=r["MAP"], name="MAP", mode="lines+markers", line=dict(color=ui.C[2]))
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        ui.show(ui.style(fig, "Mean squared error vs number of samples", "N (samples)", "MSE"))
    ui.controls([
        ("Noise σ", r"$\sigma$", "reading noise", "everything shifts up by σ²"),
        ("Prior μ₀, σ₀", r"$\mu_0, \sigma_0$", "MAP's prior", "MAP's behaviour at small N changes"),
        ("Where does A come from", "–", "is the prior right or not", "decides whether MAP beats the bound"),
        ("Monte Carlo trials", "–", "repeats per point", "less jagged curves"),
    ])
    ui.experiments([
        ("Defaults ('Drawn from the prior').", "Blue sits on the dashed line; orange is *below* it at small N.",
         "The MLE is efficient (it meets the CRLB). MAP is biased toward a correct prior, so the unbiased-only bound "
         "doesn't apply to it."),
        ("Switch to 'Fixed value', true A = 2.5, μ₀ = 0, σ₀ = 0.1.",
         "Orange stays far above the bound until N reaches several hundred.",
         "A confident wrong prior is a systematic error that only lots of data can wash out."),
        ("Increase σ from 1 to 4.", "The blue and dashed lines move up 16× (σ² = 16); MAP's advantage now lasts "
         "to much larger N.",
         "CRLB = σ²/N. Noisier data makes the same prior relatively more valuable, for longer."),
    ])
    ui.takeaway("The MLE sits on the CRLB line: it can't be beaten without extra information. MAP *does* go below "
                "the CRLB when the prior is right, because it is biased and the CRLB only covers unbiased "
                "estimators. With a wrong prior MAP is worse for small N, then both converge.")

# --------------------------------------------------------------------------- frequency
with t_freq:
    st.markdown(
        r"""
**The situation.** Estimate the frequency $f_0$ of a tone in noise from a window of $N$ samples, as in pitch
tracking or Doppler radar. The estimator is the peak of the **periodogram** (an FFT power spectrum, bottom plot).
The top plot repeats this many times at each SNR and compares the MSE with the CRLB (Theory eq. 10). Frequencies are
in cycles per sample (0.5 = the Nyquist frequency).
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        Nf = st.select_slider("Window length N", [16, 32, 64, 128, 256], 64,
                              help="Samples per estimate. The CRLB falls like N³.")
        f0 = st.slider("True f₀ (cycles/sample)", 0.05, 0.45, 0.1234, 0.0001, format="%.4f",
                       help="The tone's frequency. 0.1234 × 16 kHz ≈ 1974 Hz, for example.")
        tr = st.select_slider("Trials per SNR", [50, 100, 200, 400], 100, help="Repeats per point on the top plot.")
        demo_snr = st.slider("SNR for the example spectrum (dB)", -15, 30, 0,
                             help="SNR of the single example window in the bottom plot.")
    snrs = tuple(range(-15, 31, 3))
    r = _mc_freq(snrs, Nf, f0, tr, 5)
    rng = np.random.default_rng(int(demo_snr) + 100)
    sig_demo = np.sqrt(1 / (2 * 10 ** (demo_snr / 10)))
    xd = est.sinusoid(Nf, f0, 1.0, sig_demo, rng)
    fhat = est.freq_mle(xd)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=r["snr_db"], y=r["CRLB"], name="CRLB", line=dict(color=ui.C["truth"], dash="dash"))
        fig.add_scatter(x=r["snr_db"], y=r["MLE"], name="MLE (periodogram peak)", mode="lines+markers",
                        line=dict(color=ui.C[1]))
        fig.update_yaxes(type="log")
        ui.show(ui.style(fig, "Frequency MSE vs SNR", "SNR (dB)", "MSE (cycles/sample)²", height=330))
        nfft = 4096
        Px = np.abs(np.fft.rfft(xd - xd.mean(), nfft)) ** 2
        fx = np.fft.rfftfreq(nfft)
        fig2 = go.Figure()
        fig2.add_scatter(x=fx, y=10 * np.log10(Px / Px.max() + 1e-12), name="periodogram",
                         line=dict(color=ui.C[1]))
        fig2.add_vline(x=f0, line=dict(color=ui.C["truth"]), annotation_text="true f₀")
        fig2.add_vline(x=fhat, line=dict(color=ui.C[2], dash="dot"), annotation_text="estimate",
                       annotation_position="bottom right")
        ui.show(ui.style(fig2, f"One noisy window at {demo_snr} dB SNR", "frequency (cycles/sample)", "dB",
                         height=280, legend=False))
    ui.controls([
        ("Window length N", r"$N$", "samples per estimate", "lower bound (∝ 1/N³); threshold moves to lower SNR"),
        ("True f₀", r"$f_0$", "tone frequency", "little effect (away from 0 and 0.5)"),
        ("SNR for the example", r"$\eta$ in dB", "tone power vs noise", "a cleaner peak in the bottom plot"),
    ])
    ui.experiments([
        ("Look at the top plot with defaults.",
         "Above about 0 dB the blue line follows the dashed bound; below it the error shoots up.",
         "That's the threshold effect: a noise peak sometimes beats the true peak, and the estimate lands anywhere."),
        ("Set the example SNR to −12 dB and step through a few values around it.",
         "Sometimes the dotted 'estimate' line jumps far from the true f₀.",
         "Those are the outliers that dominate the MSE below threshold."),
        ("Change N from 64 to 256.", "The bound drops, and the threshold moves left (to lower SNR).",
         "A longer window integrates more signal energy relative to noise in the peak bin."),
    ])
    ui.takeaway("Above a threshold SNR, the MLE tracks the CRLB. Below it, a noise peak occasionally beats the "
                "true peak and the error jumps by orders of magnitude: the **threshold effect**. This is why pitch "
                "trackers produce octave errors in noise and why radar needs a minimum SNR.")

# --------------------------------------------------------------------------- outliers
with t_out:
    st.markdown(
        r"""
**The situation.** Same constant-level problem, but now a fraction of the readings are **impulses**: much larger
errors, like clicks, glitches or multipath. We compare the sample **mean** (the MLE if noise is Gaussian) with the
**median** (the MLE if noise is Laplacian, i.e. heavy-tailed). Theory §7.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        No = st.slider("Samples N", 5, 200, 50, key="o_N", help="Readings per estimate.")
        scale = st.slider("Outlier size (× σ)", 2.0, 50.0, 20.0, 1.0,
                          help="Outlier standard deviation as a multiple of the normal noise.")
    ps = tuple(np.round(np.linspace(0, 0.3, 13), 3))
    r = _mc_out(No, ps, scale, 3000, 9)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=r["p"] * 100, y=r["mean"], name="mean (Gaussian MLE)", mode="lines+markers",
                        line=dict(color=ui.C[1]))
        fig.add_scatter(x=r["p"] * 100, y=r["median"], name="median (Laplacian MLE)", mode="lines+markers",
                        line=dict(color=ui.C[2]))
        fig.update_yaxes(type="log")
        ui.show(ui.style(fig, "MSE as impulsive outliers contaminate the data", "outlier fraction (%)", "MSE"))
    ui.controls([
        ("Samples N", r"$N$", "readings per estimate", "both errors fall like 1/N"),
        ("Outlier size", "–", "how wild the impulses are", "the mean suffers far more; the median barely notices"),
    ])
    ui.experiments([
        ("Read the left edge (0% outliers).", "The mean is better, by about a factor of 1.57.",
         "With truly Gaussian noise the mean is optimal; the median's variance is π/2 times larger."),
        ("Read at 2–5% outliers.", "The mean's error is already several times the median's.",
         "A few huge impulses drag the average; the median ignores how large they are."),
        ("Reduce outlier size to 2×.", "The gap nearly closes.", "Mild contamination is almost Gaussian anyway."),
    ])
    ui.in_practice("Robust estimators (median, Huber loss) are used in GPS outlier rejection, audio click "
                   "removal and financial returns, where heavy tails are normal.")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/estimation.py`):")
    ui.source(est.dc_log_likelihood, est.dc_map, est.bernoulli_estimators, est.freq_mle, est.freq_crlb,
              est.monte_carlo_dc)
