import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui
from ssp.core import estimation as est

ui.header("Level 1", "Estimation: MLE, MAP, MMSE and the CRLB",
          "Given noisy samples, what single number should we report, and how accurate can it possibly be?")

t_like, t_three, t_crlb, t_freq, t_out, t_code = st.tabs(
    ["Likelihood → MLE & MAP", "MLE vs MAP vs MMSE", "Best possible accuracy (CRLB)",
     "Frequency & the threshold effect", "Break it: outliers", "Code"])


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
**Model:** a sensor reads a constant level $A$ (a DC offset, a battery voltage, a room's noise floor) in
white Gaussian noise: $x[n] = A + w[n]$, $w[n]\sim\mathcal N(0,\sigma^2)$.

* **MLE** picks the $A$ that makes the data most probable: $\hat A_{\text{ML}} = \arg\max_A \log p(\mathbf x\mid A)
  = \bar x$.
* **MAP** also weighs a prior $A\sim\mathcal N(\mu_0,\sigma_0^2)$: $\hat A_{\text{MAP}}=\arg\max_A
  [\log p(\mathbf x\mid A)+\log p(A)]$, a precision-weighted blend of $\mu_0$ and $\bar x$.

The **curvature** of the log-likelihood at its peak is the *Fisher information*: a sharper peak means a
more precise estimate.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        A_true = st.slider("True A (hidden)", -3.0, 3.0, 1.0, 0.1)
        sigma = st.slider("Noise σ", 0.1, 5.0, 2.0, 0.1)
        N = st.slider("Samples N", 1, 200, 5)
        mu0 = st.slider("Prior mean μ₀", -3.0, 3.0, 0.0, 0.1)
        s0 = st.slider("Prior σ₀ (smaller = more confident)", 0.1, 5.0, 0.5, 0.1)
        seed = st.number_input("Random seed", 0, 9999, 0, key="l_seed")
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
        ui.show(ui.style(fig, "Each curve scaled to peak at 1", "candidate A", "relative density"))
        m1, m2, m3 = st.columns(3)
        m1.metric("MLE error", f"{mle - A_true:+.2f}")
        m2.metric("MAP error", f"{mapv - A_true:+.2f}")
        m3.metric("Weight on the data", f"{(N / sigma**2) / (N / sigma**2 + 1 / s0**2):.0%}")
    ui.takeaway("The posterior is the product of likelihood and prior. With few or very noisy samples, MAP leans "
                "on the prior. As N grows the likelihood narrows like σ/√N and MAP converges to the MLE.")

# --------------------------------------------------------------------------- MLE vs MAP vs MMSE
with t_three:
    st.markdown(
        r"""
With a Gaussian posterior, MAP (the peak) and MMSE (the mean) coincide. With a **skewed** posterior they
differ. Example: estimating a click-through rate $\theta$ from $k$ clicks in $n$ impressions, with a
$\text{Beta}(\alpha,\beta)$ prior:

| Estimator | Formula | Minimises |
|---|---|---|
| MLE | $k/n$ | ignores the prior |
| MAP (posterior mode) | $\frac{k+\alpha-1}{n+\alpha+\beta-2}$ | 0–1 loss (most probable value) |
| MMSE (posterior mean) | $\frac{k+\alpha}{n+\alpha+\beta}$ | expected squared error |
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Impressions n", 1, 100, 5)
        k = st.slider("Clicks k", 0, 100, 1)
        k = min(k, n)
        a = st.slider("Prior α", 1.0, 20.0, 2.0, 0.5, key="bb_a")
        b = st.slider("Prior β", 1.0, 100.0, 30.0, 0.5, key="bb_b")
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
    ui.takeaway("With 1 click in 5 impressions, the MLE says 20%. A prior built from past campaigns (≈6%) pulls "
                "MAP and MMSE toward something believable. Ad platforms and recommenders shrink small-sample "
                "rates toward a prior like this so they don't overreact to new items.")

# --------------------------------------------------------------------------- CRLB
with t_crlb:
    st.markdown(
        r"""
The **Cramér–Rao lower bound** says no *unbiased* estimator can beat $\operatorname{var}(\hat A)\ge
1/I(A)$, where $I$ is the Fisher information. For the DC level, $\text{CRLB}=\sigma^2/N$. The sample mean
reaches it, so it is **efficient**. A Monte Carlo run shows this, and shows what a prior does.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        sig = st.slider("Noise σ", 0.2, 5.0, 1.0, 0.1, key="c_sig")
        pm = st.slider("Prior mean μ₀", -3.0, 3.0, 0.0, 0.1, key="c_mu")
        ps = st.slider("Prior σ₀", 0.05, 3.0, 0.3, 0.05, key="c_s0")
        mode = st.radio("Where does the true A come from?",
                        ["Drawn from the prior (prior is right)", "Fixed value (prior may be wrong)"])
        At = st.slider("Fixed true A", -3.0, 3.0, 1.0, 0.1, disabled=mode.startswith("Drawn"))
        trials = st.select_slider("Monte Carlo trials", [500, 2000, 5000, 20000], 5000)
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
    ui.takeaway("The MLE sits on the CRLB line: it can't be beaten without extra information. MAP *does* go below "
                "the CRLB when the prior is right, because it is biased and the CRLB only covers unbiased estimators. "
                "With a wrong prior MAP is worse for small N, then both converge.")
    ui.break_it("Choose 'Fixed value', set true A = 2.5, prior μ₀ = 0 and σ₀ = 0.1. A confident, wrong prior "
                "can cost you hundreds of samples.")

# --------------------------------------------------------------------------- frequency
with t_freq:
    st.markdown(
        r"""
**Model:** $x[n]=a\cos(2\pi f_0 n+\phi)+w[n]$. Estimating $f_0$ is pitch tracking, Doppler radar speed,
or engine RPM from a vibration sensor. The (approximate) MLE is the **periodogram peak**, refined by a local
search. Its CRLB is

$$\operatorname{var}(\hat f_0)\;\ge\;\frac{12}{(2\pi)^2\,\eta\,N(N^2-1)},\qquad \eta=\frac{a^2}{2\sigma^2}.$$

The bound falls as $N^{-3}$: longer windows help a lot.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        Nf = st.select_slider("Window length N", [16, 32, 64, 128, 256], 64)
        f0 = st.slider("True f₀ (cycles/sample)", 0.05, 0.45, 0.1234, 0.0001, format="%.4f")
        tr = st.select_slider("Trials per SNR", [50, 100, 200, 400], 100)
        demo_snr = st.slider("SNR for the example spectrum (dB)", -15, 30, 0)
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
    ui.takeaway("Above a threshold SNR, the MLE tracks the CRLB. Below it, a noise peak occasionally beats the "
                "true peak and the error jumps by orders of magnitude: the **threshold effect**. This is why pitch "
                "trackers produce octave errors in noise and why radar needs a minimum SNR.")

# --------------------------------------------------------------------------- outliers
with t_out:
    st.markdown(
        r"""
The MLE is only optimal **for the noise model you assumed**. The sample mean is the MLE for Gaussian noise. If
the noise is Laplacian (heavy-tailed), the MLE becomes the **median**. Real sensors produce impulses (clicks,
packet glitches, multipath), so the assumption matters.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        No = st.slider("Samples N", 5, 200, 50, key="o_N")
        scale = st.slider("Outlier size (× σ)", 2.0, 50.0, 20.0, 1.0)
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
    ui.break_it("With 0% outliers the mean wins (the median's variance is π/2 ≈ 1.57× larger). Add just 2–5% "
                "impulses and the mean falls apart. A mismatched noise model can matter more than the estimator.")
    ui.in_practice("Robust estimators (median, Huber loss) are used in GPS outlier rejection, audio click "
                   "removal and financial returns, where heavy tails are normal.")

with t_code:
    ui.source(est.dc_log_likelihood, est.dc_map, est.bernoulli_estimators, est.freq_mle, est.freq_crlb,
              est.monte_carlo_dc)
