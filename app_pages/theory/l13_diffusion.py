import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import generative as gen

ui.theory_header("l13", """
**Diffusion models** generate the images, video, music and speech you've seen from modern AI. Their idea is
almost embarrassingly simple: **destroying** data by adding noise is easy, so learn to **undo** a little bit of
noise at a time, then start from pure noise and undo all of it. Each small step is a denoising problem, the oldest
problem in signal processing, and the best denoiser in the mean-square sense is Level 1's MMSE estimator. A
diffusion model is a stack of MMSE denoisers, one for every noise level.
""")
st.markdown("**Prerequisites:** Level 10 (neural networks and training), Level 1 (MMSE = posterior mean), Level 0 "
            "(Gaussian conditioning). Level 11's ELBO helps for §6.")

ui.section("1. The idea in one picture")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled"];
  x0 [label="x₀\\nclean data", fillcolor="#fdf0e8", color="#f47230"];
  x1 [label="x₁\\na little noise", fillcolor="#f6f1ec", color="#c9a88f"];
  xd [label="…  x_t  …\\nnoisier and noisier", fillcolor="#f2f2f2", color="#b0b0b0"];
  xT [label="x_T\\npure noise", fillcolor="#eeeeee", color="#8a8a8a"];
  x0 -> x1 [label=" add noise (fixed)"]; x1 -> xd; xd -> xT;
  x0 -> x1 [dir=back, style=dashed, color="#257de0", fontcolor="#257de0", label=" remove noise (learned)"];
  x1 -> xd [dir=back, style=dashed, color="#257de0"]; xd -> xT [dir=back, style=dashed, color="#257de0"];
}
"""
)
st.markdown(
    r"""
* **Forward process** (top arrows): a fixed recipe that blends the data with more and more Gaussian noise over
  $T$ steps (typically 100–1,000) until nothing of the data remains. No learning.
* **Reverse process** (dashed): a neural network that, given a noisy sample and the noise level, removes a little
  of the noise. Chain $T$ of these steps starting from pure noise, and out comes a new sample.

Compare with a GAN, which must produce a finished image in one shot and is judged by an adversary. A diffusion
model only ever has to make a noisy image slightly less noisy, and it's trained by plain regression.
"""
)

ui.section("2. The forward process: dissolving data in noise")
st.markdown(r"Each step shrinks the signal slightly and adds a little fresh noise, with a small variance $\beta_t$ from a fixed **schedule**:")
ui.eq(r"x_t = \sqrt{1 - \beta_t}\;x_{t-1} + \sqrt{\beta_t}\;\varepsilon_t, \qquad \varepsilon_t \sim \mathcal N(0, I)", 1)
st.markdown(
    r"""
Sums of Gaussians are Gaussian, so we can jump straight from the data to **any** step. Define
$\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \alpha_1\alpha_2\cdots\alpha_t$:
"""
)
ui.eq(r"x_t = \sqrt{\bar\alpha_t}\;x_0 + \sqrt{1 - \bar\alpha_t}\;\varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)", 2)
st.markdown(
    r"""
In signal-processing terms, $x_t$ is the clean signal at power fraction $\bar\alpha_t$ plus white Gaussian noise
at power fraction $1 - \bar\alpha_t$: a **signal-to-noise ratio** $\text{SNR}_t = \bar\alpha_t / (1 - \bar\alpha_t)$
that falls from very high to essentially zero. The schedule decides how fast.
"""
)
sch_c, sch_l = gen.schedule(200, "cosine"), gen.schedule(200, "linear")
tt = np.arange(1, 201)
fig = go.Figure()
for sch, nm, c in [(sch_c, "cosine schedule", ui.C[1]), (sch_l, "linear schedule", ui.C[2])]:
    snr = 10 * np.log10(sch["abar"] / (1 - sch["abar"]))
    fig.add_scatter(x=tt, y=snr, name=nm, line=dict(color=c, width=3))
fig.add_hline(y=0, line=dict(color=ui.C["truth"], dash="dot"), annotation_text="signal power = noise power")
ui.show(ui.style(fig, "SNR of x_t along the forward process (T = 200 steps)", "step t", "SNR (dB)", height=260))
st.markdown(
    r"""
The linear schedule destroys the signal early (0 dB after a quarter of the steps) and spends most of the rest
below −10 dB, where little is left to learn; the cosine schedule spreads the SNR more evenly across steps, so this
app uses it.
"""
)

ui.section("3. The reverse step: what we'd do if we knew the clean data")
st.markdown(
    r"""
To reverse, we need $p(x_{t-1} \mid x_t)$: given a noisy sample, what did it look like one step earlier? That
depends on the whole data distribution and is unknown. But **if we also knew $x_0$**, everything is Gaussian and
Level 0's conditioning rule gives it exactly:
"""
)
ui.eq(r"q(x_{t-1} \mid x_t, x_0) = \mathcal N\!\left(\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0 + "
      r"\frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t,\;\; "
      r"\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t\,I\right)", 3)
st.markdown(
    r"""
The mean is a weighted average of "where we are" ($x_t$) and "where the clean data is" ($x_0$). So the whole
problem reduces to one question at every step: **what is our best guess of $x_0$ from $x_t$?**
"""
)

ui.section("4. The denoiser, and why it's an MMSE estimator")
st.markdown(
    r"""
Train one network $\hat\varepsilon_\theta(x_t, t)$ to predict the noise that was added. The training loop is
almost trivially simple: pick a clean sample $x_0$, a random step $t$ and fresh noise $\varepsilon$; form $x_t$
with eq. (2); regress:
"""
)
ui.eq(r"\mathcal L(\theta) = \mathbb E_{x_0,\,t,\,\varepsilon}\Big[\big\lVert \varepsilon - \hat\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + "
      r"\sqrt{1 - \bar\alpha_t}\,\varepsilon,\; t\big)\big\rVert^2\Big]", 4)
st.markdown(
    r"""
Predicting the noise is the same as predicting the clean sample, because eq. (2) can be solved for $x_0$:
$\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}$. And a network trained with a
**squared-error** loss learns the **conditional mean** (Level 1: the MMSE estimator is the posterior mean):
"""
)
ui.eq(r"\hat x_0(x_t) \;\approx\; \mathbb E[x_0 \mid x_t] \qquad \text{(the MMSE denoiser at noise level } t\text{)}", 5)
st.markdown(
    r"""
**Tweedie's formula** connects that denoiser to the **score**, the gradient of the log-density of the noisy data,
which points toward regions where data is more likely:
"""
)
ui.eq(r"\mathbb E[x_0 \mid x_t] = \frac{x_t + (1 - \bar\alpha_t)\,\nabla_{x}\log p_t(x_t)}{\sqrt{\bar\alpha_t}}"
      r"\quad\Longleftrightarrow\quad \nabla_x \log p_t(x_t) = -\frac{\hat\varepsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}", 6)
st.markdown(
    r"""
So "denoiser", "noise predictor" and "score model" are three names for the same network. Score-based models and
DDPMs are the same idea seen from two sides.

**A link to the noise-suppression demo.** The Wiener filter and MMSE-STSA estimators there are MMSE denoisers too,
but with a **hand-designed** Gaussian or Rayleigh prior on speech and a single noise level. A diffusion model
**learns** the prior from data and has a denoiser for every noise level.
"""
)

ui.section("5. Sampling: chain the denoisers")
st.markdown(r"**DDPM (ancestral sampling).** Start with $x_T \sim \mathcal N(0, I)$. For $t = T, \dots, 1$:")
ui.eq(r"\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\;\hat\varepsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}, \qquad "
      r"x_{t-1} \sim q\big(x_{t-1} \mid x_t, \hat x_0\big) \text{ from eq. (3)}", 7)
st.markdown(
    r"""
Each step: guess the clean sample, move a little toward it, add a little fresh noise. Early steps decide the
coarse structure (which digit, which cluster); late steps add fine detail. Clipping $\hat x_0$ to the valid data
range (pixels in $[-1, 1]$) is a common practical safeguard.

**DDIM.** Drop the fresh noise and jump between a few noise levels using $\hat x_0$: a **deterministic** map from
noise to data that needs 10–50 network calls instead of hundreds. Speeding up sampling is a major research area
(distillation, consistency models, flow matching).
"""
)

ui.section("6. Why diffusion models work so well")
st.markdown(
    r"""
* **Stable training:** a regression loss (4), no adversary, no mode collapse.
* **Covers the whole distribution:** the loss (4) is, up to weighting, the ELBO of a very deep latent-variable
  model ($x_1, \dots, x_T$ are the latents), so like a VAE it's penalised for ignoring parts of the data.
* **Each step is easy:** removing a little noise is a far simpler job than producing a full image in one shot.
* **Flexible conditioning:** feed extra inputs to the denoiser (a text prompt, a noisy recording) and it samples
  from the **conditional** distribution. *Classifier-free guidance* sharpens the conditioning further.

**The price:** many network evaluations per sample.
"""
)

ui.section("7. MMSE estimate vs a posterior sample: a key distinction for enhancement")
st.markdown(
    r"""
Given a noisy observation, there are two different "clean" answers:

* the **MMSE estimate** $\mathbb E[x_0 \mid x_t]$ (one step of eq. 5): lowest average squared error, but an
  **average** over all plausible clean signals, so it's smooth and blurry (or muffled, for speech)
* a **posterior sample**: run the reverse chain from $x_t$ down to 0. Each result is a single plausible clean
  signal: sharp and realistic, but with higher squared error, because it commits to one possibility

This is the **perception–distortion trade-off**: you can't minimise squared error and look or sound perfectly
natural at the same time. It's why diffusion-based speech enhancement (score-based models conditioned on the noisy
recording, such as SGMSE+) sounds more natural than MSE-trained networks, and why it can occasionally
"hallucinate" plausible phonemes at very low SNR. The Lab's digit tab shows both answers side by side.
"""
)

ui.section("8. Where diffusion is used")
st.markdown(
    r"""
* **Images and video:** latent diffusion (Stable Diffusion and successors) runs diffusion in a VAE's latent space
  (Level 11), conditioned on text.
* **Audio:** diffusion vocoders (WaveGrad, DiffWave), text-to-speech and text-to-music systems, and diffusion
  or score-based speech enhancement, dereverberation and bandwidth extension.
* **Science:** protein structure and molecule generation, weather ensembles, inverse problems in imaging (MRI
  reconstruction with a learned prior).
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l13", [
    ("Tab **Forward: dissolving data**: *step t* slider, SNR plot", r"eq. (2) and $\text{SNR}_t$"),
    ("Tab **Reverse with a perfect denoiser**: arrows", r"from $x_t$ to $\mathbb E[x_0 \mid x_t]$, eq. (5)–(6), "
     "exact for Gaussian-mixture data"),
    ("Points flowing from noise to data", "DDPM sampling, eq. (7), or DDIM"),
    ("Tab **Learn the denoiser**: loss curve", "regression loss (4)"),
    ("*DDIM steps*", "§5: fewer network calls, some loss of quality"),
    ("Tab **Digits**: 'one-step MMSE' vs 'reverse chain'", "§7: MMSE estimate vs posterior sample"),
])

ui.section("Check yourself")
ui.check([
    ("At step t, ᾱ_t = 0.5. What is the SNR of x_t?", "ᾱ/(1 − ᾱ) = 1, i.e. **0 dB**: equal signal and noise power."),
    ("What does the network actually predict, and with which loss?",
     "The noise ε that was added to make x_t (or equivalently the clean sample), with a squared-error loss."),
    ("Why is the one-step estimate x̂₀ blurry at high noise levels?",
     "It's the posterior mean E[x₀ | x_t]: an average over all clean samples consistent with x_t."),
    ("Why do diffusion models need many network evaluations per sample, while VAEs and GANs need one?",
     "Generation chains many small denoising steps from pure noise; each step is one network call."),
    ("How does a diffusion model relate to the Wiener filter?",
     "Both are MMSE denoisers. The Wiener filter assumes a Gaussian signal prior and one noise level; a diffusion "
     "model learns the prior from data and covers every noise level."),
])
