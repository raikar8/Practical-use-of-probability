import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui

ui.theory_header("l12", """
A **generative adversarial network (GAN)** learns to generate data without ever writing down a likelihood.
Two networks play a game. A **generator** turns noise into fake samples; a **discriminator** tries to tell fake
from real. Each improves by exploiting the other's mistakes, and if all goes well the fakes become
indistinguishable from real data. GANs produced the first photo-realistic faces and are still the standard way
to train the neural vocoders that turn spectrograms into speech.
""")
st.markdown("**Prerequisites:** Level 10 (noise → data, neural networks, training, and the two-sample classifier test).")

ui.section("1. The idea: a forger and a detective")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11];
  z [label="noise z ~ N(0, I)", shape=plaintext];
  G [label="generator G\\n(the forger)", shape=box, style=filled, fillcolor="#eef4fc", color="#257de0"];
  fake [label="fake x = G(z)", shape=box, style="rounded,filled", fillcolor="#fdf0e8", color="#f47230"];
  real [label="real x from data", shape=box, style="rounded,filled", fillcolor="#e6f7f2", color="#00b894"];
  D [label="discriminator D\\n(the detective)", shape=box, style=filled, fillcolor="#eef4fc", color="#257de0"];
  out [label="D(x) = P(real)", shape=plaintext];
  {rank=same; fake; real}
  z -> G -> fake -> D; real -> D; D -> out;
  D -> G [style=dashed, label=" gradient: how to look more real", constraint=false];
}
"""
)
st.markdown(
    r"""
* The **discriminator** $D(x) \in (0, 1)$ is a classifier: the probability that $x$ is real. It's exactly the
  two-sample classifier of Level 10 §7, trained continuously.
* The **generator** $G(z)$ is Level 10's noise-to-data function. It never sees real data directly. It only gets
  the discriminator's gradient: "move your sample in **this** direction and I'd be more fooled".
"""
)

ui.section("2. The game")
st.markdown("Both networks share one value function. The discriminator maximises it; the generator minimises it:")
ui.eq(r"\min_G \max_D \; V(D, G) = \mathbb E_{x \sim p_{\text{data}}}\big[\log D(x)\big] + "
      r"\mathbb E_{z \sim \mathcal N(0, I)}\big[\log\big(1 - D(G(z))\big)\big]", 1)
st.markdown(
    r"""
For the discriminator, this is ordinary logistic-regression training (cross-entropy with label 1 for real and 0
for fake). The generator wants $D(G(z))$ close to 1, which makes $\log(1 - D(G(z)))$ very negative.
"""
)

ui.section("3. What the game is really minimising")
st.markdown(r"Fix the generator, whose samples have density $p_g$. For each $x$, the discriminator maximises $p_{\text{data}}(x)\log D + p_g(x)\log(1 - D)$, which gives:")
ui.eq(r"D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}", 2)
st.markdown(
    r"""
That is Bayes' rule with equal prior odds of real and fake (Level 0). Plug $D^*$ back into (1):
"""
)
ui.eq(r"V(D^*, G) = 2\,\mathrm{JS}\big(p_{\text{data}}\,\|\,p_g\big) - \log 4", 3)
st.markdown(
    r"""
where JS is the **Jensen–Shannon divergence**, a symmetric cousin of KL that is zero only when the two
distributions are equal. So with a perfect discriminator, training the generator minimises a distance between the
model and the data, **without ever computing $p_g(x)$**. At the optimum $p_g = p_{\text{data}}$ and
$D^* = 1/2$ everywhere: the detective is reduced to guessing.

Compare with Level 10 §6: maximum likelihood minimises $\mathrm{KL}(p_{\text{data}}\,\|\,p_g)$, which
punishes missing any part of the data heavily. JS is more tolerant of missing parts, which is one reason GANs
tend to produce sharp samples but can **drop whole modes** of the data.
"""
)

ui.section("4. Training in practice")
st.markdown(
    r"""
Alternate small gradient steps:

1. **Discriminator step(s):** a batch of real samples and a batch of fakes; increase $V$ (cross-entropy on
   real-vs-fake labels).
2. **Generator step:** a batch of fakes; backpropagate the discriminator's output **through** $D$ into $G$ and
   update only $G$.

**The non-saturating trick.** Early in training, fakes are obviously fake: $D(G(z)) \approx 0$. There, the
minimax generator loss $\log(1 - D)$ is almost flat, so the generator gets almost no gradient. Almost everyone
instead trains the generator to **maximise** $\log D(G(z))$: the same fixed point, but a strong gradient exactly
when the generator is doing badly. In terms of the discriminator's raw output (the logit $\ell$, with
$D = \text{sigmoid}(\ell)$):
"""
)
d = np.linspace(0.001, 0.999, 400)
fig = go.Figure()
fig.add_scatter(x=d, y=d, name="minimax: |d log(1 − D) / dℓ| = D", line=dict(color=ui.C[1], width=3))
fig.add_scatter(x=d, y=1 - d, name="non-saturating: |d log D / dℓ| = 1 − D", line=dict(color=ui.C[2], width=3))
fig.add_vrect(x0=0, x1=0.1, fillcolor=ui.rgba(ui.C["truth"], 0.15), line_width=0,
              annotation_text="early training: fakes easy to spot", annotation_position="bottom right")
ui.show(ui.style(fig, "Gradient size the generator receives, vs how convinced D is", "D(G(z))", "gradient size",
                 height=260))

ui.section("5. Why GANs are hard to train")
st.markdown(
    r"""
A GAN isn't minimising one fixed loss; it's two players chasing each other. Common failure modes, all visible in
the Lab:

* **Mode collapse.** The generator finds a few outputs that fool the current discriminator and produces only
  those. The discriminator then learns to reject them, the generator jumps to other outputs, and so on. Often
  caused by a generator that learns much faster than the discriminator.
* **Oscillation.** The losses swing up and down instead of settling; samples keep changing.
* **Vanishing gradients.** A discriminator that wins too completely gives the generator nothing to learn from
  (§4).
* **No likelihood to monitor.** The discriminator's loss says how the duel is going, not how good the samples
  are. You have to look, or use external measures.

**Common remedies:** balanced learning rates (Adam with β₁ = 0.5 is customary, as in this app), limiting how
steep the discriminator can be (Wasserstein GANs, gradient penalties, spectral normalisation), and architectural
tricks (progressive growing, StyleGAN).
"""
)

ui.section("6. GANs vs VAEs, and where GANs are used")
st.markdown(
    r"""
| | VAE (Level 11) | GAN |
|---|---|---|
| Training signal | reconstruction + KL (a likelihood bound) | a learned critic |
| Samples | often blurry | sharp |
| Covers all modes? | yes (likelihood punishes missing data) | not guaranteed (mode collapse) |
| Encoder (data → code)? | yes | no (needs extra work) |
| Training stability | stable | delicate |

**In audio and speech:** GAN-trained neural vocoders (MelGAN, HiFi-GAN and successors) turn mel-spectrograms
into waveforms in real time and are part of most text-to-speech systems. Adversarial losses are added to
speech-enhancement and codec training to remove the muffled sound that a plain squared-error loss produces
(SEGAN, MetricGAN, adversarially trained neural codecs). In images, GANs power super-resolution and style
transfer, and inspired the discriminator-based "perceptual" losses used widely today.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l12", [
    ("Tab **The duel in 1-D**: grey area vs blue curve", r"$p_{\text{data}}$ vs the generator's $p_g$"),
    ("Orange curve D(x)", r"the discriminator; compare with $D^*$ of eq. (2). It flattens to 0.5 at the optimum"),
    ("*D(real)* and *D(fake)* curves", r"the two terms of eq. (1) during training"),
    ("Tab **2-D: modes & collapse**: background shading", r"$D(x)$ over the plane"),
    ("*Generator / discriminator learning rate*", "the balance between the two players (§5)"),
    ("*Modes covered*", "detects mode collapse"),
])

ui.section("Check yourself")
ui.check([
    ("At the ideal end of training, what does the discriminator output for every input?",
     "**1/2**: p_g = p_data, so eq. (2) gives D* = p/(p + p) = 1/2. It can only guess."),
    ("Why do practitioners train the generator to maximise log D(G(z)) instead of minimising log(1 − D(G(z)))?",
     "When fakes are easy to spot (D ≈ 0), log(1 − D) is flat and gives almost no gradient; log D gives a strong one."),
    ("The generator produces perfect samples of only 3 of the 8 clusters. What is this called, and why doesn't the "
     "game's loss forbid it outright?",
     "Mode collapse. The generator is only ever judged on the samples it makes, and JS-style objectives penalise "
     "missing modes less than maximum likelihood does."),
    ("Name one place GAN training is used in speech today.",
     "Neural vocoders such as HiFi-GAN (spectrogram → waveform) in text-to-speech; adversarial losses in speech "
     "enhancement and neural audio codecs."),
])
