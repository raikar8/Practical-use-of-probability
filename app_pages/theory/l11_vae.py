import streamlit as st

from ssp import ui

ui.theory_header("l11", """
A **variational autoencoder (VAE)** learns to describe each data point with a few hidden numbers (a **latent
code**) and to generate new data from new codes. It's the first deep generative model in this track and the most
"probabilistic" one: it is Level 8's variational inference with the posterior approximation produced by a
neural network. VAEs are also a working part of today's image generators: Stable Diffusion runs its diffusion
process inside a VAE's latent space.
""")
st.markdown("**Prerequisites:** Level 10 (noise → data, neural networks, training) and the ELBO from Level 7 §4 or Level 8 §7.")

ui.section("1. The idea in one picture")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11];
  x [label="image x\\n(64 pixels)", shape=box, style="rounded,filled", fillcolor="#fdf0e8", color="#f47230"];
  enc [label="encoder\\nneural net", shape=box, style=filled, fillcolor="#eef4fc", color="#257de0"];
  z [label="code z\\n(2 numbers)\\n+ uncertainty", shape=ellipse, style=filled, fillcolor="#e6f7f2", color="#00b894"];
  dec [label="decoder\\nneural net", shape=box, style=filled, fillcolor="#eef4fc", color="#257de0"];
  xh [label="reconstruction\\nx̂", shape=box, style="rounded,filled", fillcolor="#fdf0e8", color="#f47230"];
  noise [label="new z ~ N(0, I)", shape=plaintext];
  x -> enc -> z -> dec -> xh; noise -> dec [style=dashed, label=" generate"];
}
"""
)
st.markdown(
    r"""
* The **encoder** squeezes an image into a tiny code, here just 2 numbers so we can plot it.
* The **decoder** rebuilds the image from the code.
* Training makes reconstructions good **and** forces the codes of all images to fill a standard Gaussian cloud
  with no holes. Then **any** point drawn from $\mathcal N(0, I)$ decodes to a plausible image: that's the
  generator.

A plain autoencoder (no probability) also compresses and reconstructs, but its codes can be scattered anywhere
with gaps between them, and decoding a random point gives garbage. The probabilistic treatment below is what
makes a VAE generative.
"""
)

ui.section("2. The generative model")
st.markdown("Write down how data is supposed to be produced (Level 10's recipe, eq. 1):")
ui.eq(r"z \sim \mathcal N(0, I), \qquad x \mid z \sim p_\theta(x \mid z), \qquad "
      r"p_\theta(x \mid z) = \prod_{i=1}^{64} \text{Bernoulli}\big(x_i;\; \pi_i(z)\big)", 1)
st.markdown(
    r"""
The decoder network outputs 64 numbers $\pi_i(z) \in (0, 1)$: the probability that pixel $i$ is ink. (For
real-valued data such as audio features, use a Gaussian with mean $\mu_\theta(z)$ instead.) This is a
**latent-variable model**, like the Gaussian mixture of Level 7, except that the hidden variable is continuous
and the link from $z$ to $x$ is a neural network.

Two quantities we'd like are intractable, for the same reason as in Level 10 eq. (6):
"""
)
ui.eq(r"p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz, \qquad "
      r"p_\theta(z \mid x) = \frac{p_\theta(x \mid z)\,p(z)}{p_\theta(x)}", 2)
st.markdown("We need the first to train (maximum likelihood) and the second to find the code of a given image.")

ui.section("3. The encoder is an approximate posterior")
st.markdown(
    r"""
Level 8's variational inference fitted one Gaussian $q$ to one posterior by optimisation. A dataset has
thousands of images, each with its own posterior $p(z \mid x)$, so instead we train **one network** that outputs
a Gaussian for **any** $x$. This is called **amortised** inference:
"""
)
ui.eq(r"q_\phi(z \mid x) = \mathcal N\big(z;\; \mu_\phi(x),\; \mathrm{diag}\,\sigma^2_\phi(x)\big)", 3)
st.markdown(
    r"""
The encoder outputs a **mean** (the best-guess code) and a **standard deviation** (how unsure it is) for each
latent dimension.
"""
)

ui.section("4. The training objective: the ELBO")
st.markdown("The same identity as in Levels 7 and 8, now for one image $x$:")
ui.eq(r"\log p_\theta(x) = \underbrace{\mathbb E_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - "
      r"\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{ELBO}(x;\,\theta,\phi)} + "
      r"\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\big)", 4)
st.markdown(
    r"""
The last KL is never negative, so the ELBO is a **lower bound** on $\log p_\theta(x)$, and the gap is exactly how
far the encoder's Gaussian is from the true posterior. Maximising the ELBO over **both** networks does two things
at once: raises the likelihood (decoder) and tightens the bound (encoder).

The ELBO has a plain-language reading:

* **Reconstruction term** $\mathbb E_q[\log p_\theta(x \mid z)]$: decode codes drawn from the encoder's Gaussian;
  how well do they rebuild $x$? For Bernoulli pixels this is minus the binary cross-entropy.
* **KL term** $\mathrm{KL}(q_\phi(z \mid x)\,\|\,\mathcal N(0, I))$: how far this image's code distribution is
  from the standard Gaussian prior. It pulls every code toward the origin and keeps it fuzzy, which packs codes
  into one hole-free cloud.

For two Gaussians the KL has a closed form (per latent dimension, then summed):
"""
)
ui.eq(r"\mathrm{KL}\big(\mathcal N(\mu, \sigma^2)\,\|\,\mathcal N(0, 1)\big) = \tfrac12\left(\mu^2 + \sigma^2 - 1 - \log\sigma^2\right)", 5)
st.markdown(
    r"""
**Worked example.** $\mu = 1$, $\sigma = 0.5$: $\tfrac12(1 + 0.25 - 1 - \log 0.25) = \tfrac12(0.25 + 1.386) =
0.82$ nats. A code exactly at the prior ($\mu = 0$, $\sigma = 1$) costs 0. A very precise code ($\sigma \to 0$)
costs a lot: precision must be paid for by better reconstruction.
"""
)

ui.section("5. The reparameterisation trick")
st.markdown(
    r"""
The reconstruction term is an average over random $z$, and we need its gradient with respect to the encoder's
outputs. Sampling isn't differentiable, but we can move the randomness outside:
"""
)
ui.eq(r"z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)", 6)
st.markdown(
    r"""
Now $z$ is an ordinary differentiable function of $\mu$ and $\sigma$, with $\varepsilon$ just an extra input.
Backpropagation flows from the reconstruction error, through the decoder, through $z$, into the encoder:
$\partial z/\partial\mu = 1$ and $\partial z/\partial\sigma = \varepsilon$. This is exactly the trick that
made Level 8's VI work, now inside a network.

**One training step:** take a mini-batch of images → encode to $(\mu, \sigma)$ → draw one $\varepsilon$ per image
→ decode $z$ → compute $-\text{ELBO}$ = cross-entropy + KL → backpropagate → Adam step.
"""
)

ui.section("6. What you get, and the knob β")
st.markdown(
    r"""
* **A map of the data.** Plot the encoder means $\mu_\phi(x)$ of all images: similar digits land together,
  without any labels being used.
* **Generation.** Draw $z \sim \mathcal N(0, I)$ and decode. Walking along a straight line in $z$ morphs one digit
  smoothly into another (**interpolation**).
* **β-VAE.** Multiply the KL term by $\beta$:
"""
)
ui.eq(r"\mathcal L_\beta = \mathbb E_q\big[\log p_\theta(x \mid z)\big] - \beta\;\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)", 7)
st.markdown(
    r"""
  Small $\beta$ allows precise, informative codes (sharper reconstructions, a less regular map). Large $\beta$
  makes codes cheap only if they say little. Push it too far and the encoder gives up entirely: every image gets
  the code $\mathcal N(0, I)$, KL = 0, and the decoder outputs the average digit for everything. That failure is
  called **posterior collapse**.
"""
)

ui.section("7. Strengths and weaknesses")
st.markdown(
    r"""
* **Blurry samples.** With a simple pixel-wise likelihood, the decoder hedges between plausible images, and the
  average of several sharp images is a blurry one.
* **The bound isn't the likelihood.** If the true posterior is not Gaussian (e.g. two different codes explain
  an image equally well), the gap in eq. (4) never closes. The Lab shows this on a toy model.
* **But:** stable training, a fast one-pass sampler, a meaningful latent space, and an approximate likelihood.

**Where VAEs are used:** as the image compressor inside latent diffusion models (Stable Diffusion), as neural
audio codecs and speech-representation learners (VQ-VAE variants), for anomaly detection (low ELBO = unusual
input), and in molecule and material design (optimise in the latent space, then decode).
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l11", [
    ("Tab **The ELBO up close**: *Gaussian q mean / std* sliders", r"$\mu_\phi(x)$, $\sigma_\phi(x)$ for one image, eq. (3)"),
    ("The three numbers log p(x), ELBO, gap", r"eq. (4): the gap is $\mathrm{KL}(q\,\|\,p(z \mid x))$"),
    ("Tab **A VAE on digits**: latent map", r"encoder means $\mu_\phi(x)$ of every digit"),
    ("Grid of decoded images", r"decoder output $\pi(z)$ on a grid of codes"),
    ("*KL weight β*", r"$\beta$ in eq. (7)"),
    ("Reconstruction and KL curves", "the two terms of the ELBO during training"),
    ("Tab **Generate & interpolate**", r"$z \sim \mathcal N(0, I)$ → decoder; straight lines in $z$"),
])

ui.section("Check yourself")
ui.check([
    ("What does the encoder output for each image?",
     "The mean and standard deviation of a Gaussian over the latent code: q(z | x) = N(μ(x), σ²(x))."),
    ("Why is the ELBO a lower bound on log p(x)?",
     "log p(x) = ELBO + KL(q(z|x) ‖ p(z|x)), and a KL is never negative."),
    ("Compute KL(N(0, 0.5²) ‖ N(0, 1)).", "½(0 + 0.25 − 1 − ln 0.25) = ½(−0.75 + 1.386) ≈ **0.32 nats**."),
    ("Why do we write z = μ + σε instead of sampling z directly?",
     "So that z is a differentiable function of the encoder outputs and backpropagation can reach the encoder."),
    ("After training with a huge β, every reconstruction looks like the same blurry average digit. What happened?",
     "Posterior collapse: the KL penalty made it cheapest to ignore the input, so z carries no information."),
])
