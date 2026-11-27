import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui

ui.theory_header("l10", """
This level starts the **deep generative models** track and assumes no previous knowledge of neural networks.
By the end of Level 13 you'll know how variational autoencoders, GANs and diffusion models work, the
probability behind each, and why they look so different. This page builds the shared foundation: what a
generative model is, how random noise can be turned into realistic data, what a neural network is, and how one
is trained.
""")

ui.section("1. What is a generative model?")
st.markdown(
    r"""
Most of machine learning you've heard of is **discriminative**: given an input $x$ (an image, a sound), predict a
label $y$ ("cat", "speech"). It learns $p(y \mid x)$.

A **generative model** learns the distribution of the data itself, $p(x)$: what real images, voices or sensor
signals look like. Once you have it you can:

* **sample** new data: synthetic speech, images, test signals, data augmentation
* **score** data: how typical is this input? (anomaly and fault detection)
* use it as a **prior** to clean up measurements: denoising, filling in missing audio, super-resolution. Level 1's
  MAP estimation with a far richer prior.

You've already used generative models in this app. An HMM generates dice rolls; a state-space model generates
noisy GPS readings. Those were **designed by hand** with a few parameters. That's impossible for images or speech:
nobody can write down $p(x)$ for "a handwritten digit" with 64 pixels, let alone a photo with millions. So we
**learn** it from examples.
"""
)

ui.section("2. Two ways to 'have' a distribution")
st.markdown(
    r"""
* **A density** you can evaluate: give it any $x$, get $p(x)$. Needed for scoring and for maximum likelihood.
* **A sampler**: a procedure that produces new $x$'s distributed like the data.

For a Gaussian you have both. For complex data you often get only one cheaply, and the three famous model
families differ exactly in which one they make easy.
"""
)

ui.section("3. The central trick: transform simple noise")
st.markdown(
    r"""
Computers can draw standard Gaussian noise $z \sim \mathcal N(0, I)$. Push it through a function and you get a
new distribution:
"""
)
ui.eq(r"z \sim \mathcal N(0, I), \qquad x = g(z)", 1)
st.markdown(
    r"""
If $g$ is flexible enough, $x$ can follow almost any distribution. Every model in Levels 10–13 is a recipe for
**learning** a $g$ that turns noise into realistic data.

In 1-D with an increasing $g$, the density of $x$ follows from conservation of probability: the probability in
a small interval $dz$ ends up in an interval $dx = g'(z)\,dz$, so it's **diluted** where $g$ stretches and
**concentrated** where $g$ is flat:
"""
)
ui.eq(r"p_x(x) = p_z(z)\,\left|\frac{dz}{dx}\right| = \frac{p_z\big(g^{-1}(x)\big)}{\big|g'\big(g^{-1}(x)\big)\big|}", 2)
z = np.linspace(-3.5, 3.5, 600)
gz = 0.35 * z + 1.8 * np.tanh(2.5 * z)
gp_ = 0.35 + 1.8 * 2.5 / np.cosh(2.5 * z) ** 2
pz = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
fig = make_subplots(rows=1, cols=2, subplot_titles=["g(z) = 0.35 z + 1.8 tanh(2.5 z)",
                                                    "density of x = g(z): two peaks from one bell"])
fig.add_scatter(x=z, y=gz, line=dict(color=ui.C[1], width=3), showlegend=False, row=1, col=1)
fig.add_scatter(x=gz, y=pz / gp_, line=dict(color=ui.C[2], width=3), showlegend=False, row=1, col=2)
fig.update_xaxes(title="z (Gaussian noise)", row=1, col=1)
fig.update_yaxes(title="x", row=1, col=1)
fig.update_xaxes(title="x", row=1, col=2)
ui.show(ui.style(fig, None, height=280, legend=False))
st.markdown(
    r"""
The function is steep near $z = 0$, so the many $z$'s near zero get spread thinly, and flat for large $|z|$, so
probability piles up near $x = \pm 2$. A single Gaussian becomes a two-peaked distribution.
"""
)

ui.section("4. Neural networks: flexible functions with knobs")
st.markdown(
    r"""
We need a family of functions $g_\theta$ flexible enough to describe "how noise becomes a digit", with
parameters $\theta$ we can tune. A **neural network** is such a family. One **layer** takes a vector $h$,
mixes its entries linearly, and bends the result with a fixed nonlinear function $\sigma$ (the **activation**):
"""
)
ui.eq(r"h^{(\ell+1)} = \sigma\big(W^{(\ell)}\,h^{(\ell)} + b^{(\ell)}\big)", 3)
st.markdown(
    r"""
* $W$ (a matrix of **weights**) and $b$ (a vector of **biases**) are the knobs; all of them together are $\theta$.
* $\sigma$ is applied to each entry: $\tanh$, or $\text{ReLU}(a) = \max(a, 0)$. Without it, stacking layers would
  only ever give a linear function.
* A **multilayer perceptron (MLP)** stacks a few layers: input → hidden layers → output. Each hidden entry is a
  "unit" or "neuron".

How flexible is it? Each ReLU unit adds one bend to a piecewise-linear function, so with enough units an MLP can
approximate any reasonable function (the *universal approximation* property). The networks in this app have a
few hundred units and run on a laptop CPU; image generators have billions of parameters, but the idea is the
same.
"""
)

ui.section("5. Training: walk downhill on a loss")
st.markdown(
    r"""
Choose a **loss** $\mathcal L(\theta)$ that measures how bad the current network is: the squared error of a fit,
or minus the log-likelihood of the data. Then repeatedly move the knobs a little in the direction that reduces
the loss fastest:
"""
)
ui.eq(r"\theta \;\leftarrow\; \theta - \eta\,\nabla_\theta \mathcal L(\theta)", 4)
st.markdown(
    r"""
* $\eta$ is the **learning rate** (step size). Too small: slow. Too large: overshoot and diverge. This is the
  same trade-off as the LMS step size in Level 2. LMS *is* gradient descent on a one-layer linear network.
* $\nabla_\theta \mathcal L$, the gradient, is computed by **backpropagation**: the chain rule from calculus,
  applied layer by layer from the output back to the input. Example with one unit: $y = w_2 \tanh(w_1 x)$ and
  $\mathcal L = (y - t)^2$. Then $\partial\mathcal L/\partial w_2 = 2(y - t)\tanh(w_1x)$ and
  $\partial\mathcal L/\partial w_1 = 2(y - t)\,w_2\,(1 - \tanh^2(w_1 x))\,x$. A computer does this
  bookkeeping for millions of weights.
* **Stochastic** gradient descent estimates the gradient from a random **mini-batch** of examples instead of
  the whole dataset: noisier steps, much cheaper.
* **Adam**, used everywhere in this app, adapts the step size of each parameter using running averages of its
  gradient and squared gradient.

Every network in Levels 10–13 is implemented in `ssp/core/nn.py` with hand-written backpropagation, checked
against numerical derivatives by the tests. There's no deep-learning library underneath.
"""
)

ui.section("6. Learning a distribution = maximum likelihood")
st.markdown(
    r"""
How should we choose $\theta$ so that $g_\theta$ produces realistic data? The natural answer is Level 1's
**maximum likelihood**: make the training examples as probable as possible under the model. That is also the
same as making the model distribution close to the data distribution:
"""
)
ui.eq(r"\max_\theta \frac{1}{N}\sum_{n=1}^{N} \log p_\theta(x_n) \;\;\Longleftrightarrow\;\; "
      r"\min_\theta \mathrm{KL}\big(p_{\text{data}}\,\|\,p_\theta\big)", 5)
st.markdown(
    r"""
For an **invertible** $g$, eq. (2) gives $\log p_\theta(x)$ exactly, and we can train by gradient descent on
(5). That's a **normalising flow**, and the Lab trains a small one in 1-D.

**The obstacle.** For realistic generators $g$ is not invertible: the noise $z$ usually has fewer dimensions
than $x$ (a few dozen numbers can describe a face, while an image has millions of pixels), and networks aren't
built to be invertible. Then
"""
)
ui.eq(r"p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz", 6)
st.markdown(
    r"""
is an integral over all possible noise vectors, which is exactly the intractable normalising integral of Level 8.
The three big families are three ways around it:

| Family | Idea | Density? | Sampling | Typical strength |
|---|---|---|---|---|
| **VAE** (Level 11) | learn an *encoder* that guesses $z$ from $x$, and maximise a **lower bound** on $\log p(x)$ (Level 8's ELBO) | a bound | one network pass | smooth, organised latent space; fast |
| **GAN** (Level 12) | skip the density: train a second network to tell real from fake and make the generator fool it | no | one network pass | sharp samples; fast |
| **Diffusion** (Level 13) | gradually add noise to data, learn to **undo it step by step** with a denoiser | a bound | many network passes | best quality and diversity; stable training |
| **Normalising flow** | an invertible $g$, exact likelihood via eq. (2) | exact | one pass | exact likelihoods; constrained architecture |
"""
)

ui.section("7. How do you judge a generator?")
st.markdown(
    r"""
In 1-D you can compare histograms. In 64 or a million dimensions you can't. Common approaches:

* **Likelihood on held-out data**, if the model provides one (flows; bounds for VAEs and diffusion).
* **A two-sample classifier test:** train a classifier to tell real examples from generated ones. If its
  accuracy on new examples is about 50%, it can't tell them apart. This idea **is** the GAN of Level 12.
* **Feature-based distances** such as FID, which compare statistics of real and generated images in the feature
  space of a pretrained network. Fine for images, but they measure what that network notices.
* **Listening or viewing tests**: still the final word for speech and images.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l10", [
    ("Tab **Noise in, data out**: the function sliders", r"$g$ in eq. (1); the orange curve is eq. (2)"),
    ("Tab **A neural network learns a curve**: hidden units, learning rate", r"eq. (3) and (4)"),
    ("Loss curve going down", r"gradient descent on the squared error"),
    ("Tab **Learning a distribution**: blue density moving toward the histogram", r"maximum likelihood (5) for an "
     "invertible g (a normalising flow)"),
    ("*Two-sample classifier accuracy*", "§7: ≈ 50% means real and generated are indistinguishable to it"),
])

ui.section("Check yourself")
ui.check([
    ("What does a generative model learn, compared with a classifier?",
     "The distribution of the data, p(x), instead of p(label | x). It can produce and score data, not only label it."),
    ("x = 2z + 1 with z ~ N(0, 1). What is the distribution of x, using eq. (2)?",
     "g′ = 2 everywhere, so p_x(x) = N((x − 1)/2; 0, 1)/2: **N(1, 2²)**."),
    ("Why does a stack of layers need a nonlinear activation?",
     "A composition of linear maps is linear. Without σ, any depth gives just one linear function."),
    ("Why can't we simply maximise log p_θ(x) for a GAN-style generator x = g_θ(z)?",
     "When g isn't invertible, p_θ(x) is an integral over all z (eq. 6) with no formula. VAEs bound it, GANs avoid "
     "it, diffusion models break it into easy denoising steps."),
])
