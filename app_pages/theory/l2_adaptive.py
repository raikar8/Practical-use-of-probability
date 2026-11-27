import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui

ui.theory_header("l2", """
Level 1 estimated **one fixed number** from a batch of data. Real systems need more: estimate **a whole filter**
(dozens of coefficients), **one sample at a time**, while it **keeps changing**. The textbook case is the
**acoustic echo canceller** in a car or a phone. This page derives the three classic answers, **LMS, NLMS and RLS**,
from one idea: minimise the squared error.
""")

ui.section("1. The problem: identify an unknown filter on the fly")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", fillcolor="#f4f4f4"];
  x [label="far-end voice x[n]\\n(loudspeaker)", fillcolor="#eef4fc"];
  h [label="echo path h\\n(cabin acoustics, unknown)"];
  w [label="adaptive filter w\\n(our estimate of h)", fillcolor="#eef4fc"];
  s [label="+", shape=circle]; m [label="−", shape=circle];
  v [label="noise v[n]"]; e [label="error e[n]\\n(sent to the far end)", fillcolor="#fdf0e8"];
  x -> h -> s; v -> s; s -> m [label="mic d[n]"]; x -> w; w -> m [label="echo estimate y[n]"]; m -> e;
  e -> w [style=dashed, label="adapt"];
}
"""
)
st.markdown(
    r"""
The loudspeaker plays $x[n]$. The microphone hears it through the cabin's **echo path**: a short impulse
response $\mathbf h$ with $M$ taps (one per sample of delay). We collect the last $M$ input samples in the
**tap-input vector** $\mathbf u[n] = [x[n], x[n-1], \dots, x[n-M+1]]^T$. Then
"""
)
ui.eq(r"d[n] = \mathbf h^T\mathbf u[n] + v[n], \qquad y[n] = \mathbf w^T\mathbf u[n], \qquad e[n] = d[n] - y[n]", 1)
st.markdown(
    r"""
If $\mathbf w = \mathbf h$, the echo is removed and $e[n]$ is just the noise (plus any near-end speech). Our job:
**adjust $\mathbf w$ every sample so that $e[n]$ becomes as small as possible**, and keep doing it when
$\mathbf h$ changes (someone moves, a window opens).
"""
)

ui.section("2. The ideal answer: the Wiener filter")
st.markdown(r"""Choose $\mathbf w$ to minimise the mean squared error $J(\mathbf w) = E[e[n]^2]$. Expanding $e^2$ with (1):""")
ui.eq(r"J(\mathbf w) = \sigma_d^2 - 2\,\mathbf w^T\mathbf p + \mathbf w^T R\,\mathbf w,"
      r"\qquad R = E[\mathbf u\mathbf u^T],\;\; \mathbf p = E[\mathbf u\, d]", 2)
st.markdown(
    r"""
$J$ is a **bowl** (a quadratic) in the $M$-dimensional space of weights. $R$ is the input **autocorrelation
matrix** and $\mathbf p$ the cross-correlation between input and microphone. The bottom of the bowl is where
the gradient $\nabla J = 2(R\mathbf w - \mathbf p)$ is zero:
"""
)
ui.eq(r"\mathbf w_o = R^{-1}\mathbf p \qquad (\text{Wiener–Hopf equations})", 3)
st.markdown(
    r"""
That's the best possible linear filter, but it needs the statistics $R, \mathbf p$ (so all the data) and a
matrix inverse, and it can't follow a changing echo path. The adaptive filters below **reach (3) recursively and
keep tracking it**.
"""
)

ui.section("3. LMS: roll downhill using one sample at a time")
st.markdown(r"""**Steepest descent** steps against the gradient: $\mathbf w \leftarrow \mathbf w - \tfrac{\mu}{2}\nabla J$. LMS (Widrow & Hoff, 1960) replaces the unknown expectation in the gradient by its **instantaneous value** $-2e[n]\mathbf u[n]$:""")
ui.eq(r"\mathbf w[n+1] = \mathbf w[n] + \mu\, e[n]\,\mathbf u[n]", 4)
st.markdown(
    r"""
In words: **nudge each tap in proportion to (error × the input that tap saw)**. That's $2M$ multiply-adds per
sample. The **step size** $\mu$ trades speed against noise:

* **too large → divergence.** A rough stability limit is $0 < \mu < 2/(M\,P_x)$, where $P_x$ is the input power.
* **small → slow, but precise.** In steady state, $\mathbf w$ jitters around $\mathbf w_o$ by an amount
  proportional to $\mu$ (the *misadjustment*).

**Why coloured input is slow.** The bowl's shape is set by $R$. For white input, $R \propto I$ and the bowl is
round, so the downhill direction points straight at the minimum. For **coloured** input like speech (strongly
correlated neighbouring samples), the bowl is a long narrow valley. Steps zig-zag across it and crawl along it.
The speed depends on the **eigenvalue spread** $\lambda_{\max}/\lambda_{\min}$ of $R$.
"""
)


def _lms_path(rho, mu, n=400, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n + 1)
    x = np.zeros(n + 1)
    for i in range(1, n + 1):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * v[i]
    h = np.array([1.0, -0.5])
    w = np.array([-1.0, 1.0])
    path = [w.copy()]
    for i in range(1, n + 1):
        u = np.array([x[i], x[i - 1]])
        e = h @ u - w @ u
        w = w + mu * e * u
        path.append(w.copy())
    return np.array(path), h, rho


fig = go.Figure()
g1, g2 = np.meshgrid(np.linspace(-1.5, 2, 120), np.linspace(-1.5, 1.6, 120))
for rho, col, nm in ((0.0, ui.C[1], "white input"), (0.9, ui.C[2], "coloured input (ρ = 0.9)")):
    R = np.array([[1, rho], [rho, 1]])
    d1, d2 = g1 - 1.0, g2 + 0.5
    J = R[0, 0] * d1**2 + 2 * R[0, 1] * d1 * d2 + R[1, 1] * d2**2
    fig.add_contour(x=g1[0], y=g2[:, 0], z=J, contours=dict(coloring="none", start=0.05, end=4, size=0.4),
                    line=dict(color=col, width=1), showscale=False, name=f"error bowl, {nm}", hoverinfo="skip",
                    showlegend=True)
    path, h, _ = _lms_path(rho, 0.03)
    fig.add_scatter(x=path[:, 0], y=path[:, 1], mode="lines", name=f"LMS path, {nm}", line=dict(color=col, width=2))
fig.add_scatter(x=[1.0], y=[-0.5], mode="markers", name="optimum w = h", marker=dict(size=12, color=ui.C["truth"]))
fig.update_layout(hovermode="closest")
fig.update_yaxes(scaleanchor="x", scaleratio=1)
ui.show(ui.style(fig, "Two taps, 400 LMS steps: the round bowl is easy, the stretched bowl is slow",
                 "tap w₀", "tap w₁", height=380))

ui.section("4. NLMS: normalise the step")
st.markdown(r"""LMS's step depends on how loud the input is: a loud passage gives huge updates. **Normalised** LMS divides by the current input energy:""")
ui.eq(r"\mathbf w[n+1] = \mathbf w[n] + \frac{\mu}{\epsilon + \lVert\mathbf u[n]\rVert^2}\, e[n]\,\mathbf u[n]", 5)
st.markdown(
    r"""
Now $\mu$ is dimensionless and **stable for $0 < \mu < 2$** whatever the signal level. $\epsilon$ is a tiny
constant that avoids dividing by zero in silence. NLMS is the workhorse of real echo and feedback cancellers. It
still slows down on coloured input, which is why products often add pre-whitening or work in sub-bands.
"""
)

ui.section("5. RLS: solve least squares exactly, every sample")
st.markdown(r"""Instead of stepping downhill, **Recursive Least Squares** minimises an exponentially weighted sum of *all* past squared errors:""")
ui.eq(r"\mathbf w[n] = \arg\min_{\mathbf w}\sum_{k=0}^{n}\lambda^{\,n-k}\big(d[k]-\mathbf w^T\mathbf u[k]\big)^2,"
      r"\qquad 0 \ll \lambda \le 1", 6)
st.markdown(r"""The solution can be updated recursively with a matrix $P \approx (\text{weighted } R)^{-1}$:""")
ui.eq(r"\mathbf k = \frac{P\mathbf u}{\lambda + \mathbf u^TP\mathbf u},\quad "
      r"\mathbf w \leftarrow \mathbf w + \mathbf k\,e,\quad "
      r"P \leftarrow \frac{1}{\lambda}\big(P - \mathbf k\,\mathbf u^TP\big)", 7)
st.markdown(
    r"""
* **Converges in about $2M$ samples, whatever the input colour.** $P$ effectively whitens the input.
* **Costs $O(M^2)$ per sample.** For $M=32$ that's ~1,000 multiplies per sample, against ~64 for NLMS.
* **Forgetting factor $\lambda$** sets the memory: roughly the last $1/(1-\lambda)$ samples count. With
  $\lambda=0.99$ that's ~100 samples (fast tracking, noisier); with $0.999$ it's ~1,000 (precise, slow to follow a
  change).

RLS is exactly a **Kalman filter whose state is the weight vector**. $\lambda$ plays the role of process noise.
That's the bridge to Level 3.
"""
)

ui.section("6. How we measure success")
ui.eq(r"\text{misalignment}(n) = 10\log_{10}\frac{\lVert\mathbf h - \mathbf w[n]\rVert^2}{\lVert\mathbf h\rVert^2}\;\text{dB}", 8)
st.markdown(
    r"""
0 dB means "no better than doing nothing"; −30 dB means the filter matches the echo path to about 3%. Plotted
over time, this is the **learning curve**. It falls as the filter converges, jumps back up when the echo path
changes, then falls again. How fast it re-converges is the filter's **tracking** ability.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l2", [
    ("*Filter taps M*", r"length $M$ of $\mathbf h$ and $\mathbf w$"),
    ("*Echo-to-noise ratio*", r"power of $\mathbf h^T\mathbf u$ vs noise $v$ in (1); sets the best reachable misalignment"),
    ("*Input colour: AR(1) coefficient*", r"0 = white (round bowl); near 1 = speech-like, large eigenvalue spread"),
    ("*Echo path changes halfway*", r"$\mathbf h$ jumps to a new random response at the middle sample"),
    ("*LMS step μ*, *NLMS step μ*, *RLS forgetting λ*", r"$\mu$ in (4), $\mu$ in (5), $\lambda$ in (6)–(7)"),
    ("Top plot", r"the learning curve (8) for each algorithm"),
    ("Bottom plot (bars vs dots)", r"true $\mathbf h$ vs the final $\mathbf w$"),
    ("Tab **Wiener: the target**", r"the batch solution (3), computed from the first half of the data"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **μ too large.** LMS blows up. The lab shows the misalignment shooting up and the run stopping.
* **Filter too short.** If $M$ is shorter than the real echo path, part of the echo can never be cancelled.
* **Double-talk (real products).** When the near-end person speaks, $e[n]$ contains their voice and the filter
  would adapt to it. Echo cancellers detect this and freeze adaptation. That's a detection problem, and HMM/VAD
  ideas from Level 4 apply.
"""
)

ui.section("Check yourself")
ui.check([
    ("RLS with λ = 0.995: roughly how many past samples does it remember?",
     "1/(1 − 0.995) = **200 samples** (12.5 ms at 16 kHz)."),
    ("Unit-power white input, M = 16 taps. Roughly what is the largest stable LMS step?",
     "μ < 2/(M·Pₓ) = 2/16 ≈ **0.125**. In practice use much less, e.g. 0.01."),
    ("Why does NLMS slow down on speech but RLS doesn't?",
     "Speech is coloured, so R has a large eigenvalue spread and gradient steps zig-zag. RLS uses P ≈ R⁻¹, which "
     "undoes the colouring."),
    ("Your echo canceller stalls at −15 dB misalignment even with tiny μ. Name two possible causes.",
     "Echo-to-noise ratio around 15 dB (the noise floor), or a filter shorter than the echo path (undermodelling)."),
])
