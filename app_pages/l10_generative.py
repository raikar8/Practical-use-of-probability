import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import generative as gen
from ssp.core import nn

ui.lab_intro(
    "l10",
    goal="Build intuition for the three ingredients of every deep generative model: turning noise into data with "
         "a function, neural networks as tunable functions, and fitting a distribution by maximum likelihood.",
    on_screen=[
        "**Noise in, data out:** shape a function by hand and see the distribution it turns Gaussian noise into "
        "(Theory §3)",
        "**A neural network learns a curve:** gradient descent tuning a small network, frame by frame "
        "(Theory §4–5)",
        "**Learning a distribution:** a tiny invertible generator (a normalising flow) trained by maximum "
        "likelihood, and a classifier test of how good it is (Theory §6–7)",
    ],
    start_here="In *Noise in, data out*, raise the *tanh height* slider and watch one bell split into two peaks.",
)

t_push, t_nn, t_flow, t_code = st.tabs(["Noise in, data out", "A neural network learns a curve",
                                         "Learning a distribution", "Code"])

# --------------------------------------------------------------------------- push noise through g
with t_push:
    st.markdown(
        r"""
**The situation.** We draw 5,000 Gaussian noise values $z$ and compute $x = g(z)$ with
$g(z) = a\,z + b\,\tanh\big(c\,(z - s)\big) + m$. Every term increases with $z$, so $g$ is invertible and the
density of $x$ follows Theory eq. 2 (orange curve). Your challenge: make the samples match the grey target, then
check whether a classifier can still tell them apart.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        a = st.slider("Slope a", 0.05, 2.0, 1.0, 0.05, help="Linear part: stretches the whole distribution.")
        b = st.slider("tanh height b", 0.0, 3.0, 0.0, 0.1, help="Size of the S-shaped step. Large = two peaks.")
        cc = st.slider("tanh sharpness c", 0.2, 6.0, 2.0, 0.1, help="How abrupt the step is.")
        sh = st.slider("Step position s", -1.5, 1.5, 0.0, 0.1, help="Where (in z) the step happens: moves "
                                                                    "probability between the two peaks.")
        mshift = st.slider("Shift m", -3.0, 3.0, 0.0, 0.1, help="Moves everything left or right.")
        target_on = st.toggle("Show the target and test the match", True,
                              help="Compare with samples from the 'two peaks' dataset.")
    rng = np.random.default_rng(0)
    z = rng.standard_normal(5000)

    def g(zz):
        return a * zz + b * np.tanh(cc * (zz - sh)) + mshift

    def gprime(zz):
        return a + b * cc / np.cosh(cc * (zz - sh)) ** 2
    x = g(z)
    zg = np.linspace(-4, 4, 800)
    with c2:
        fig = make_subplots(rows=1, cols=3, column_widths=[0.22, 0.33, 0.45],
                            subplot_titles=["noise z ~ N(0, 1)", "the function x = g(z)", "the result: x"])
        fig.add_histogram(x=z, nbinsx=60, histnorm="probability density", marker_color=ui.rgba(ui.C[7], 0.6),
                          showlegend=False, row=1, col=1)
        fig.add_scatter(x=zg, y=g(zg), line=dict(color=ui.C[1], width=3), showlegend=False, row=1, col=2)
        if target_on:
            tgt = gen.data_1d("two peaks", 5000, np.random.default_rng(1))
            fig.add_histogram(x=tgt, nbinsx=80, histnorm="probability density", name="target data",
                              marker_color=ui.rgba(ui.C["truth"], 0.45), row=1, col=3)
        fig.add_histogram(x=x, nbinsx=80, histnorm="probability density", name="samples of x",
                          marker_color=ui.rgba(ui.C[1], 0.5), row=1, col=3)
        fig.add_scatter(x=g(zg), y=np.exp(-0.5 * zg**2) / np.sqrt(2 * np.pi) / gprime(zg),
                        name="density by eq. (2)", line=dict(color=ui.C[2], width=2.5), row=1, col=3)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="z", row=1, col=1)
        fig.update_xaxes(title="z", row=1, col=2)
        fig.update_xaxes(title="x", range=[-6, 6], row=1, col=3)
        ui.show(ui.style(fig, None, height=330, legend_below=True))
        if target_on:
            acc = gen.two_sample_classifier(tgt[:, None], x[:, None], steps=500)["accuracy"]
            st.metric("Two-sample classifier accuracy (real vs yours)", f"{acc:.0%}",
                      help="A small network trained to tell target samples from yours, tested on new ones. "
                           "50% = it can't tell them apart; 100% = obviously different.")
    ui.takeaway("A smooth function can turn a single bell curve into almost any shape: where g is steep, "
                "probability is spread thin; where it's flat, probability piles up. Generative models learn this "
                "function instead of hand-tuning it.")
    ui.controls([
        ("Slope a", "$a$", "overall stretch", "wider distribution"),
        ("tanh height b", "$b$", "size of the S-step", "a deeper gap, two separate peaks"),
        ("tanh sharpness c", "$c$", "how sudden the step is", "narrower peaks, emptier gap"),
        ("Step position s", "$s$", "which share of z goes to each side", "more mass in the left peak"),
        ("Shift m", "$m$", "location", "moves everything right"),
    ])
    ui.experiments([
        ("Start from the defaults (b = 0).", "x is just a Gaussian, and the classifier tells it from the target "
         "easily (≈ 79% accuracy).", "A straight line can only stretch and shift a bell."),
        ("Set b = 1.8, a = 0.35, c = 2.5.", "Two peaks appear, matching the theory figure.",
         "The flat parts of tanh collect probability."),
        ("Now tune the sliders to match the target.", "a = 0.5, b = 1.4, c = 4, s = 0, m = −0.2 brings the "
         "accuracy down to ≈ 57%. Getting to 50% by hand is hard.", "That's the job of training: adjust the "
         "function automatically until a test can't tell the difference (third tab)."),
    ])

# --------------------------------------------------------------------------- neural net regression
with t_nn:
    st.markdown(
        r"""
**The situation.** 200 noisy samples of an unknown curve. A small neural network (Theory eq. 3) starts with random
weights, and Adam (a refined version of eq. 4) adjusts them to reduce the squared error. Press ▶ to watch
training.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        target = st.selectbox("Curve to learn", ["wave", "step", "two bumps"], help="The unknown function.")
        width = st.select_slider("Hidden units per layer", [2, 4, 8, 16, 32, 64], 16, help="Width of each layer.")
        depth = st.slider("Hidden layers", 1, 3, 2, help="Number of hidden layers.")
        act = st.radio("Activation σ", ["tanh", "relu"], horizontal=True, help="The bend applied in each unit.")
        lr = st.select_slider("Learning rate η", [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], 0.01,
                              help="Step size of gradient descent (Theory eq. 4).")
        steps = st.select_slider("Training steps", [100, 300, 1000, 3000], 1000, help="Gradient steps.")
        nseed = st.number_input("Initialisation seed", 0, 9999, 0, key="nn_seed", help="Different random weights.")
    rng = np.random.default_rng(5)
    xs = np.sort(rng.uniform(-3, 3, 200))
    f = {"wave": lambda v: np.sin(2 * v) + 0.3 * v, "step": lambda v: np.where(v > 0.5, 1.0, -0.5),
         "two bumps": lambda v: np.exp(-4 * (v + 1.2) ** 2) - 0.8 * np.exp(-6 * (v - 1.3) ** 2)}[target]
    ys = f(xs) + 0.1 * rng.standard_normal(200)
    xg = np.linspace(-3.3, 3.3, 300)

    @st.cache_data(show_spinner="Training…")
    def _fit(target, width, depth, act, lr, steps, seed):
        r = nn.fit_regression(xs, ys, [width] * depth, steps, lr=lr, act=act, seed=seed, snapshots=40, x_grid=xg)
        return r["loss"], r["snaps"], r["snap_steps"], r["net"].n_params()
    loss, snaps, ssteps, npar = _fit(target, width, depth, act, lr, steps, int(nseed))
    with c2:
        def fr(sn):
            return [go.Scatter(x=xg, y=sn, line=dict(color=ui.C[1], width=3), name="network output")]
        fig = go.Figure()
        fig.add_scatter(x=xs, y=ys, mode="markers", name="data", marker=dict(color=ui.C[2], size=5, opacity=0.6))
        fig.add_scatter(x=xg, y=f(xg), name="true curve", line=dict(color=ui.C["truth"], width=2, dash="dash"))
        fig.add_trace(fr(snaps[-1])[0])
        fig.frames = [go.Frame(data=fr(s), traces=[2], name=str(k)) for s, k in zip(snaps, ssteps)]
        fig.update_yaxes(range=[-2.6, 2.6])
        ui.play_controls(fig, [str(k) for k in ssteps], 120, "step ", start_at_end=True)
        ui.show(ui.style(fig, "The network's curve during training", "x", "y", height=380))
        f2 = go.Figure(go.Scatter(y=loss, line=dict(color=ui.C[1]), name="loss"))
        f2.add_hline(y=0.01, line=dict(color=ui.C["truth"], dash="dot"), annotation_text="noise level (0.1²)")
        f2.update_yaxes(type="log")
        ui.show(ui.style(f2, "Mean squared error during training", "step", "loss", height=220, legend=False))
        m = st.columns(3)
        m[0].metric("Parameters (weights + biases)", npar)
        m[1].metric("Final loss", f"{np.mean(loss[-20:]):.4f}")
        m[2].metric("Loss / noise floor", f"{np.mean(loss[-20:]) / 0.01:.1f}×",
                    help="1× = as good as possible: the remaining error is the measurement noise.")
    ui.takeaway("Training is nothing but many small downhill steps on the loss. The learning rate sets the "
                "step size: too small and nothing happens in time, too large and the loss jumps around. "
                "Width and depth set how many bends the curve can have.")
    ui.controls([
        ("Hidden units, layers", r"sizes of $W^{(\ell)}$", "network capacity", "more bends possible"),
        ("Activation σ", r"$\sigma$", "tanh (smooth) or ReLU (piecewise linear)", "–"),
        ("Learning rate η", r"$\eta$", "gradient step size", "faster, until it becomes unstable"),
        ("Training steps", "–", "how long to train", "closer to the noise floor"),
        ("Initialisation seed", "–", "random starting weights", "a different path, sometimes a different result"),
    ])
    ui.experiments([
        ("Defaults (wave, 2×16 tanh, η = 0.01).", "The curve starts nearly flat and bends into shape; the loss "
         "falls to near the noise floor (0.01).", "Each step nudges all 321 weights to reduce the error."),
        ("Set the learning rate to 0.0003.", "After 1,000 steps the curve is still far off.",
         "Steps too small to get anywhere in the time allowed."),
        ("Set 2 hidden units, 1 layer.", "The network can't follow the wave, however long it trains.",
         "Two tanh bends are all it has: too little capacity."),
        ("Choose *step* and compare tanh with ReLU.", "Both approximate the jump with a steep ramp.",
         "A smooth network can only approximate a discontinuity with a steep slope."),
    ])

# --------------------------------------------------------------------------- 1-D flow by MLE
with t_flow:
    st.markdown(
        r"""
**The situation.** Now let the computer find $g$. We use an **invertible** generator, so its density is exact
(Theory eq. 2), and maximise the log-likelihood of 2,000 data points (eq. 5) by gradient descent. Technically
this is a small **normalising flow**: $z = f(x) = e^{\alpha}x + \sum_i e^{\beta_i}\tanh(e^{\gamma_i}x + d_i)$, with
$f$ always increasing. Sampling runs it backwards: draw $z$, solve $f(x) = z$.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        ds = st.selectbox("Data", gen.DATA_1D, help="The distribution to learn.")
        units = st.select_slider("tanh units", [1, 2, 4, 8, 16], 8, help="How many S-shaped pieces f can use.")
        fsteps = st.select_slider("Training steps", [100, 300, 1000, 2000], 2000, help="Gradient steps.")
        flr = st.select_slider("Learning rate", [0.005, 0.02, 0.05, 0.1], 0.05, help="Adam step size.")
    X = gen.data_1d(ds, 2000, np.random.default_rng(2))
    grid = np.linspace(X.min() - 1.5, X.max() + 1.5, 400)

    @st.cache_data(show_spinner="Training the flow…")
    def _flow(ds, units, steps, lr):
        r = gen.train_flow1d(X, units, steps, lr, 0, snapshots=40, grid=grid)
        S = r["flow"].sample(2000, np.random.default_rng(3))
        gauss = np.random.default_rng(3).normal(X.mean(), X.std(), 2000)
        acc_f = gen.two_sample_classifier(X[:, None], S[:, None])["accuracy"]
        acc_g = gen.two_sample_classifier(X[:, None], gauss[:, None])["accuracy"]
        return r["loss"], r["snaps"], r["snap_steps"], S, acc_f, acc_g
    floss, fsn, fst, S, acc_f, acc_g = _flow(ds, units, fsteps, flr)
    gauss_nll = 0.5 * np.log(2 * np.pi * X.var()) + 0.5
    with c2:
        def fr(d):
            return [go.Scatter(x=grid, y=d, line=dict(color=ui.C[1], width=3), name="learned density p_θ(x)")]
        fig = go.Figure()
        fig.add_histogram(x=X, nbinsx=70, histnorm="probability density", name="training data",
                          marker_color=ui.rgba(ui.C["truth"], 0.5))
        fig.add_trace(fr(fsn[-1])[0])
        fig.frames = [go.Frame(data=fr(d), traces=[1], name=str(k)) for d, k in zip(fsn, fst)]
        fig.update_yaxes(range=[0, max(d.max() for d in fsn[-5:]) * 1.25])
        ui.play_controls(fig, [str(k) for k in fst], 120, "step ", start_at_end=True)
        ui.show(ui.style(fig, "Maximum likelihood, step by step", "x", "density", height=360))
        sub = make_subplots(rows=1, cols=2, subplot_titles=["negative log-likelihood per point",
                                                            "samples generated by the trained flow"])
        sub.add_scatter(y=floss, line=dict(color=ui.C[1]), showlegend=False, row=1, col=1)
        sub.add_hline(y=gauss_nll, line=dict(color=ui.C[2], dash="dash"), row=1, col=1,
                      annotation_text="best single Gaussian")
        sub.add_histogram(x=X, nbinsx=60, histnorm="probability density", marker_color=ui.rgba(ui.C["truth"], 0.45),
                          showlegend=False, row=1, col=2)
        sub.add_histogram(x=S, nbinsx=60, histnorm="probability density", marker_color=ui.rgba(ui.C[1], 0.5),
                          showlegend=False, row=1, col=2)
        sub.update_layout(barmode="overlay")
        sub.update_xaxes(title="step", row=1, col=1)
        sub.update_xaxes(range=[grid[0], grid[-1]], row=1, col=2)
        ui.show(ui.style(sub, None, height=260, legend=False, legend_below=True))
        m = st.columns(3)
        m[0].metric("Final NLL per point", f"{floss[-1]:.3f}", help="Lower = the data is more probable.")
        m[1].metric("Classifier: data vs flow samples", f"{acc_f:.0%}", help="50% = indistinguishable.")
        m[2].metric("Classifier: data vs best Gaussian", f"{acc_g:.0%}")
    ui.takeaway("Maximum likelihood pulls the model's density up where data is and (because it must integrate to "
                "1) down everywhere else. With an invertible generator the likelihood is exact, and the trained "
                "flow's samples fool a classifier. In many dimensions, invertibility is the hard part; Levels 11–13 "
                "drop it.")
    ui.controls([
        ("Data", r"$p_{\text{data}}$", "the distribution to learn", "–"),
        ("tanh units", "number of $i$ terms", "flexibility of f", "more peaks and shapes possible"),
        ("Training steps, learning rate", "eq. (4)", "optimisation", "–"),
    ])
    ui.experiments([
        ("Defaults (two peaks, 8 units), press ▶.", "The density starts as one broad bump, then splits; the "
         "classifier accuracy ends at ≈ 49%, versus ≈ 74% for the best single Gaussian.",
         "The flow learns the function that maps one bell onto two peaks."),
        ("Set 1 tanh unit with *three peaks*.", "It can't make three peaks: the NLL stays at 2.24, no better than a "
         "single Gaussian, versus 1.64 with 8 units.",
         "Each tanh unit can create about one 'step' in f, which is one gap between peaks."),
        ("Choose *skewed (log-normal)*.", "The fit captures the long right tail.",
         "Flows can stretch one side more than the other."),
    ])

with t_code:
    st.markdown("The exact code this page runs (from `ssp/core/nn.py` and `ssp/core/generative.py`):")
    ui.source(nn.MLP, nn.Adam, gen.Flow1D, gen.train_flow1d, gen.two_sample_classifier)
