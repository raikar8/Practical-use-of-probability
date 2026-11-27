import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import approx, mixtures as mx

ui.lab_intro(
    "l7",
    goal="Watch EM uncover hidden structure: clusters nobody labelled, the right number of clusters, and "
         "which simple model explains which part of a curve.",
    on_screen=[
        "**EM step by step:** press ▶ and watch soft labels and Gaussian ellipses update together, with the "
        "log-likelihood climbing (Theory §3–5)",
        "**How many clusters?** likelihood keeps rising with K while BIC finds the right K (Theory §5)",
        "**Mixture of experts:** straight-line experts and a gate that learns where each one applies (Theory §6)",
    ],
    start_here="In EM step by step, press ▶ Play. Then choose *five clusters*, *Initialisation = random points*, "
               "and try a few seeds.",
)

COL = [ui.C[i] for i in (1, 2, 3, 4, 5, 6, 7, 8)]
t_em, t_k, t_moe, t_code = st.tabs(["EM step by step", "How many clusters?", "Mixture of experts", "Code"])


@st.cache_data(show_spinner=False)
def _data(name):
    return mx.blobs(name, 600, np.random.default_rng(0))


@st.cache_data(show_spinner=False)
def _em(name, K, init, seed, n_iter):
    X, _ = _data(name)
    r = mx.gmm_em(X, K, n_iter, seed=seed, init=init)
    frames = []
    for h in r["hist"]:
        logr, _ = mx.gmm_log_resp(X, h["w"], h["means"], h["covs"])
        frames.append((np.exp(logr), h))
    return frames


# --------------------------------------------------------------------------- EM animation
with t_em:
    st.markdown(
        r"""
**The situation.** 600 unlabelled 2-D points that came from a few hidden groups. EM must find the groups' centres,
shapes and sizes. Each point is coloured by its most likely component, and faded when EM is unsure
(responsibility near 1/K). Ellipses show each Gaussian's 2σ outline.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        name = st.selectbox("Dataset", ["three round clusters", "stretched & overlapping", "five clusters"],
                            help="The hidden groups that generated the points.")
        K = st.slider("Components K", 1, 8, 3, help="How many Gaussians EM fits. The data's true number is in "
                                                   "the dataset name.")
        init = st.radio("Initialisation", ["random points", "k-means++"], horizontal=True,
                        help="Random points: start the means at random data points. k-means++: spread them out.")
        seed = st.number_input("Random seed", 0, 9999, 0, key="em_seed", help="A different starting guess.")
        n_iter = st.slider("EM iterations", 5, 60, 30, help="Number of E+M steps shown in the animation.")
    X, true_lab = _data(name)
    frames = _em(name, K, init, int(seed), n_iter)
    with c2:
        def traces(resp, h):
            lab = resp.argmax(1)
            conf = resp.max(1)
            out = [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                              marker=dict(size=6, color=[COL[k % 8] for k in lab],
                                          opacity=np.clip((conf - 1 / max(K, 2)) / (1 - 1 / max(K, 2)), 0.15, 1)
                                          if K > 1 else 0.6))]
            for k in range(8):
                if k < K:
                    e = approx.ellipse(h["means"][k], h["covs"][k])
                    out.append(go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", line=dict(color=COL[k], width=2.5),
                                          name=f"component {k + 1}: w = {h['w'][k]:.2f}", showlegend=False))
                else:
                    out.append(go.Scatter(x=[None], y=[None], mode="lines", showlegend=False))
            return out
        fig = go.Figure(data=traces(*frames[0]),
                        frames=[go.Frame(data=traces(*f), name=str(i)) for i, f in enumerate(frames)])
        fig.update_xaxes(range=[X[:, 0].min() - 1, X[:, 0].max() + 1])
        fig.update_yaxes(range=[X[:, 1].min() - 1, X[:, 1].max() + 1], scaleanchor="x")
        ui.play_controls(fig, [str(i) for i in range(len(frames))], 350, "iteration ")
        ui.show(ui.style(fig, "Soft labels (colours) and components (ellipses), iteration by iteration", height=480,
                         legend=False))
        ll = [f[1]["loglik"] for f in frames]
        f2 = go.Figure(go.Scatter(x=np.arange(len(ll)), y=ll, mode="lines+markers", line=dict(color=ui.C[1]),
                                  name="log-likelihood"))
        ui.show(ui.style(f2, "Log-likelihood after each iteration (never goes down)", "iteration",
                         "log p(data)", height=220, legend=False))
        m = st.columns(3)
        m[0].metric("Final log-likelihood", f"{ll[-1]:.1f}")
        m[1].metric("Iterations to get within 0.1 of it", next(i for i, v in enumerate(ll) if ll[-1] - v < 0.1))
        m[2].metric("Smallest change between iterations", f"{np.min(np.diff(ll)) if len(ll) > 1 else 0:.2g}",
                    help="Never negative: EM can't decrease the likelihood (Theory §4).")
    ui.takeaway("Each iteration does two easy things: label points softly with Bayes' rule, then refit each Gaussian "
                "to its share. The likelihood rises every time, but where it ends depends on where it started.")
    ui.controls([
        ("Dataset", "–", "the hidden groups", "–"),
        ("Components K", "$K$", "how many Gaussians", "more flexible; can split real clusters"),
        ("Initialisation", "–", "where the means start", "k-means++ avoids most bad starts"),
        ("Random seed", "–", "a different starting guess", "possibly a different local optimum"),
        ("EM iterations", "–", "how long to run", "–"),
    ])
    ui.experiments([
        ("Defaults (three round clusters, K = 3).", "The log-likelihood climbs from ≈ −2370 to ≈ −1891 in about 9 "
         "iterations.", "Well-separated clusters: the soft labels become nearly certain within a few steps."),
        ("Choose *stretched & overlapping*.", "It takes about twice as many iterations; points in the overlap stay "
         "faded.", "Where components overlap, responsibilities stay mixed and each M-step moves the ellipses only a "
         "little."),
        ("Choose *five clusters*, K = 5, random points; try seeds 0–9.", "Some seeds end with two ellipses on one "
         "cluster and one ellipse spanning two (a lower final likelihood).",
         "A local optimum: about half of random starts miss the best solution here, versus about 1 in 10 with "
         "k-means++."),
        ("Set K = 1.", "One big ellipse around everything.", "A single Gaussian can only describe the overall "
         "mean and spread."),
    ])

# --------------------------------------------------------------------------- choosing K
with t_k:
    st.markdown(
        r"""
**The situation.** Same data, but now we don't know how many groups there are. For each $K$ we keep the best of 5
k-means++ restarts, then compare two scores: the log-likelihood (higher = better fit) and BIC (Theory eq. 6, lower
= better trade-off between fit and number of parameters).
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        name2 = st.selectbox("Dataset", ["three round clusters", "stretched & overlapping", "five clusters"],
                             key="k_ds", help="The true number of groups is in the name.")
        kmax = st.slider("Largest K to try", 4, 10, 8, help="Fits K = 1 … this value.")

    @st.cache_data(show_spinner="Fitting mixtures…")
    def _sweep(name, kmax):
        X, _ = _data(name)
        out = []
        for K in range(1, kmax + 1):
            best = max((mx.gmm_em(X, K, 150, seed=s, init="k-means++") for s in range(5)), key=lambda r: r["loglik"])
            out.append((best["loglik"], mx.gmm_bic(best["loglik"], K, 2, len(X))))
        return np.array(out)
    res = _sweep(name2, kmax)
    with c2:
        Ks = np.arange(1, kmax + 1)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["log-likelihood (higher = better fit)",
                                                            "BIC (lower = better)"])
        fig.add_scatter(x=Ks, y=res[:, 0], mode="lines+markers", line=dict(color=ui.C[1]), showlegend=False, row=1,
                        col=1)
        fig.add_scatter(x=Ks, y=res[:, 1], mode="lines+markers", line=dict(color=ui.C[2]), showlegend=False, row=1,
                        col=2)
        kb = int(Ks[np.argmin(res[:, 1])])
        fig.add_scatter(x=[kb], y=[res[kb - 1, 1]], mode="markers", marker=dict(size=14, color=ui.C[2],
                        symbol="star"), showlegend=False, row=1, col=2)
        fig.update_xaxes(title="K")
        ui.show(ui.style(fig, None, height=300, legend=False, legend_below=True))
        st.metric("BIC's choice of K", kb)
    ui.takeaway("The likelihood never stops improving with more components, so it can't pick K. BIC charges "
                "log N per parameter, and its minimum lands on the true number of groups for these datasets.")
    ui.experiments([
        ("Look at the likelihood curve for *five clusters*.", "Large jumps up to K = 5, then tiny gains.",
         "After the real groups are covered, extra components only fit noise."),
        ("Compare BIC's choice with the dataset name for all three.", "BIC picks 3, 3 and 5.",
         "BIC approximates the Bayesian evidence, which rewards fit but charges for every extra parameter."),
    ])

# --------------------------------------------------------------------------- mixture of experts
with t_moe:
    st.markdown(
        r"""
**The situation.** A device whose output follows **different straight-line laws in different input ranges**
(linear, then a steeper region, then saturation and fold-back). One straight line can't fit it. A mixture of
$K$ linear **experts** plus a **gate** that chooses between them can (Theory eq. 7), and EM fits both without
being told where the regions are.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        Ke = st.slider("Experts K", 1, 5, 3, help="Number of straight-line experts.")
        it = st.slider("EM iterations", 1, 100, 60, help="Number of E+M steps.")
        noise = st.select_slider("Measurement noise", [0.05, 0.15, 0.4, 0.8], 0.15, help="Noise std on y.")
        mseed = st.number_input("Initialisation seed", 0, 9999, 0, key="moe_seed", help="A different start.")
    x, y = mx.piecewise_data(400, np.random.default_rng(1), noise)
    res = mx.moe_em(x, y, Ke, it, seed=int(mseed))
    xg = np.linspace(-3, 3, 300)
    g, lines, mean = mx.moe_predict(xg, res)
    with c2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.07)
        lab = res["resp"].argmax(1)
        fig.add_scatter(x=x, y=y, mode="markers", marker=dict(size=5, color=[COL[k] for k in lab], opacity=0.6),
                        showlegend=False, row=1, col=1)
        for k in range(Ke):
            on = g[:, k] > 0.5
            fig.add_scatter(x=xg, y=lines[:, k], line=dict(color=COL[k], dash="dot", width=1.5), showlegend=False,
                            row=1, col=1)
            fig.add_scatter(x=xg, y=np.where(on, lines[:, k], np.nan), name=f"expert {k + 1}",
                            line=dict(color=COL[k], width=3), row=1, col=1)
            fig.add_scatter(x=xg, y=g[:, k], line=dict(color=COL[k]), showlegend=False, row=2, col=1)
        fig.add_scatter(x=xg, y=mean, name="mixture mean prediction", line=dict(color=ui.C["truth"], width=2),
                        row=1, col=1)
        fig.update_yaxes(title="y", row=1, col=1)
        fig.update_yaxes(title="gate g_k(x)", range=[-0.05, 1.05], row=2, col=1)
        fig.update_xaxes(title="input x", row=2, col=1)
        ui.show(ui.style(fig, "Experts (solid where the gate trusts them) and the gate", height=460))
        ll = [h["loglik"] for h in res["hist"]]
        m = st.columns(3)
        m[0].metric("Log-likelihood", f"{ll[-1]:.1f}")
        m[1].metric("Experts' noise σ", ", ".join(f"{v:.2f}" for v in np.sqrt(res["s2"])))
        m[2].metric("True noise σ", f"{noise:.2f}")
    ui.takeaway("EM discovers both the regions and the lines: the E-step decides which expert owns each point, the "
                "M-step refits each line to its points and teaches the gate where each region is.")
    ui.controls([
        ("Experts K", "$K$", "number of linear pieces", "more flexibility; extra experts can overfit"),
        ("EM iterations", "–", "how long EM runs", "–"),
        ("Measurement noise", "$s$", "noise on the data", "region boundaries get blurrier"),
        ("Initialisation seed", "–", "starting lines and gate", "possibly a different local optimum"),
    ])
    ui.experiments([
        ("Defaults (K = 3).", "Three lines with slopes ≈ 0.4, 1.6 and −0.8, and noise σ ≈ 0.14–0.17 each.",
         "EM recovers the three hidden laws and where they switch."),
        ("Set K = 1, then 2.", "The log-likelihood jumps from ≈ −400 to ≈ −75, then to ≈ +170 at K = 3.",
         "Each extra expert that matches a real region explains a large share of the data."),
        ("Set K = 4 or 5.", "The likelihood barely changes, and extra experts claim a handful of points with a "
         "tiny σ (0.01–0.05).", "More components always fit better on training data; the gain no longer reflects real "
         "structure (compare BIC in the previous tab)."),
        ("Set iterations to 1, then 5, 20.", "The gate's switch points move into place over the iterations.",
         "The gate learns from the responsibilities, which sharpen as the experts improve."),
    ])
    ui.in_practice("Gaussian mixtures model speakers and noise types in classic speech systems, background pixels "
                   "in video, and customer segments. Mixtures of experts appear in piecewise system identification "
                   "and, with neural experts and a learned router, in today's largest language models.")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/mixtures.py`):")
    ui.source(mx.gmm_log_resp, mx.gmm_em, mx.gmm_bic, mx.moe_em)
