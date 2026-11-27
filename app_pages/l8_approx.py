import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import approx as ap

ui.lab_intro(
    "l8",
    goal="Compare the standard ways to approximate a posterior you can't compute, on 2-D posteriors where the "
         "exact answer is known, so you can see exactly how and where each method goes wrong.",
    on_screen=[
        "**MCMC samplers:** a Metropolis random walk, Gibbs sampling and Hamiltonian Monte Carlo exploring the "
        "same posterior with the same budget of density evaluations (Theory §3–6)",
        "**Variational vs Laplace:** a Gaussian fitted by maximising the ELBO (watch it move), the Laplace "
        "Gaussian at the peak, and the best 'cover everything' Gaussian (Theory §2, §7)",
    ],
    start_here="In MCMC samplers, keep the *banana* target and switch the sampler between Metropolis and HMC. "
               "Compare the 'effective samples per 1,000 evaluations' metric.",
)


def contours(fig, t, g):
    fig.add_contour(x=g["x"], y=g["y"], z=g["p"], showscale=False, ncontours=10, hoverinfo="skip",
                    contours=dict(coloring="lines"), line=dict(width=1),
                    colorscale=[[0, "#d0d0d0"], [1, "#606060"]], name="exact posterior")
    fig.update_xaxes(range=t.box[:2])
    fig.update_yaxes(range=t.box[2:], scaleanchor="x")


t_mc, t_vi, t_code = st.tabs(["MCMC samplers", "Variational vs Laplace", "Code"])

# --------------------------------------------------------------------------- MCMC
with t_mc:
    st.markdown(
        r"""
**The situation.** A 2-D posterior whose exact shape (grey contours) we computed on a grid, which is possible only
because it's 2-D. Each sampler gets the **same budget of log-density evaluations** (the expensive part in real
models), and we measure how much independent information it produced: the **effective sample size** (Theory
eq. 3).

* **banana:** a curved ridge, as when two parameters are only identified through a product
* **correlated Gaussian:** a narrow diagonal ridge (correlation 0.95)
* **two modes:** two separate peaks, as with label switching or two competing explanations
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        tname = st.selectbox("Target posterior", list(ap.TARGETS), help="The 2-D posterior to sample.")
        samplers = ["Metropolis", "Hamiltonian MC"] + (["Gibbs"] if tname == "correlated Gaussian" else [])
        smp = st.radio("Sampler", samplers, horizontal=True,
                       help="Gibbs needs exact conditionals, which we have only for the Gaussian.")
        if smp == "Metropolis":
            step = st.select_slider("Step size", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0], 0.5,
                                    help="Standard deviation of the random proposal step.")
        elif smp == "Hamiltonian MC":
            step = st.select_slider("Leapfrog step size ε", [0.05, 0.1, 0.2, 0.3, 0.5], 0.2,
                                    help="Time step of the simulated sliding. Too big and the simulation blows up.")
            L = st.slider("Leapfrog steps L", 1, 40, 15, help="How long each slide lasts (L gradient evaluations).")
        budget = st.select_slider("Budget (log-density evaluations)", [1000, 2000, 4000, 10000], 4000,
                                  help="Metropolis uses 1 per sample, HMC L+1 per sample, Gibbs 1 per sample.")
        seed = st.number_input("Random seed", 0, 9999, 0, key="mc_seed", help="A different chain.")
    t = ap.target(tname)
    g = ap.grid_density(t)
    rng = np.random.default_rng(int(seed))
    if smp == "Metropolis":
        res = ap.metropolis(t, budget, step, rng)
    elif smp == "Hamiltonian MC":
        res = ap.hmc(t, max(20, budget // (L + 1)), step, L, rng, keep_paths=6)
    else:
        res = ap.gibbs_gaussian(0.95, budget, rng)
    ch = res["chain"]
    burn = len(ch) // 10
    kept = ch[burn:]
    e = ap.ess(kept)
    with c2:
        fig = go.Figure()
        contours(fig, t, g)
        fig.add_scatter(x=kept[:, 0], y=kept[:, 1], mode="markers", name="samples (after burn-in)",
                        marker=dict(size=3, color=ui.C[1], opacity=0.35))
        if smp == "Gibbs":
            z = res["zigzag"][:120]
            fig.add_scatter(x=z[:, 0], y=z[:, 1], mode="lines", name="first 60 Gibbs sweeps",
                            line=dict(color=ui.C[2], width=1.5))
        else:
            n_path = 150 if smp == "Metropolis" else 30
            fig.add_scatter(x=ch[:n_path, 0], y=ch[:n_path, 1], mode="lines+markers",
                            name=f"first {n_path} moves", line=dict(color=ui.C[2], width=1.5), marker=dict(size=4))
        if smp == "Hamiltonian MC":
            for j, (path, ok) in enumerate(res["paths"]):
                fig.add_scatter(x=path[:, 0], y=path[:, 1], mode="lines", showlegend=j == 0,
                                name="leapfrog paths (dotted = rejected)",
                                line=dict(color=ui.C[7], width=2, dash="solid" if ok else "dot"))
        fig.add_scatter(x=[t.start[0]], y=[t.start[1]], mode="markers", name="start",
                        marker=dict(symbol="x", size=12, color=ui.C[8]))
        ui.show(ui.style(fig, f"{smp} on the {tname} posterior", "θ₁", "θ₂", height=430))
        f2 = go.Figure(go.Scatter(y=ch[:, 0], mode="lines", line=dict(color=ui.C[1], width=1), name="θ₁"))
        f2.add_hline(y=g["mean"][0], line=dict(color=ui.C["truth"], dash="dash"))
        f2.add_vrect(x0=0, x1=burn, fillcolor=ui.rgba(ui.C["truth"], 0.15), line_width=0,
                     annotation_text="burn-in", annotation_position="top left")
        ui.show(ui.style(f2, "Trace of θ₁ (dashed = exact posterior mean)", "sample number", "θ₁", height=220,
                         legend=False))
        m = st.columns(4)
        m[0].metric("Acceptance rate", f"{res['accept']:.0%}")
        m[1].metric("Effective samples", f"{e:.0f} of {len(kept)}")
        m[2].metric("Effective per 1,000 evaluations", f"{1000 * e / budget:.1f}",
                    help="The fair comparison: information gained per unit of computation.")
        m[3].metric("Error in posterior mean", f"{np.abs(kept.mean(0) - g['mean']).max():.2f}",
                    help="Largest error of the sample mean vs the exact mean (worst of the two coordinates).")
    ui.takeaway("A random walk with small steps is almost always accepted and almost useless. HMC follows the "
                "curve of the posterior using its gradient, so each sample is nearly independent of the last. "
                "But no local sampler crosses a deep valley between two peaks easily.")
    ui.controls([
        ("Target posterior", r"$p(\theta \mid x)$", "the shape to explore", "–"),
        ("Step size (Metropolis)", "proposal scale, eq. (4)", "how far each random proposal goes",
         "fewer acceptances but bigger moves"),
        ("Leapfrog step size ε (HMC)", r"$\epsilon$", "simulation time step", "longer moves; too big → "
         "energy errors → rejections"),
        ("Leapfrog steps L (HMC)", "$L$", "slide duration", "longer moves, more gradients per sample"),
        ("Budget", "–", "log-density evaluations allowed", "more samples, lower Monte Carlo error"),
    ])
    ui.experiments([
        ("Banana, Metropolis, step 0.05.", "≈ 90% acceptance but only ≈ 5 effective samples, and the mean is "
         "badly off.", "Tiny steps: the chain hasn't even travelled the length of the banana."),
        ("Banana, Metropolis, step 2.", "≈ 20% acceptance, ≈ 30 effective samples.",
         "Bigger steps are rejected more often, but the accepted ones go somewhere."),
        ("Banana, Hamiltonian MC (ε = 0.2, L = 15).", "≈ 99% acceptance and ≈ 19 effective samples per 1,000 "
         "evaluations, versus ≈ 1–8 for Metropolis; the mean error drops to ≈ 0.03.",
         "Gradient-guided slides follow the ridge."),
        ("Correlated Gaussian, HMC, ε = 0.5.", "0% acceptance.", "The narrowest direction has standard deviation "
         "≈ 0.22; the leapfrog simulation is unstable once ε exceeds about twice that."),
        ("Correlated Gaussian, Gibbs, then Metropolis with step 0.5.", "Gibbs: the orange zig-zag moves in small "
         "axis-aligned steps, ≈ 180 effective of 3,600. Metropolis: ≈ 45. HMC (ε = 0.2): every sample effectively "
         "independent (225 of 225).", "Each conditional is narrow (std √(1−0.95²) ≈ 0.31), so the chain creeps along the "
         "diagonal."),
        ("Two modes: Metropolis with step 0.2, then HMC.", "Both chains stay in the right-hand peak; the mean is "
         "off by ≈ 2.7 (Metropolis) and ≈ 1.7 (HMC), even though HMC accepts 98% of its moves.",
         "Crossing the valley needs a long jump. Run several chains from different starts to detect this."),
        ("Two modes: Metropolis with step 4.", "Only ≈ 7% acceptance, yet the most effective samples of any "
         "setting here, and a small mean error.", "Big jumps occasionally land in the other peak; for this shape "
         "that's what matters."),
    ])

# --------------------------------------------------------------------------- VI vs Laplace
with t_vi:
    st.markdown(
        r"""
**The situation.** Instead of sampling, fit one **Gaussian** to the same posteriors. Three different Gaussians
are shown:

* **blue, variational inference:** starts at the red ×, then gradient ascent on the ELBO (Theory eq. 6–7). Press ▶.
* **orange dashed, Laplace:** at the peak found from the same start, sized by the curvature there (eq. 2)
* **aqua, moment matching:** the exact mean and covariance, which is the best Gaussian under $\mathrm{KL}(p\,\|\,q)$
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        tn = st.selectbox("Target posterior", list(ap.TARGETS), key="vi_t", help="The 2-D posterior to approximate.")
        t2 = ap.target(tn)
        sx = st.slider("Start θ₁", float(t2.box[0]) + 0.5, float(t2.box[1]) - 0.5, float(t2.start[0]), 0.5,
                       help="Where the optimisers start.")
        sy = st.slider("Start θ₂", float(t2.box[2]) + 0.5, float(t2.box[3]) - 0.5, float(t2.start[1]), 0.5,
                       help="Where the optimisers start.")
        steps = st.select_slider("VI iterations", [50, 200, 500, 1500], 1500, help="Gradient-ascent steps.")
        lr = st.select_slider("VI learning rate", [0.01, 0.02, 0.05, 0.1], 0.05, help="Adam step size.")
    g2 = ap.grid_density(t2)
    v = ap.vi_gaussian(t2, steps, lr, 64, 0, init=(sx, sy), snapshots=30)
    la = ap.laplace(t2, (sx, sy))
    with c2:
        def frame(m, C):
            e = ap.ellipse(m, C)
            return [go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", line=dict(color=ui.C[1], width=3),
                               name="VI Gaussian (2σ)"),
                    go.Scatter(x=[m[0]], y=[m[1]], mode="markers", marker=dict(color=ui.C[1], size=8),
                               showlegend=False)]
        fig = go.Figure()
        contours(fig, t2, g2)
        el = ap.ellipse(la["mean"], la["cov"])
        fig.add_scatter(x=el[:, 0], y=el[:, 1], mode="lines", name="Laplace (2σ)",
                        line=dict(color=ui.C[2], dash="dash", width=2.5))
        em = ap.ellipse(g2["mean"], g2["cov"])
        fig.add_scatter(x=em[:, 0], y=em[:, 1], mode="lines", name="moment matching (2σ)",
                        line=dict(color=ui.C[3], width=2.5))
        fig.add_scatter(x=[sx], y=[sy], mode="markers", name="start", marker=dict(symbol="x", size=12,
                                                                                    color=ui.C[8]))
        n_static = len(fig.data)
        first = frame(*v["hist"][-1][:2])
        for tr in first:
            fig.add_trace(tr)
        fig.frames = [go.Frame(data=frame(m, C), traces=[n_static, n_static + 1], name=str(k))
                      for m, C, k in v["hist"]]
        ui.play_controls(fig, [str(k) for _, _, k in v["hist"]], 150, "VI iteration ", start_at_end=True)
        ui.show(ui.style(fig, "Three Gaussian approximations of the same posterior", "θ₁", "θ₂", height=460))
        f2 = go.Figure(go.Scatter(y=np.convolve(v["elbo"], np.ones(20) / 20, mode="valid"),
                                  line=dict(color=ui.C[1]), name="ELBO"))
        ui.show(ui.style(f2, "ELBO during VI (smoothed over 20 steps)", "iteration", "ELBO", height=200, legend=False))
        rows = []
        for nm, (mm, CC) in [("Variational (VI)", (v["mean"], v["cov"])), ("Laplace", (la["mean"], la["cov"])),
                             ("Moment matching", (g2["mean"], g2["cov"]))]:
            a, b = ap.gauss_kl_to_grid(mm, CC, g2)
            rows.append({"approximation": nm, "mean": f"({mm[0]:.2f}, {mm[1]:.2f})",
                         "std": f"({np.sqrt(CC[0, 0]):.2f}, {np.sqrt(CC[1, 1]):.2f})",
                         "KL(q‖p)": round(a, 2), "KL(p‖q)": round(b, 2)})
        rows.append({"approximation": "exact posterior", "mean": f"({g2['mean'][0]:.2f}, {g2['mean'][1]:.2f})",
                     "std": f"({np.sqrt(g2['cov'][0, 0]):.2f}, {np.sqrt(g2['cov'][1, 1]):.2f})", "KL(q‖p)": 0.0,
                     "KL(p‖q)": 0.0})
        st.dataframe(rows, hide_index=True, width="stretch")
    ui.takeaway("VI is an optimisation, so it's fast and its answer depends on the start. Minimising KL(q‖p) "
                "makes q stay inside the posterior: on a banana it hugs the middle of the ridge, on two peaks it "
                "picks one. Both VI and Laplace report less uncertainty than there really is.")
    ui.controls([
        ("Target posterior", r"$p(\theta \mid x)$", "the shape to approximate", "–"),
        ("Start θ₁, θ₂", r"$m$ at iteration 0", "where VI and the Laplace optimiser begin", "may change which peak "
         "they find"),
        ("VI iterations, learning rate", "–", "gradient ascent on the ELBO", "closer to the optimum"),
    ])
    ui.experiments([
        ("Correlated Gaussian.", "All three ellipses coincide; every KL ≈ 0.", "The posterior is Gaussian, so the "
         "Gaussian family contains it exactly."),
        ("Banana.", "VI's standard deviations ≈ (0.94, 0.50) vs exact (1.97, 1.40); Laplace has the right width "
         "along θ₁ (2.0) but is centred at the peak, not the mean.",
         "Reverse KL keeps q inside the curved ridge, so it shrinks; Laplace only sees the peak's curvature."),
        ("Two modes, default start (0, 3).", "VI and Laplace both find the smaller right-hand peak (KL(q‖p) ≈ 1.06).",
         "The start is closer to that peak. Gradient methods find a local optimum."),
        ("Two modes, start θ₁ = −3.", "Both switch to the bigger left peak (KL(q‖p) ≈ 0.44); the moment-matched "
         "Gaussian sits in the valley between.", "Mode-seeking vs mass-covering (Theory §7)."),
    ])
    ui.in_practice("Stan and PyMC run HMC (NUTS) for scientific models such as epidemiology, pharmacology and "
                   "astronomy. Variational inference powers VAEs, topic models and Bayesian deep learning, where "
                   "datasets and parameter counts are too large for sampling. Laplace approximations give quick "
                   "error bars in large-data fits, including on neural networks.")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/approx.py`):")
    ui.source(ap.metropolis, ap.hmc, ap.gibbs_gaussian, ap.ess, ap.laplace, ap.vi_gaussian)
