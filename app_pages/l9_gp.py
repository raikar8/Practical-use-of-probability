import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import gp

ui.lab_intro(
    "l9",
    goal="Put uncertainty on whole curves: see error bars that grow away from the data, let the evidence choose "
         "model complexity, and use that uncertainty to find the best setting of an expensive-to-test system in "
         "a handful of trials.",
    on_screen=[
        "**Bayesian regression:** a polynomial fit with its uncertainty band, and the evidence vs the training "
        "error for every degree (Theory §1–2)",
        "**Gaussian processes:** random functions from the prior, the posterior after seeing data, and "
        "hyperparameters learned from the evidence (Theory §3–4)",
        "**Bayesian optimisation:** searching for the best tuning setting, one simulated listening test at a time "
        "(Theory §5)",
    ],
    start_here="In Bayesian regression, drag the degree from 1 to 12 and watch the two curves in the right-hand "
               "plot: the training error keeps falling, the evidence doesn't.",
)


def truth(x):
    return np.sin(1.3 * x) + 0.3 * x


def data(n, noise, seed, gap=False):
    r = np.random.default_rng(seed)
    x = r.uniform(-3, 3, n)
    if gap:
        x = np.where((x > -0.5) & (x < 1.5), x - 2.0, x)        # leave a hole in the data
    x = np.sort(x)
    return x, truth(x) + noise * r.standard_normal(n)


t_blr, t_gp, t_bo, t_code = st.tabs(["Bayesian regression", "Gaussian processes", "Bayesian optimisation", "Code"])
xg = np.linspace(-4, 4, 300)

# --------------------------------------------------------------------------- BLR
with t_blr:
    st.markdown(
        r"""
**The situation.** Noisy measurements of an unknown smooth curve (grey). We fit polynomials of different degrees,
but the Bayesian way: a Gaussian prior on the coefficients gives a whole **distribution of curves**
(Theory eq. 1), drawn as a mean and a ±2σ band (eq. 2). On the right, the **evidence** (eq. 3) for every degree.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        deg = st.slider("Polynomial degree", 0, 12, 3, help="Number of features minus one (1, x, x², …).")
        n = st.slider("Number of points", 5, 80, 15, help="How many noisy measurements.")
        noise = st.select_slider("Noise std", [0.05, 0.1, 0.25, 0.5, 1.0], 0.25, help="Measurement noise (known "
                                                                                       "to the model).")
        alpha = st.select_slider("Prior precision α", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0,
                                 help="How strongly coefficients are pulled toward 0 (1/prior variance).")
        gap = st.toggle("Leave a gap in the data", False, help="Remove points between x = −0.5 and 1.5.")
        seed = st.number_input("Random seed", 0, 9999, 0, key="blr_seed", help="A different dataset.")
    x, y = data(n, noise, int(seed), gap)
    beta = 1 / noise**2
    post = gp.blr(gp.poly_features(x, deg), y, alpha, beta)
    mean, var = gp.blr_predict(gp.poly_features(xg, deg), post)
    degs = np.arange(0, 13)
    evs, trs = [], []
    for d in degs:
        P = gp.poly_features(x, d)
        evs.append(gp.blr(P, y, alpha, beta)["log_evidence"])
        m_d = gp.blr(P, y, alpha, beta)["m"]
        trs.append(np.sqrt(np.mean((P @ m_d - y) ** 2)))
    with c2:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                            subplot_titles=[f"Degree {deg}: mean and ±2σ predictive band", "Choosing the degree"])
        sd = np.sqrt(var)
        fig.add_scatter(x=np.r_[xg, xg[::-1]], y=np.r_[mean + 2 * sd, (mean - 2 * sd)[::-1]], fill="toself",
                        fillcolor=ui.rgba(ui.C[1], 0.18), line=dict(width=0), name="±2σ", row=1, col=1)
        rs = np.random.default_rng(1)
        Pg = gp.poly_features(xg, deg)
        L = np.linalg.cholesky(post["S"] + 1e-12 * np.eye(len(post["m"])))
        for j in range(3):
            w = post["m"] + L @ rs.standard_normal(len(post["m"]))
            fig.add_scatter(x=xg, y=Pg @ w, line=dict(color=ui.C[1], width=1, dash="dot"), showlegend=j == 0,
                            name="curves drawn from the posterior", row=1, col=1)
        fig.add_scatter(x=xg, y=truth(xg), name="true curve", line=dict(color=ui.C["truth"], width=3), row=1, col=1)
        fig.add_scatter(x=xg, y=mean, name="posterior mean", line=dict(color=ui.C[1], width=2.5), row=1, col=1)
        fig.add_scatter(x=x, y=y, mode="markers", name="data", marker=dict(color=ui.C[2], size=7), row=1, col=1)
        fig.update_yaxes(range=[-3.5, 3.5], row=1, col=1)
        ev = np.array(evs)
        fig.add_scatter(x=degs, y=ev - ev.max(), mode="lines+markers", name="log evidence (rel. to best)",
                        line=dict(color=ui.C[3]), row=1, col=2)
        fig.add_scatter(x=degs, y=-20 * np.array(trs), mode="lines+markers", name="−20 × training RMS",
                        line=dict(color=ui.C[4], dash="dash"), row=1, col=2)
        fig.add_vline(x=deg, line=dict(color=ui.C["truth"], dash="dot"), row=1, col=2)
        fig.update_yaxes(range=[-40, 2], row=1, col=2)
        fig.update_xaxes(title="x", row=1, col=1)
        fig.update_xaxes(title="degree", row=1, col=2)
        ui.show(ui.style(fig, None, height=400, legend_below=True))
        m = st.columns(3)
        m[0].metric("Evidence's choice of degree", int(degs[np.argmax(ev)]))
        m[1].metric("RMS error vs the true curve (−3…3)",
                    f"{np.sqrt(np.mean((mean[(xg > -3) & (xg < 3)] - truth(xg[(xg > -3) & (xg < 3)])) ** 2)):.3f}")
        m[2].metric("Predictive std at x = 3.8", f"{np.sqrt(var[np.argmin(np.abs(xg - 3.8))]):.2f}",
                    help="Outside the data range: how unsure the model admits to being.")
    ui.takeaway("The band is narrow where there's data and fans out where there isn't. The training error always "
                "prefers more parameters; the evidence stops at the simplest degree that explains the data, and "
                "moves up only when more data justifies it.")
    ui.controls([
        ("Polynomial degree", r"$\dim \boldsymbol\phi - 1$", "model complexity", "more flexible; wider band away "
         "from data"),
        ("Number of points", "$N$", "amount of data", "narrower band; evidence supports higher degrees"),
        ("Noise std", r"$1/\sqrt\beta$", "measurement noise", "wider band; simpler models preferred"),
        ("Prior precision α", r"$\alpha$", "pull of coefficients toward 0", "smoother, flatter fits"),
        ("Leave a gap", "–", "remove data in the middle", "the band bulges in the gap"),
    ])
    ui.experiments([
        ("Defaults (15 points), then degree 12.", "The mean wiggles and the band explodes beyond x = ±3; the "
         "evidence plot marks degree 3 as best.", "Twelve free coefficients can't be pinned down by 15 points; "
         "the prior stops them going wild only where there is data."),
        ("Set Number of points = 80.", "The evidence's choice moves up (to 5 here).",
         "With more data, the extra wiggles of sin(1.3x) become detectable, so a richer model earns its keep."),
        ("Turn on *Leave a gap*.", "The band widens inside the gap, at every degree.",
         "Uncertainty about the curve grows where no measurements constrain it (eq. 2)."),
        ("Set α = 100 with degree 6.", "The curve flattens and misses the data.",
         "A very strong prior says 'coefficients are almost 0' and overrides the data (Level 1's MAP lesson)."),
    ])

# --------------------------------------------------------------------------- GP
with t_gp:
    st.markdown(
        r"""
**The situation.** Same kind of data, no features at all: a **Gaussian process** prior over functions, set by a
kernel (Theory eq. 4). The left plot shows random functions the prior believes in *before* seeing data; the right
plot shows the posterior (eq. 5) after seeing it. Tick the box to choose the kernel's settings by maximising the
evidence (eq. 6).
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        kname = st.selectbox("Kernel", gp.KERNELS, help="What kind of functions the prior expects.")
        auto = st.toggle("Fit hyperparameters by maximising evidence", False,
                         help="Choose ℓ, σ_f and σ_n automatically (Theory §4).")
        ell = st.slider("Length-scale ℓ", 0.05, 5.0, 1.0, 0.05, disabled=auto,
                        help="Distance over which the function changes appreciably.")
        sf = st.slider("Signal std σ_f", 0.1, 3.0, 1.0, 0.1, disabled=auto, help="Typical size of the function.")
        sn = st.slider("Noise std σ_n", 0.01, 1.0, 0.25, 0.01, disabled=auto, help="Assumed measurement noise.")
        n2 = st.slider("Number of points", 2, 60, 12, key="gp_n", help="How many noisy measurements.")
        gap2 = st.toggle("Leave a gap in the data", True, key="gp_gap", help="Remove points between x = −0.5 and 1.5.")
    x2, y2 = data(n2, 0.25, 0, gap2)
    if auto:
        hp = gp.gp_fit_hypers(x2, y2, kname)
        ell, sf, sn = hp["ell"], hp["sf"], hp["sn"]
    post2 = gp.gp_posterior(x2, y2, xg, kname, ell, sf, max(sn, 1e-3))
    with c2:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62],
                            subplot_titles=["Prior: functions before data", "Posterior: after the data"])
        pri = gp.gp_prior_samples(xg, kname, ell, sf, 4, np.random.default_rng(2))
        for j in range(4):
            fig.add_scatter(x=xg, y=pri[j], line=dict(color=ui.C[7], width=1.2), showlegend=j == 0,
                            name="prior samples", row=1, col=1)
        sd = np.sqrt(post2["var"])
        fig.add_scatter(x=np.r_[xg, xg[::-1]], y=np.r_[post2["mean"] + 2 * sd, (post2["mean"] - 2 * sd)[::-1]],
                        fill="toself", fillcolor=ui.rgba(ui.C[1], 0.18), line=dict(width=0), name="±2σ (function)",
                        row=1, col=2)
        Ks = gp.kernel(kname, x2, xg, ell, sf)
        K = gp.kernel(kname, x2, x2, ell, sf) + max(sn, 1e-3) ** 2 * np.eye(len(x2))
        Cpost = gp.kernel(kname, xg, xg, ell, sf) - Ks.T @ np.linalg.solve(K, Ks)
        Lp = np.linalg.cholesky(Cpost + 1e-6 * np.eye(len(xg)))
        smp = post2["mean"][None] + (Lp @ np.random.default_rng(3).standard_normal((len(xg), 3))).T
        for j in range(3):
            fig.add_scatter(x=xg, y=smp[j], line=dict(color=ui.C[1], width=1, dash="dot"), showlegend=j == 0,
                            name="posterior samples", row=1, col=2)
        fig.add_scatter(x=xg, y=truth(xg), name="true curve", line=dict(color=ui.C["truth"], width=3), row=1, col=2)
        fig.add_scatter(x=xg, y=post2["mean"], name="posterior mean", line=dict(color=ui.C[1], width=2.5), row=1,
                        col=2)
        fig.add_scatter(x=x2, y=y2, mode="markers", name="data", marker=dict(color=ui.C[2], size=7), row=1, col=2)
        fig.update_yaxes(range=[-4, 4])
        ui.show(ui.style(fig, None, height=400, legend_below=True))
        m = st.columns(4)
        m[0].metric("ℓ", f"{ell:.2f}")
        m[1].metric("σ_f", f"{sf:.2f}")
        m[2].metric("σ_n", f"{sn:.2f}")
        m[3].metric("Log evidence", f"{post2['lml']:.1f}", help="Theory eq. 6 for the current settings.")
    ui.takeaway("The kernel is a statement about what functions look like. Near data the posterior hugs the points; "
                "away from them it relaxes back to the prior, and the band shows it. The evidence picks settings "
                "that explain the data without being able to explain everything.")
    ui.controls([
        ("Kernel", r"$k(x, x')$", "smoothness assumption", "–"),
        ("Length-scale ℓ", r"$\ell$", "how far correlation reaches", "smoother, straighter functions"),
        ("Signal std σ_f", r"$\sigma_f$", "typical function size", "bigger wiggles allowed; wider band"),
        ("Noise std σ_n", r"$\sigma_n$", "assumed noise", "mean ignores individual points more"),
        ("Fit hyperparameters", "eq. (6)", "maximise the evidence", "–"),
    ])
    ui.experiments([
        ("Set ℓ = 0.1.", "The prior samples are jagged; the posterior mean dips back toward 0 between points and the "
         "band balloons in the gap.", "With a short length-scale, each point only informs its immediate "
         "neighbourhood."),
        ("Set ℓ = 5.", "Prior samples are almost straight lines; the posterior can't follow the curve.",
         "Too smooth a prior underfits."),
        ("Tick *Fit hyperparameters* for each kernel.", "RBF gets the highest evidence (≈ −6.7 vs −7.2 and −8.2); "
         "ℓ settles at ≈ 1.3 for RBF and longer for the rougher Matérn kernels.", "The true curve is very smooth, and the evidence "
         "can tell."),
        ("Set σ_n = 0.01 with ℓ = 0.5.", "The mean passes through every point, with a wiggle between them.",
         "Assuming no noise forces an exact fit: overfitting, GP-style."),
    ])

# --------------------------------------------------------------------------- BO
with t_bo:
    st.markdown(
        r"""
**The situation.** You're tuning one parameter of a product (say an equaliser setting) between 0 and 10. Each test
is a **listening session** that returns a noisy score, so tests are expensive. The score curve is unknown: here
it has a broad hill and a narrower, higher peak. **Bayesian optimisation** fits a GP to the tests so far and runs
the next test where the **acquisition function** (Theory eq. 7) is highest. Press ▶ to step through the tests.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        acq = st.radio("Acquisition", ["Expected improvement", "Upper confidence bound"], horizontal=True,
                       help="How the next test point is chosen.")
        kappa = st.slider("UCB exploration κ", 0.0, 5.0, 2.0, 0.5, disabled=acq != "Upper confidence bound",
                          help="Weight on uncertainty in μ + κσ.")
        bell = st.select_slider("GP length-scale ℓ", [0.3, 0.5, 0.8, 1.0, 1.5, 3.0], 1.0,
                                help="How smooth you believe the score curve is.")
        n_init = st.slider("Initial random tests", 1, 6, 3, help="Tests at random settings before BO takes over.")
        n_it = st.slider("BO tests", 3, 20, 10, help="Tests chosen by the acquisition function.")
        show_truth = st.toggle("Show the true score curve", True, help="Hidden in real life.")
        bseed = st.number_input("Random seed", 0, 9999, 0, key="bo_seed", help="Different initial tests and noise.")

    @st.cache_data(show_spinner=False)
    def _bo(acq, kappa, bell, n_init, n_it, seed):
        return gp.bayes_opt(gp.tuning_objective, 0.0, 10.0, n_init, n_it, 0.05, np.random.default_rng(seed),
                            ell=bell, acq=acq, kappa=kappa)
    res = _bo(acq, kappa, bell, n_init, n_it, int(bseed))
    grid = res["grid"]
    tv = gp.tuning_objective(grid)
    with c2:
        def tr(s):
            sd = s["sd"]
            return [
                go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[s["mean"] + 2 * sd, (s["mean"] - 2 * sd)[::-1]],
                           fill="toself", fillcolor=ui.rgba(ui.C[1], 0.18), line=dict(width=0), name="±2σ",
                           xaxis="x", yaxis="y"),
                go.Scatter(x=grid, y=s["mean"], line=dict(color=ui.C[1], width=2.5), name="GP mean", xaxis="x",
                           yaxis="y"),
                go.Scatter(x=s["X"], y=s["Y"], mode="markers", marker=dict(color=ui.C[2], size=9),
                           name="tests so far", xaxis="x", yaxis="y"),
                go.Scatter(x=grid, y=s["acq"] / (s["acq"].max() - s["acq"].min() + 1e-12), name="acquisition (scaled)",
                           line=dict(color=ui.C[3]), xaxis="x2", yaxis="y2"),
                go.Scatter(x=[s["next"]], y=[1.0], mode="markers", marker=dict(symbol="star", size=16, color=ui.C[8]),
                           name="next test", xaxis="x2", yaxis="y2"),
            ]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
        base = []
        if show_truth:
            fig.add_scatter(x=grid, y=tv, name="true score (hidden)", line=dict(color=ui.C["truth"], width=3),
                            row=1, col=1)
            base = [0]
        for t_ in tr(res["steps"][0]):
            fig.add_trace(t_)
        idx = list(range(len(base), len(base) + 5))
        fig.frames = [go.Frame(data=tr(s), traces=idx, name=str(len(s["X"]))) for s in res["steps"]]
        fig.update_yaxes(range=[-2.5, 4.5], row=1, col=1, title="score")
        fig.update_yaxes(range=[-0.1, 1.15], row=2, col=1, title="acq.")
        fig.update_xaxes(title="tuning parameter", row=2, col=1)
        ui.play_controls(fig, [str(len(s["X"])) for s in res["steps"]], 700, "tests done: ")
        ui.show(ui.style(fig, "GP belief about the score, and where to test next", height=470))
        last = res["steps"][-1]
        best_x = last["X"][np.argmax(last["Y"])]
        m = st.columns(3)
        m[0].metric("Best setting found", f"{best_x:.2f}", help="Setting with the highest measured score.")
        m[1].metric("True best setting", f"{grid[np.argmax(tv)]:.2f}")
        hits = np.where(np.abs(last["X"] - grid[np.argmax(tv)]) < 0.5)[0]
        m[2].metric("Test that first hit the high peak", f"#{hits[0] + 1}" if len(hits) else "never",
                    help=f"Out of {len(last['X'])} tests in total.")
    ui.takeaway("The acquisition is high where the GP mean is promising or the uncertainty is large. That makes BO "
                "check unexplored regions instead of polishing the first good hill. With the right smoothness "
                "assumption it finds the narrow peak in about a dozen tests.")
    ui.controls([
        ("Acquisition", "EI or UCB, eq. (7)", "how to trade off exploring vs exploiting", "–"),
        ("UCB exploration κ", r"$\kappa$", "weight on uncertainty", "more exploration"),
        ("GP length-scale ℓ", r"$\ell$", "assumed smoothness of the score", "the GP is surer about untested "
         "regions"),
        ("Initial random tests", "–", "tests before BO starts", "better first GP, fewer BO tests left"),
        ("BO tests", "–", "test budget", "–"),
    ])
    ui.experiments([
        ("Defaults, press ▶.", "BO first refines the broad hill near 2.5; once the band there has shrunk, it jumps "
         "right and finds the higher peak near 7.6 (test #12 here).",
         "EI pays for uncertainty as well as for a good mean, so unexplored regions eventually win."),
        ("Set ℓ = 3.", "BO often polishes the broad hill near 2.5 and never tests near 7.6.",
         "A long length-scale makes the GP confident that unexplored regions are like the tested ones. Over 30 "
         "seeds, only ≈ 40% of runs find the high peak, versus ≈ 97–100% with ℓ between 0.8 and 1.5."),
        ("Set ℓ = 0.3.", "The band stays wide between tests and BO samples many regions.",
         "Short length-scales make every gap look uncertain: lots of exploration, slow convergence."),
        ("Switch to UCB and set κ = 0.", "BO keeps testing next to its first good result.",
         "Pure exploitation: no reward for learning about other regions."),
    ])
    ui.in_practice("Bayesian optimisation tunes the hyperparameters of machine-learning models, the settings of "
                   "audio products from listening panels, chemical and materials experiments, and A/B-tested product "
                   "parameters, wherever each evaluation costs real time or money.")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/gp.py`):")
    ui.source(gp.blr, gp.blr_predict, gp.kernel, gp.gp_posterior, gp.gp_fit_hypers, gp.expected_improvement,
              gp.bayes_opt)
