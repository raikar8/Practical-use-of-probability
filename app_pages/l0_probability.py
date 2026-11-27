import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui
from ssp.core import probability as prob
from ssp.core.kalman import cov_ellipse

ui.lab_intro(
    "l0",
    goal="See Bayes' rule, Markov chains and Gaussian conditioning *move*. These four tabs are the building "
         "blocks of every filter later in the app.",
    on_screen=[
        "**Bayesian updating:** a belief about a defect rate sharpening batch by batch (Theory §4)",
        "**Bayes' rule & base rates:** why most alarms for rare faults are false (Theory §2–3)",
        "**Markov chains:** forecasting the next state, and the long-run behaviour (Theory §5)",
        "**Gaussians & conditioning:** how one measurement shrinks uncertainty, the core of the Kalman filter "
        "(Theory §6)",
    ],
    start_here="Open **Bayesian updating**, leave the defaults, and drag *Batches inspected* from 0 up to 20. "
               "Watch the curve narrow around the orange line (the truth).",
)

tab_bayes, tab_rule, tab_markov, tab_gauss, tab_code = st.tabs(
    ["Bayesian updating", "Bayes' rule & base rates", "Markov chains", "Gaussians & conditioning", "Code"])

# --------------------------------------------------------------------------- Bayesian updating
with tab_bayes:
    st.markdown(
        r"""
**The situation.** A factory line makes parts with an unknown defect rate $\theta$. We start with a **prior**
belief, inspect parts in batches, and after each batch update the belief with Bayes' rule:
Beta($\alpha$, $\beta$) → Beta($\alpha + k$, $\beta + n - k$). **The posterior after one batch becomes the prior for
the next.** The data is simulated from the true θ you set, so you can see whether the belief finds it.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        theta = st.slider("True defect rate θ (hidden)", 0.01, 0.5, 0.08, 0.01,
                          help="The real rate used to simulate the inspections. The estimator never sees it.")
        a0 = st.slider("Prior α (pseudo-defects)", 0.5, 50.0, 2.0, 0.5,
                       help="Before any data, believe as if you had already seen α defective parts.")
        b0 = st.slider("Prior β (pseudo-good parts)", 0.5, 50.0, 20.0, 0.5,
                       help="…and β good parts. Prior mean = α/(α+β); larger α+β = a more confident prior.")
        batch = st.slider("Parts per batch", 1, 200, 20, help="Number of parts n inspected in each batch.")
        nb = st.slider("Batches inspected", 0, 20, 5, help="How many Bayes updates to apply, one per batch.")
        seed = st.number_input("Random seed", 0, 9999, 1, key="b_seed",
                               help="Changes the random inspection results; same seed = same data.")
    rng = np.random.default_rng(int(seed))
    grid = np.linspace(0.0005, 0.9995, 800)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=stats.beta(a0, b0).pdf(grid), name="prior",
                    line=dict(color=ui.C["truth"], dash="dash"))
    a, b = a0, b0
    for i in range(nb):
        k = rng.binomial(batch, theta)
        a, b = prob.beta_bernoulli_update(a, b, k, batch - k)
        last = i == nb - 1
        fig.add_scatter(x=grid, y=stats.beta(a, b).pdf(grid),
                        name=f"after batch {i + 1}" if last else None, showlegend=last,
                        line=dict(color=ui.rgba(ui.C[1], 0.25 + 0.75 * (i + 1) / nb), width=3 if last else 1.5),
                        hovertemplate=f"batch {i + 1}<br>θ=%{{x:.3f}}<extra></extra>")
    fig.add_vline(x=theta, line=dict(color=ui.C[2], width=2), annotation_text="true θ")
    with c2:
        ui.show(ui.style(fig, "Belief about θ, batch by batch (faint = early batches, bold = latest)",
                         "defect rate θ", "density"))
        s = prob.beta_summary(a, b)
        m1, m2, m3 = st.columns(3)
        m1.metric("Posterior mean", f"{s['mean']:.3f}", help="α/(α+β) of the latest posterior")
        m2.metric("95% credible interval", f"{s['lo']:.3f} – {s['hi']:.3f}",
                  help="Given the data, θ lies in this range with 95% probability")
        m3.metric("Parts inspected", f"{nb * batch}")
    ui.controls([
        ("True defect rate θ", r"$\theta$", "the hidden truth that generates the data", "the curves move right"),
        ("Prior α, Prior β", r"$\alpha, \beta$", "pseudo-counts of defective / good parts", "a narrower, more stubborn prior"),
        ("Parts per batch", r"$n$", "inspections per Bayes update", "each update is more informative"),
        ("Batches inspected", "–", "number of updates", "more curves; the latest gets narrower"),
    ])
    ui.experiments([
        ("Set Batches inspected = 0, then step it up one at a time.",
         "Each new curve is narrower than the last, and centred closer to the orange line.",
         "Each batch adds k to α and n−k to β (Theory eq. 7). More counts = less uncertainty."),
        ("Set a confident, wrong prior: α = 2, β = 48 (prior mean 4%), true θ = 0.30, 5 parts per batch.",
         "Early curves stay near 4%; it takes several batches to move to 30%.",
         "The prior is worth 50 parts of 'experience'. The data must outweigh it."),
        ("Now set α = 0.5, β = 0.5 (a very weak prior) with the same true θ.",
         "The first batch already puts the belief near the truth, but it's wide and jumpy.",
         "With no prior information, the posterior follows the data, noise included."),
    ])
    ui.takeaway("The prior acts like α+β parts you have 'already seen'. With little data it dominates; "
                "with lots of data the likelihood takes over and the curve narrows around the truth.")
    ui.in_practice("A/B tests of click-through rates, quality control, conversion tracking, and the "
                   "'Bayesian bandits' used to choose which ad or recommendation to show next.")

# --------------------------------------------------------------------------- Bayes' rule
with tab_rule:
    st.markdown(
        r"""
**The situation.** A vibration monitor raises an alarm. What is the chance the machine is actually faulty? Three
numbers decide it: how common faults are (the **prior**), how often the alarm catches a fault (**sensitivity**),
and how often it stays quiet on a healthy machine (**specificity**). Bayes' rule (Theory eq. 4) combines them:
"""
    )
    ui.eq(r"P(\text{fault}\mid\text{alarm}) = \frac{P(\text{alarm}\mid\text{fault})\,P(\text{fault})}"
          r"{P(\text{alarm}\mid\text{fault})P(\text{fault}) + P(\text{alarm}\mid\text{ok})P(\text{ok})}")
    c1, c2 = st.columns([1, 2])
    with c1:
        prior = st.slider("Fault prevalence P(fault)", 0.001, 0.5, 0.01, 0.001, format="%.3f",
                          help="Fraction of machines that are actually faulty: the base rate (prior).")
        sens = st.slider("Sensitivity P(alarm | fault)", 0.5, 1.0, 0.99, 0.01,
                         help="Of the faulty machines, the fraction that trigger the alarm.")
        spec = st.slider("Specificity P(no alarm | ok)", 0.5, 0.999, 0.95, 0.001, format="%.3f",
                         help="Of the healthy machines, the fraction that correctly stay silent. "
                              "1 − specificity is the false-alarm rate.")
        n_alarms = st.slider("Independent alarms in a row", 1, 4, 1,
                             help="Apply Bayes' rule repeatedly: each posterior becomes the next prior.")
    p = prior
    seq = [prior]
    for _ in range(n_alarms):
        p = prob.posterior_given_alarm(p, sens, spec)
        seq.append(p)
    with c2:
        st.metric(f"P(fault | {n_alarms} alarm{'s' if n_alarms > 1 else ''})", f"{p:.1%}")
        N = 10_000
        faulty = prior * N
        st.markdown(
            f"Out of **{N:,} machines**: about **{faulty:,.0f}** are faulty and **{sens * faulty:,.0f}** of those "
            f"alarm. Of the **{N - faulty:,.0f}** healthy ones, **{(1 - spec) * (N - faulty):,.0f}** also alarm. "
            f"So among all alarms only **{sens * faulty / (sens * faulty + (1 - spec) * (N - faulty)):.1%}** are real."
        )
        fig = go.Figure(go.Bar(x=["prior"] + [f"after alarm {i}" for i in range(1, n_alarms + 1)], y=seq,
                               marker_color=ui.C[1], text=[f"{v:.1%}" for v in seq], textposition="outside"))
        fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
        ui.show(ui.style(fig, "Each alarm is evidence; the posterior becomes the next prior", None,
                         "P(fault)", legend=False, height=320))
    ui.controls([
        ("Fault prevalence", r"$P(\text{fault})$", "base rate = prior", "alarms become more trustworthy"),
        ("Sensitivity", r"$P(\text{alarm}\mid\text{fault})$", "detection rate", "slightly more trustworthy alarms"),
        ("Specificity", r"$1 - P(\text{alarm}\mid\text{ok})$", "1 − false-alarm rate", "far fewer false alarms: the big lever"),
        ("Alarms in a row", "–", "sequential Bayes updates", "the probability climbs toward 1"),
    ])
    ui.experiments([
        ("Leave the defaults (1% faulty, 99% sensitivity, 95% specificity).",
         "P(fault | alarm) is only about 17%.",
         "495 healthy machines false-alarm against 99 real faults. Rare events are swamped by false alarms."),
        ("Raise specificity to 0.999.",
         "P(fault | alarm) jumps to about 90%.",
         "False alarms drop from 495 to ~10. For rare events, specificity matters far more than sensitivity."),
        ("Back to 0.95, and set alarms in a row = 2, then 3.",
         "17% → ~80% → ~99%.",
         "Independent evidence multiplies. Yesterday's posterior is today's prior."),
    ])
    ui.takeaway("When the event is rare, even a good detector produces mostly false alarms. The base "
                "rate (the prior) matters as much as the detector.")
    ui.in_practice("Fraud detection, predictive maintenance, intrusion detection, medical screening, and "
                   "voice wake-word detection all have to deal with the base rate.")

# --------------------------------------------------------------------------- Markov chains
with tab_markov:
    st.markdown(
        r"""
**The situation.** Tomorrow's weather depends only on today's: that's the **Markov property**. The table on the
left is the **transition matrix** $A$: the row is today's weather, the column tomorrow's, and each entry is the
probability of that change. Forecasting $n$ days ahead is matrix multiplication, $\mathbf p_n = \mathbf p_0 A^n$
(Theory eq. 9). In Level 4 the state becomes *hidden* and this chain becomes an HMM.
"""
    )
    names = ["Sunny", "Cloudy", "Rainy"]
    default = pd.DataFrame([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], index=names, columns=names)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Transition matrix A** (rows = today, columns = tomorrow). Double-click a cell to edit it; "
                    "each row is re-scaled to add up to 1.")
        P_df = st.data_editor(default, key="P_edit")
        P = np.clip(P_df.to_numpy(dtype=float), 0, None)
        P = P / P.sum(axis=1, keepdims=True)
        start = st.selectbox("Start state", names, help="Today's weather: the forecast starts here with certainty.")
        steps = st.slider("Days to simulate", 20, 1000, 200,
                          help="Length of one random simulated weather history, used for the bar chart.")
        mseed = st.number_input("Random seed", 0, 9999, 2, key="m_seed")
    s0 = names.index(start)
    path = prob.sample_markov_chain(P, s0, steps, np.random.default_rng(int(mseed)))
    pi = prob.stationary_distribution(P)
    dist = prob.n_step_distribution(P, np.eye(3)[s0], 15)
    with c2:
        fig = go.Figure()
        for j, nm in enumerate(names):
            fig.add_scatter(x=np.arange(16), y=dist[:, j], name=nm, mode="lines+markers",
                            line=dict(color=ui.C[j + 1]))
            fig.add_hline(y=pi[j], line=dict(color=ui.C[j + 1], dash="dot", width=1))
        fig.update_yaxes(range=[0, 1])
        ui.show(ui.style(fig, f"Forecast P(state) starting from {start}; dotted = stationary π",
                         "days ahead", "probability", height=320))
        emp = np.bincount(path, minlength=3) / steps
        fig2 = go.Figure()
        fig2.add_bar(x=names, y=pi, name="stationary π (theory)", marker_color=ui.C[1])
        fig2.add_bar(x=names, y=emp, name=f"fraction of {steps} simulated days", marker_color=ui.C[2])
        ui.show(ui.style(fig2, "Long-run behaviour: theory vs simulation", None, "probability", height=280))
    ui.controls([
        ("Transition matrix", r"$A_{ij}$", "probability of moving from state i to j", "diagonal up = 'stickier' weather"),
        ("Start state", r"$\mathbf p_0$", "today's weather", "changes only the early forecast"),
        ("Days to simulate", "–", "length of one simulated history", "bars converge to π"),
    ])
    ui.experiments([
        ("Leave the defaults and read the forecast plot from left to right.",
         "On day 0 one state is certain; by about day 10 the lines sit on the dotted lines.",
         "The chain forgets its start; p₀Aⁿ converges to the stationary π."),
        ("Change the diagonal to 0.95 in every row (spread the rest over the other two columns).",
         "The forecast takes much longer to reach π.",
         "Sticky states remember. Convergence speed is set by the second-largest eigenvalue of A."),
        ("Set Days to simulate to 20, then 1000.",
         "With 20 days the orange bars are far from the blue; with 1000 they match.",
         "π is a long-run average; short histories are noisy samples of it."),
    ])
    ui.takeaway("Forecasts lose information about the starting state geometrically fast. How fast depends on "
                "the second-largest eigenvalue of A: sticky states (large diagonal) remember for longer.")
    ui.in_practice("Text prediction (n-gram models), Google's original PageRank, customer churn models, "
                   "weather generators, and the hidden layer of every HMM.")

# --------------------------------------------------------------------------- Gaussians
with tab_gauss:
    st.markdown(
        r"""
**The situation.** A hidden state $x$ and a sensor reading $y$ are related: they're jointly Gaussian with
correlation $\rho$. Each grey dot is one possible (sensor, state) pair. When the sensor reads a particular value
(orange vertical line), only the dots near that line remain plausible. Their spread is the **new, smaller
uncertainty** about $x$ (Theory eq. 11):
"""
    )
    ui.eq(r"x \mid y \sim \mathcal N\!\Big(\rho\,\frac{\sigma_x}{\sigma_y}\,y,\;\; \sigma_x^2(1-\rho^2)\Big)")
    c1, c2 = st.columns([1, 2])
    with c1:
        rho = st.slider("Correlation ρ between state and sensor", -0.99, 0.99, 0.8, 0.01,
                        help="How strongly the sensor follows the state. ±1 = perfect sensor, 0 = useless sensor.")
        sx = st.slider("σ of state x", 0.5, 3.0, 1.0, 0.1, help="Uncertainty about x before reading the sensor.")
        sy = st.slider("σ of sensor y", 0.5, 3.0, 1.5, 0.1, help="Spread of the sensor readings.")
        y_obs = st.slider("Observed sensor value y", -4.0, 4.0, 2.0, 0.1,
                          help="The reading we actually got. The orange bar shows what it implies about x.")
    cov = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
    pts = np.random.default_rng(0).multivariate_normal([0, 0], cov, 1500)
    cm = rho * sx / sy * y_obs
    cs = sx * np.sqrt(1 - rho**2)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=pts[:, 1], y=pts[:, 0], mode="markers", name="samples",
                        marker=dict(size=4, color=ui.rgba(ui.C["truth"], 0.35)))
        for k, nm in ((1, "1σ"), (2, "2σ")):
            ex, ey = cov_ellipse(np.zeros(2), cov[::-1, ::-1], k)
            fig.add_scatter(x=ex, y=ey, mode="lines", name=f"{nm} ellipse", line=dict(color=ui.C[1]))
        fig.add_vline(x=y_obs, line=dict(color=ui.C[2]))
        fig.add_scatter(x=[y_obs, y_obs], y=[cm - 2 * cs, cm + 2 * cs], mode="lines+markers",
                        name="x | y: mean ± 2σ", line=dict(color=ui.C[2], width=5))
        fig.update_layout(hovermode="closest")
        ui.show(ui.style(fig, "Conditioning a joint Gaussian on an observation", "sensor y", "state x"))
        m1, m2 = st.columns(2)
        m1.metric("E[x | y]", f"{cm:.2f}")
        m2.metric("σ before → after", f"{sx:.2f} → {cs:.2f}")
    ui.controls([
        ("Correlation ρ", r"$\rho$", "how informative the sensor is", "the orange bar gets shorter (more certainty)"),
        ("σ of state x", r"$\sigma_x$", "prior uncertainty about x", "everything scales up"),
        ("σ of sensor y", r"$\sigma_y$", "spread of readings", "the same reading moves x less"),
        ("Observed y", r"$y$", "the measurement", "the estimate E[x|y] moves along the ellipse's axis"),
    ])
    ui.experiments([
        ("Set ρ = 0.", "The ellipse is upright and the orange bar is as tall as the prior (σ unchanged).",
         "An uncorrelated sensor carries no information about x."),
        ("Set ρ = 0.95.", "The ellipse becomes a thin diagonal and the orange bar is short.",
         "Variance shrinks by (1 − ρ²) = 0.0975, so σ drops to about 31% of its prior value."),
        ("Move the observed y from −4 to 4.", "The bar slides along the ellipse; its length stays the same.",
         "In a Gaussian world the *mean* depends on the reading, but the *uncertainty* doesn't. The Kalman filter "
         "uses exactly this."),
    ])
    ui.takeaway("A correlated measurement is information. |ρ| near 1 means the sensor nearly pins down the "
                "state; ρ = 0 means the reading tells you nothing and the belief does not move.")

with tab_code:
    st.markdown("The exact functions this page runs (from `ssp/core/probability.py`):")
    ui.source(prob.beta_bernoulli_update, prob.posterior_given_alarm, prob.sample_markov_chain,
              prob.n_step_distribution, prob.stationary_distribution)
