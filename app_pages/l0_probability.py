import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui
from ssp.core import probability as prob
from ssp.core.kalman import cov_ellipse

ui.header("Level 0", "Probability toolkit",
          "How should a belief change when data arrives, and how do we predict the next state?")

tab_bayes, tab_rule, tab_markov, tab_gauss, tab_code = st.tabs(
    ["Bayesian updating", "Bayes' rule & base rates", "Markov chains", "Gaussians & conditioning", "Code"])

# --------------------------------------------------------------------------- Bayesian updating
with tab_bayes:
    st.markdown(
        r"""
A factory line produces parts with an unknown defect rate $\theta$. Before inspecting anything we hold a
**prior** $\theta \sim \text{Beta}(\alpha, \beta)$. After seeing $k$ defects in $n$ parts, Bayes' rule gives

$$p(\theta \mid k) \propto \underbrace{\theta^k (1-\theta)^{n-k}}_{\text{likelihood}}\;
\underbrace{\theta^{\alpha-1}(1-\theta)^{\beta-1}}_{\text{prior}}
\;\Rightarrow\; \theta \mid k \sim \text{Beta}(\alpha + k,\; \beta + n - k).$$

The posterior after one batch is the prior for the next. That **recursive** update is the seed of
every filter in this app.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        theta = st.slider("True defect rate θ (hidden)", 0.01, 0.5, 0.08, 0.01)
        a0 = st.slider("Prior α (pseudo-defects)", 0.5, 50.0, 2.0, 0.5)
        b0 = st.slider("Prior β (pseudo-good parts)", 0.5, 50.0, 20.0, 0.5)
        batch = st.slider("Parts per batch", 1, 200, 20)
        nb = st.slider("Batches inspected", 0, 20, 5)
        seed = st.number_input("Random seed", 0, 9999, 1, key="b_seed")
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
        ui.show(ui.style(fig, "Belief about θ, batch by batch", "defect rate θ", "density"))
        s = prob.beta_summary(a, b)
        m1, m2, m3 = st.columns(3)
        m1.metric("Posterior mean", f"{s['mean']:.3f}")
        m2.metric("95% credible interval", f"{s['lo']:.3f} – {s['hi']:.3f}")
        m3.metric("Parts inspected", f"{nb * batch}")
    ui.takeaway("The prior acts like α+β parts you have 'already seen'. With little data it dominates; "
                "with lots of data the likelihood takes over and the curve narrows around the truth.")
    ui.break_it("Set a confident but wrong prior (α=2, β=48 means 'about 4%') with true θ=0.30 and small "
                "batches. Count how many batches it takes for the posterior to admit it was wrong.")
    ui.in_practice("A/B tests of click-through rates, quality control, conversion tracking, and the "
                   "'Bayesian bandits' used to choose which ad or recommendation to show next.")

# --------------------------------------------------------------------------- Bayes' rule
with tab_rule:
    st.markdown(
        r"""
A vibration monitor raises an alarm. What is the chance the machine is actually faulty?

$$P(\text{fault}\mid\text{alarm}) = \frac{P(\text{alarm}\mid\text{fault})\,P(\text{fault})}
{P(\text{alarm}\mid\text{fault})P(\text{fault}) + P(\text{alarm}\mid\text{ok})P(\text{ok})}$$
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        prior = st.slider("Fault prevalence P(fault)", 0.001, 0.5, 0.01, 0.001, format="%.3f")
        sens = st.slider("Sensitivity P(alarm | fault)", 0.5, 1.0, 0.99, 0.01)
        spec = st.slider("Specificity P(no alarm | ok)", 0.5, 0.999, 0.95, 0.001, format="%.3f")
        n_alarms = st.slider("Independent alarms in a row", 1, 4, 1)
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
        fig = go.Figure(go.Bar(x=[f"prior"] + [f"after alarm {i}" for i in range(1, n_alarms + 1)], y=seq,
                               marker_color=ui.C[1], text=[f"{v:.1%}" for v in seq], textposition="outside"))
        fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
        ui.show(ui.style(fig, "Each alarm is evidence; the posterior becomes the next prior", None,
                         "P(fault)", legend=False, height=320))
    ui.takeaway("When the event is rare, even a good detector produces mostly false alarms. The base "
                "rate (the prior) matters as much as the detector.")
    ui.in_practice("Fraud detection, predictive maintenance, intrusion detection, medical screening, and "
                   "voice wake-word detection all have to deal with the base rate.")

# --------------------------------------------------------------------------- Markov chains
with tab_markov:
    st.markdown(
        r"""
A **Markov chain** says the next state depends only on the current one: $P(s_{t+1}=j \mid s_t=i) = P_{ij}$.
Predicting $n$ steps ahead is just matrix multiplication, $\mathbf p_{t+n} = \mathbf p_t P^n$, and the chain
forgets where it started, settling at the **stationary distribution** $\boldsymbol\pi = \boldsymbol\pi P$.
In Level 4 the state becomes *hidden* and this chain turns into an HMM.
"""
    )
    names = ["Sunny", "Cloudy", "Rainy"]
    default = pd.DataFrame([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], index=names, columns=names)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("Transition matrix (rows are *from*, columns are *to*). Edit it: rows are re-normalised.")
        P_df = st.data_editor(default, key="P_edit")
        P = np.clip(P_df.to_numpy(dtype=float), 0, None)
        P = P / P.sum(axis=1, keepdims=True)
        start = st.selectbox("Start state", names)
        steps = st.slider("Days to simulate", 20, 1000, 200)
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
    ui.takeaway("Forecasts lose information about the starting state geometrically fast. How fast depends on "
                "the second-largest eigenvalue of P: sticky states (large diagonal) remember for longer.")
    ui.in_practice("Text prediction (n-gram models), Google's original PageRank, customer churn models, "
                   "weather generators, and the hidden layer of every HMM.")

# --------------------------------------------------------------------------- Gaussians
with tab_gauss:
    st.markdown(
        r"""
Two jointly Gaussian quantities, say a hidden state $x$ and a sensor reading $y$. Observing $y$ changes
what we believe about $x$:

$$x \mid y \sim \mathcal N\!\Big(\mu_x + \rho\frac{\sigma_x}{\sigma_y}(y-\mu_y),\;\; \sigma_x^2(1-\rho^2)\Big).$$

The new mean moves toward the evidence and the variance **shrinks** by $(1-\rho^2)$. This formula, in
matrix form, *is* the Kalman update.
"""
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        rho = st.slider("Correlation ρ between state and sensor", -0.99, 0.99, 0.8, 0.01)
        sx = st.slider("σ of state x", 0.5, 3.0, 1.0, 0.1)
        sy = st.slider("σ of sensor y", 0.5, 3.0, 1.5, 0.1)
        y_obs = st.slider("Observed sensor value y", -4.0, 4.0, 2.0, 0.1)
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
    ui.takeaway("A correlated measurement is information. |ρ| near 1 means the sensor nearly pins down the "
                "state; ρ = 0 means the reading tells you nothing and the belief does not move.")

with tab_code:
    ui.source(prob.beta_bernoulli_update, prob.posterior_given_alarm, prob.sample_markov_chain,
              prob.n_step_distribution, prob.stationary_distribution)
