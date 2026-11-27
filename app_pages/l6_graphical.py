import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import graphical as gm
from ssp.core import hmm, kalman

ui.lab_intro(
    "l6",
    goal="See inference as messages flowing through a graph: first exact reasoning in a small Bayesian network, "
         "then the messages on a chain that make up forward–backward, Viterbi and the Kalman smoother.",
    on_screen=[
        "**Explaining away:** the burglar-alarm network; set what you've observed and watch every other "
        "probability update (Theory §1–2)",
        "**Messages on a chain:** for one roll of the casino HMM, the message from the past, the local evidence, "
        "the message from the future, and their product (Theory §3)",
        "**Gaussian messages:** the same idea with bell curves, which reproduces the Kalman filter and smoother "
        "(Theory §4)",
    ],
    start_here="In Explaining away, set *Alarm* to **yes** and read the Burglary bar. Then also set *Earthquake* "
               "to **yes**.",
)

t_bn, t_chain, t_gauss, t_code = st.tabs(["Explaining away", "Messages on a chain", "Gaussian messages", "Code"])

# --------------------------------------------------------------------------- Bayesian network
with t_bn:
    st.markdown(
        r"""
**The situation.** Burglaries and earthquakes are rare and unrelated. Either can set off your home alarm, and
your neighbours John and Mary sometimes call when they hear it (John also calls by mistake sometimes). You set
what you know; the page computes $P(\text{each variable} = \text{yes} \mid \text{what you know})$ exactly by
summing the joint distribution over everything unknown (Theory eq. 2).
"""
    )
    c1, c2 = st.columns([1, 2.2])
    names = ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]
    labels = {"Burglary": "Burglary", "Earthquake": "Earthquake", "Alarm": "Alarm", "JohnCalls": "John calls",
              "MaryCalls": "Mary calls"}
    with c1:
        st.markdown("**What you've observed**")
        ev = {}
        for v in names:
            choice = st.segmented_control(labels[v], ["unknown", "yes", "no"], default="unknown", key=f"bn_{v}",
                                          help=f"Set {labels[v]} as observed, or leave it unknown.")
            if choice == "yes":
                ev[v] = 1
            elif choice == "no":
                ev[v] = 0
        st.markdown("**The model**")
        pb = st.select_slider("P(burglary) per night", [0.001, 0.005, 0.01, 0.05, 0.1], 0.01,
                              help="Prior probability of a burglary.")
        pe = st.select_slider("P(earthquake) per night", [0.001, 0.005, 0.02, 0.05, 0.1], 0.02,
                              help="Prior probability of an earthquake.")
        pq = st.slider("P(alarm | earthquake only)", 0.05, 0.95, 0.29, 0.01,
                       help="How easily an earthquake sets off the alarm. P(alarm | burglary only) is 0.94.")
    bn = gm.alarm_network(p_burglary=pb, p_quake=pe, p_alarm_quake=pq)
    try:
        post, pev = bn.marginals(ev)
        prior, _ = bn.marginals({})
        ok = pev > 0
    except ZeroDivisionError:
        ok = False
    with c2:
        if not ok:
            st.error("That combination of observations is impossible under the model (probability 0).")
        else:
            col = {v: ("#fdf0e8" if v in ev else "#eef4fc") for v in names}
            st.graphviz_chart(
                "digraph { rankdir=LR; node [fontname=Helvetica, fontsize=11, shape=ellipse, style=filled];"
                + "".join(f'{v} [label="{labels[v]}\\nP(yes) = {post[v]:.3f}", fillcolor="{col[v]}", '
                          f'color="{"#f47230" if v in ev else "#257de0"}"];' for v in names)
                + "Burglary -> Alarm; Earthquake -> Alarm; Alarm -> JohnCalls; Alarm -> MaryCalls; }"
            )
            fig = go.Figure()
            fig.add_bar(y=[labels[v] for v in names], x=[prior[v] for v in names], orientation="h",
                        name="before any observation", marker_color=ui.C["truth"])
            fig.add_bar(y=[labels[v] for v in names], x=[post[v] for v in names], orientation="h",
                        name="given what you've observed", marker_color=ui.C[1])
            fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"), hovermode="y unified")
            ui.show(ui.style(fig, "P(yes) for every variable", "probability", None, height=300))
            st.caption("Orange nodes are observed. Probabilities of observed variables are 0 or 1.")
    ui.takeaway("Observing the alarm makes a burglary 58× more likely. Adding 'there was an earthquake' brings it "
                "almost back to the prior: the earthquake *explains away* the alarm. Two causes that were "
                "independent become linked once you observe their common effect.")
    ui.controls([
        ("Observation buttons", "evidence $e$", "what you know for sure", "–"),
        ("P(burglary), P(earthquake)", r"$p(B)$, $p(E)$", "prior rates of the two causes", "that cause becomes the "
         "default explanation"),
        ("P(alarm | earthquake only)", r"$p(A{=}1 \mid B{=}0, E{=}1)$", "how good an excuse an earthquake is",
         "stronger explaining away"),
    ])
    ui.experiments([
        ("Set John calls = yes.", "Burglary rises from 1% to ≈ 13%; Alarm to ≈ 23%.",
         "John sometimes calls by mistake (5%), so his call alone is weak evidence."),
        ("Add Mary calls = yes.", "Burglary jumps to ≈ 56%, Alarm to ≈ 95%.",
         "Two independent witnesses agreeing is much stronger than one."),
        ("Now add Earthquake = yes.", "Burglary collapses to ≈ 3%.",
         "Explaining away (the collider pattern, Theory §2)."),
        ("Reset, set Alarm = yes, then toggle John calls between yes and no.",
         "Mary calls stays at 0.70 whatever John does.",
         "Given the alarm, John and Mary are independent (a fork with the middle observed)."),
    ])
    ui.in_practice("Bayesian networks are used for fault diagnosis (aircraft, telecom networks, cars' on-board "
                   "diagnostics), medical decision support and risk analysis, where experts can state local "
                   "cause–effect probabilities but not a full joint table.")

# --------------------------------------------------------------------------- chain messages
with t_chain:
    st.markdown(
        r"""
**The situation.** The dishonest casino from Level 4, with a short game so you can see every roll. Each roll
$s_t$ (fair or loaded die) is a node on a chain. To know roll $t$'s die, combine three things
(Theory eq. 4):

* the **message from the past** $m_{\rightarrow t}$: what rolls $1..t-1$ imply, pushed through the switching probabilities
* the **local evidence** $\phi_t$: how likely this roll's face is under each die
* the **message from the future** $m_{\leftarrow t}$: what rolls $t+1..T$ imply
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        T = st.slider("Number of rolls", 10, 60, 30, help="Length of the chain.")
        p_sw = st.slider("P(switch dice) per roll", 0.01, 0.3, 0.08, 0.01,
                         help="Both directions. Lower = longer spells on each die = stronger messages.")
        p6 = st.slider("Loaded die: P(6)", 0.2, 0.9, 0.6, 0.05, help="How strongly the loaded die favours 6.")
        mode = st.segmented_control("Algorithm", ["Sum-product", "Max-product"], default="Sum-product",
                                    help="Sum = forward–backward marginals. Max = Viterbi's best path.")
        seed = st.number_input("Random seed", 0, 9999, 3, key="ch_seed", help="A different game.")
        t_sel = st.slider("Inspect roll t", 1, T, min(12, T), help="Which node's messages to show.")
    s, rolls, true = hmm.sample_casino(T, p_sw, p_sw, p6, np.random.default_rng(int(seed)))
    lik = np.exp(true.loglik_matrix(rolls))
    sp = gm.chain_sum_product(true.pi, true.A, lik)
    mp = gm.chain_max_product(true.pi, true.A, lik)
    fb = hmm.posteriors(true.pi, true.A, np.log(lik))
    vit, _ = hmm.viterbi(true.pi, true.A, np.log(lik))
    with c2:
        faces = " ".join(f"**:red[{r + 1}]**" if r == 5 else str(r + 1) for r in rolls)
        st.markdown(f"Rolls: {faces}")
        t = t_sel - 1
        loc = lik[t] / lik[t].sum()
        if mode == "Sum-product":
            parts = [sp["forward"][t], loc, sp["backward"][t], sp["marginal"][t]]
        else:
            mm = np.exp(mp["max_marginal"][t])
            parts = [None, loc, None, mm / mm.sum()]
        fig = make_subplots(rows=1, cols=4, subplot_titles=["from the past", "seen here (roll "
                                                            f"{rolls[t] + 1})", "from the future", "belief"])
        for i, pv in enumerate(parts):
            if pv is None:
                continue
            fig.add_bar(x=["fair", "loaded"], y=pv, marker_color=[ui.C[1], ui.C[2]], showlegend=False, row=1,
                        col=i + 1)
        fig.update_yaxes(range=[0, 1])
        st.markdown(f"**Messages at roll {t_sel}** (each normalised to sum to 1)")
        ui.show(ui.style(fig, None, height=230, legend=False, legend_below=True))
        k = np.arange(1, T + 1)
        f2 = go.Figure()
        f2.add_scatter(x=k, y=s, name="truth: loaded?", line=dict(color=ui.C["truth"], shape="hv", width=6),
                       opacity=0.35)
        filt = sp["forward"] * lik
        filt /= filt.sum(1, keepdims=True)
        f2.add_scatter(x=k, y=filt[:, 1], name="past + here (filter)", line=dict(color=ui.C[1]))
        f2.add_scatter(x=k, y=sp["marginal"][:, 1], name="past + here + future (smoother)",
                       line=dict(color=ui.C[2]))
        f2.add_scatter(x=k, y=mp["path"], name="max-product path", line=dict(color=ui.C[3], dash="dot",
                                                                              shape="hv"))
        f2.add_vline(x=t_sel, line=dict(color=ui.C["truth"], dash="dash"))
        f2.update_yaxes(range=[-0.05, 1.05])
        ui.show(ui.style(f2, "P(loaded) along the chain", "roll t", "P(loaded)", height=280))
        m = st.columns(3)
        m[0].metric("Sum-product vs Level 4 forward–backward", f"{np.abs(sp['marginal'] - fb['gamma']).max():.0e}",
                    help="Largest difference between the two methods' marginals. ~1e-16 = identical.")
        m[1].metric("Max-product vs Level 4 Viterbi", "identical" if np.all(mp["path"] == vit) else "differs")
        m[2].metric("Smoother accuracy", f"{np.mean((sp['marginal'][:, 1] > 0.5) == s):.0%}")
    ui.takeaway("The belief at any roll is past × here × future. Use only the first two and you have the live "
                "filter; add the third and you have the smoother. Swap sums for maxima and the same messages give "
                "Viterbi's best path. These match the Level 4 code to machine precision.")
    ui.controls([
        ("Number of rolls", r"$T$", "chain length", "–"),
        ("P(switch dice)", r"$A_{12} = A_{21}$", "how often the casino swaps dice", "messages carry less "
         "information from far away"),
        ("Loaded die: P(6)", r"$\phi_t$", "how informative each roll is", "sharper local evidence"),
        ("Algorithm", "sum vs max", "marginals vs the best joint path", "–"),
        ("Inspect roll t", "$t$", "which node's messages to show", "–"),
    ])
    ui.experiments([
        ("Pick a roll that is a 6 inside a run of non-sixes.",
         "'Seen here' says loaded, but the messages from both sides say fair, and the belief stays fair.",
         "Neighbouring evidence outvotes a single surprising observation, because dice rarely switch."),
        ("Pick the first roll of a loaded spell (grey step up).",
         "'From the past' still says fair; 'from the future' already says loaded.",
         "The filter can only notice the switch after a few sixes; the smoother sees them coming."),
        ("Set P(switch dice) = 0.3.", "The messages flatten toward 50/50 and the belief follows 'seen here'.",
         "If the die can change at any roll, other rolls say little about this one."),
    ])

# --------------------------------------------------------------------------- Gaussian messages
with t_gauss:
    st.markdown(
        r"""
**The situation.** A slowly drifting quantity (say a sensor offset) $x_t = x_{t-1} + w_t$, measured with noise
$z_t = x_t + v_t$. Now every message is a **bell curve** (Theory eq. 5). Pick a time step and see the three
Gaussians whose product is the smoothed belief. The forward stream alone is the Kalman filter.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        q = st.select_slider("Process noise q", [0.01, 0.05, 0.1, 0.3, 1.0], 0.1,
                             help="How much x can drift per step (variance).")
        r = st.select_slider("Measurement noise r", [0.1, 0.3, 1.0, 3.0], 1.0, help="Sensor noise variance.")
        seed2 = st.number_input("Random seed", 0, 9999, 2, key="g_seed", help="A different run.")
        tg = st.slider("Inspect step t", 1, 40, 20, help="Which step's messages to plot.")
    x, z = kalman.simulate_random_walk(40, q, r, 0.0, np.random.default_rng(int(seed2)))
    b = gm.gaussian_chain_bp(z, q, r, 0.0, 1.0 + q)
    kf = kalman.KalmanFilter([[1.0]], [[1.0]], [[q]], [[r]], [0.0], [[1.0]])
    ref = kalman.run_kalman(kf, list(z))
    xs, Ps = kalman.rts_smoother(ref["x"], ref["P"], ref["x_pred"], ref["P_pred"], np.eye(1))
    with c2:
        t = tg - 1
        grid = np.linspace(min(z.min(), x.min()) - 2, max(z.max(), x.max()) + 2, 500)

        def g(m, v):
            return np.exp(-0.5 * (grid - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
        f = go.Figure()
        f.add_scatter(x=grid, y=g(b["fwd_mean"][t], b["fwd_var"][t]), name="from the past (prediction)",
                      line=dict(color=ui.C[1], dash="dash"))
        f.add_scatter(x=grid, y=g(z[t], r), name=f"seen here: z = {z[t]:.2f}", line=dict(color=ui.C[4], dash="dot"))
        if np.isfinite(b["bwd_var"][t]):
            f.add_scatter(x=grid, y=g(b["bwd_mean"][t], b["bwd_var"][t]), name="from the future",
                          line=dict(color=ui.C[7], dash="dash"))
        f.add_scatter(x=grid, y=g(b["smooth_mean"][t], b["smooth_var"][t]), name="belief (product)",
                      line=dict(color=ui.C[2], width=3))
        f.add_vline(x=x[t], line=dict(color=ui.C["truth"]), annotation_text="truth")
        ui.show(ui.style(f, f"The three messages at step {tg} and their product", "x", "density", height=280))
        k = np.arange(1, 41)
        f2 = go.Figure()
        for mean, var, nm, c in [(b["filt_mean"], b["filt_var"], "filter (past + here)", ui.C[1]),
                                 (b["smooth_mean"], b["smooth_var"], "smoother (+ future)", ui.C[2])]:
            sd = np.sqrt(var)
            f2.add_scatter(x=np.r_[k, k[::-1]], y=np.r_[mean + 2 * sd, (mean - 2 * sd)[::-1]], fill="toself",
                           fillcolor=ui.rgba(c, 0.15), line=dict(width=0), showlegend=False, hoverinfo="skip")
            f2.add_scatter(x=k, y=mean, name=nm, line=dict(color=c))
        f2.add_scatter(x=k, y=x, name="truth", line=dict(color=ui.C["truth"], width=3))
        f2.add_scatter(x=k, y=z, name="measurements", mode="markers", marker=dict(color=ui.C[4], size=7))
        f2.add_vline(x=tg, line=dict(color=ui.C["truth"], dash="dash"))
        ui.show(ui.style(f2, "Over time: filter and smoother with ±2σ bands", "step t", "x", height=300))
        m = st.columns(3)
        m[0].metric("Max |message passing − Kalman filter|", f"{np.abs(b['filt_mean'] - ref['x'][:, 0]).max():.0e}")
        m[1].metric("Max |message passing − RTS smoother|", f"{np.abs(b['smooth_mean'] - xs[:, 0]).max():.0e}")
        m[2].metric(f"Std at step {tg}: filter → smoother",
                    f"{np.sqrt(b['filt_var'][t]):.2f} → {np.sqrt(b['smooth_var'][t]):.2f}")
    ui.takeaway("Multiplying Gaussians adds their precisions, so the belief is always narrower than any single "
                "message. The forward messages reproduce the Kalman filter and forward × backward reproduces the RTS "
                "smoother, to machine precision.")
    ui.controls([
        ("Process noise q", "$q$", "drift per step", "messages from far away fade faster (wider)"),
        ("Measurement noise r", "$r$", "sensor noise", "'seen here' gets wider; neighbours matter more"),
        ("Inspect step t", "$t$", "which step to show", "at t = 40 there's no future message"),
    ])
    ui.experiments([
        ("Move t to the last step (40).", "The 'from the future' curve disappears and filter = smoother there.",
         "Nothing comes after the last step, so the smoother has nothing extra to add."),
        ("Set r = 3 and q = 0.01.", "'Seen here' is very wide; the belief is set by the past and future messages.",
         "A noisy sensor and a slowly drifting state: averaging over neighbours beats any single reading."),
        ("Set r = 0.1 and q = 1.", "The belief sits almost on the measurement.",
         "A precise sensor and a fast-moving state: neighbours say little."),
    ])
    ui.in_practice("Message passing is how LDPC and turbo codes are decoded in 5G, Wi-Fi and satellite links "
                   "(loopy belief propagation), how speech recognisers search word lattices (max-product), and how "
                   "large sensor networks and robot maps are fused (Gaussian belief propagation, factor-graph SLAM).")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/graphical.py`):")
    ui.source(gm.BayesNet, gm.chain_sum_product, gm.chain_max_product, gm.gaussian_chain_bp)
