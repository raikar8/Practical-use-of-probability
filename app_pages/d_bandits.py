import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import beta as beta_dist

from ssp import ui
from ssp.core import bandits as bd

ui.lab_intro(
    "bandits",
    goal="Run an online experiment the way modern sites do: learn which variant is best while losing as few "
         "clicks as possible, and compare that with a classic A/B test.",
    on_screen=[
        "**One experiment, live:** the Beta posterior for each variant updating as visitors arrive, the traffic "
        "each variant gets, and the clicks lost (Theory §2–3)",
        "**Policies compared:** average regret of four policies over many repeated experiments (Theory §3–4)",
    ],
    start_here="Press ▶ in *One experiment, live* with Thompson sampling and watch the orange curve (the best "
               "variant) pull ahead and take the traffic. Then switch to *A/B test, then commit*.",
)

NAMES = ["A", "B", "C", "D", "E"]
t_live, t_cmp, t_code = st.tabs(["One experiment, live", "Policies compared", "Code"])

# --------------------------------------------------------------------------- live
with t_live:
    st.markdown(
        r"""
**The situation.** A site tests variants of a headline. Each visitor sees one variant and clicks or doesn't.
The true click rates are set by you below; the policy doesn't know them. For each variant we keep a
$\text{Beta}(1 + \text{clicks}, 1 + \text{non-clicks})$ belief (Theory eq. 2).
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        K = st.slider("Number of variants", 2, 5, 3, help="How many versions are being tested.")
        defaults = [4.0, 5.0, 4.5, 3.5, 4.8]
        p = [st.slider(f"True click rate of {NAMES[k]} (%)", 1.0, 10.0, defaults[k], 0.1, key=f"p{k}",
                       help="Unknown to the policy.") / 100 for k in range(K)]
        pol = st.selectbox("Policy", bd.POLICIES, index=3, help="How each visitor's variant is chosen.")
        eps = st.slider("ε (ε-greedy)", 0.01, 0.3, 0.1, 0.01, disabled=pol != "ε-greedy",
                        help="Fraction of visitors shown a random variant.")
        ab_n = st.select_slider("A/B test length (visitors per variant)", [100, 200, 500, 1000, 2000, 5000], 500,
                                disabled=pol != "A/B test, then commit",
                                help="Equal split for this many visitors per variant, then commit.")
        T = st.select_slider("Visitors", [1000, 3000, 10000, 30000], 10000, help="Length of the experiment.")
        seed = st.number_input("Random seed", 0, 9999, 0, key="bd_seed", help="A different run of the experiment.")
    res = bd.run(pol, p, T, np.random.default_rng(int(seed)), eps=eps, ab_n=ab_n, record_every=max(1, T // 40))
    best = int(np.argmax(p))
    with c2:
        xs = np.linspace(0, max(p) * 1.8, 300)

        def frame(snap):
            t, a, b = snap
            return [go.Scatter(x=xs * 100, y=beta_dist.pdf(xs, a[k], b[k]) / 100, line=dict(color=ui.C[k + 1],
                               width=3 if k == best else 2), name=f"{NAMES[k]}: {int(a[k] + b[k] - 2)} visitors")
                    for k in range(K)]
        snaps = res["snaps"][1:]                # the first snapshot (1 visitor) is just the flat prior
        fig = go.Figure(data=frame(snaps[-1]), frames=[go.Frame(data=frame(s), name=str(s[0])) for s in snaps])
        for k in range(K):
            fig.add_vline(x=p[k] * 100, line=dict(color=ui.C[k + 1], dash="dot", width=1))
        top = max(beta_dist.pdf(xs, snaps[-1][1][k], snaps[-1][2][k]).max() for k in range(K)) / 100
        fig.update_yaxes(range=[0, top * 1.1])
        ui.play_controls(fig, [str(s[0]) for s in snaps], 150, "visitors: ", start_at_end=True)
        ui.show(ui.style(fig, "Beliefs about each click rate (dotted = true rate)", "click rate (%)", "density",
                         height=380))
        sub = make_subplots(rows=1, cols=2, subplot_titles=["Share of traffic (rolling 500 visitors)",
                                                            "Clicks lost so far (regret)"])
        w = min(500, T // 4)
        for k in range(K):
            share = np.convolve((res["arms"] == k).astype(float), np.ones(w) / w, mode="valid")
            sub.add_scatter(x=np.arange(len(share)) + w, y=share, line=dict(color=ui.C[k + 1]), name=NAMES[k],
                            showlegend=False, row=1, col=1)
        sub.add_scatter(x=np.arange(1, T + 1), y=res["regret"], line=dict(color=ui.C[8]), showlegend=False, row=1,
                        col=2)
        sub.update_yaxes(range=[0, 1.02], row=1, col=1)
        sub.update_xaxes(title="visitor")
        ui.show(ui.style(sub, None, height=260, legend=False, legend_below=True))
        pb = bd.prob_best(snaps[-1][1], snaps[-1][2], np.random.default_rng(0))
        m = st.columns(4)
        m[0].metric("Clicks", f"{int(res['rewards'].sum())}",
                    help=f"Always showing the best variant would give ≈ {p[best] * T:.0f} on average.")
        m[1].metric("Clicks lost (regret)", f"{res['regret'][-1]:.0f}")
        m[2].metric(f"Traffic to the best ({NAMES[best]})", f"{np.mean(res['arms'] == best):.0%}")
        m[3].metric(f"P({NAMES[best]} is best | data)", f"{pb[best]:.0%}",
                    help="From the final posteriors (Theory eq. 3).")
        if pol == "A/B test, then commit" and res["committed"] is not None:
            (st.success if res["committed"] == best else st.error)(
                f"After the test, the policy committed to **{NAMES[res['committed']]}**"
                + (" (the true best)." if res["committed"] == best else f", but the best is **{NAMES[best]}**."))
    ui.takeaway("Thompson sampling never stops learning, but it spends less and less on variants that are clearly "
                "worse. An A/B test spends a fixed amount, then bets everything on a winner chosen from noisy data.")
    ui.controls([
        ("True click rates", r"$p_k$", "how good each variant really is", "–"),
        ("Policy", "Theory §3", "how each visitor's variant is chosen", "–"),
        ("ε", r"$\varepsilon$", "ε-greedy's exploration rate", "more exploration forever"),
        ("A/B test length", "$n$", "visitors per variant before committing", "fewer wrong commits, more test cost"),
        ("Visitors", "$T$", "experiment length", "–"),
    ])
    ui.experiments([
        ("Defaults (Thompson sampling, seed 0).", "≈ 85% of all traffic goes to B (≈ 96% in the last 2,000 "
         "visitors); only ≈ 10 clicks lost; P(B is best) ≈ 96%.",
         "As B's posterior separates from the others, its draws win almost every time."),
        ("Switch to *A/B test, then commit* (500 per variant).", "With seed 0 it commits to the wrong variant and "
         "loses ≈ 50 clicks.", "With rates of 4–5%, 500 visitors per variant can't reliably tell them apart: about "
         "40% of such tests pick a non-best variant."),
        ("Switch to UCB1.", "Traffic stays spread out for a long time (only ≈ 35% to B); ≈ 47 clicks lost.",
         "UCB1's exploration bonus is designed for the worst case and is far too cautious for small differences."),
        ("Make the rates very different (A = 2%, B = 8%).", "Every policy finds B quickly; the differences between "
         "policies shrink.", "Large differences are easy. Bandits earn their keep when differences are small, "
         "which is the usual case."),
    ])

# --------------------------------------------------------------------------- comparison
with t_cmp:
    st.markdown(
        r"""
**The situation.** One run is luck-dependent, so here each policy runs the same experiment (the rates set in the
first tab) many times and we average the regret curves. Lower is better.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        runs = st.select_slider("Repeated experiments", [10, 20, 50], 20, help="Runs averaged per policy.")
        T2 = st.select_slider("Visitors per experiment", [2000, 10000, 30000], 10000, key="T2",
                              help="Length of each experiment.")

    @st.cache_data(show_spinner="Running experiments…")
    def _cmp(p, T, runs, eps, ab_n):
        return {pol: bd.mean_regret(pol, list(p), T, runs, seed=0, eps=eps, ab_n=ab_n) for pol in bd.POLICIES}
    curves = _cmp(tuple(p), T2, runs, eps, ab_n)
    with c2:
        fig = go.Figure()
        for i, (pol_, cv) in enumerate(curves.items()):
            fig.add_scatter(x=np.arange(1, T2 + 1)[::10], y=cv[::10], name=f"{pol_}: {cv[-1]:.0f}",
                            line=dict(color=ui.C[i + 1]))
        ui.show(ui.style(fig, f"Average clicks lost over {runs} experiments", "visitor", "regret", height=380))
    ui.takeaway("Thompson sampling has the lowest regret and its curve flattens (regret grows like log T). "
                "ε-greedy keeps paying its ε forever. The A/B test pays for the test, then either flattens (right "
                "winner) or climbs in a straight line (wrong winner), and averaged over runs it climbs.")
    ui.experiments([
        ("Defaults (3 variants, 10,000 visitors, 20 runs).", "Thompson ≈ 30, A/B ≈ 35, ε-greedy ≈ 39, UCB1 ≈ 46.",
         "Thompson sampling matches traffic to the probability of being best."),
        ("Set 30,000 visitors.", "Thompson ≈ 53, A/B ≈ 93, ε-greedy ≈ 76, UCB1 ≈ 129.",
         "Over longer horizons, policies that keep exploring at a fixed rate or commit to wrong winners fall "
         "further behind."),
    ])
    ui.in_practice("Bandits choose news headlines and thumbnails, ads, recommendation slots and email subject lines, "
                   "tune game difficulty, and allocate patients in adaptive clinical trials. Contextual bandits add "
                   "visitor features, so the best variant can differ per person.")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/bandits.py`):")
    ui.source(bd.run, bd.prob_best)
