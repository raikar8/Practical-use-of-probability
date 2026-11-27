import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import hmm

ui.lab_intro(
    "l4",
    goal="Use an HMM to uncover a hidden switching process from its observations alone: first with the model "
         "known (filtering, smoothing, Viterbi), then learning the model itself from unlabelled data (Baum–Welch).",
    on_screen=[
        "**The dishonest casino:** the true hidden die (grey), three different HMM answers about it, and the "
        "observed sixes below (Theory §1–5)",
        "**Learning with Baum–Welch:** the likelihood climbing each EM iteration, learned vs true dice and switching "
        "probabilities, and local optima across random restarts (Theory §6)",
    ],
    start_here="In the casino tab, find a grey block (the loaded die in play) and compare how quickly the blue "
               "(live) and orange (hindsight) curves rise at its start.",
)

t_casino, t_learn, t_code = st.tabs(["The dishonest casino", "Learning with Baum–Welch", "Code"])

# --------------------------------------------------------------------------- casino
with t_casino:
    st.markdown(
        r"""
**The situation.** A casino secretly swaps a **fair die** for a **loaded die** that favours 6, then swaps back. You
see only the rolls. Here the HMM is given the *true* model, and we compare its three answers (Theory §2):

* **blue, forward** $P(\text{loaded} \mid \text{rolls so far})$: what a live observer could know
* **orange, smoothed** $P(\text{loaded} \mid \text{all rolls})$: what's known in hindsight
* **dotted aqua, Viterbi:** the single most probable sequence of dice
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        T = st.slider("Number of rolls", 100, 5000, 2000, 100,
                      help="Length of the game. The plot shows the first 600; accuracies use all rolls.")
        p_fl = st.slider("P(switch fair → loaded) per roll", 0.001, 0.2, 0.02, 0.001, format="%.3f",
                         help="A₁₂. Fair spells last 1/this rolls on average (0.02 → 50 rolls).")
        p_lf = st.slider("P(switch loaded → fair) per roll", 0.01, 0.5, 0.10, 0.01,
                         help="A₂₁. Loaded spells last 1/this rolls on average (0.1 → 10 rolls).")
        p6 = st.slider("Loaded die: P(6)", 0.2, 0.9, 0.5, 0.05,
                       help="How strongly the loaded die favours 6 (a fair die gives 1/6 ≈ 0.17). The other faces "
                            "share the rest.")
        seed = st.number_input("Random seed", 0, 9999, 1, key="cas_seed")
    s, rolls, true = hmm.sample_casino(T, p_fl, p_lf, p6, np.random.default_rng(int(seed)))
    logB = true.loglik_matrix(rolls)
    post = hmm.posteriors(true.pi, true.A, logB)
    path, _ = hmm.viterbi(true.pi, true.A, logB)
    filt = post["alpha"][:, 1]
    smooth = post["gamma"][:, 1]
    with c2:
        faces = "".join(f"**:red[{r + 1}]**" if r == 5 else str(r + 1) for r in rolls[:90])
        st.markdown(f"First 90 rolls (sixes in red): {faces}")
        n = min(T, 600)
        k = np.arange(n)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
        fig.add_scatter(x=k, y=s[:n], name="truth: loaded?", line=dict(color=ui.C["truth"], shape="hv", width=6),
                        opacity=0.35, row=1, col=1)
        fig.add_scatter(x=k, y=filt[:n], name="forward P(loaded | past)", line=dict(color=ui.C[1]), row=1, col=1)
        fig.add_scatter(x=k, y=smooth[:n], name="smoothed P(loaded | all)", line=dict(color=ui.C[2]), row=1, col=1)
        fig.add_scatter(x=k, y=path[:n], name="Viterbi path", line=dict(color=ui.C[3], shape="hv", dash="dot"),
                        row=1, col=1)
        fig.add_scatter(x=k, y=(rolls[:n] == 5).astype(int), mode="markers", name="rolled a 6",
                        marker=dict(size=4, color=ui.C[8]), row=2, col=1)
        fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1, title="P(loaded)")
        fig.update_yaxes(tickvals=[0, 1], ticktext=["other", "six"], row=2, col=1)
        fig.update_xaxes(title="roll number", row=2, col=1)
        ui.show(ui.style(fig, f"Hidden state vs three HMM answers (first {n} rolls)", height=460))
        m = st.columns(4)
        m[0].metric("Rolls on the loaded die", f"{s.mean():.0%}")
        m[1].metric("Forward accuracy (real time)", f"{np.mean((filt > 0.5) == s):.1%}")
        m[2].metric("Smoothed accuracy", f"{np.mean((smooth > 0.5) == s):.1%}")
        m[3].metric("Viterbi accuracy", f"{np.mean(path == s):.1%}")
    ui.takeaway("The forward probability reacts only after a run of sixes, because it can't see the future. The "
                "smoother uses later rolls and snaps to the switch points, so it's more accurate but only available "
                "after the fact. Viterbi gives one consistent story (no single-roll flips), which is what you want "
                "when the output is a label sequence.")
    ui.controls([
        ("Number of rolls", r"$T$", "sequence length", "more data; accuracies stabilise"),
        ("P(switch fair → loaded)", r"$A_{12}$", "how often cheating starts", "more, shorter loaded spells"),
        ("P(switch loaded → fair)", r"$A_{21}$", "how quickly cheating stops", "shorter loaded spells, harder to catch"),
        ("Loaded die: P(6)", r"$b_{\text{loaded}}(6)$", "how different the dice are", "easier detection"),
    ])
    ui.experiments([
        ("Defaults: compare the four metrics.",
         "Forward ≈ 87%, smoothed ≈ 92%, Viterbi ≈ 91%. The die is loaded 20% of the time.",
         "Hindsight helps: rolls after a switch confirm it. The forward curve needs a few sixes before it believes."),
        ("Count the jumps in the blue curve vs the dotted Viterbi steps.",
         "Thresholding the forward curve at 0.5 would flip about 150 times; Viterbi switches about 20 times "
         "(the truth: 56).",
         "Viterbi scores whole sequences, so it won't flip for one lucky six. Short loaded spells get missed."),
        ("Set Loaded die P(6) = 0.25.", "Viterbi never calls 'loaded'; accuracy ≈ 80% = 'always say fair'.",
         "When the dice barely differ, there's too little information per roll. No algorithm can beat that."),
        ("Set P(6) = 0.8.", "Every method improves; smoothed and Viterbi reach about 96%.",
         "Each six is now strong evidence, so switches are detected within a roll or two."),
        ("Set the switching probabilities to 0.2 and 0.5.", "Accuracies fall to ~73%, barely above 'always fair'.",
         "Spells last 2–5 rolls, too short to accumulate evidence. The 'sticky' prior that makes HMMs work is gone."),
    ])
    ui.in_practice("Speech recognition (phoneme HMMs, pre-deep-learning and still inside hybrid systems), gene "
                   "finding, part-of-speech tagging, fault diagnosis, regime-switching finance models, and the "
                   "decoding in mobile phones (Viterbi decoding of convolutional codes).")

# --------------------------------------------------------------------------- learning
with t_learn:
    st.markdown(
        r"""
**The situation.** Now pretend nobody tells you the dice or the switching probabilities. **Baum–Welch** starts
from a random guess and alternates "estimate where the switches probably were" (E-step) and "re-count the dice
and switches" (M-step), Theory §6. It uses the rolls generated in the casino tab, so change those settings there.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        iters = st.slider("EM iterations (max)", 1, 300, 150,
                          help="How many E/M rounds to run (it stops early once the likelihood stops improving).")
        init_seed = st.number_input("Initial-guess seed", 0, 9999, 3,
                                    help="A different random starting guess for the main run.")
        n_restarts = st.slider("Random restarts to compare", 1, 12, 8,
                               help="Extra runs from different random starts; the bottom plot compares where they end.")
        st.caption("Uses the rolls from the casino tab.")

    @st.cache_data(show_spinner="Running EM…")
    def _fit(rolls_bytes, iters, init_seed, n_restarts):
        r = np.frombuffer(rolls_bytes, dtype=np.int64)
        main = hmm.baum_welch(r, hmm.init_categorical(2, 6, np.random.default_rng(init_seed)), n_iter=iters,
                              tol=1e-10)
        finals = [hmm.baum_welch(r, hmm.init_categorical(2, 6, np.random.default_rng(1000 + i)), n_iter=iters,
                                 tol=1e-10)["loglik"][-1] for i in range(n_restarts)]
        return main, finals

    fit, finals = _fit(rolls.astype(np.int64).tobytes(), iters, int(init_seed), n_restarts)
    p = fit["params"]
    loaded = int(np.argmax(p.E[:, 5]))
    order = [1 - loaded, loaded]                       # present as [fair-like, loaded-like]
    true_ll = hmm.forward(true.pi, true.A, logB)[2]
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=np.arange(len(fit["loglik"])), y=fit["loglik"], name="EM log-likelihood",
                        line=dict(color=ui.C[1]))
        fig.add_hline(y=true_ll, line=dict(color=ui.C["truth"], dash="dash"),
                      annotation_text="likelihood of the true parameters")
        ui.show(ui.style(fig, "Log-likelihood climbs every iteration", "iteration", "log p(rolls)", height=300))
        fig2 = go.Figure()
        faces = [str(i) for i in range(1, 7)]
        fig2.add_bar(x=faces, y=true.E[1], name="true loaded die", marker_color=ui.C["truth"])
        fig2.add_bar(x=faces, y=p.E[loaded], name="learned 'loaded' state", marker_color=ui.C[2])
        fig2.add_bar(x=faces, y=p.E[1 - loaded], name="learned 'fair' state", marker_color=ui.C[1])
        ui.show(ui.style(fig2, "Emission probabilities learned without labels", "die face", "probability",
                         height=300))
        tbl = pd.DataFrame(
            {"true": [true.A[0, 1], true.A[1, 0]],
             "learned": [p.A[order[0], order[1]], p.A[order[1], order[0]]]},
            index=["P(fair → loaded)", "P(loaded → fair)"])
        st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
        fig3 = go.Figure(go.Scatter(x=np.arange(1, len(finals) + 1), y=finals, mode="markers",
                                    marker=dict(size=11, color=ui.C[1])))
        fig3.add_hline(y=max(finals), line=dict(color=ui.C["truth"], dash="dot"), annotation_text="best restart")
        ui.show(ui.style(fig3, "Final log-likelihood from different random starts", "restart", "log-likelihood",
                         height=260, legend=False))
    ui.takeaway("EM can beat the true parameters' likelihood slightly: it fits *this* sample, including its noise. "
                "The state labels are arbitrary ('label switching'), so we name the states after the fact by "
                "looking at what they emit.")
    ui.controls([
        ("EM iterations", "–", "E-step + M-step rounds", "the likelihood curve extends (it never goes down)"),
        ("Initial-guess seed", "–", "random starting parameters", "may end in a different local optimum"),
        ("Random restarts", "–", "independent EM runs", "more chances to find the best optimum"),
    ])
    ui.experiments([
        ("Set EM iterations to 1, then 5, 20, 150.", "The learned 'loaded' bars move from random toward the grey truth.",
         "Each iteration re-counts the dice using better soft labels."),
        ("Look at the restarts plot.", "Most restarts end at almost the same likelihood; occasionally one is lower.",
         "EM finds a local optimum. Keep the best of several restarts."),
        ("In the casino tab, set Number of rolls = 300, then come back.",
         "The dice are still learned roughly, but the switching probabilities can be several times off.",
         "With few loaded spells there are few switch *events* to count. EM needs events, not just samples."),
    ])

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/hmm.py`):")
    ui.source(hmm.forward, hmm.backward, hmm.posteriors, hmm.viterbi, hmm.baum_welch)
