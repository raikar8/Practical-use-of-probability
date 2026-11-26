import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import hmm

ui.header("Level 4", "Hidden Markov models",
          "The state is discrete and never observed directly. How do we track it, decode it, and learn the model "
          "without labels?")

st.markdown(
    r"""
An HMM is the Level 0 Markov chain with the state **hidden**: each state $k$ emits observations with likelihood
$b_k(x_t)=p(x_t\mid s_t=k)$. One model answers three different questions:

| Question | Algorithm | Uses data from | Typical use |
|---|---|---|---|
| Which state am I in **now**? $P(s_t\mid x_{1:t})$ | **Forward** (filtering) | past only → runs in real time | live VAD, live regime alarm |
| Which state was I in **then**? $P(s_t\mid x_{1:T})$ | **Forward–backward** (smoothing) | past *and* future | offline labelling, EM |
| What is the single best **path**? $\arg\max P(s_{1:T}\mid x_{1:T})$ | **Viterbi** | everything | speech recognition, gene finding, decoding |
| What are $\pi, A, b$? | **Baum–Welch** (EM) | unlabelled data | training |

**Forward recursion:** predict with the chain, then weigh by the evidence and normalise. This is the same
predict/update loop as the Kalman filter, for a discrete state:

$$\alpha_t(j) \;\propto\; b_j(x_t)\sum_i \alpha_{t-1}(i)\,A_{ij}.$$
"""
)
with st.expander("HMM vs Kalman filter, side by side"):
    st.markdown(
        r"""
| | Kalman filter (L3) | HMM forward (L4) |
|---|---|---|
| hidden state | continuous vector, Gaussian belief $\mathcal N(\hat x, P)$ | one of $K$ labels, belief vector $\alpha_t$ |
| predict | $\hat x^- = F\hat x$, $P^- = FPF^T+Q$ | $\alpha^-_t = \alpha_{t-1}A$ |
| update | Kalman gain on the innovation | multiply by $b_k(x_t)$, normalise |
| smoother | RTS | backward pass |
| best path | the smoother mean | Viterbi |
| when nothing is Gaussian or discrete | → particle filter (Level 5) | → particle filter (Level 5) |
"""
    )

t_casino, t_learn, t_code = st.tabs(["The dishonest casino", "Learning with Baum–Welch", "Code"])

# --------------------------------------------------------------------------- casino
with t_casino:
    st.markdown(
        "A casino secretly swaps a **fair die** for a **loaded die** that favours 6, then swaps back. You only see "
        "the rolls. When was the loaded die in play?"
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        T = st.slider("Number of rolls", 100, 5000, 2000, 100)
        p_fl = st.slider("P(switch fair → loaded) per roll", 0.001, 0.2, 0.02, 0.001, format="%.3f")
        p_lf = st.slider("P(switch loaded → fair) per roll", 0.01, 0.5, 0.10, 0.01)
        p6 = st.slider("Loaded die: P(6)", 0.2, 0.9, 0.5, 0.05)
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
    ui.break_it("Set P(6) on the loaded die to 0.25: the two dice look almost the same and no algorithm can "
                "separate them. Information in the emissions limits everything. Then make switching very frequent "
                "(0.2 / 0.5): the 'sticky' prior disappears and decoding collapses toward roll-by-roll guessing.")
    ui.in_practice("Speech recognition (phoneme HMMs, pre-deep-learning and still inside hybrid systems), gene "
                   "finding, part-of-speech tagging, fault diagnosis, regime-switching finance models, and the "
                   "decoding in mobile phones (Viterbi decoding of convolutional codes).")

# --------------------------------------------------------------------------- learning
with t_learn:
    st.markdown(
        r"""
In reality nobody hands you $A$ and the dice probabilities. **Baum–Welch** (EM) learns them from the rolls alone:

1. **E-step:** with the current guess, run forward–backward to get the expected time in each state
   ($\gamma_t$) and the expected number of each transition ($\xi_t$).
2. **M-step:** re-estimate by counting. $A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t\gamma_t(i)}$, and
   $b_k(m)$ = the expected fraction of time in $k$ that emitted symbol $m$.

Each iteration can only **increase** the likelihood. It can still get stuck in a local optimum, so practitioners
restart from several initial guesses.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        iters = st.slider("EM iterations (max)", 1, 300, 150)
        init_seed = st.number_input("Initial-guess seed", 0, 9999, 3)
        n_restarts = st.slider("Random restarts to compare", 1, 12, 8)
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
    ui.break_it("Drop to 300–600 rolls: the emissions are still roughly right, but the switching probabilities "
                "can be off several-fold, because there are only a handful of loaded runs to count. EM needs enough "
                "*events*, not just enough samples.")

with t_code:
    ui.source(hmm.forward, hmm.backward, hmm.posteriors, hmm.viterbi, hmm.baum_welch)
