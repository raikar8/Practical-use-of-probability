import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import adaptive as ad

ui.lab_intro(
    "l2",
    goal="Race three adaptive filters (LMS, NLMS, RLS) at learning an unknown echo path from scratch, then "
         "re-learning it after it suddenly changes. See the speed–precision trade-off each one makes.",
    on_screen=[
        "**Top plot, the learning curve:** misalignment in dB (how far each filter's taps are from the true echo "
        "path; lower is better, −30 dB ≈ within 3%). Theory eq. 8.",
        "**Bottom plot:** the true echo path (grey bars) and each filter's final taps (dots).",
        "**Wiener tab:** the batch optimum $R^{-1}\\mathbf p$ that all three are chasing (Theory §2).",
    ],
    start_here="Keep the defaults and read the top plot left to right: who reaches −20 dB first, who goes lowest, "
               "and who recovers fastest after the dashed line where the echo path changes?",
)

t_play, t_wiener, t_code = st.tabs(["Playground", "Wiener: the target", "Code"])


@st.cache_data(show_spinner="Running the adaptive filters…")
def _run(M, N, snr_db, ar, change, mu_lms, mu_nlms, lam, seed):
    rng = np.random.default_rng(seed)
    h1 = ad.room_like_ir(M, 4.0 / M, rng)
    h2 = ad.room_like_ir(M, 4.0 / M, rng) if change else None
    noise = 10 ** (-snr_db / 20)
    p = ad.make_system_id(N, h1, h2, N // 2 if change else None, noise, ar, rng)
    out = {"lms": ad.lms(p["x"], p["d"], M, mu_lms), "nlms": ad.nlms(p["x"], p["d"], M, mu_nlms),
           "rls": ad.rls(p["x"], p["d"], M, lam)}
    mis = {k: ad.misalignment_db(v["W"], p["H"]) for k, v in out.items()}
    return p, out, mis, h1, h2


with t_play:
    st.markdown(
        r"""
**The situation.** A loudspeaker plays a signal $x[n]$; the microphone hears it through an unknown **echo path**
$\mathbf h$ ($M$ taps) plus noise. Each filter adjusts its own taps $\mathbf w$ every sample to make its prediction
of the echo match the microphone (Theory §1). Halfway through, someone moves in the car and **$\mathbf h$ changes**.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("**The world**")
        M = st.select_slider("Filter taps M", [4, 8, 16, 32], 16,
                             help="Length of the echo path and of each adaptive filter. Longer = more to learn.")
        N = st.select_slider("Samples", [2000, 4000, 8000], 4000, help="Length of the simulation.")
        snr = st.slider("Echo-to-noise ratio (dB)", 0, 60, 30,
                        help="Echo power vs microphone noise. Sets the best misalignment any filter can reach "
                             "(about −ENR dB).")
        ar = st.slider("Input colour: AR(1) coefficient (0 = white, 0.95 = speech-like)", 0.0, 0.98, 0.0, 0.01,
                       help="How correlated neighbouring input samples are. Speech is strongly coloured, and that "
                            "slows LMS/NLMS down (Theory §3).")
        change = st.toggle("Echo path changes halfway", True, help="Replace h by a new random echo path at the midpoint.")
        st.markdown("**The filters**")
        mu_lms = st.select_slider("LMS step μ", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], 0.01,
                                  help="LMS step size. Larger = faster but noisier; too large = divergence.")
        mu_nlms = st.select_slider("NLMS step μ", [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9], 0.5,
                                   help="NLMS normalised step, stable between 0 and 2. 1 is the fastest.")
        lam = st.select_slider("RLS forgetting λ", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 0.99,
                               help="RLS memory ≈ 1/(1−λ) samples. Close to 1 = precise but slow to track changes.")
        seed = st.number_input("Random seed", 0, 9999, 4, help="Different echo paths and signals.")
    p, out, mis, h1, h2 = _run(M, N, snr, ar, change, mu_lms, mu_nlms, lam, int(seed))
    names = {"lms": "LMS", "nlms": "NLMS", "rls": "RLS"}
    with c2:
        fig = go.Figure()
        n = np.arange(N)
        for i, (k, nm) in enumerate(names.items()):
            fig.add_scatter(x=n[::4], y=mis[k][::4], name=nm, line=dict(color=ui.C[i + 1]))
        if change:
            fig.add_vline(x=N // 2, line=dict(color=ui.C["truth"], dash="dash"), annotation_text="echo path changes")
        fig.update_yaxes(range=[-max(snr, 10) - 15, 10])
        ui.show(ui.style(fig, "Learning curve: misalignment ‖h − w‖² / ‖h‖² (lower is better)", "sample n", "dB"))
        final_h = h2 if change else h1
        fig2 = go.Figure()
        fig2.add_bar(x=np.arange(M), y=final_h, name="true echo path", marker_color=ui.C["truth"])
        for i, (k, nm) in enumerate(names.items()):
            w = out[k]["W"][-1]
            fig2.add_scatter(x=np.arange(M), y=w, name=nm, mode="markers",
                             marker=dict(color=ui.C[i + 1], size=9, line=dict(width=2, color="white")))
        ui.show(ui.style(fig2, "Final taps vs the true echo path", "tap index (delay in samples)", "coefficient",
                         height=300))
        cols = st.columns(3)
        for col, (k, nm) in zip(cols, names.items()):
            v = mis[k][-200:]
            col.metric(f"{nm}: final misalignment", "diverged" if np.isnan(v).all() else f"{np.nanmean(v):.1f} dB")
    ui.controls([
        ("Filter taps M", r"$M$", "number of coefficients to learn", "slower convergence; RLS cost grows as M²"),
        ("Echo-to-noise ratio", "–", "echo vs noise power", "a lower floor (better final misalignment)"),
        ("Input colour", "AR coefficient", "correlation between neighbouring input samples",
         "LMS (and NLMS) slow down; RLS doesn't"),
        ("Echo path changes", r"$\mathbf h \to \mathbf h'$", "a sudden change mid-way", "tests tracking"),
        ("LMS step μ", r"$\mu$ (eq. 4)", "step size", "faster, noisier, eventually diverges"),
        ("NLMS step μ", r"$\mu$ (eq. 5)", "normalised step", "faster up to μ ≈ 1, noisier floor"),
        ("RLS forgetting λ", r"$\lambda$ (eq. 6)", "memory ≈ 1/(1−λ) samples", "more precise, slower to track"),
    ])
    ui.experiments([
        ("Keep the defaults and compare the three curves.",
         "RLS reaches −20 dB in ~20 samples, NLMS in ~90, LMS in ~240. After the change RLS needs ~250.",
         "RLS solves least squares exactly from the start. After a change it must first *forget* the old path, "
         "which takes about 1/(1−λ) = 100 samples."),
        ("Set RLS λ = 0.9999.", "Before the change RLS goes very low (≈ −49 dB); after it, it barely recovers.",
         "A 10,000-sample memory averages noise beautifully but is anchored to the old echo path."),
        ("Set RLS λ = 0.9.", "RLS now re-converges in ~30 samples, but its floor rises to ≈ −30 dB.",
         "Short memory = fast tracking + noisy estimate. This is the same trade-off as Q in a Kalman filter."),
        ("Reset λ to 0.99 and set Input colour = 0.95.",
         "LMS stalls around −10 dB; NLMS slows a lot; RLS is almost unaffected.",
         "Coloured input stretches the error bowl (eigenvalue spread). RLS's matrix P undoes it."),
        ("Set M = 32 and LMS μ = 0.2.", "The LMS curve disappears and its metric says 'diverged'.",
         "μ above ~2/(M·input power) makes each step overshoot more than the last."),
    ])
    ui.takeaway("RLS converges in roughly 2M samples regardless of input colour, at O(M²) cost. LMS is cheap but "
                "slows down badly with coloured input, because its speed depends on the eigenvalue spread of R. "
                "NLMS is the practical middle ground and is the workhorse of real echo cancellers.")
    ui.in_practice("Acoustic echo cancellation in cars and conference phones, feedback cancellation in hearing "
                   "aids, channel equalisation in modems, and active noise control in headphones.")

with t_wiener:
    st.markdown(
        r"""
**What this shows.** The **Wiener solution** $\mathbf w_o = R^{-1}\mathbf p$ (Theory eq. 3) computed in one go from
the first half of the data. It's the bottom of the error bowl: the best any linear filter can do. It needs all the
data and a matrix inverse, so it can't run in real time or follow changes. The adaptive filters reach it
recursively.
"""
    )
    w_o = ad.wiener_solution(p["x"][: N // 2], p["d"][: N // 2], M)
    fig = go.Figure()
    fig.add_bar(x=np.arange(M), y=h1, name="true h (first half)", marker_color=ui.C["truth"])
    fig.add_scatter(x=np.arange(M), y=w_o, name="Wiener R⁻¹p from the first half", mode="markers",
                    marker=dict(color=ui.C[2], size=10, line=dict(width=2, color="white")))
    ui.show(ui.style(fig, "The batch optimum that adaptive filters converge to", "tap index", "coefficient"))

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/adaptive.py`):")
    ui.source(ad.lms, ad.nlms, ad.rls, ad.wiener_solution, ad.misalignment_db)
