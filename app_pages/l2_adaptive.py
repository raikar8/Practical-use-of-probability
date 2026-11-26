import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import adaptive as ad

ui.header("Level 2", "Adaptive filters: LMS, NLMS, RLS",
          "How do we keep estimating something that keeps changing, one sample at a time?")

st.markdown(
    r"""
**Scenario: an echo canceller.** A loudspeaker plays $x[n]$; the microphone hears it through an unknown echo path
$\mathbf h$ plus noise: $d[n]=\mathbf h^T\mathbf u[n]+v[n]$, with $\mathbf u[n]=[x[n],\dots,x[n-M+1]]^T$.
We learn $\mathbf w\approx\mathbf h$ online and subtract the predicted echo. Halfway through, someone moves in
the car and **the echo path changes**.
"""
)
with st.expander("The three update rules (open for the math)"):
    st.markdown(
        r"""
| Algorithm | Update | Cost / sample | Idea |
|---|---|---|---|
| **LMS** | $\mathbf w \leftarrow \mathbf w + \mu\, e[n]\,\mathbf u[n]$ | $O(M)$ | one stochastic-gradient step on $E[e^2]$ |
| **NLMS** | $\mathbf w \leftarrow \mathbf w + \dfrac{\mu}{\epsilon+\lVert\mathbf u\rVert^2}\, e[n]\,\mathbf u[n]$ | $O(M)$ | LMS with the step normalised by input power; stable for $0<\mu<2$ |
| **RLS** | $\mathbf k=\dfrac{\mathbf P\mathbf u}{\lambda+\mathbf u^T\mathbf P\mathbf u},\;\mathbf w\leftarrow\mathbf w+\mathbf k\,e,\;\mathbf P\leftarrow\dfrac{\mathbf P-\mathbf k\mathbf u^T\mathbf P}{\lambda}$ | $O(M^2)$ | exact weighted least squares; $\mathbf P$ is (scaled) inverse input correlation |

The batch **Wiener** solution $\mathbf w_o = \mathbf R^{-1}\mathbf p$ is where they all head.
**RLS is a Kalman filter** whose state is the weight vector, with the forgetting factor $\lambda$ playing the
role of process noise. That bridge leads straight to Level 3.
"""
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
    c1, c2 = st.columns([1, 2.2])
    with c1:
        M = st.select_slider("Filter taps M", [4, 8, 16, 32], 16)
        N = st.select_slider("Samples", [2000, 4000, 8000], 4000)
        snr = st.slider("Echo-to-noise ratio (dB)", 0, 60, 30)
        ar = st.slider("Input colour: AR(1) coefficient (0 = white, 0.95 = speech-like)", 0.0, 0.98, 0.0, 0.01)
        change = st.toggle("Echo path changes halfway", True)
        st.divider()
        mu_lms = st.select_slider("LMS step μ", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], 0.01)
        mu_nlms = st.select_slider("NLMS step μ", [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9], 0.5)
        lam = st.select_slider("RLS forgetting λ", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], 0.99)
        seed = st.number_input("Random seed", 0, 9999, 4)
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
        ui.show(ui.style(fig, "Misalignment ‖h − w‖² / ‖h‖² (lower is better)", "sample n", "dB"))
        final_h = h2 if change else h1
        fig2 = go.Figure()
        fig2.add_bar(x=np.arange(M), y=final_h, name="true echo path", marker_color=ui.C["truth"])
        for i, (k, nm) in enumerate(names.items()):
            w = out[k]["W"][-1]
            fig2.add_scatter(x=np.arange(M), y=w, name=nm, mode="markers",
                             marker=dict(color=ui.C[i + 1], size=9, line=dict(width=2, color="white")))
        ui.show(ui.style(fig2, "Final taps vs the true echo path", "tap index", "coefficient", height=300))
        cols = st.columns(3)
        for col, (k, nm) in zip(cols, names.items()):
            v = mis[k][-200:]
            col.metric(f"{nm}: final misalignment", "diverged" if np.isnan(v).all() else f"{np.nanmean(v):.1f} dB")
    ui.takeaway("RLS converges in roughly 2M samples regardless of input colour, at O(M²) cost. LMS is cheap but "
                "slows down badly with coloured input, because its speed depends on the eigenvalue spread of R. "
                "NLMS is the practical middle ground and is the workhorse of real echo cancellers.")
    ui.break_it("(1) Push LMS μ to 0.2 with M=32: it diverges. (2) Set AR = 0.95: LMS crawls while RLS barely "
                "notices. (3) Set λ = 0.9999: RLS is very precise but slow to re-converge after the change. "
                "λ trades tracking against noise, just like Q in a Kalman filter.")
    ui.in_practice("Acoustic echo cancellation in cars and conference phones, feedback cancellation in hearing "
                   "aids, channel equalisation in modems, and active noise control in headphones.")

with t_wiener:
    w_o = ad.wiener_solution(p["x"][: N // 2], p["d"][: N // 2], M)
    fig = go.Figure()
    fig.add_bar(x=np.arange(M), y=h1, name="true h (first half)", marker_color=ui.C["truth"])
    fig.add_scatter(x=np.arange(M), y=w_o, name="Wiener R⁻¹p from the first half", mode="markers",
                    marker=dict(color=ui.C[2], size=10, line=dict(width=2, color="white")))
    ui.show(ui.style(fig, "The batch optimum adaptive filters converge to", "tap index", "coefficient"))
    st.markdown(
        "The Wiener solution needs all the data and a matrix inverse. It is the *target*, not something you can "
        "run in real time. Adaptive filters reach it recursively and keep following it when it moves."
    )

with t_code:
    ui.source(ad.lms, ad.nlms, ad.rls, ad.wiener_solution, ad.misalignment_db)
