import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.sims import enhance as en
from ssp.sims import feedback as fb
from ssp.sims import speech as sp

ui.header("Domain demo · Speech & hearing", "Noise suppression and hearing-aid feedback cancellation",
          "Two estimation problems inside every hearing aid and car hands-free system: how loud is the speech under "
          "this noise, and what is the sound leaking from the receiver back into the microphone?")

t_ns, t_fb = st.tabs(["Noise suppression (MMSE)", "Feedback cancellation (hearing aid)"])

# --------------------------------------------------------------------------- noise suppression
with t_ns:
    with st.expander("The estimators (open for the math)"):
        st.markdown(
            r"""
In each STFT bin, $Y = S + N$. With noise power $\lambda_N$, define the **a-posteriori SNR**
$\gamma = |Y|^2/\lambda_N$ (observed) and the **a-priori SNR** $\xi = E|S|^2/\lambda_N$ (hidden). The output is
$\hat S = G(\xi,\gamma)\,Y$.

| Gain | Estimates | Formula |
|---|---|---|
| Spectral subtraction | power, by moment matching | $\sqrt{\max(1-1/\gamma,\,0)}$ |
| Wiener | complex $S$, MMSE | $\xi/(1+\xi)$ |
| MMSE-STSA | amplitude $\lvert S\rvert$, MMSE (Ephraim–Malah 1984) | $\tfrac{\sqrt{\pi v}}{2\gamma}\,e^{-v/2}\big[(1+v)I_0(\tfrac v2)+vI_1(\tfrac v2)\big],\ v=\tfrac{\xi\gamma}{1+\xi}$ |
| MMSE-LSA | $\log\lvert S\rvert$, MMSE (1985) | $\tfrac{\xi}{1+\xi}\exp\!\big(\tfrac12 E_1(v)\big)$ |

**Decision-directed** $\xi$: $\hat\xi_l = a\,\frac{G_{l-1}^2|Y_{l-1}|^2}{\lambda_N} + (1-a)\max(\gamma_l-1,0)$. The
second term alone ($a=0$) is the **ML estimate**. It fluctuates wildly in noise-only bins and causes *musical
noise*. $a\approx0.98$ adds a strong prior from the previous frame.

**Noise tracking** (SPP-MMSE, Gerkmann & Hendriks 2012): for each bin, compute the posterior probability that
speech is present, then update $\lambda_N$ with $E[|N|^2\mid Y] = (1-p)|Y|^2 + p\,\lambda_N$. It's Bayes' rule
from L0, run about 16,000 times a second (257 bins × 62.5 frames).
"""
        )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        noise = st.selectbox("Noise", ["car", "white", "babble"], key="ns_noise", format_func={
            "car": "Car cabin", "white": "White", "babble": "Babble"}.get)
        snr = st.slider("Input SNR (dB)", -5, 20, 5, key="ns_snr")
        ramp = st.slider("Noise gets louder half-way by (dB)", 0, 15, 10,
                         help="The car accelerates: a noise estimate taken at the start goes stale")
        gain = st.selectbox("Gain rule", list(en.GAINS), index=3)
        tracker = st.radio("Noise estimate", ["spp", "initial", "oracle"], format_func={
            "spp": "SPP-MMSE tracker (adaptive)", "initial": "First 0.15 s only (frozen)",
            "oracle": "True average noise spectrum (cheating)"}.get)
        dd = st.select_slider("Decision-directed a (0 = ML estimate of ξ)", [0.0, 0.5, 0.9, 0.95, 0.98, 0.99], 0.98)
        gmin = st.slider("Gain floor (dB)", -40, -5, -25)
        nseed = st.number_input("Random seed", 0, 9999, 1, key="ns_seed")

    @st.cache_data(show_spinner="Enhancing…")
    def _ns(noise, snr, ramp, gain, tracker, dd, gmin, seed):
        sc = en.make_scene(10.0, float(snr), noise, float(ramp), int(seed))
        r = en.enhance(sc["noisy"], gain, tracker, dd_alpha=dd, g_min_db=float(gmin), noise_oracle=sc["noise"])
        return sc, r

    sc, r = _ns(noise, snr, ramp, gain, tracker, dd, gmin, int(nseed))
    with c2:
        def _spec(P):
            return 10 * np.log10(P[:, ::2] + 1e-10)
        fo, to = r["f"], r["t"][::2]
        _, _, Yc = en.stft(sc["clean"])
        _, _, Ye = en.stft(r["out"])
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=("noisy input", "enhanced output", "gain G (white = pass)"))
        vmax = _spec(r["P"]).max()
        for i, Z in enumerate((_spec(r["P"]), _spec(np.abs(Ye) ** 2))):
            fig.add_heatmap(x=to, y=fo, z=Z, zmin=vmax - 70, zmax=vmax, colorscale="Viridis", showscale=False,
                            hoverinfo="skip", row=i + 1, col=1)
        fig.add_heatmap(x=to, y=fo, z=20 * np.log10(r["G"][:, ::2]), zmin=gmin, zmax=0, colorscale="Greys_r",
                        showscale=False, hoverinfo="skip", row=3, col=1)
        for i in range(1, 4):
            fig.update_yaxes(title="Hz", range=[0, 6000], row=i, col=1)
        fig.update_xaxes(title="time (s)", row=3, col=1)
        if ramp:
            for i in range(1, 4):
                fig.add_vline(x=5.0, line=dict(color="white", dash="dot"), row=i, col=1)
        ui.show(ui.style(fig, None, height=620, legend=False))
        res = {"input": sc["noisy"], "output": r["out"]}
        tbl = pd.DataFrame({
            "SNR (dB)": {k: en.snr_db(sc["clean"], v) for k, v in res.items()},
            "segmental SNR (dB)": {k: en.segmental_snr_db(sc["clean"], v) for k, v in res.items()},
            "noise reduction in pauses (dB)": {"input": 0.0,
                                               "output": en.noise_attenuation_db(sc["noisy"], r["out"], sc["mask"])}})
        st.dataframe(tbl.style.format("{:.1f}"), width="stretch")
        a1, a2, a3 = st.columns(3)
        norm = np.abs(sc["noisy"]).max() + 1e-9
        a1.caption("Noisy")
        a1.audio(np.clip(sc["noisy"] / norm, -1, 1), sample_rate=sp.FS)
        a2.caption("Enhanced")
        a2.audio(np.clip(r["out"] / norm, -1, 1), sample_rate=sp.FS)
        a3.caption("Clean (hidden truth)")
        a3.audio(np.clip(sc["clean"] / norm, -1, 1), sample_rate=sp.FS)
        f2 = go.Figure()
        k500 = int(np.argmin(np.abs(r["f"] - 500)))
        f2.add_scatter(x=r["t"], y=10 * np.log10(r["P"][k500] + 1e-12), name="|Y|² at 500 Hz",
                       line=dict(color=ui.C["truth"], width=1))
        f2.add_scatter(x=r["t"], y=10 * np.log10(r["noise_psd"][k500]), name="noise estimate λ_N",
                       line=dict(color=ui.C[1], width=2.5))
        ui.show(ui.style(f2, "Noise tracking in one frequency bin", "time (s)", "dB", height=260))
    ui.takeaway("The MMSE gains act like a soft, per-bin Bayesian decision: strong bins pass, weak ones are pulled "
                "down to the floor. The decision-directed prior makes the gain smooth in time. The noise tracker "
                "matters as much as the gain rule: when the car speeds up, a frozen estimate under-suppresses "
                "everything that follows.")
    ui.break_it("(1) a = 0 (ML ξ): listen for 'musical noise', isolated tones twinkling in the pauses. (2) Noise "
                "estimate 'first 0.15 s only' with a 10 dB ramp: the second half stays noisy. (3) Gain floor −40 dB "
                "at 0 dB SNR: more suppression, but speech gets thin and watery. Real products keep the floor "
                "around −12 to −20 dB.")
    ui.in_practice("Variants of these estimators sit in phone and car hands-free chains, hearing aids, and as the "
                   "statistical front end (or the training target) of DNN noise suppressors.")

# --------------------------------------------------------------------------- feedback cancellation
with t_fb:
    st.markdown(
        r"""
A hearing aid amplifies by $K$ and its receiver sits millimetres from the microphone. Sound leaks back through
the **feedback path** $F$ and the loop howls once $K$ exceeds the **maximum stable gain**
$\text{MSG} = -20\log_{10}\max_\omega|F(\omega)|$. A canceller subtracts $\hat F * u$ from the microphone
signal, so the loop only sees $F-\hat F$. Half-way through, the user raises a phone to the ear and $F$ jumps.

* **None:** no canceller.
* **Fixed:** $\hat F$ measured once at the fitting appointment (what a fitting tool does), never updated.
* **NLMS:** starts from the fitting measurement and keeps adapting (L2's algorithm in a closed loop).
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        src = st.selectbox("Incoming sound", ["speech", "music", "noise"],
                           format_func={"speech": "Speech", "music": "Music (sustained tones)",
                                        "noise": "Broadband noise"}.get)
        margin = st.slider("Hearing-aid gain above the no-canceller MSG (dB)", 0, 20, 10)
        jump = st.slider("Feedback path increase at 2.5 s (dB)", 0, 12, 6)
        mu = st.select_slider("NLMS step μ", [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02], 0.002)
        fseed = st.number_input("Random seed", 0, 9999, 1, key="fb_seed")

    @st.cache_data(show_spinner="Running the hearing-aid loop…")
    def _fb(src, margin, jump, mu, seed):
        cfg = fb.HAConfig(source=src, gain_margin_db=float(margin), path_change_db=float(jump), mu=mu, seed=int(seed))
        return {m: fb.simulate(cfg, m) for m in ("none", "fixed", "nlms")}

    out = _fb(src, margin, jump, mu, int(fseed))
    names = {"none": "No canceller", "fixed": "Fixed (fitting measurement)", "nlms": "Adaptive NLMS"}
    cols = {"none": ui.C[8], "fixed": ui.C[2], "nlms": ui.C[1]}
    with c2:
        r0 = out["nlms"]
        fig = go.Figure()
        for m, res in out.items():
            fig.add_scatter(x=res["t"], y=res["msg"], name=f"MSG with {names[m].lower()}", line=dict(color=cols[m]))
        fig.add_hline(y=r0["K_db"], line=dict(color=ui.C["truth"], dash="dash"),
                      annotation_text="hearing-aid gain: unstable below this line")
        fig.add_vline(x=r0["change_t"], line=dict(color=ui.C["truth"], dash="dot"),
                      annotation_text="phone to ear", annotation_position="bottom right")
        ui.show(ui.style(fig, "Maximum stable gain of the remaining loop (higher = safer)", "time (s)", "dB",
                         height=330))
        f2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=[names[m] for m in out])
        for i, (m, res) in enumerate(out.items()):
            tt = np.arange(res["out"].size) / sp.FS
            f2.add_scatter(x=tt[::8], y=res["out"][::8], line=dict(color=cols[m], width=1), showlegend=False,
                           row=i + 1, col=1)
            f2.update_yaxes(range=[-1.05, 1.05], row=i + 1, col=1)
        f2.update_xaxes(title="time (s)", row=3, col=1)
        ui.show(ui.style(f2, "Receiver output (±1 = saturated: howling)", height=420, legend=False))
        tbl = pd.DataFrame({names[m]: {"time howling": fb.howling_fraction(res),
                                       "final misalignment (dB)": res["misalignment"][-1]}
                            for m, res in out.items()}).T
        st.dataframe(tbl.style.format({"time howling": "{:.0%}", "final misalignment (dB)": "{:.1f}"}),
                     width="stretch")
        st.caption("Listen to the receiver output (lower the volume first: howling is loud)")
        ac = st.columns(3)
        for col_, (m, res) in zip(ac, out.items()):
            col_.caption(names[m])
            col_.audio(np.clip(res["out"], -1, 1) * 0.3, sample_rate=sp.FS)
    ui.takeaway("A fixed canceller is only as good as the last measurement. The adaptive filter re-identifies the "
                "new path within a fraction of a second and restores the stability margin. The cost is a closed-loop "
                "**bias**: the receiver signal is a delayed copy of the input, so the filter partly models the input "
                "itself. In the first half nothing changes, yet the NLMS margin sags below the fixed canceller's. "
                "That's the bias, and it's worst for tonal input like music. (MSG ignores phase, so it's a "
                "conservative bound: dipping slightly below the gain line doesn't always howl.)")
    ui.break_it("Music with μ = 0.02: fast re-convergence, but the margin collapses during tones (entrainment, "
                "heard as warbling artefacts). μ = 0.0005: no bias problems, but the howl after the phone-to-ear "
                "event lasts much longer. Products break the input/output correlation with decorrelation tricks "
                "(prediction-error prewhitening, small frequency shifts, probe noise) and run howling detectors "
                "alongside.")
    ui.in_practice("Every modern hearing aid runs an adaptive feedback canceller. The same structure (an adaptive "
                   "FIR on the loudspeaker signal) is the car's acoustic echo canceller from L2, here inside a "
                   "closed loop. A real-time C++ version of this exact loop is in `cpp/include/ssp/afc.hpp`: "
                   "block-based, allocation-free, and tested to reproduce this page's simulation.")
