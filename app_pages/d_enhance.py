import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.sims import enhance as en
from ssp.sims import feedback as fb
from ssp.sims import speech as sp

ui.lab_intro(
    "enhance",
    goal="Two hearing-aid / hands-free problems, hands-on: (1) suppress background noise by estimating, in every "
         "time–frequency bin, how much is speech; (2) stop a hearing aid howling by cancelling the sound that leaks "
         "from its loudspeaker back into its microphone.",
    on_screen=[
        "**Noise suppression tab:** spectrograms of the noisy input, the enhanced output and the gain applied (white "
        "= kept, black = suppressed), quality numbers, audio players, and the noise tracker in one frequency bin "
        "(Theory Part A)",
        "**Feedback cancellation tab:** how much gain is safe over time with no canceller, a fixed one and an "
        "adaptive one; the loudspeaker output (±1 = howling); audio (Theory Part B)",
    ],
    start_here="In the noise tab, play *Noisy* then *Enhanced*. Then look at the gain spectrogram: speech harmonics "
               "show up as white horizontal stripes, and pauses are black.",
)

t_ns, t_fb = st.tabs(["Noise suppression (MMSE)", "Feedback cancellation (hearing aid)"])

# --------------------------------------------------------------------------- noise suppression
with t_ns:
    st.markdown(
        r"""
**The situation.** Speech is mixed with noise. The signal is split into short frames and frequency bins (the
STFT), and every bin is multiplied by a gain between 0 and 1 chosen by a statistical estimator (Theory A3).
The estimator needs two things: the **noise power** in each bin (tracked over time, Theory A5) and the
**a-priori SNR** ξ (estimated with the decision-directed rule, Theory A4). The x-axis of each spectrogram is time,
the y-axis frequency, and the colour is power.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        ns_src = st.radio("Audio", ["Synthetic speech in noise (truth known)", "Your own voice: record or upload"],
                          key="ns_src", help="Synthetic audio has a known clean signal, so SNR can be measured.")
        synthetic = ns_src.startswith("Synthetic")
        rec = None
        if synthetic:
            noise = st.selectbox("Noise", ["car", "white", "babble"], key="ns_noise", format_func={
                "car": "Car cabin", "white": "White", "babble": "Babble"}.get, help="Type of background noise.")
            snr = st.slider("Input SNR (dB)", -5, 20, 5, key="ns_snr", help="Speech vs noise level at the start.")
            ramp = st.slider("Noise gets louder half-way by (dB)", 0, 15, 10,
                             help="The car accelerates: a noise estimate taken at the start goes stale")
        else:
            rec = ui.real_audio("ns")
            ramp = 0
        gain = st.selectbox("Gain rule", list(en.GAINS), index=3,
                            help="Which estimator turns the SNRs into a gain (Theory A3).")
        trackers = ["spp", "initial", "oracle"] if synthetic else ["spp", "initial"]
        tracker = st.radio("Noise estimate", trackers, format_func={
            "spp": "SPP-MMSE tracker (adaptive)", "initial": "First 0.15 s only (frozen)",
            "oracle": "True average noise spectrum (cheating)"}.get,
            help="How λ_N is obtained: tracked continuously (SPP), measured once at the start, or the true value.")
        dd = st.select_slider("Decision-directed a (0 = ML estimate of ξ)", [0.0, 0.5, 0.9, 0.95, 0.98, 0.99], 0.98,
                              help="Weight on the previous frame's clean-speech estimate in Theory eq. 3. 0 = no "
                                   "smoothing (musical noise).")
        gmin = st.slider("Gain floor (dB)", -40, -5, -25,
                         help="The gain never goes below this, so some natural background remains.")
        if synthetic:
            nseed = st.number_input("Random seed", 0, 9999, 1, key="ns_seed")

    @st.cache_data(show_spinner="Enhancing…")
    def _ns(noise, snr, ramp, gain, tracker, dd, gmin, seed):
        sc = en.make_scene(10.0, float(snr), noise, float(ramp), int(seed))
        r = en.enhance(sc["noisy"], gain, tracker, dd_alpha=dd, g_min_db=float(gmin), noise_oracle=sc["noise"])
        return sc, r

    @st.cache_data(show_spinner="Enhancing your recording…")
    def _ns_real(x_bytes, gain, tracker, dd, gmin):
        x = np.frombuffer(x_bytes)
        return {"noisy": x, "clean": None}, en.enhance(x, gain, tracker, dd_alpha=dd, g_min_db=float(gmin))

    if synthetic:
        sc, r = _ns(noise, snr, ramp, gain, tracker, dd, gmin, int(nseed))
    elif rec is not None:
        sc, r = _ns_real(rec.astype(float).tobytes(), gain, tracker, dd, gmin)
    else:
        sc = r = None
        with c2:
            st.info("Record a few seconds (quiet first, then speech with pauses) or upload a WAV. Your recording is "
                    "processed here and not stored.")
    if r is not None:
        with c2:
            def _spec(P):
                return 10 * np.log10(P[:, ::2] + 1e-10)
            fo, to = r["f"], r["t"][::2]
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
            ui.show(ui.style(fig, None, height=620, legend=False, legend_below=True))
            if synthetic:
                res = {"input": sc["noisy"], "output": r["out"]}
                tbl = pd.DataFrame({
                    "SNR (dB)": {k: en.snr_db(sc["clean"], v) for k, v in res.items()},
                    "segmental SNR (dB)": {k: en.segmental_snr_db(sc["clean"], v) for k, v in res.items()},
                    "noise reduction in pauses (dB)": {"input": 0.0,
                                                       "output": en.noise_attenuation_db(sc["noisy"], r["out"], sc["mask"])}})
                st.dataframe(tbl.style.format("{:.1f}"), width="stretch")
            else:
                bl = en.blind_levels(sc["noisy"], r["out"])
                m1, m2 = st.columns(2)
                m1.metric("Noise reduction in pauses", f"{bl['noise reduction in pauses (dB)']:.1f} dB",
                          help="Quietest 20% of frames, which are almost surely pauses")
                m2.metric("Level change on loud speech", f"{bl['level change on loud speech (dB)']:+.1f} dB",
                          help="Loudest 20% of frames: close to 0 dB means speech is preserved")
                st.caption("No clean reference exists for a real recording, so these two numbers stand in for SNR: "
                           "lots of reduction in pauses with little change on loud speech is what you want.")
            cols_a = st.columns(3 if synthetic else 2)
            norm = np.abs(sc["noisy"]).max() + 1e-9
            cols_a[0].caption("Noisy")
            cols_a[0].audio(np.clip(sc["noisy"] / norm, -1, 1), sample_rate=sp.FS)
            cols_a[1].caption("Enhanced")
            cols_a[1].audio(np.clip(r["out"] / norm, -1, 1), sample_rate=sp.FS)
            if synthetic:
                cols_a[2].caption("Clean (hidden truth)")
                cols_a[2].audio(np.clip(sc["clean"] / norm, -1, 1), sample_rate=sp.FS)
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
    ui.controls([
        ("Noise, Input SNR", "–", "what is mixed with the speech", "lower SNR = harder"),
        ("Noise gets louder half-way", "–", "a change in noise level at 5 s", "tests the noise tracker"),
        ("Gain rule", r"$G(\xi,\gamma)$", "which estimator", "–"),
        ("Noise estimate", r"$\lambda_N$", "how the noise power is obtained", "–"),
        ("Decision-directed a", r"$a$ (eq. 3)", "smoothing of ξ over time", "smoother gain, less musical noise"),
        ("Gain floor", r"$G_{\min}$", "maximum attenuation", "more natural background, less suppression"),
    ])
    ui.experiments([
        ("Defaults (car noise, 5 dB, noise +10 dB at 5 s): read the table.",
         "SNR improves by about 9 dB (≈ −4 → ≈ +4); pauses get ≈ 9 dB quieter.",
         "The MMSE-LSA gain plus the SPP noise tracker, all running causally frame by frame."),
        ("Switch Noise estimate to 'First 0.15 s only'.", "The SNR improvement collapses (output ≈ −2 dB), and "
         "after 5 s the output spectrogram stays noisy.",
         "A frozen noise estimate can't know the car sped up, so the louder noise is treated as speech."),
        ("Switch to 'True average noise spectrum'.", "Output SNR ≈ 7 dB and ≈ 17 dB quieter pauses: the best case.",
         "This is the cheating upper bound; the gap to SPP is the price of having to *estimate* the noise."),
        ("Back to SPP; set a = 0 and listen.", "Twinkling tones appear in the pauses; the gain map turns speckled.",
         "The ML estimate of ξ jumps with every random noise fluctuation: musical noise."),
        ("Try Spectral subtraction.", "Pauses get only ≈ 2 dB quieter, and the output SNR drops below zero.",
         "Without the smoothed ξ, subtraction is noisy and weak. Statistical gains do much better."),
    ])
    ui.in_practice("Variants of these estimators sit in phone and car hands-free chains, hearing aids, and as the "
                   "statistical front end (or the training target) of DNN noise suppressors.")

# --------------------------------------------------------------------------- feedback cancellation
with t_fb:
    st.markdown(
        r"""
**The situation.** A hearing aid amplifies sound, and its loudspeaker (receiver) leaks back into its microphone
through the **feedback path** $F$. Above the **maximum stable gain** (MSG, Theory eq. 6) the loop howls. A
canceller subtracts its estimate of the leak. At 2.5 s the user raises a phone to the ear and $F$ jumps. Three
cancellers compete:

* **None:** no canceller.
* **Fixed:** $\hat F$ measured once at the fitting appointment (what a fitting tool does), never updated.
* **NLMS:** starts from the fitting measurement and keeps adapting (L2's algorithm in a closed loop).
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        src = st.selectbox("Incoming sound", ["speech", "music", "noise"],
                           format_func={"speech": "Speech", "music": "Music (sustained tones)",
                                        "noise": "Broadband noise"}.get,
                           help="What the hearing aid is listening to. Tonal sounds are the hardest for the canceller.")
        margin = st.slider("Hearing-aid gain above the no-canceller MSG (dB)", 0, 20, 10,
                           help="How much more gain than is safe without a canceller. 0 = right at the edge.")
        jump = st.slider("Feedback path increase at 2.5 s (dB)", 0, 12, 6,
                         help="How much stronger the leak gets when the phone is raised to the ear.")
        mu = st.select_slider("NLMS step μ", [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02], 0.002,
                              help="Adaptation speed of the NLMS canceller: faster tracking vs more closed-loop bias.")
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
    ui.controls([
        ("Incoming sound", "x", "speech, music or noise", "tonal input causes more bias"),
        ("Gain above MSG", "K vs eq. 6", "how hard the aid amplifies", "a smaller stability margin"),
        ("Feedback path increase", "change in F", "the phone-to-ear event", "a bigger jump to re-learn"),
        ("NLMS step μ", r"$\mu$", "adaptation speed", "faster recovery, more bias"),
    ])
    ui.experiments([
        ("Defaults: read 'time howling'.", "No canceller ≈ 99%, fixed ≈ 51% (all after 2.5 s), adaptive ≈ 15%.",
         "Fixed works until the path changes; the adaptive filter re-learns the new path within a fraction of a second."),
        ("Set μ = 0.0005.", "The adaptive canceller now howls ≈ 50% of the time.",
         "Too slow to re-learn after the change. Same trade-off as LMS's μ in Level 2."),
        ("Set μ = 0.02.", "Howling after the change drops to ≈ 5%, but before 2.5 s its MSG curve sits *below* the "
         "fixed canceller's.",
         "Fast adaptation also chases the incoming sound itself (closed-loop bias), eating into the margin."),
        ("Switch to music with μ = 0.02.", "The adaptive MSG sags even further during the first half.",
         "Sustained tones are strongly self-correlated, the worst case for bias. Products add decorrelation tricks."),
        ("Set the path increase to 0 dB.", "Fixed and adaptive both stay stable (≈ 3% howling).",
         "Without a change, the fitting-time measurement is good enough."),
    ])
    ui.in_practice("Every modern hearing aid runs an adaptive feedback canceller. The same structure (an adaptive "
                   "FIR on the loudspeaker signal) is the car's acoustic echo canceller from L2, here inside a "
                   "closed loop. A real-time C++ version of this exact loop is in `cpp/include/ssp/afc.hpp`: "
                   "block-based, allocation-free, and tested to reproduce this page's simulation.")
