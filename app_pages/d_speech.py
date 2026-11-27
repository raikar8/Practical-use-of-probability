import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core.kalman import cov_ellipse
from ssp.sims import speech as sp

ui.lab_intro(
    "speech",
    goal="Detect when someone is speaking in noisy audio, synthetic or your own recording, and compare a simple "
         "energy threshold with an HMM that learns speech/noise from the recording and knows speech comes in "
         "segments.",
    on_screen=[
        "**Top:** the waveform. **Middle:** the band energy feature with the energy detector's threshold. "
        "**Bottom lanes:** each detector's decision (high = speech), with the true speech on top for synthetic audio.",
        "**Score table:** accuracy, missed speech, false alarms, and how often each detector flips",
        "**Feature scatter:** every 10 ms frame as a dot in (energy, flatness) space, with the two Gaussians the HMM "
        "learned",
        "**Listen:** the noisy input and the audio gated by each detector",
    ],
    start_here="Compare the 'Energy threshold' lane (orange, flickering) with the 'HMM Viterbi' lane (aqua, whole "
               "segments) against the grey Truth lane, then play the two gated versions.",
)

c1, c2 = st.columns([1, 2.4])
with c1:
    src = st.radio("Audio", ["Synthetic speech in noise (truth known)", "Your own voice: record or upload"],
                   help="Synthetic audio has known true speech segments, so the detectors can be scored.")
    truth = None
    if src.startswith("Synthetic"):
        noise = st.selectbox("Noise", ["car", "white", "babble"], format_func={
            "car": "Car cabin (rumble + engine + road hiss)", "white": "White noise",
            "babble": "Babble (other talkers)"}.get, help="Type of background noise mixed with the speech.")
        snr = st.slider("SNR (dB)", -10, 30, 5, help="Speech level vs noise level during speech. 0 dB = equally loud.")
        dur = st.slider("Length (s)", 6, 30, 12, help="Length of the synthetic recording.")
        seed = st.number_input("Random seed", 0, 9999, 1, help="Different sentences and noise.")

        @st.cache_data(show_spinner="Synthesising speech…")
        def _scene(dur, snr, noise, seed):
            return sp.make_scene(sp.VadScene(duration=float(dur), snr_db=float(snr), noise=noise, seed=int(seed)))

        scene = _scene(dur, snr, noise, seed)
        audio, clean, mask = scene["noisy"], scene["clean"], scene["mask"]
    else:
        rec = ui.real_audio("vad")
        if rec is None:
            st.info("Record a few seconds of speech with pauses, or upload a WAV (e.g. from a car "
                    "voice-assistant test). There's no ground truth for real audio, so the page shows the "
                    "detectors' decisions and lets you listen to them.")
            st.stop()
        audio, clean, mask = rec, None, None
    st.divider()
    st.markdown("**Energy detector**")
    margin = st.slider("Energy VAD: dB above the noise floor", 1.0, 20.0, 6.0, 0.5,
                       help="A frame counts as speech if its band energy exceeds the noise floor (10th percentile) "
                            "by this many dB.")
    st.markdown("**HMM detector**")
    learn_A = st.toggle("Use the duration prior learned by EM", False,
                        help="Off: set the duration prior yourself below")
    stay = st.select_slider("Duration prior: P(stay) per 10 ms frame", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999], 0.99,
                            disabled=learn_A,
                            help="Probability of staying in the same state for the next frame. Expected segment "
                                 "length = 10 ms / (1 − P).")
    kappa = st.select_slider("Evidence scale κ", [0.05, 0.1, 0.2, 0.3, 0.5, 1.0], 0.1,
                             help="Down-weights each frame's evidence, because overlapping frames aren't independent "
                                  "(Theory §4c). 1 = no correction.")
    st.caption(f"Expected segment length ≈ {10 / (1 - stay):.0f} ms" if not learn_A else "Using EM's estimate")

feats = sp.frame_features(audio)
hop_s = feats["hop"] / sp.FS
h = sp.hmm_vad(feats, None if learn_A else stay, None if learn_A else stay, evidence_scale=kappa)
energy = sp.energy_vad(feats["energy"], margin)
dets = {"Energy threshold": energy, "HMM forward (real time)": h["forward"], "HMM Viterbi (offline)": h["viterbi"]}
colors = {"Energy threshold": ui.C[2], "HMM forward (real time)": ui.C[1], "HMM Viterbi (offline)": ui.C[3]}
if mask is not None:
    truth = sp.frame_labels(mask, feats)

with c2:
    blk = 160
    nb = len(audio) // blk
    env_hi = audio[: nb * blk].reshape(nb, blk).max(axis=1)
    env_lo = audio[: nb * blk].reshape(nb, blk).min(axis=1)
    tb = (np.arange(nb) + 0.5) * blk / sp.FS
    t = feats["t"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.38, 0.3, 0.32], vertical_spacing=0.04)
    fig.add_scatter(x=np.r_[tb, tb[::-1]], y=np.r_[env_hi, env_lo[::-1]], fill="toself", name="noisy waveform",
                    fillcolor=ui.rgba(ui.C["truth"], 0.5), line=dict(width=0), hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=t, y=feats["energy"], name="band energy (dB)", line=dict(color=ui.C[1], width=1), row=2, col=1)
    fig.add_hline(y=np.percentile(feats["energy"], 10) + margin, line=dict(color=ui.C[2], dash="dot"),
                  annotation_text="energy threshold", row=2, col=1)
    lanes = ({"Truth": truth} if truth is not None else {}) | dets
    for i, (nm, lab) in enumerate(lanes.items()):
        y0 = len(lanes) - 1 - i
        col = ui.C["truth"] if nm == "Truth" else colors[nm]
        fig.add_scatter(x=t, y=np.where(lab == 1, y0 + 0.8, y0 + 0.1), mode="lines", name=nm if nm == "Truth" else None,
                        showlegend=False, line=dict(color=col, width=2, shape="hv"), row=3, col=1)
    fig.update_yaxes(tickvals=[len(lanes) - 1 - i + 0.45 for i in range(len(lanes))], ticktext=list(lanes),
                     row=3, col=1, showgrid=False)
    fig.update_yaxes(title="amplitude", row=1, col=1)
    fig.update_yaxes(title="dB", row=2, col=1)
    fig.update_xaxes(title="time (s)", row=3, col=1)
    ui.show(ui.style(fig, "Waveform, the main feature, and each detector's decision (high = speech)", height=560))

    if truth is not None:
        res = pd.DataFrame({nm: sp.scores(lab, truth, hop_s) for nm, lab in dets.items()}).T
        st.dataframe(res.style.format({"accuracy": "{:.1%}", "missed speech": "{:.1%}", "false alarm": "{:.1%}",
                                       "switches / s": "{:.1f}"}), width="stretch")
        st.caption(f"True speech covers {truth.mean():.0%} of frames, with "
                   f"{sp.hmm.count_switches(truth) / (len(truth) * hop_s):.1f} switches per second.")
    else:
        st.dataframe(pd.DataFrame({nm: {"speech fraction": lab.mean(),
                                        "switches / s": sp.hmm.count_switches(lab) / (len(lab) * hop_s)}
                                   for nm, lab in dets.items()}).T.style.format("{:.2f}"), width="stretch")

c3, c4 = st.columns([1.2, 1])
with c3:
    F = h["F"]
    p = h["params"]
    fig2 = go.Figure()
    lab = truth if truth is not None else h["viterbi"]
    for k, nm, col in ((0, "noise frames", ui.C[1]), (1, "speech frames", ui.C[2])):
        sel = lab == k
        fig2.add_scatter(x=F[sel, 0], y=F[sel, 1], mode="markers", name=nm + (" (truth)" if truth is not None else
                         " (decoded)"), marker=dict(size=4, color=ui.rgba(col, 0.45)))
        ex, ey = cov_ellipse(p.means[k], np.diag(p.variances[k]), 2.0)
        fig2.add_scatter(x=ex, y=ey, mode="lines", name=f"learned {'speech' if k else 'noise'} state (2σ)",
                         line=dict(color=col, width=3))
    fig2.update_layout(hovermode="closest")
    ui.show(ui.style(fig2, "Feature space: what Baum–Welch learned without labels",
                     "band energy (standardised)", "spectral flatness (standardised)", height=380))
with c4:
    la = h["learned_A"]
    st.markdown("**Learned transition matrix** (10 ms frames)")
    st.dataframe(pd.DataFrame(la, index=["from noise", "from speech"], columns=["to noise", "to speech"])
                 .style.format("{:.3f}"), width="stretch")
    st.caption(f"EM's implied mean durations: noise {10 / (1 - la[0, 0]):.0f} ms, speech {10 / (1 - la[1, 1]):.0f} ms. "
               "EM fits at full evidence weight, so brief dips inside words count as pauses and the learned "
               "segments come out shorter than real utterances.")
    st.markdown("**Listen**")
    st.caption("Noisy input")
    st.audio(np.clip(audio / (np.abs(audio).max() + 1e-9), -1, 1), sample_rate=sp.FS)
    if clean is not None:
        st.caption("Clean speech (hidden truth)")
        st.audio(np.clip(clean / (np.abs(clean).max() + 1e-9), -1, 1), sample_rate=sp.FS)
    g = sp.gate(audio, h["viterbi"], feats)
    st.caption("Gated by the HMM Viterbi VAD (non-speech attenuated by 25 dB)")
    st.audio(np.clip(g / (np.abs(audio).max() + 1e-9), -1, 1), sample_rate=sp.FS)
    g2 = sp.gate(audio, energy, feats)
    st.caption("Gated by the energy VAD: listen for choppy word endings")
    st.audio(np.clip(g2 / (np.abs(audio).max() + 1e-9), -1, 1), sample_rate=sp.FS)

ui.takeaway("The energy detector decides each frame alone, so it chops the quiet gaps and soft endings inside "
            "words. The HMM keeps a belief and needs sustained evidence to change its mind, so speech comes out as "
            "whole segments. The forward version does this causally, which is what runs in a real-time product.")
ui.controls([
    ("Audio / Noise / SNR", "–", "what the detectors hear", "higher SNR = easier"),
    ("Energy VAD margin", "–", "threshold above the noise floor", "fewer false alarms, more missed speech"),
    ("Use the duration prior learned by EM", r"$a$ from Baum–Welch", "let the data set the segment length", "–"),
    ("P(stay)", r"$a$ (Theory eq. 3)", "duration prior", "longer segments, less flicker"),
    ("Evidence scale κ", r"$\kappa$", "weight of each frame's evidence", "more responsive, more flicker"),
])
ui.experiments([
    ("Defaults (car noise, 5 dB): read the score table.",
     "Energy ≈ 88% with ~8 flips/s; HMM forward ≈ 93% with ~3/s; HMM Viterbi ≈ 99% with < 1/s.",
     "Frame-by-frame decisions chop the quiet bits inside words; the HMM's duration prior bridges them."),
    ("Set κ = 1.", "Both HMM lanes start flickering, and accuracy falls to the energy detector's level (≈ 88%).",
     "Full-weight evidence from overlapping frames overwhelms the duration prior."),
    ("Set κ back to 0.1 and P(stay) to 0.9.", "The HMM lanes flicker more (forward ≈ 5 flips/s).",
     "A 100 ms expected duration is too short for speech, so the prior no longer bridges gaps between syllables."),
    ("Set the energy margin to 3 dB.", "The energy lane flips ~20 times per second and false alarms jump to ~30%.",
     "A low threshold catches more speech but also noise peaks. Hand-tuned thresholds are fragile."),
    ("Set Noise = babble.", "Every detector calls ~45% of noise frames 'speech'.",
     "Babble *is* speech: energy and flatness can't separate it. Products add pitch, modulation or neural features."),
    ("Set Noise = white at 0 dB.", "The energy detector misses almost all speech; the HMM still finds part of it.",
     "Broadband noise raises the energy floor, but speech's peaky spectrum still shows up in flatness."),
])
ui.in_practice("VADs gate echo-cancellers and noise-suppressors (adapt the noise estimate only in pauses), "
               "trigger wake-word and ASR engines in cars and phones, and drive discontinuous transmission in "
               "speech codecs. Statistical-model VADs with HMM-style hangover (Sohn et al., 1999) are the classic "
               "design. Hearing aids use similar speech/noise classifiers to switch programs.")

with st.expander("Code: features and the HMM VAD (from `ssp/sims/speech.py`)"):
    ui.source(sp.frame_features, sp.hmm_vad, sp.energy_vad)
