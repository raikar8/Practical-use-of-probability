import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.sims import speech as sp

ui.theory_header("speech", """
**Prerequisites:** Level 0 (Gaussians), Level 4 (HMM: forward, Viterbi, Baum–Welch).

A **voice activity detector (VAD)** answers "is someone talking in this 10 ms frame?". Every speech product needs
one: echo cancellers freeze adaptation during double-talk, noise suppressors update their noise estimate in
pauses, wake-word and speech-recognition engines start and stop on it, and codecs stop transmitting in silence.
This page builds a VAD from scratch: audio framing, two features, a simple threshold detector, and why an HMM does
much better.
""")

ui.section("1. From a waveform to frames")
st.markdown(
    r"""
* **Sampling:** the microphone signal is sampled at $f_s = 16{,}000$ samples per second, enough for speech up to 8 kHz.
* **Framing:** speech properties change every few tens of milliseconds, so we cut the signal into short
  overlapping **frames**: 25 ms long (400 samples), starting every 10 ms (the **hop**). Each frame is multiplied
  by a smooth **window** (Hann) to avoid edge clicks, then transformed with the **FFT** into a power spectrum
  $P_t(f)$.

**What speech looks like.** *Voiced* sounds (vowels) are **harmonic**: a fundamental (pitch) of roughly 100–250 Hz
plus multiples of it, shaped by the vocal tract into peaks called **formants**. *Unvoiced* sounds ("s", "f") are
noise-like bursts at high frequencies. **Car noise** is mostly a low-frequency rumble (below ~400 Hz) plus engine
harmonics and some road hiss.
"""
)
rng = np.random.default_rng(1)
s_, _ = sp.synth_speech(3.0, rng)
f = np.fft.rfftfreq(512, 1 / sp.FS)
starts = np.arange(0, s_.size - 400, 160)
loudest = starts[np.argmax([np.sum(s_[i:i + 400] ** 2) for i in starts])]     # a frame in the middle of a vowel
fr_speech = np.abs(np.fft.rfft(s_[loudest:loudest + 400] * np.hanning(400), 512)) ** 2
noise_only = sp.car_noise(4000, np.random.default_rng(2), "car")
fr_noise = np.abs(np.fft.rfft(noise_only[:400] * np.hanning(400), 512)) ** 2
fig = go.Figure()
fig.add_scatter(x=f, y=10 * np.log10(fr_speech / fr_speech.max() + 1e-9), name="a voiced speech frame (harmonic peaks)",
                line=dict(color=ui.C[1]))
fig.add_scatter(x=f, y=10 * np.log10(fr_noise / fr_noise.max() + 1e-9), name="a car-noise frame (low-frequency rumble)",
                line=dict(color=ui.C[2]))
fig.add_vrect(x0=300, x1=4000, fillcolor=ui.rgba(ui.C["truth"], 0.12), line_width=0,
              annotation_text="feature band 300–4000 Hz", annotation_position="top left")
fig.update_xaxes(range=[0, 6000])
fig.update_yaxes(range=[-70, 3])
ui.show(ui.style(fig, "Two 25 ms frames: speech is peaky, car noise is low and smooth", "frequency (Hz)",
                 "power (dB, relative)", height=300))

ui.section("2. Two features per frame")
st.markdown(r"""Both features are computed only in the **300–4000 Hz band**, where speech carries most of its energy and car rumble is weak.""")
ui.eq(r"E_t = 10\log_{10}\sum_{f\in\text{band}} P_t(f) \qquad\text{(band energy, dB)}", 1)
ui.eq(r"\text{SF}_t = 10\log_{10}\frac{\exp\!\big(\frac1K\sum_f \log P_t(f)\big)}{\frac1K\sum_f P_t(f)}"
      r"\qquad\text{(spectral flatness, dB)}", 2)
st.markdown(
    r"""
* **Energy** is high when someone talks, but also when the noise gets louder.
* **Flatness** is the geometric mean over the arithmetic mean. It's ≈ 0 dB for a flat (noise-like) spectrum and
  strongly **negative** for a peaky, harmonic spectrum like voiced speech. It helps exactly where energy fails:
  loud noise.
"""
)

ui.section("3. The simple detector, and why it flickers")
st.markdown(
    r"""
**Energy threshold:** estimate the noise floor as a low percentile (10th) of all frame energies, and call a frame
speech if its energy exceeds the floor by a **margin** in dB. Each frame is decided **alone**, so the quiet gaps
inside and between words, and the soft word endings, drop below the threshold. The output **flickers** several
times per second. Products traditionally add an ad-hoc "hangover" counter to hold speech on a little longer.
"""
)

ui.section("4. The HMM detector")
st.markdown(
    r"""
Two hidden states, **noise** and **speech**, follow a Markov chain with self-transition probability $a$. Each
state emits the 2-D feature vector $[E_t, \text{SF}_t]$ (standardised) from its own Gaussian. Three ideas make it
work:

**(a) Learned without labels.** Baum–Welch (Level 4) fits both Gaussians and the transition matrix to *this*
recording. The state with the higher mean energy is called "speech".

**(b) A duration prior instead of a hangover hack.** The expected time in a state is
"""
)
ui.eq(r"\text{expected duration} = \frac{\text{hop}}{1-a} = \frac{10\ \text{ms}}{1-a}", 3)
st.markdown(
    r"""
so $a = 0.99$ means segments of about 1 s. A single quiet frame can't flip the decision; it takes sustained
evidence.

**(c) The evidence scale κ.** Consecutive frames overlap by 15 of their 25 ms, so they're far from independent.
Treating each frame's likelihood as fresh evidence **over-counts** it and drowns the duration prior. Decoding with
scaled log-likelihoods $\kappa \log b_k(\mathbf o_t)$, $\kappa < 1$, corrects this. Speech recognisers use the
same trick (the "acoustic scale").

**Two decoders:**

* **Forward** $P(\text{speech}_t \mid \mathbf o_{1:t}) > 0.5$ is **causal**: it can run live in a phone or a car
  head unit.
* **Viterbi** gives the best overall segmentation, but needs the whole recording (or a delay).
"""
)

ui.section("5. How the detectors are scored")
st.markdown(
    r"""
With synthetic audio the true speech segments are known, frame by frame:

* **accuracy:** fraction of frames labelled correctly
* **missed speech:** speech frames called noise (clipped words, the worst error for a speech recogniser)
* **false alarm:** noise frames called speech (wasted processing; noise passed through a gate)
* **switches per second:** how often the decision flips. Real speech switches about once a second; a detector
  switching 7–10 times a second is flickering.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("speech", [
    ("*Audio*: synthetic / your own voice", r"synthetic gives the truth for scoring; recordings are scored by listening"),
    ("*Noise*, *SNR*", r"the background type and level (car is low-frequency, babble is other voices)"),
    ("*Energy VAD: dB above the noise floor*", r"the margin of the section 3 detector"),
    ("*Use the duration prior learned by EM* / *P(stay)*", r"$a$ in (3), learned or set by hand"),
    ("*Evidence scale κ*", r"$\kappa$ in section 4(c)"),
    ("Middle plot", r"band energy (1) and the energy threshold"),
    ("Coloured lanes", r"each detector's decision per frame (high = speech)"),
    ("Feature-space scatter + ellipses", r"frames in ($E$, SF) space and the two Gaussians Baum–Welch learned"),
])

ui.section("Check yourself")
ui.check([
    ("P(stay) = 0.995 with a 10 ms hop. What duration does that encode?", "10 ms / 0.005 = **2 s** per segment on average."),
    ("Why does flatness help in loud white noise but not in babble?",
     "White noise is flat (≈ 0 dB flatness) while speech is peaky, so they separate. Babble *is* speech: it's "
     "harmonic too, so both features look speech-like."),
    ("Your forward VAD clips the first syllable of every utterance. Why, and what's the trade-off?",
     "A sticky prior needs several frames of evidence to switch on. Lower P(stay) or raise κ for faster onsets, at "
     "the cost of more flicker. Products often keep a short look-back buffer and prepend it."),
])
