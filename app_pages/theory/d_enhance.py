import streamlit as st

from ssp import ui

ui.theory_header("enhance", """
**Prerequisites:** Level 1 (MLE, MAP, MMSE), Level 2 (NLMS). The speech-presence idea also uses Level 0's Bayes rule.

Two estimation problems run inside every hearing aid, phone and car hands-free system. **Noise suppression**:
how much of each little piece of sound is speech? **Feedback cancellation**: what is leaking from the loudspeaker
back into the microphone? This page explains both from the ground up.
""")

# ---------------------------------------------------------------------------
ui.section("Part A · Noise suppression")
st.subheader("A1. Working in time–frequency: the STFT")
st.markdown(
    r"""
Speech and noise overlap in time, but at any moment they usually occupy **different frequencies**. So we cut the
signal into overlapping frames (here 32 ms = 512 samples at 16 kHz, a new frame every 16 ms), window each one and
take its FFT. This is the **short-time Fourier transform (STFT)**. The result is a grid of complex numbers
$Y(k,l)$, one per frequency bin $k$ (257 bins, 31.25 Hz apart) and frame $l$. Plotted as $\lvert Y\rvert^2$ in dB,
it is the **spectrogram** shown in the Lab.

Because the STFT is linear, the noisy signal $y = s + n$ becomes, in every bin,
"""
)
ui.eq(r"Y(k,l) = S(k,l) + N(k,l)", 1)
st.markdown(
    r"""
The suppressor multiplies every bin by a **gain** $0 \le G(k,l) \le 1$, then transforms back (inverse STFT with
overlap-add): $\hat S = G\,Y$. Bins dominated by speech should get $G \approx 1$, and bins dominated by noise
$G \approx$ a small floor. **The whole problem is choosing G**, and that's an estimation problem in every bin.
"""
)
st.subheader("A2. Two SNRs per bin")
ui.eq(r"\gamma(k,l) = \frac{\lvert Y(k,l)\rvert^2}{\lambda_N(k,l)} \;\;(\text{a-posteriori SNR, observed}),"
      r"\qquad \xi(k,l) = \frac{E\lvert S(k,l)\rvert^2}{\lambda_N(k,l)} \;\;(\text{a-priori SNR, hidden})", 2)
st.markdown(
    r"""
$\lambda_N$ is the **noise power** in that bin (estimated, see A5). $\gamma$ is what we see; $\xi$ is what we'd
like to know. A large $\xi$ means "this bin is mostly speech".
"""
)
st.subheader("A3. Gain rules are estimators")
st.markdown(
    r"""
Assume speech and noise are zero-mean Gaussian in each bin with variances $\xi\lambda_N$ and $\lambda_N$. Then:

| Rule | It estimates | Gain |
|---|---|---|
| Spectral subtraction | power, by subtracting the noise estimate | $\sqrt{\max(1-1/\gamma,\,0)}$ |
| **Wiener** | the complex $S$, MMSE (Level 1) | $\xi/(1+\xi)$ |
| **MMSE-STSA** (Ephraim & Malah 1984) | the amplitude $\lvert S\rvert$, MMSE | a Bessel-function formula in $\xi, \gamma$ |
| **MMSE-LSA** (1985) | $\log\lvert S\rvert$, MMSE: closer to how loudness is perceived | $\frac{\xi}{1+\xi}\exp\!\big(\tfrac12E_1(v)\big)$, $v=\frac{\xi\gamma}{1+\xi}$ |

The Wiener gain is the Level 1 "weight on the data" again: signal variance over total variance. The
Ephraim–Malah gains are the same idea, optimised for amplitude or log-amplitude errors.
"""
)
st.subheader("A4. Estimating ξ: the decision-directed rule")
st.markdown(r"""The simplest estimate of $\xi$ is the **maximum-likelihood** one, $\max(\gamma - 1, 0)$. It's very noisy: in a noise-only bin $\gamma$ fluctuates randomly, so isolated bins briefly get a high gain and you hear twinkling tones called **musical noise**. The **decision-directed** estimate adds a prior from the previous frame's clean-speech estimate:""")
ui.eq(r"\hat\xi(k,l) = a\,\frac{G(k,l-1)^2\,\lvert Y(k,l-1)\rvert^2}{\lambda_N(k,l)} + (1-a)\,\max\big(\gamma(k,l)-1,\,0\big)", 3)
st.markdown(
    r"""
With $a \approx 0.98$, $\hat\xi$ changes smoothly, and musical noise disappears. That's the Level 1 lesson (a prior
reduces variance) in an audio product. $a = 0$ gives back the ML estimate.
"""
)
st.subheader("A5. Tracking the noise power")
st.markdown(r"""$\lambda_N$ is never observed directly and changes (the car speeds up, the fan turns on). The **speech-presence-probability (SPP)** tracker applies Bayes' rule in every bin and frame:""")
ui.eq(r"P(\text{speech}\mid Y) = \left[1 + (1+\xi_{H_1})\,\exp\!\Big(-\gamma\,\frac{\xi_{H_1}}{1+\xi_{H_1}}\Big)\right]^{-1}", 4)
ui.eq(r"E[\lvert N\rvert^2 \mid Y] = \big(1-P\big)\lvert Y\rvert^2 + P\,\lambda_N, \qquad "
      r"\lambda_N \leftarrow \alpha\,\lambda_N + (1-\alpha)\,E[\lvert N\rvert^2\mid Y]", 5)
st.markdown(
    r"""
where $\xi_{H_1}$ is a typical SNR when speech is present (15 dB). If a bin is probably noise, its power updates
the noise estimate; if it's probably speech, the old estimate is kept. The simpler alternative, measuring the
noise once at the start, fails as soon as the noise level changes.

**Gain floor.** Real products never let $G$ go to zero. A floor of −12 to −25 dB keeps a little natural
background and avoids "watery" speech.
"""
)
st.subheader("A6. Measuring the result")
st.markdown(
    r"""
With synthetic audio we know the clean speech, so we can compute **SNR** (overall error energy) and **segmental
SNR** (averaged over speech frames, closer to perception), plus **noise reduction in pauses**. For your own
recordings there is no clean reference. The Lab then reports the **level drop in the quietest frames** (almost
surely pauses) and the **level change on the loudest frames** (speech should change little).
"""
)

# ---------------------------------------------------------------------------
ui.section("Part B · Feedback cancellation in a hearing aid")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", fillcolor="#f4f4f4"];
  x [label="incoming sound x", fillcolor="#eef4fc"]; m [label="+", shape=circle];
  c [label="−", shape=circle]; g [label="hearing aid\\ngain K, delay D", fillcolor="#eef4fc"];
  r [label="receiver\\n(loudspeaker) u"]; F [label="feedback path F\\n(leak back to the mic)", fillcolor="#fdf0e8"];
  H [label="adaptive canceller Ĥ ≈ F", fillcolor="#eef4fc"];
  x -> m [label="mic"]; m -> c; c -> g [label="e"]; g -> r; r -> F; F -> m; r -> H; H -> c;
  c -> H [style=dashed, label="adapt (NLMS)"];
}
"""
)
st.markdown(
    r"""
A hearing aid amplifies sound by a gain $K$ (often 20–40 dB), and its loudspeaker (the **receiver**) sits
millimetres from its microphone. Some of the output leaks back through the **feedback path** $F$, gets amplified
again, and so on. If the loop gain exceeds 1 at some frequency, the device **howls** (whistles). The largest
safe gain without any cancellation is the **maximum stable gain**:
"""
)
ui.eq(r"\text{MSG} = -20\log_{10}\max_\omega \lvert F(\omega)\rvert \;\;\text{dB}", 6)
st.markdown(
    r"""
**The canceller.** An adaptive FIR filter $\hat H$ runs on the receiver signal and subtracts its prediction of the
leak from the microphone. It's exactly the echo canceller of Level 2 (NLMS), so the loop now only sees the
**residual** $F - \hat H$, and the usable gain rises to $-20\log_{10}\max_\omega\lvert F - \hat H\rvert$.

**In practice:**

* At the **fitting appointment** the audiologist's software measures $F$, and that measurement initialises
  $\hat H$. A *fixed* canceller stops there.
* $F$ **changes** when you put on a hat, hug someone or hold a phone to your ear (often by 6–12 dB). Only an
  *adaptive* canceller can follow.
* **Closed-loop bias.** The receiver signal is a delayed, amplified copy of the incoming sound, so input and
  output are correlated. That breaks the least-squares assumption from Level 2, and the filter partly models the
  incoming sound instead of $F$. It's worst for tonal sounds (music, which is why music + hearing aids can
  "warble"). A larger step size $\mu$ follows changes faster but suffers more bias. Products add decorrelation:
  small frequency shifts, prediction-error pre-whitening, probe noise.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("enhance", [
    ("Noise tab: three spectrograms", r"$\lvert Y\rvert^2$, $\lvert GY\rvert^2$, and the gain $G$ (white = pass, black = floor)"),
    ("*Gain rule*", r"the rows of the table in A3"),
    ("*Noise estimate*: SPP / first 0.15 s / true average", r"(4)–(5), a frozen start-up estimate, or cheating with the true noise"),
    ("*Decision-directed a*", r"$a$ in (3); 0 = ML estimate (musical noise)"),
    ("*Gain floor*", r"the minimum $G$"),
    ("*Noise gets louder half-way*", r"a change that a frozen noise estimate can't follow"),
    ("Feedback tab: *Hearing-aid gain above MSG*", r"$K$ relative to (6) without a canceller"),
    ("*Feedback path increase*", r"the phone-to-ear change in $F$"),
    ("*NLMS step μ*", r"speed vs bias, Level 2 equation (5)"),
    ("MSG-over-time plot", r"$-20\log_{10}\max\lvert F-\hat H\rvert$ for each canceller vs the gain line"),
])

ui.section("Check yourself")
ui.check([
    ("ξ = 1 (0 dB) in a bin. What is the Wiener gain, in dB?", "1/(1+1) = 0.5, i.e. **−6 dB** in amplitude."),
    ("Why does a = 0 in the decision-directed rule cause musical noise?",
     "ξ then follows γ − 1 frame by frame. In noise-only bins γ fluctuates randomly, so isolated bins pop to a high "
     "gain for one frame: short random tones."),
    ("A feedback path peaks at −25 dB. What is the MSG without cancellation? With a canceller leaving a residual "
     "peak of −40 dB?",
     "**25 dB** without; **40 dB** with, so the canceller added 15 dB of stable gain."),
])
