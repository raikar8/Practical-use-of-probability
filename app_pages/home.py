import streamlit as st

st.title("Practical Use of Probability")
st.markdown(
    "An interactive lab for **statistical signal processing**: from Bayes' rule to Kalman filters, "
    "then into the real systems that use them today. Every algorithm is written from scratch in "
    "NumPy (see `ssp/core/`), checked by tests against theory and reference libraries, and every "
    "page has a **Code** tab that shows the exact source it runs."
)

st.subheader("The one idea behind everything here")
st.markdown(
    """
All of these algorithms run the same loop:

1. **Believe** something about a hidden quantity, including how unsure you are (a prior / a covariance).
2. **Predict** how it changes (a model). Uncertainty grows.
3. **Observe** noisy data (a likelihood).
4. **Correct** the belief, weighting the data by how much you trust it (Bayes). Uncertainty shrinks.

MLE and MAP do step 4 once. RLS and the Kalman filter do steps 2–4 forever, in real time.
HMMs do it with discrete hidden states. Particle filters do it when nothing is Gaussian.
"""
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Learning ladder")
    st.markdown(
        """
| Level | You will be able to… |
|---|---|
| **L0** Probability | update beliefs with Bayes, predict with Markov chains |
| **L1** Estimation | derive MLE / MAP / MMSE and know the best possible accuracy (CRLB) |
| **L2** Adaptive filters | track a changing system with LMS, NLMS, RLS (echo cancellation) |
| **L3** Kalman | fuse a model with noisy sensors, and test whether your filter is honest |
| **L4** HMM | track, decode and learn hidden discrete states: forward, Viterbi, Baum–Welch |
| **L5** Advanced | particle filters, IMM, Bayesian online change-point detection |
"""
    )
with c2:
    st.subheader("Domain demos")
    st.markdown(
        """
| Demo | Algorithm | Status |
|---|---|---|
| Car GPS + IMU through a tunnel | EKF with bias estimation | ready |
| Online game: where is the player *now*? | Kalman prediction vs. netcode heuristics | ready |
| Voice activity detection in a car | unsupervised 2-state HMM, duration prior | ready |
| Market regimes, volatility, hedge ratios | Gaussian HMM, GARCH by MLE, Kalman regression | ready |
| Noise suppression & hearing-aid feedback | MMSE/Wiener gains, SPP noise tracking, adaptive feedback cancellation | ready |
| Sports skill ratings & live win probability | Elo vs Kalman ratings, Poisson/Gamma live model | ready |
"""
    )

st.subheader("How each page is laid out")
st.markdown(
    "**Intuition & math** → **Playground** (sliders, live plots) → **Break it** (mis-tune on purpose and "
    "watch it fail, which is how you learn what each knob does) → **Code** (the exact source)."
)
st.caption("All examples use simulated data so you can compare every estimate with the hidden truth. The speech and "
           "market pages also accept your own WAV / CSV files. C++ ports of the Kalman filter and Viterbi decoder "
           "live in `cpp/`.")
