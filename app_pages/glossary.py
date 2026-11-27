import streamlit as st

st.caption("Start here")
st.title("Notation & glossary")
st.markdown("Keep this page open in a second browser tab while you read the Theory pages.")

st.header("Symbols")
st.markdown(
    r"""
| Symbol | Read as | Meaning |
|---|---|---|
| $x$ (or $x_t$, $\theta$) | "x", "theta" | the **hidden** quantity we want: a state at time $t$, or a fixed parameter |
| $z$ (or $z_t$, $y$) | "z" | a **measurement** (observation) |
| $\hat x$ | "x hat" | an **estimate** of $x$ |
| $\hat x^-$, $P^-$ | "x hat minus" | a **predicted** estimate, before the newest measurement is used |
| $P(A)$ | "probability of A" | probability of an event (a number between 0 and 1) |
| $p(x)$ | "p of x" | a probability **density** (continuous) or mass function (discrete) |
| $p(x \mid z)$ | "p of x given z" | probability of $x$ **once we know** $z$ (conditional) |
| $x \sim \mathcal N(\mu, \sigma^2)$ | "x is distributed normal…" | $x$ is Gaussian with mean $\mu$ and variance $\sigma^2$ |
| $E[x]$ | "expectation of x" | the average value of $x$ over its distribution |
| $\operatorname{var}(x)$, $\sigma^2$ | "variance", "sigma squared" | average squared spread around the mean; $\sigma$ is the standard deviation |
| $\propto$ | "proportional to" | equal up to a constant factor (normalise afterwards so it sums to 1) |
| $\arg\max_x f(x)$ | "arg max" | the value of $x$ at which $f$ is largest (not the largest value itself) |
| $\sum_t$, $\prod_t$ | "sum", "product" | add up, or multiply together, over all $t$ |
| $A^T$, $A^{-1}$ | "A transpose", "A inverse" | flip rows/columns; matrix inverse |
| $\lVert v\rVert^2$ | "norm squared" | sum of squares of the entries of a vector |
| $\log$ | "log" | natural logarithm (base $e$) unless written $\log_{10}$ |
| dB | "decibel" | $10\log_{10}$ of a power ratio; +10 dB = 10× power, +3 dB ≈ 2× power |
"""
)
st.info("**Watch out:** in the Kalman-filter pages $P$ is a **covariance matrix**, not a probability. It's the "
        "standard notation in that field, so the app keeps it.", icon=":material/warning:")

st.header("Words")
terms = {
    "Prior": "What you believe about the hidden quantity **before** seeing the current data.",
    "Likelihood": "How probable the observed data would be **if** the hidden quantity had a particular value. "
                  "It's a function of the hidden value, with the data held fixed.",
    "Posterior": "Your belief **after** combining the prior with the data (Bayes' rule).",
    "Evidence": "The total probability of the data, summed over every possible hidden value. It's the "
                "normalising constant in Bayes' rule.",
    "Estimator": "A recipe that turns data into a guess, e.g. 'take the average'.",
    "Bias": "The average error of an estimator over many repeats. Zero bias = right on average.",
    "Variance (of an estimator)": "How much the estimate scatters from one repeat to the next.",
    "MSE (mean squared error)": "Average squared error. Equals bias² + variance.",
    "MLE": "Maximum-likelihood estimate: the value that makes the observed data most probable.",
    "MAP": "Maximum a-posteriori estimate: the peak of the posterior (likelihood × prior).",
    "MMSE": "Minimum mean-squared-error estimate: the mean of the posterior.",
    "CRLB": "Cramér–Rao lower bound: the smallest variance any unbiased estimator can achieve.",
    "Fisher information": "How sharply the likelihood peaks. More information means a smaller CRLB.",
    "SNR": "Signal-to-noise ratio: signal power ÷ noise power, often in dB.",
    "Adaptive filter": "A filter whose coefficients update themselves sample by sample to track a changing system.",
    "State": "The minimal set of numbers that describes a system now (e.g. position and velocity).",
    "State-space model": "Two equations: how the state evolves (motion model) and how measurements relate to it "
                         "(measurement model).",
    "Process noise (Q)": "Randomness in how the state evolves: the part of the motion the model can't predict.",
    "Measurement noise (R)": "Randomness in the sensor.",
    "Innovation": "Measurement minus predicted measurement: the 'surprise'.",
    "Kalman gain (K)": "How much of the innovation to add to the estimate (0 = ignore the sensor, 1 = trust it fully).",
    "Covariance": "A matrix of variances and co-variances: the filter's own error bars, including correlations.",
    "Filtering / smoothing / prediction": "Estimating the state now from past data / at a past time using later "
                                          "data too / at a future time.",
    "Markov property": "The next state depends only on the current one, not on the whole history.",
    "Hidden Markov model": "A Markov chain you can't see directly; each hidden state produces observations.",
    "Viterbi": "Algorithm for the single most probable sequence of hidden states.",
    "EM / Baum–Welch": "Learn a model's parameters from unlabelled data by alternating 'guess the hidden states' "
                       "and 're-fit the parameters'.",
    "Particle filter": "Represent a belief by many weighted random samples; works for any model.",
    "ESS (effective sample size)": "How many particles are really contributing. It collapses when a few particles "
                                   "carry all the weight.",
    "Calibration": "Probabilities are calibrated if events given 70% happen about 70% of the time.",
    "Monte Carlo": "Estimating something by simulating it many times with random numbers and averaging.",
}
for k, v in terms.items():
    st.markdown(f"**{k}.** {v}")
