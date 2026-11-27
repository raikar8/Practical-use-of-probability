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
| $\mathrm{KL}(q\,\Vert\,p)$ | "KL from q to p" | Kullback–Leibler divergence: how different $q$ is from $p$ ($\ge 0$, zero only if equal; not symmetric) |
| $\nabla_\theta \mathcal L$ | "gradient of L" | the vector of partial derivatives of $\mathcal L$ with respect to every parameter in $\theta$ |
| $\mathcal{GP}(m, k)$ | "Gaussian process" | a distribution over functions with mean $m$ and kernel (covariance function) $k$ |
| $q_\phi(z \mid x)$, $p_\theta(x \mid z)$ | "encoder", "decoder" | in Part III: a network with weights $\phi$ giving a distribution over codes, and one with weights $\theta$ giving a distribution over data |
| $z$ (Part III) | "latent", "noise" | in Levels 10–13, $z$ is the random input or hidden code of a generator, **not** a measurement |
| $\bar\alpha_t$ | "alpha bar t" | in diffusion: the fraction of signal power left after $t$ noising steps |
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
    "Bayesian network": "A directed graph whose nodes are random variables and whose arrows say which variables "
                        "directly influence which; the joint distribution is the product of one small table per node.",
    "Message passing / belief propagation": "Exact (on trees) or approximate inference by passing local summaries "
                                            "between neighbouring variables. Forward–backward, Viterbi and the "
                                            "Kalman smoother are special cases.",
    "Latent variable": "A hidden variable that's part of the model but never observed (a cluster label, a code).",
    "Responsibility": "In EM: the probability that a given component generated a given data point.",
    "BIC": "Bayesian information criterion: −2 log-likelihood + (number of parameters) × log N. Lower is better.",
    "MCMC": "Markov chain Monte Carlo: draw correlated samples from a posterior known only up to a constant.",
    "Burn-in": "The first MCMC samples, discarded because they still depend on the starting point.",
    "Variational inference": "Approximate a posterior by the closest member of a simple family, found by "
                             "maximising the ELBO.",
    "ELBO": "Evidence lower bound: log p(x) minus the KL from the approximation to the true posterior.",
    "Reparameterisation trick": "Write a random sample as a deterministic function of parameters and fixed noise "
                                "(z = μ + σε) so gradients can flow through it.",
    "Kernel": "A function k(x, x′) that says how similar the outputs at x and x′ should be; it defines a GP.",
    "Marginal likelihood": "Another name for the evidence: the probability of the data averaged over a model's "
                           "parameters. Used to compare models and tune hyperparameters.",
    "Acquisition function": "In Bayesian optimisation: a score saying how useful it would be to evaluate next at x.",
    "Regret": "In bandits: reward lost compared with always choosing the best option.",
    "Thompson sampling": "Choose an action with the probability that it's the best: sample each option's "
                         "posterior once and pick the highest sample.",
    "Neural network (MLP)": "Layers of 'linear mix, then bend with a nonlinearity'. A flexible function with many "
                            "tunable weights.",
    "Gradient descent / Adam": "Training by repeatedly stepping the weights downhill on the loss; Adam adapts the "
                               "step size per weight.",
    "Backpropagation": "The chain rule applied layer by layer to get the gradient of the loss for every weight.",
    "Generative model": "A model of the data distribution p(x) that can produce new samples (and sometimes score "
                        "them).",
    "Normalising flow": "A generator built from invertible functions, so its exact likelihood can be computed.",
    "VAE": "Variational autoencoder: an encoder giving a Gaussian over codes and a decoder from codes to data, "
           "trained by maximising the ELBO.",
    "Posterior collapse": "A VAE failure where the code carries no information and the decoder ignores it.",
    "GAN": "Generative adversarial network: a generator trained to fool a discriminator that is trained to tell "
           "real from fake.",
    "Mode collapse": "A GAN failure where the generator only produces a few kinds of output.",
    "Diffusion model": "A generator that starts from noise and applies a learned denoiser many times, reversing a "
                       "fixed noising process.",
    "Score": "The gradient of the log-density, ∇ log p(x): a direction toward more probable data.",
    "Tweedie's formula": "Links the MMSE denoiser to the score of the noisy data distribution.",
}
for k, v in terms.items():
    st.markdown(f"**{k}.** {v}")
