# Practical Use of Probability

An interactive Streamlit course in **statistical signal processing and probabilistic machine learning**. It
starts with Bayes' rule, works up to Kalman filters and HMMs, applies them to real systems that still use them
today, then continues through graphical models, EM, MCMC, variational inference and Gaussian processes to
**VAEs, GANs and diffusion models**, built from scratch.

* **Part I · Estimation and tracking:** Levels 0–5 and six domain demos
* **Part II · Probabilistic machine learning:** Levels 6–9 and a bandit (A/B-testing) demo
* **Part III · Deep generative models:** Levels 10–13, assuming no prior knowledge of neural networks

* Every algorithm is written **from scratch in NumPy** (`ssp/core/`). Each page has a **Code** tab showing the
  exact source it runs.
* Every algorithm is **checked by a test**, either against theory (CRLB, χ² consistency, Wiener solution) or
  against a reference library (filterpy).
* Every topic has two pages, meant to be read in order:
  * **Theory**: what problem we're solving and why, the model and its numbered equations, worked numeric
    examples, small figures, common pitfalls, check-yourself questions, and a table mapping each idea to what
    you'll see in the Lab.
  * **Lab**: a short prologue (what the lab is for, what each plot shows, where to start), a tooltip on every
    control, a "Controls explained" table (symbol, meaning, what turning it up does), and **guided experiments**
    with the numbers you should see, then a **Code** tab.
* Start at **Overview & how to use**; **Notation & glossary** collects every symbol in one place.

## Run it

```bash
git clone https://github.com/raikar8/Practical-use-of-probability.git
cd Practical-use-of-probability
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the tests with `pytest`. This includes a smoke test that renders every page headlessly.

## What's inside

| Topic (Theory + Lab) | Concepts | Try this |
|---|---|---|
| **L0 · Probability toolkit** | Beta–Bernoulli updating, Bayes' rule and base rates, Markov chains, Gaussian conditioning | A confident wrong prior vs. the data |
| **L1 · MLE, MAP, CRLB** | Likelihood, MLE / MAP / MMSE, Monte Carlo vs. CRLB, frequency estimation and the threshold effect, robust estimation | MAP beating the CRLB, and why that's allowed |
| **L2 · LMS, NLMS, RLS** | Echo-path identification with an abrupt change, coloured input, Wiener target | Make LMS diverge; coloured input vs. RLS |
| **L3 · Kalman filter** | Predict/update, steady-state gain, 2D constant-velocity tracking, sensor blackout, RTS smoother, NEES/NIS consistency | Mis-tune Q and watch NEES leave the χ² band |
| **Car GPS + IMU fusion** | 5-state EKF with gyro-bias estimation through a GPS-denied tunnel (animated) | Turn off bias estimation → overconfident filter |
| **L4 · Hidden Markov models** | Forward (filtering), forward–backward (smoothing), Viterbi, Baum–Welch on the dishonest casino; HMM vs Kalman | Filtered vs smoothed vs Viterbi accuracy; EM restarts |
| **L5 · Particles, IMM, change points** | Bootstrap particle filter vs EKF vs exact grid posterior; road localisation from a barometer (animated); IMM on a manoeuvring target; Bayesian online change-point detection | Resampling off → weight collapse; IMM switching too eagerly |
| **Voice activity detection** | Unsupervised 2-state HMM on band energy + spectral flatness, duration prior, evidence scale; car / white / babble noise; WAV upload; listen to gated audio | κ = 1 makes the HMM flicker; babble breaks every detector |
| **Market regimes & volatility** | Gaussian-HMM regimes (filtered vs smoothed = look-ahead bias), GARCH(1,1) by MLE with forecasts, Kalman hedge ratio with MLE-fitted noise; CSV upload | Trade on smoothed regimes = cheating; mis-set δ |
| **Noise suppression & hearing aids** | Spectral subtraction / Wiener / MMSE-STSA / MMSE-LSA gains, decision-directed SNR, SPP-MMSE noise tracking; hearing-aid feedback loop with fixed vs adaptive NLMS canceller, MSG over time | ML ξ → musical noise; μ vs closed-loop bias |
| **Sports ratings & win probability** | Elo vs Kalman (Glicko-style) ratings with a strong newcomer; live football win/draw/loss from Poisson goals with Gamma-learned rates, red cards, calibration check | Kalman q too small; over-eager in-game learning |
| **Online game lag prediction** | Snap vs. extrapolation vs. interpolation vs. Kalman prediction under ping, jitter and loss (animated, 60 Hz) | Noisy snapshots break naive extrapolation |
| **L6 · Graphical models** | Bayesian networks and explaining away; sum-product = forward–backward, max-product = Viterbi, Gaussian BP = Kalman smoother | Add the earthquake and watch the burglary explanation vanish |
| **L7 · Hidden variables & EM** | Gaussian mixtures by EM (animated), local optima, BIC for K, mixture of experts | Five clusters with random starts |
| **L8 · MCMC & VI** | Metropolis, Gibbs, HMC with equal compute budgets; Laplace vs reverse-KL VI vs moment matching on exact 2-D posteriors | VI on two peaks: which one it picks depends on the start |
| **L9 · GPs & Bayesian optimisation** | Bayesian regression and the evidence, GP kernels and type-II ML, Bayesian optimisation of an expensive tuning curve | Too long a length-scale stops BO exploring |
| **A/B tests & recommendations** | Beta posteriors, A/B-then-commit vs ε-greedy vs UCB1 vs Thompson sampling, regret | How often a 500-per-arm A/B test picks the wrong winner |
| **L10 · Generative models from zero** | Noise → data by change of variables, an MLP trained by hand-written backprop, a 1-D normalising flow by maximum likelihood, two-sample classifier test | Match a target distribution by hand, then let MLE do it |
| **L11 · VAE** | Exact ELBO on a 1-D latent model, a VAE trained live on 8×8 digits (latent map, decoder manifold), generation and interpolation | β = 4: posterior collapse |
| **L12 · GAN** | 1-D duel with D vs the ideal D*, 2-D ring/moons/spiral with a mode counter | Fast generator, slow discriminator → mode collapse |
| **L13 · Diffusion** | Forward noising and SNR, exact-score diffusion with Tweedie arrows, a learned denoiser with DDPM/DDIM, pre-trained digit model, MMSE estimate vs posterior sample | DDIM with 3 steps; MMSE blur at high noise |

## Layout

```
streamlit_app.py        entry point (navigation)
app_pages/              Lab pages (one per topic), home, glossary
app_pages/theory/       Theory pages (one per topic)
ssp/core/               from-scratch algorithms
    probability.py      Bayes updates, Markov chains
    estimation.py       MLE, MAP, MMSE, CRLB, frequency MLE
    adaptive.py         Wiener, LMS, NLMS, RLS
    kalman.py           KF, EKF, RTS smoother, NEES/NIS, motion models
    hmm.py              forward, backward, posteriors, Viterbi, Baum–Welch
    finance.py          GARCH(1,1) MLE, Kalman time-varying regression (+ MLE of its noise levels)
    particle.py         SIR particle filter, resampling schemes, grid (exact) filter, benchmarks
    imm.py              interacting multiple model filter
    changepoint.py      Bayesian online change-point detection (Normal–Inverse-Gamma)
    graphical.py        Bayesian network by enumeration, chain sum/max-product, Gaussian belief propagation
    mixtures.py         Gaussian-mixture EM, BIC, mixture of experts
    approx.py           Metropolis, Gibbs, HMC, ESS, Laplace, variational inference
    gp.py               Bayesian linear regression + evidence, GP regression, Bayesian optimisation
    bandits.py          A/B-then-commit, ε-greedy, UCB1, Thompson sampling
    nn.py               MLP with hand-written backprop, Adam
    generative.py       1-D flow, VAE, GAN, DDPM/DDIM, exact mixture score, Tweedie
ssp/data/               8×8 digits (UCI optdigits) and the pre-trained digit diffusion weights
scripts/                train_digits_diffusion.py (re-creates the pre-trained weights, ~3 min on a CPU)
ssp/sims/               domain simulators (vehicle, netcode, speech, market, enhance, feedback, sports)
cpp/                    header-only C++17 Kalman filter, Viterbi decoder, feedback canceller + exercises
ssp/ui.py               shared plotting / layout helpers
tests/                  correctness proofs, page smoke tests, and a lint that catches KaTeX-breaking multi-line $$
```

## C++ ports

`cpp/` holds allocation-free C++17 versions of the Kalman filter, the Viterbi decoder and the hearing-aid
**adaptive feedback canceller** (block-based, real-time safe, about 0.1% of the per-sample budget at 16 kHz),
with a CMake build, benchmarks and interview-style exercises on real-time constraints. `pytest` checks that they
reproduce the Python, including the closed loop while it howls, and that the canceller's output doesn't depend
on the audio block size.

## Where to go next

* Record yourself in a car, next to a fan or with music playing, and compare the detectors and suppressors on real noise.
* Particle-filter the netcode demo, or put an IMM into it ("running" vs "juking" models).
* Template the C++ canceller on `float`, and add a howling detector (exercise 4 follow-ups in `cpp/README.md`).

Data is simulated by default, so every estimate can be compared with the hidden truth. The audio pages also let you **record from your microphone** in the browser (or upload a WAV), and the market page accepts a CSV of prices. This is for learning only;
nothing here is financial advice.
