# Practical Use of Probability

An interactive Streamlit lab for **statistical signal processing**. It starts with Bayes' rule, works up to
Kalman filters, and then applies them to real systems that still use them today.

* Every algorithm is written **from scratch in NumPy** (`ssp/core/`). Each page has a **Code** tab showing the
  exact source it runs.
* Every algorithm is **checked by a test**, either against theory (CRLB, χ² consistency, Wiener solution) or
  against a reference library (filterpy).
* Every page follows the same pattern: **Intuition & math → Playground → Break it → Code**.

## Run it

```bash
git clone https://github.com/raikar8/Practical-use-of-probability.git
cd Practical-use-of-probability
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the tests with `pytest`. This includes a smoke test that renders every page headlessly.

## What's inside (Sprint 1)

| Page | Concepts | Try this |
|---|---|---|
| **L0 · Probability toolkit** | Beta–Bernoulli updating, Bayes' rule and base rates, Markov chains, Gaussian conditioning | A confident wrong prior vs. the data |
| **L1 · MLE, MAP, CRLB** | Likelihood, MLE / MAP / MMSE, Monte Carlo vs. CRLB, frequency estimation and the threshold effect, robust estimation | MAP beating the CRLB, and why that's allowed |
| **L2 · LMS, NLMS, RLS** | Echo-path identification with an abrupt change, coloured input, Wiener target | Make LMS diverge; coloured input vs. RLS |
| **L3 · Kalman filter** | Predict/update, steady-state gain, 2D constant-velocity tracking, sensor blackout, RTS smoother, NEES/NIS consistency | Mis-tune Q and watch NEES leave the χ² band |
| **Car GPS + IMU fusion** | 5-state EKF with gyro-bias estimation through a GPS-denied tunnel (animated) | Turn off bias estimation → overconfident filter |
| **Online game lag prediction** | Snap vs. extrapolation vs. interpolation vs. Kalman prediction under ping, jitter and loss (animated, 60 Hz) | Noisy snapshots break naive extrapolation |

## Layout

```
streamlit_app.py        entry point (navigation)
app_pages/              one file per page
ssp/core/               from-scratch algorithms
    probability.py      Bayes updates, Markov chains
    estimation.py       MLE, MAP, MMSE, CRLB, frequency MLE
    adaptive.py         Wiener, LMS, NLMS, RLS
    kalman.py           KF, EKF, RTS smoother, NEES/NIS, motion models
ssp/sims/               domain simulators (vehicle, netcode)
ssp/ui.py               shared plotting / layout helpers
tests/                  correctness proofs + page smoke tests
```

## Roadmap

* **Sprint 2:** L4 HMMs (forward, Viterbi, forward–backward, Baum–Welch). Market-regime demo (HMM, Kalman hedge
  ratio, GARCH-MLE). Speech demo: HMM voice-activity detection.
* **Sprint 3:** L5 particle filter, IMM, Bayesian online change-point detection. Sports skill ratings and live
  win probability. Audio MMSE noise suppression and hearing-aid feedback cancellation. C++ ports of the
  Kalman filter and Viterbi.

All data is simulated, so every estimate can be compared with the hidden truth. This is for learning only;
nothing here is financial advice.
