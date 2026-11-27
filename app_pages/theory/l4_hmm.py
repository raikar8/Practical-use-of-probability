import streamlit as st

from ssp import ui

ui.theory_header("l4", """
Sometimes the hidden thing isn't a number but a **situation** that switches: *someone is speaking / nobody is*,
*the market is calm / turbulent*, *the casino is using the fair die / the loaded die*. We never see the situation,
only noisy observations it produces. A **hidden Markov model (HMM)** describes this, and four classic algorithms
answer everything we want to ask of it.
""")

ui.section("1. The model: a Markov chain you can't see")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11];
  node [shape=circle, style=filled, fillcolor="#eef4fc", color="#257de0"];
  s1 [label="s₁"]; s2 [label="s₂"]; s3 [label="s₃"]; s4 [label="…", shape=plaintext, style=""];
  node [shape=box, style="rounded,filled", fillcolor="#fdf0e8", color="#f47230"];
  x1 [label="x₁"]; x2 [label="x₂"]; x3 [label="x₃"];
  s1 -> s2 [label=" A"]; s2 -> s3 [label=" A"]; s3 -> s4;
  s1 -> x1 [label=" b"]; s2 -> x2 [label=" b"]; s3 -> x3 [label=" b"];
  {rank=same; s1; x1} {rank=same; s2; x2} {rank=same; s3; x3}
}
"""
)
st.markdown(
    r"""
* **Hidden states** $s_t \in \{1,\dots,K\}$ follow a Markov chain (Level 0): initial probabilities $\pi_k$ and
  transition matrix $A_{ij} = P(s_{t+1}=j \mid s_t=i)$.
* **Emissions.** In state $k$, the observation $x_t$ is drawn from $b_k(x_t) = p(x_t \mid s_t = k)$. That could
  be a die-face table, or a Gaussian over audio features or daily returns.

**Running example: the dishonest casino.** State 1 = fair die (each face 1/6). State 2 = loaded die (P(6) = 0.5,
other faces 0.1). The casino switches fair → loaded with probability 0.02 per roll and back with 0.1. You see
only the rolls.

**Duration.** A state with self-transition $A_{kk}$ lasts on average $1/(1-A_{kk})$ steps: 50 rolls for the
fair die and 10 for the loaded one. The transition matrix is a **prior on how long situations last**. In voice
activity detection that is exactly what stops the output flickering.
"""
)

ui.section("2. Four questions, four algorithms")
st.markdown(
    r"""
| Question | Name | Algorithm | Uses |
|---|---|---|---|
| Which state am I in **now**, given everything so far? $P(s_t \mid x_{1:t})$ | filtering | **forward** | live decisions (real time) |
| Which state was I in **at time t**, given *all* the data? $P(s_t \mid x_{1:T})$ | smoothing | **forward–backward** | offline labelling, learning |
| What is the single most probable **sequence** of states? | decoding | **Viterbi** | speech recognition, segmentation |
| What are $\pi$, $A$ and $b$? | learning | **Baum–Welch (EM)** | training without labels |
"""
)

ui.section("3. The forward algorithm: predict, then weigh by the evidence")
st.markdown(r"""Let $\alpha_t(j) = P(s_t = j \mid x_{1:t})$, the filtered belief. One step is the same two moves as the Kalman filter:""")
ui.eq(r"\underbrace{\alpha^-_t(j) = \sum_i \alpha_{t-1}(i)\,A_{ij}}_{\text{predict with the chain}} \qquad"
      r"\underbrace{\alpha_t(j) = \frac{b_j(x_t)\,\alpha^-_t(j)}{\sum_k b_k(x_t)\,\alpha^-_t(k)}}_{\text{Bayes update}}", 1)
st.markdown(
    r"""
**Worked example.** Start 50/50. Roll a **6**: the likelihoods are $b_{\text{fair}}(6) = 1/6$ and
$b_{\text{loaded}}(6) = 0.5$, so $\alpha_1 \propto [0.5/6,\; 0.5\times0.5] = [0.083, 0.25] \to [0.25, 0.75]$.

Roll another **6**. Predict: $\alpha^-_2 = [0.25\times0.98 + 0.75\times0.1,\; 0.25\times0.02 + 0.75\times0.9]
= [0.32, 0.68]$. Weigh: $[0.32/6,\; 0.68\times0.5] = [0.053, 0.34] \to [0.14, 0.86]$.

Two sixes and we're 86% sure the die is loaded. The normalising sums (the denominators of (1)) multiply up to
the **likelihood of the whole sequence**, $p(x_{1:T})$, which is what learning maximises. In code we normalise
at every step and add up the logs, because a product of 10,000 small numbers underflows to zero.
"""
)

ui.section("4. Backward pass and smoothing: let the future vote")
st.markdown(r"""A **backward** recursion computes how well each state at time $t$ explains the *future* observations:""")
ui.eq(r"\beta_t(i) \propto \sum_j A_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j), \qquad \beta_T(i) = 1", 2)
ui.eq(r"\gamma_t(k) = P(s_t = k \mid x_{1:T}) \;\propto\; \alpha_t(k)\,\beta_t(k)", 3)
st.markdown(
    r"""
The smoothed $\gamma$ is more accurate than the filtered $\alpha$, especially around switch points, because the
rolls that come *after* a switch confirm it. It's only available after the fact, though. **Using smoothed
results to simulate a live decision is cheating** (look-ahead bias). The market demo shows this.
"""
)

ui.section("5. Viterbi: the best whole path")
st.markdown(r"""Picking the most likely state at each $t$ separately can produce a sequence that is impossible or very unlikely as a whole. **Viterbi** finds the single best path by dynamic programming. Replace the sum in (1) by a max and remember which predecessor won:""")
ui.eq(r"\delta_t(j) = b_j(x_t)\,\max_i\big[\delta_{t-1}(i)\,A_{ij}\big], \qquad "
      r"\psi_t(j) = \arg\max_i\big[\delta_{t-1}(i)\,A_{ij}\big]", 4)
st.markdown(
    r"""
At the end, take the best final state and follow the back-pointers $\psi$ backwards. The cost is
$O(TK^2)$ instead of checking all $K^T$ paths. It's done in logs (sums instead of products) for the same underflow
reason. The result is **one consistent story** with no single-step flickers, which is what you want when the
output is a label sequence (speech / silence segments, words).
"""
)

ui.section("6. Baum–Welch: learning without labels")
st.markdown(
    r"""
Nobody tells you the casino's matrices. **Expectation–maximisation** alternates two steps:

1. **E-step:** with the current guess of $(\pi, A, b)$, run forward–backward to get $\gamma_t(k)$ (expected
   time in each state) and $\xi_t(i,j)$ (expected $i\to j$ transitions).
2. **M-step:** re-estimate by **counting**, as if those soft labels were true:
"""
)
ui.eq(r"A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}, \qquad "
      r"b_k(m) = \frac{\sum_{t:\,x_t=m}\gamma_t(k)}{\sum_t\gamma_t(k)}, \qquad "
      r"\mu_k = \frac{\sum_t \gamma_t(k)\,x_t}{\sum_t\gamma_t(k)}", 5)
st.markdown(
    r"""
(The last formula is for Gaussian emissions, with a similar weighted formula for the variance.)

* Each iteration **never decreases** the likelihood $p(x_{1:T})$.
* It can stop at a **local** optimum. Restart from several initial guesses and keep the best.
* **Label switching:** EM doesn't know which state you *call* "loaded". Name states afterwards by what they emit.
* It needs enough **events**, not just samples. With few switches, the transition probabilities are poorly
  determined.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l4", [
    ("*P(switch fair → loaded)*, *P(switch loaded → fair)*", r"$A_{12}$, $A_{21}$ (and the expected durations $1/(1-A_{kk})$)"),
    ("*Loaded die: P(6)*", r"the emission $b_{\text{loaded}}(6)$; the closer to 1/6, the harder the problem"),
    ("Blue line", r"filtered $\alpha_t(\text{loaded})$, eq. (1), which is real-time"),
    ("Orange line", r"smoothed $\gamma_t(\text{loaded})$, eq. (3), which uses the future"),
    ("Dotted aqua steps", r"the Viterbi path, eq. (4)"),
    ("Tab **Learning with Baum–Welch**: *EM iterations*, *Initial-guess seed*, *Random restarts*", r"section 6: the likelihood curve, learned $b$ and $A$, local optima"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Filtering vs smoothing** answer different questions. Live systems can only use filtering (or fixed-lag
  smoothing with a delay).
* **Too-frequent switching** in $A$ removes the 'stickiness' that makes HMMs better than frame-by-frame
  thresholds.
* **Overlapping emissions** (a loaded die barely different from a fair one) make every algorithm guess. No
  algorithm can create information that isn't in the data.
"""
)

ui.section("Check yourself")
ui.check([
    ("A_kk = 0.95 for a state. How long does a visit last on average?", "1/(1 − 0.95) = **20 steps**."),
    ("Belief [0.5, 0.5]; you roll a 3 (fair 1/6, loaded 0.1). New belief, before the next transition?",
     "∝ [0.5/6, 0.5·0.1] = [0.083, 0.05] → **[0.625, 0.375]**: a non-six is evidence for the fair die."),
    ("Why can the per-step most likely state differ from the Viterbi path?",
     "Per-step choices ignore how the steps connect. They can string together switches that the transition "
     "matrix makes very unlikely. Viterbi scores the whole sequence."),
    ("EM's final likelihood is higher than that of the true parameters. Is something wrong?",
     "No. EM fits *this* sample, including its randomness (mild overfitting). It will usually be close to the truth "
     "with enough data."),
])
