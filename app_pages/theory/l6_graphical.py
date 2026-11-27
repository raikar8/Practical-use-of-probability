import streamlit as st

from ssp import ui

ui.theory_header("l6", """
Levels 0–5 used a handful of models: a coin, a Kalman filter, an HMM. Real problems have **many** uncertain
quantities that influence each other: faults and alarms, words and sounds, transmitted bits and received
signals. **Graphical models** draw those influences as a graph, and **message passing** turns the graph into an
inference algorithm. The payoff of this level: the HMM forward–backward algorithm, Viterbi and the Kalman
smoother turn out to be **one algorithm**, run on different kinds of numbers.
""")

ui.section("1. Why draw a graph?")
st.markdown(
    r"""
A full joint distribution over $n$ yes/no variables needs $2^n - 1$ numbers: about a million for 20 variables.
Nobody can specify that, and no amount of data would pin it down. But most variables only influence a few others
**directly**. The chain rule of probability plus "who directly affects whom" gives a factorisation:
"""
)
ui.eq(r"p(x_1, \dots, x_n) = \prod_{i=1}^{n} p\big(x_i \mid \text{parents}(x_i)\big)", 1)
st.markdown(
    r"""
Each factor is a small table (a **conditional probability table**, CPT). That is a **Bayesian network**: a
directed graph with no cycles, with one CPT per node.

**Running example: Pearl's burglar alarm.** A burglary (B) or an earthquake (E) can set off the alarm (A).
Two neighbours, John (J) and Mary (M), may call you when they hear it.
"""
)
st.graphviz_chart(
    """
digraph {
  rankdir=TB; node [fontname="Helvetica", fontsize=11, shape=ellipse, style=filled, fillcolor="#eef4fc",
  color="#257de0"];
  B [label="Burglary\\nP = 0.01"]; E [label="Earthquake\\nP = 0.02"];
  A [label="Alarm"]; J [label="John calls"]; M [label="Mary calls"];
  B -> A; E -> A; A -> J; A -> M;
}
"""
)
st.markdown(
    r"""
$p(B,E,A,J,M) = p(B)\,p(E)\,p(A \mid B,E)\,p(J \mid A)\,p(M \mid A)$ needs $1+1+4+2+2 = 10$ numbers instead of 31.
The saving grows exponentially with the number of variables.
"""
)

ui.section("2. Reading a graph: three patterns")
st.markdown(
    r"""
Everything about dependence in a Bayesian network follows from three small patterns. "Observed" means we know
the value of the middle node.

| Pattern | Example | Ends dependent when the middle is **unknown**? | …when the middle is **observed**? |
|---|---|---|---|
| **Chain** $X \to Z \to Y$ | Burglary → Alarm → John calls | yes (a burglary makes a call likelier) | **no**: once you know the alarm rang, a burglary tells you nothing more about John |
| **Fork** $X \leftarrow Z \to Y$ | John ← Alarm → Mary | yes (John calling makes Mary likelier) | **no**: given the alarm, the two neighbours are independent |
| **Collider** $X \to Z \leftarrow Y$ | Burglary → Alarm ← Earthquake | **no**: burglaries and earthquakes are unrelated | **yes!** Observing the alarm couples them |

The collider is the surprising one, and it has a name: **explaining away**.
"""
)
st.markdown("**Worked example: explaining away.** Compute posteriors exactly by summing the joint (1) over the unknowns:")
ui.eq(r"P(B=1 \mid A=1) = \frac{\sum_{e} p(B{=}1)\,p(e)\,p(A{=}1 \mid B{=}1, e)}{\sum_{b,e} p(b)\,p(e)\,p(A{=}1 \mid b, e)}", 2)
st.markdown(
    r"""
* Before anything happens: $P(\text{burglary}) = 1\%$.
* The alarm rings: $P(\text{burglary} \mid A) = 58\%$. The alarm is strong evidence.
* You then hear on the radio there was an earthquake: $P(\text{burglary} \mid A, E) = 3\%$.

The earthquake doesn't make burglaries less common. It **explains** the alarm, so the alarm stops being evidence
of a burglary. The same reasoning appears in diagnosis (two possible faults, one symptom) and in sensor fusion
(a reading explained by a known disturbance).
"""
)

ui.section("3. Factor graphs and messages")
st.markdown(
    r"""
Summing the joint over all unknowns (eq. 2) costs $2^n$ terms. The trick that makes inference fast is to **push
the sums inside the products** so each sum only touches the factors that involve its variable. Done
systematically on a tree-shaped graph, this becomes **message passing**: every node tells its neighbour what it
knows about the shared variable, summarising everything on its own side of the graph.

On a **chain** $s_1 - s_2 - \dots - s_T$ with local evidence $\phi_t(s_t) = p(x_t \mid s_t)$ and links
$\psi(s_{t-1}, s_t) = A_{s_{t-1}s_t}$, there are two message streams:
"""
)
ui.eq(r"m_{\rightarrow t}(k) = \sum_j m_{\rightarrow t-1}(j)\,\phi_{t-1}(j)\,A_{jk}, \qquad "
      r"m_{\leftarrow t}(k) = \sum_j A_{kj}\,\phi_{t+1}(j)\,m_{\leftarrow t+1}(j)", 3)
ui.eq(r"P(s_t = k \mid x_{1:T}) \;\propto\; m_{\rightarrow t}(k)\;\phi_t(k)\;m_{\leftarrow t}(k)", 4)
st.markdown(
    r"""
In words: **what the past says × what I see here × what the future says**. This is the **sum-product**
algorithm. Compare with Level 4:

* $m_{\rightarrow t}\,\phi_t$ is exactly the forward variable $\alpha_t$, and $m_{\leftarrow t}$ is the backward
  variable $\beta_t$. Forward–backward **is** sum-product on a chain.
* Replace every $\sum$ by $\max$ and you get **max-product**, which finds the single best joint assignment.
  That **is** Viterbi.
"""
)

ui.section("4. Gaussian messages: the Kalman smoother")
st.markdown(
    r"""
If the variables are continuous and everything is linear-Gaussian, every message is a Gaussian and can be carried
as a (mean, variance) pair. For the random walk $x_t = x_{t-1} + w_t$, $z_t = x_t + v_t$ (Level 3):

* **Combining** two Gaussian beliefs about the same variable multiplies them: precisions (1/variance) add, and
  the mean is the precision-weighted average.
* **Passing through the link** $x_{t-1} \to x_t$ adds the process noise $q$ to the variance.
"""
)
ui.eq(r"\frac{1}{v} = \frac{1}{v_1} + \frac{1}{v_2}, \qquad m = v\left(\frac{m_1}{v_1} + \frac{m_2}{v_2}\right)"
      r"\qquad\text{then}\qquad (m, v) \to (m, v + q)", 5)
st.markdown(
    r"""
The forward stream is the Kalman filter (prediction, then update), and forward × local × backward is the
**RTS smoother**. Same graph, same algorithm; only the type of message changed (a table of probabilities in an
HMM, a Gaussian in a Kalman filter, a weighted cloud of particles in a particle filter).
"""
)

ui.section("5. Beyond chains")
st.markdown(
    r"""
* **Trees:** sum-product is exact on any tree, in two sweeps (leaves to root, root to leaves).
* **Graphs with loops:** run the same message updates anyway and iterate: **loopy belief propagation**. It's not
  guaranteed to converge or be exact, but it is how **turbo codes** and **LDPC codes** are decoded in 4G/5G, Wi-Fi
  and satellite links, and it works remarkably well there.
* **Exact inference on loopy graphs** groups variables into clusters until the graph is a tree (the junction tree
  algorithm). The cost grows exponentially with cluster size, which is why approximate inference (Level 8) exists.
"""
)

ui.section("6. A unifying fact: exponential families and conjugacy")
st.markdown(
    r"""
Why were the updates in Levels 0–3 so tidy? Beta–Bernoulli, Gamma–Poisson and Gaussian–Gaussian are all
**exponential families**:
"""
)
ui.eq(r"p(x \mid \eta) = h(x)\,\exp\!\big(\eta^\top T(x) - A(\eta)\big)", 6)
st.markdown(
    r"""
* $T(x)$ are the **sufficient statistics**: counts of heads and tails, the number of goals, the sum and sum of
  squares. They're all the data you need to keep.
* With a **conjugate prior**, the posterior has the same form as the prior, and updating means **adding the
  sufficient statistics** to the prior's parameters. That is why Beta(α, β) became Beta(α + heads, β + tails).
* For message passing, conjugacy means messages stay in the same family, so they can be carried as a few numbers.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l6", [
    ("Tab **Explaining away**: evidence buttons, posterior bars", r"exact inference by summing eq. (1), as in eq. (2)"),
    ("*Burglary* bar dropping when *Earthquake = yes* is added", "the collider pattern of §2"),
    ("Tab **Messages on a chain**: 'from the past', 'seen here', 'from the future', 'belief'",
     r"the three factors of eq. (4) and their normalised product"),
    ("*Sum-product / Max-product* switch", r"forward–backward vs Viterbi (§3)"),
    ("Tab **Gaussian messages**: three bell curves and their product", r"eq. (5); the blue/orange lines are the "
     r"Kalman filter and the RTS smoother"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Arrows are not necessarily causes.** A Bayesian network encodes conditional independences. Reading arrows as
  causes is an extra assumption (that's the subject of causal inference).
* **Conditioning on a collider creates dependence.** Selecting only cases where "the alarm rang" (or "the patient
  was admitted") can make unrelated causes look negatively correlated (Berkson's paradox).
* **Loopy BP isn't exact.** On graphs with short loops it can be overconfident or oscillate.
"""
)

ui.section("Check yourself")
ui.check([
    ("How many numbers specify p(B)p(E)p(A|B,E)p(J|A)p(M|A) with binary variables?",
     "1 + 1 + 4 + 2 + 2 = **10**, versus 2⁵ − 1 = 31 for an unrestricted joint."),
    ("The alarm is known to have rung. Does learning that John called change P(Mary calls)?",
     "**No.** Given the alarm, John and Mary are independent (a fork with the middle observed)."),
    ("Burglary and earthquake are independent. Why does observing the alarm make them dependent?",
     "The alarm is a collider. Either cause explains it, so confirming one cause makes the other less needed."),
    ("Which algorithm do you get if you replace sums by maxima in sum-product on an HMM chain?",
     "**Viterbi** (max-product): it finds the single most probable joint state sequence."),
])
