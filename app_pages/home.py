import streamlit as st

from ssp.ui import TOPICS

st.title("Practical Use of Probability")
st.markdown(
    "#### Learn statistical signal processing from scratch, by seeing it work on real problems"
)
st.markdown(
    """
This app is for someone who wants to understand **how machines make good guesses from noisy data**: how a phone
knows where it is in a tunnel, how a hearing aid separates your voice from traffic, how a game draws another
player who is 100 ms away. It assumes school algebra and a little calculus. Everything else is built up from zero.
"""
)

st.header("The one problem behind everything here")
st.markdown(
    """
There is something we **want to know** but **can't observe directly**: a position, a speech signal, a team's
true strength, the state of a machine. We call it the **hidden state** or the **parameter**.

What we *can* observe are **measurements**: sensor readings, audio samples, match results. They are related to
the hidden thing, but corrupted by **noise**.

Statistical signal processing is the craft of turning those noisy measurements into the best possible estimate
of the hidden thing, **and an honest statement of how uncertain that estimate is**.
"""
)
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#257de0", fontname="Helvetica"];
  edge [color="#555555", fontname="Helvetica", fontsize=10];
  B [label="Belief\\n(what we think + how unsure)"];
  P [label="Predict\\n(use a model of how things change)"];
  O [label="Observe\\n(a noisy measurement arrives)", fillcolor="#fdf0e8", color="#f47230"];
  C [label="Correct\\n(Bayes: weigh the data by how much we trust it)"];
  B -> P [label="time passes:\\nuncertainty grows"];
  P -> C; O -> C;
  C -> B [label="uncertainty shrinks"];
}
"""
)
st.markdown(
    """
Almost every algorithm in this app is that loop. They differ in **what the belief looks like**:

* one number with an error bar: MLE and MAP (Level 1)
* a set of filter weights that keep adapting: LMS and RLS (Level 2)
* a Gaussian over a state vector: the Kalman filter (Level 3)
* probabilities over a few discrete situations: hidden Markov models (Level 4)
* a cloud of samples, when nothing is Gaussian: particle filters (Level 5)
"""
)

st.header("How to use this app")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
**Every topic has two pages** in the sidebar:

1. **Theory.** Read this first. It explains the problem in plain words, builds the maths step by step with
   every symbol defined, works through a numeric example, and ends with a table that maps each equation to a
   control or plot in the Lab. There are short self-check questions at the bottom.
2. **Lab.** The interactive part. It opens with **what the lab is for**, **what you're looking at** and **where to
   start**. Every control has a **?** tooltip, a **Controls explained** table, and a numbered list of **Guided
   experiments**: what to change, what to watch, and why it happens.
"""
    )
with c2:
    st.markdown(
        """
**Colour conventions in every plot**

* **grey** is the hidden truth (only known because the data is simulated)
* coloured lines are estimates; blue is usually the method the page is about
* shaded bands are uncertainty: the algorithm's own claim of "I'm probably within this range"

**Callout boxes**

* 💡 **Takeaway**: the one idea to remember
* 🔧 **Break it**: a deliberate mistake to learn from
* 🏭 **In practice**: where this runs in real products today
"""
    )

st.header("Suggested path")
st.markdown(
    "Go through Levels 0 → 5 in order; each one uses ideas from the one before. The **demos** each need only "
    "some of the levels, so you can jump to a demo as soon as you've done its prerequisites:"
)
st.graphviz_chart(
    """
digraph {
  rankdir=LR; nodesep=0.25; ranksep=0.5;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  node [fillcolor="#eef4fc", color="#257de0"];
  L0 [label="L0 Probability"]; L1 [label="L1 Estimation"]; L2 [label="L2 Adaptive filters"];
  L3 [label="L3 Kalman"]; L4 [label="L4 HMM"]; L5 [label="L5 Particles, IMM,\\nchange points"];
  node [fillcolor="#fdf0e8", color="#f47230"];
  V [label="Car GPS+IMU"]; N [label="Game lag"]; S [label="Voice activity"];
  E [label="Noise suppression\\n& hearing aids"]; M [label="Markets"]; SP [label="Sports"];
  L0 -> L1 -> L2 -> L3 -> L4 -> L5;
  L3 -> V; L3 -> N; L4 -> S; L1 -> E; L2 -> E; L4 -> M; L3 -> M; L3 -> SP; L1 -> SP;
}
"""
)
cols = st.columns(3)
for i, (key, (level, title, theory, lab)) in enumerate(TOPICS.items()):
    with cols[i % 3].container(border=True):
        st.markdown(f"**{level if level != 'Domain' else 'Demo'} · {title}**")
        a, b = st.columns(2)
        a.page_link(theory, label="Theory", icon=":material/menu_book:")
        b.page_link(lab, label="Lab", icon=":material/science:")

st.header("Good to know")
st.markdown(
    """
* **Simulated data.** All demos use simulated data by default, so the hidden truth is known and every estimate
  can be checked against it. The speech pages can also record your microphone, and the market page accepts a CSV
  of prices.
* **No black boxes.** Every algorithm is written from scratch in NumPy (folder `ssp/core/`), and every Lab has a
  **Code** section showing the exact function it runs. Tests compare each one with theory or a reference library.
* **C++.** The folder `cpp/` has real-time-safe C++ versions of the Kalman filter, the Viterbi decoder and the
  hearing-aid feedback canceller, with interview-style exercises.
* **Notation** is collected on the *Notation & glossary* page; keep it open in a second tab.
"""
)
