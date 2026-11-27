import streamlit as st

from ssp import ui

ui.theory_header("netcode", """
**Prerequisites:** Level 3 (Kalman filter, constant-velocity model, missing measurements).

In an online game you see other players through a network. Their position reaches you **late**, **irregularly**
and **sometimes not at all**, yet your screen must show where they are **now**, 60 times a second. That is a
real-time prediction problem, and a small Kalman filter solves it in a principled way.
""")

ui.section("1. How positions travel")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", fillcolor="#f4f4f4"];
  S [label="Server\\n(knows the true position)"]; N [label="Network\\ndelay = ping/2 + jitter\\nsome packets lost",
     fillcolor="#fdf0e8"];
  C [label="Your client\\ndraws a frame every 16.7 ms", fillcolor="#eef4fc"];
  S -> N [label="snapshot every\\n1/(snapshot rate) s"]; N -> C;
}
"""
)
st.markdown(
    r"""
* **Snapshot rate:** how often the server sends the player's position (e.g. 20 Hz = every 50 ms).
* **Ping:** the round-trip time; a one-way trip takes about half.
* **Jitter:** random extra delay, so packets don't arrive evenly spaced.
* **Packet loss:** some snapshots never arrive.
* **Snapshot noise:** positions are compressed or quantised before sending, so each one is slightly off.

**How big is the problem?** A player running at 7 m/s with 110 ms of total delay is $7 \times 0.11 \approx 0.8$ m
from where the newest snapshot says, more than a player's width (≈0.6 m).
"""
)

ui.section("2. Four ways to draw the other player")
st.markdown(
    r"""
| Strategy | Draws | Error comes from |
|---|---|---|
| **Snap** | the newest snapshot as-is | always behind by the delay: error ≈ speed × delay |
| **Interpolate** | a blend of two snapshots, at a time (ping/2 + buffer) in the **past** | smooth and correct about the past, but even further behind the present |
| **Extrapolate** (dead reckoning) | newest position + velocity × elapsed time, with velocity from the last two snapshots | snapshot noise ÷ small time gap = a very noisy velocity; overshoots on turns |
| **Kalman** | a constant-velocity filter updated by each snapshot, then **predicted** to now | can't foresee sudden direction changes, but averages out the noise |
"""
)

ui.section("3. The Kalman model")
st.markdown(r"""State per axis: position and velocity. Snapshots arrive at **irregular** times $t_k$, so the motion model uses the actual gap $\Delta t = t_k - t_{k-1}$ (Level 3, equation 8):""")
ui.eq(r"\mathbf x = [p_x, v_x, p_y, v_y]^T,\qquad F(\Delta t) = \begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}\otimes I_2,"
      r"\qquad Q(\Delta t) = q\begin{bmatrix}\Delta t^3/3&\Delta t^2/2\\ \Delta t^2/2&\Delta t\end{bmatrix}\otimes I_2", 1)
st.markdown(
    r"""
* **Update** when a snapshot arrives (in send-time order; late duplicates are dropped), with $H$ picking out
  positions and $R$ = snapshot noise².
* **Every frame**, predict from the last update time $t_k$ to the current time $t$ **without** updating:
"""
)
ui.eq(r"\hat{\mathbf x}(t) = F(t - t_k)\,\hat{\mathbf x}_k, \qquad P(t) = F\,P_k\,F^T + Q(t - t_k)", 2)
st.markdown(
    r"""
$P(t)$ gives the **uncertainty radius** drawn in the Lab. It grows like $q\,\tau^3/3$ while packets are missing,
so a game can use it to stop extrapolating, or a server can use it to judge whether a hit was plausible.

**Choosing q.** $q$ says how violently players change velocity. Too small: the filter believes players glide in
straight lines and overshoots every turn. Too large: it trusts each noisy snapshot and jitters.

**The unavoidable error.** If a player changes direction (a "juke") after the last snapshot was sent, *nobody*
can know it yet. Prediction error from an acceleration $a$ over a delay $\tau$ is about $\tfrac12 a\tau^2$. This is
why every method gets worse as ping rises.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("netcode", [
    ("*Ping*, *Jitter σ*, *Packet loss*", r"the network delays and losses in section 1"),
    ("*Snapshot rate*", r"how often $t_k$ arrives; larger gaps mean larger $\Delta t$ in (1)"),
    ("*Snapshot position noise σ*", r"$\sqrt R$: the reason naive extrapolation goes wild"),
    ("*Player agility*", r"how often the true player changes direction: the unpredictable part"),
    ("*Interpolation buffer*", r"how far in the past interpolation renders"),
    ("*Kalman manoeuvre noise q*", r"$q$ in (1)"),
    ("Left panel / right zoom panel", r"the arena, and each method's error with the true player held at the centre"),
    ("Dotted line in the error plot", r"the Kalman filter's own 95% radius from $P(t)$ in (2)"),
])

ui.section("Check yourself")
ui.check([
    ("Ping 200 ms, player at 6 m/s. Roughly how far behind is 'snap'?",
     "One-way ≈ 100 ms plus up to one snapshot interval of age: at least **0.6 m**, often ~0.9 m."),
    ("Snapshot noise 0.2 m, snapshots 50 ms apart. How noisy is a two-point velocity estimate?",
     "≈ √2 · 0.2 / 0.05 ≈ **5.7 m/s** of noise, nearly the running speed itself. That's why naive "
     "extrapolation jitters."),
    ("Why render remote players in the past at all?",
     "Interpolation never shows a position that didn't happen, so motion looks smooth. Many engines accept being "
     "~100 ms behind for that, and extrapolate only when packets stop."),
])
