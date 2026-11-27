import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui

ui.theory_header("l3", """
Now the hidden quantity **moves**: a car's position, a battery's charge, a target on radar. We have a rough
**model of how it moves** and a **noisy sensor**. The Kalman filter (1960) combines them optimally, one
measurement at a time, and keeps an honest error bar. It runs in every phone, car and aircraft. If Levels 0–1
clicked, there is nothing new here except bookkeeping.
""")

ui.section("1. The state-space model: two equations")
st.markdown(r"""The **state** $\mathbf x_k$ is the smallest set of numbers that describes the system at time step $k$ (for a moving object: position and velocity). Two equations describe the world:""")
ui.eq(r"\mathbf x_k = F\,\mathbf x_{k-1} + \mathbf w_k, \qquad \mathbf w_k \sim \mathcal N(\mathbf 0, Q)"
      r"\qquad\text{(motion model)}", 1)
ui.eq(r"\mathbf z_k = H\,\mathbf x_k + \mathbf v_k, \qquad \mathbf v_k \sim \mathcal N(\mathbf 0, R)"
      r"\qquad\text{(measurement model)}", 2)
st.markdown(
    r"""
| Symbol | Meaning | 1-D example: room temperature | Tracking example |
|---|---|---|---|
| $F$ | how the state moves on, if nothing random happens | 1 (stays the same) | position += velocity × Δt |
| $Q$ | **process noise**: how much the model can be wrong per step | small $q$: temperature drifts slowly | random accelerations |
| $H$ | what the sensor sees of the state | 1 (we read temperature directly) | picks out position only |
| $R$ | **measurement noise** of the sensor | thermometer noise $r$ | GPS / radar error |

$Q$ and $R$ are the two knobs a Kalman-filter engineer spends their time on.
"""
)

ui.section("2. The filter: predict, then update")
st.markdown(r"""The filter keeps a **Gaussian belief** about the state: a best guess $\hat{\mathbf x}$ and a **covariance** $P$ (its own error bars, including correlations). Each step has two halves.""")
st.markdown("**Predict.** Push the belief through the motion model. The guess moves, and the uncertainty **grows** by $Q$:")
ui.eq(r"\hat{\mathbf x}^- = F\,\hat{\mathbf x}, \qquad P^- = F\,P\,F^T + Q", 3)
st.markdown("**Update.** Compare the measurement with what we expected to see:")
ui.eq(r"\boldsymbol\nu = \mathbf z - H\hat{\mathbf x}^- \;\;(\text{innovation}),\qquad "
      r"S = H P^- H^T + R \;\;(\text{its covariance})", 4)
ui.eq(r"K = P^-H^T S^{-1}, \qquad \hat{\mathbf x} = \hat{\mathbf x}^- + K\boldsymbol\nu, \qquad P = (I - KH)\,P^-", 5)
st.markdown(
    r"""
The **Kalman gain** $K$ decides how much of the surprise $\boldsymbol\nu$ to believe. In one dimension with
$H=1$ it is simply
"""
)
ui.eq(r"K = \frac{P^-}{P^- + r}", 6)
st.markdown(
    r"""
which is exactly the **"weight on the data"** from Level 1, equation (6): prediction uncertainty versus sensor
uncertainty. If the prediction is vague ($P^- \gg r$), $K \to 1$ and we trust the sensor. If the sensor is noisy
($r \gg P^-$), $K\to 0$ and we trust the model.

**Worked example (1-D).** Belief $\hat x = 20.0$ °C with $P = 1.0$; drift $q = 0.01$; thermometer noise $r = 1$;
new reading $z = 21.0$.

1. Predict: $\hat x^- = 20.0$, $P^- = 1.01$.
2. Gain: $K = 1.01/2.01 \approx 0.502$.
3. Update: $\hat x = 20.0 + 0.502 \times 1.0 = 20.50$, and $P = (1-0.502)\times1.01 \approx 0.503$.

One reading halved the variance. Repeat forever.
"""
)
xg = np.linspace(16, 25, 400)
fig = go.Figure()
fig.add_scatter(x=xg, y=stats.norm(20, np.sqrt(1.01)).pdf(xg), name="prediction N(20, 1.01)",
                line=dict(color=ui.C["truth"], dash="dash"))
fig.add_scatter(x=xg, y=stats.norm(21, 1).pdf(xg), name="measurement N(21, 1)", line=dict(color=ui.C[2]))
fig.add_scatter(x=xg, y=stats.norm(20.502, np.sqrt(0.503)).pdf(xg), name="updated belief N(20.50, 0.50)",
                line=dict(color=ui.C[1], width=3))
ui.show(ui.style(fig, "The update is a product of Gaussians: narrower than either", "temperature (°C)", "density",
                 height=260))

ui.section("3. Steady state: only the ratio q / r matters")
st.markdown(r"""For the 1-D random walk, $P$ and $K$ converge after a few steps to constants. Solving $P^-_\infty = (1-K_\infty)P^-_\infty + q$:""")
ui.eq(r"P^-_\infty = \frac{q + \sqrt{q^2 + 4qr}}{2}, \qquad K_\infty = \frac{P^-_\infty}{P^-_\infty + r}", 7)
st.markdown(
    r"""
A filter with constant gain is just an **exponential moving average** $\hat x \leftarrow \hat x + K(z - \hat x)$.
The Kalman filter picks the averaging constant **optimally** from $q/r$, and adapts it during start-up or
after missing data.
"""
)

ui.section("4. Tracking in 2-D: velocity you never measure")
st.markdown(r"""For a moving target, use state $[p, v]$ per axis and assume the velocity changes by random accelerations of intensity $q$ (the **constant-velocity model**):""")
ui.eq(r"F = \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix}, \qquad "
      r"Q = q\begin{bmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{bmatrix}, \qquad "
      r"H = \begin{bmatrix}1 & 0\end{bmatrix}", 8)
st.markdown(
    r"""
The sensor measures **only position**, yet the filter estimates **velocity**. The off-diagonal terms of $P$ record
that position errors and velocity errors are correlated, so a position surprise also corrects the velocity.

* **Missing measurements** (a tunnel, a blackout): skip the update and only predict. $P$ grows every step,
  which shows up as a **growing uncertainty ellipse**. When data returns, one update shrinks it again.
* **Smoothing (RTS).** If you can wait until the end, a backward pass uses *future* measurements too. That's
  better for offline analysis, but not for live control.
"""
)

ui.section("5. Is my filter honest? NEES and NIS")
st.markdown(r"""A filter claims its error covariance is $P$. Two tests check that claim:""")
ui.eq(r"\text{NEES}_k = (\mathbf x_k - \hat{\mathbf x}_k)^T P_k^{-1}(\mathbf x_k - \hat{\mathbf x}_k), \qquad "
      r"\text{NIS}_k = \boldsymbol\nu_k^T S_k^{-1}\boldsymbol\nu_k", 9)
st.markdown(
    r"""
If the model is right, both follow a **chi-square distribution** whose mean equals the dimension (4 for the
2-D state, 2 for the 2-D measurement). Averaged over many runs, they should stay inside a $\chi^2$ band.

* **Above the band: overconfident.** Real errors are larger than $P$ claims, usually because $Q$ is too small.
  This is how filters **diverge** in practice.
* **Below the band: too pessimistic.** $P$ is too large ($Q$ or $R$ too big), and the filter wastes information.

NEES needs the truth, so it only works in simulation. **NIS needs only the innovations**, so it is how engineers
tune $Q$ and $R$ on real data.
"""
)

ui.section("6. When the model is nonlinear: the extended Kalman filter")
st.markdown(
    r"""
A car's motion ($p_x \mathrel{+}= v\cos\theta\,\Delta t$) or a radar's range-and-bearing measurement are
nonlinear: $\mathbf x_k = f(\mathbf x_{k-1}) + \mathbf w$, $\mathbf z = h(\mathbf x) + \mathbf v$. The
**EKF** uses the nonlinear $f, h$ for the means, and their **Jacobians** $F = \partial f/\partial\mathbf x$,
$H = \partial h/\partial\mathbf x$ (evaluated at the current estimate) in place of $F, H$ in (3)–(5). It works
when the nonlinearity is mild over the size of the uncertainty. When it isn't, see Level 5. The **Car GPS + IMU**
demo uses an EKF.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("l3", [
    ("Tab **1D: recursive Bayes**: *True process noise q*, *True sensor noise r*", r"the real $Q$ and $R$ used to simulate (1)–(2)"),
    ("*Filter's assumed q*, *Filter's assumed r*", r"the $Q, R$ the filter *believes*, used in (3)–(5). Set them wrong on purpose"),
    ("Blue line with shaded band", r"$\hat x$ ± 2√P"),
    ("Gain plot and *steady state K∞*", r"$K$ from (5)–(6) converging to (7)"),
    ("Tab **2D tracking**: *manoeuvre intensity q*, *position noise r*, *Filter q ÷ true q*", r"$q$ in (8), $R$, and a deliberate mismatch"),
    ("Ellipses", r"95% region of the position part of $P$"),
    ("*Sensor blackout*, *Show RTS smoother*", r"predict-only steps, and the backward smoother of section 4"),
    ("Tab **Is my filter honest?**", r"average NEES and NIS (9) with the 95% χ² band"),
])

ui.section("Common pitfalls")
st.markdown(
    r"""
* **Q too small is the classic failure.** The filter becomes overconfident, ignores the sensor and loses the target.
* **Tuning by eye.** A smooth-looking estimate isn't necessarily a good one. Check NIS.
* **Wrong units.** $q$ in (8) is a spectral density (units²/s³), not a variance per step, so changing $\Delta t$
  changes $Q$.
"""
)

ui.section("Check yourself")
ui.check([
    ("P⁻ = 4, r = 1, z − x̂⁻ = 2. What are K, the updated estimate shift, and P?",
     "K = 4/5 = **0.8**; the estimate moves by **1.6**; P = 0.2 · 4 = **0.8**."),
    ("q = 0.01, r = 1. What is the steady-state gain?",
     "P⁻∞ = (0.01 + √(0.0001 + 0.04))/2 ≈ 0.105; K∞ = 0.105/1.105 ≈ **0.095**."),
    ("The NIS average sits far above the χ² band. What do you change first?",
     "**Increase Q.** The filter is overconfident; real motion is more random than it assumes. (Or check R is not "
     "too small.)"),
    ("Why does the filter know velocity when only position is measured?",
     "The motion model links them (position += velocity·Δt), and P carries the position–velocity correlation, so "
     "position innovations update velocity through K."),
])
