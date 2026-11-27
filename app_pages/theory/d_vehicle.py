import streamlit as st

from ssp import ui

ui.theory_header("vehicle", """
**Prerequisites:** Level 3 (Kalman filter; section 6 on the EKF).

Every car navigation system, phone and drone fuses satellite positioning with inertial sensors. Neither sensor is
good enough alone, but their errors are **complementary**. This page explains why, writes down the model the
demo uses, and shows how an extended Kalman filter even learns a sensor error it never measures directly.
""")

ui.section("1. The two sensors and how they fail")
st.markdown(
    r"""
| | GNSS (GPS) | IMU (gyroscope + accelerometer) |
|---|---|---|
| What it measures | **absolute position** | **rates of change**: yaw rate $\omega$, acceleration $a$ |
| Rate | slow: 1–10 Hz | fast: 100 Hz and more |
| Error type | noisy (a few metres), but **doesn't drift** | smooth, but errors **accumulate** when integrated |
| Fails when | tunnels, parking garages, urban canyons | never "drops out", but **drifts** without correction |

Following the IMU alone (**dead reckoning**) means integrating: heading from yaw rate, then position from heading
and speed. A small constant **gyro bias** $b$ (a MEMS gyro's offset when nothing is rotating) makes the heading
error grow linearly, and the sideways position error grow roughly **quadratically** with time:
"""
)
ui.eq(r"\delta\theta(t) \approx b\,t, \qquad \text{lateral error}(t) \approx v\int_0^t b\,\tau\,d\tau = \tfrac12\,v\,b\,t^2", 1)
st.markdown(
    r"""
**Worked example.** $b = 0.6\,°/\text{s} = 0.0105$ rad/s, $v = 12$ m/s, a 25 s tunnel:
$\tfrac12 \times 12 \times 0.0105 \times 25^2 \approx 39$ m off. GPS alone would be ±3 m noisy but unbiased.
Fusion should beat both.
"""
)

ui.section("2. The model the EKF uses")
st.markdown(r"""**State** (five numbers): position, heading, speed and the unknown gyro bias:""")
ui.eq(r"\mathbf x = [\,p_x,\; p_y,\; \theta,\; v,\; b_\omega\,]^T", 2)
st.markdown(r"""**Motion model**, run at the IMU rate with the measured yaw rate $\omega_m$ and acceleration $a_m$ as inputs:""")
ui.eq(r"p_x' = p_x + v\cos\theta\,\Delta t, \quad p_y' = p_y + v\sin\theta\,\Delta t, \quad "
      r"\theta' = \theta + (\omega_m - b_\omega)\,\Delta t, \quad v' = v + a_m\,\Delta t, \quad b_\omega' = b_\omega", 3)
st.markdown(
    r"""
The IMU noise becomes the **process noise** $Q$: gyro noise enters the heading, accelerometer noise the speed,
and a tiny random walk lets the bias drift slowly.

**Measurement model**, at the GPS rate (none in the tunnel):
"""
)
ui.eq(r"\mathbf z = \begin{bmatrix}p_x\\ p_y\end{bmatrix} + \mathbf v, \qquad \mathbf v \sim \mathcal N(\mathbf 0,\,\sigma_{\text{GPS}}^2 I)", 3.1)
st.markdown(
    r"""
(3) is **nonlinear** in $\theta$ ($\cos\theta$, $\sin\theta$), so we use the **EKF**: the mean goes through (3)
itself, and the covariance through its Jacobian, for example $\partial p_x'/\partial\theta = -v\sin\theta\,\Delta t$
and $\partial\theta'/\partial b_\omega = -\Delta t$.
"""
)

ui.section("3. How the filter learns the bias it never measures")
st.markdown(
    r"""
GPS measures only position, yet the bias estimate converges. The chain of reasoning is all inside $P$:

1. A bias error makes the predicted heading drift, so the predicted positions drift **sideways** in a consistent
   direction.
2. The motion model creates **correlation** in $P$ between position errors and the bias.
3. Each GPS innovation (position surprise) is therefore partly explained by the bias, and $K$ corrects $b_\omega$
   accordingly.

This is **state augmentation**: add the unknown sensor error to the state and let the filter estimate it. Without
it (toggle off in the Lab), the filter believes the gyro is perfect. In the tunnel it drifts *and* stays
**confident**: its claimed 95% region stops containing the truth. That's the worst kind of failure.
"""
)

ui.section("4. What happens in the tunnel")
st.markdown(
    r"""
No GPS means no update, so the filter only **predicts**: the covariance grows every step and the 95% region
widens. A well-modelled filter grows it at the right rate, so the truth stays inside. When GPS returns, the first
fixes snap the estimate and the region back. Real systems also use wheel-speed sensors and map matching (a
particle-filter idea from Level 5) to shrink tunnel errors further.
"""
)

ui.section("How this maps to the Lab")
ui.lab_map("vehicle", [
    ("*GPS rate*, *GPS noise σ*", r"how often (3.1) is applied, and $\sigma_{\text{GPS}}$"),
    ("*Tunnel entry*, *Tunnel length*", r"the interval with no updates (section 4)"),
    ("*True gyro bias*", r"the hidden $b$ in (1). It drives the dead-reckoning drift"),
    ("*EKF estimates the gyro bias*", r"whether $b_\omega$ is in the state (2) or assumed zero"),
    ("Map: grey road, orange dots, red line, blue line + shaded region", r"truth, GPS fixes, dead reckoning, EKF mean and 95% region"),
    ("Error plot (log scale) with dotted bound", r"actual EKF error vs the filter's own 95% claim"),
    ("Bias plot", r"$\hat b_\omega$ ± 2σ converging to the true bias"),
])

ui.section("Check yourself")
ui.check([
    ("Bias 0.3 °/s, speed 20 m/s, 30 s without GPS. Roughly how far does dead reckoning drift sideways?",
     "b = 0.3·π/180 ≈ 0.00524 rad/s; ½·20·0.00524·900 ≈ **47 m**."),
    ("Why does the EKF need the IMU at all if GPS is unbiased?",
     "Between fixes and in outages the IMU propagates the state smoothly. It also averages GPS noise, because the "
     "motion model says the car can't jump 3 m sideways between fixes."),
    ("With bias estimation off, the error leaves the 95% bound in the tunnel. What does that mean?",
     "The filter is **overconfident**: its model omits a real error source, so P is too small. Downstream systems "
     "that trust P (lane-level guidance, ADAS) would be misled."),
])
