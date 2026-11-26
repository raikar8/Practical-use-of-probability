import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core.kalman import cov_ellipse
from ssp.sims import vehicle as veh

ui.header("Domain demo · Automotive", "Car GPS + IMU fusion through a tunnel",
          "GPS is noisy and slow, the IMU is fast but drifts. Can an EKF beat both, and survive losing GPS?")

with st.expander("The model (open for the math)", expanded=False):
    st.markdown(
        r"""
**State** $\mathbf x=[p_x,\,p_y,\,\theta,\,v,\,b_\omega]$: position, heading, speed and the **gyro bias**, which
is never measured directly.

**Predict at 100 Hz from the IMU** (yaw rate $\omega_m$, acceleration $a_m$), which is nonlinear in $\theta$:

$$p_x' = p_x + v\cos\theta\,\Delta t,\quad p_y' = p_y + v\sin\theta\,\Delta t,\quad
\theta' = \theta + (\omega_m - b_\omega)\Delta t,\quad v' = v + a_m\Delta t,\quad b_\omega'=b_\omega$$

The EKF linearises this with the Jacobian $F=\partial f/\partial\mathbf x$ at the current estimate
(e.g. $\partial p_x'/\partial\theta = -v\sin\theta\,\Delta t$).

**Update at the GPS rate:** $\mathbf z = [p_x, p_y] + \mathbf v$. In the tunnel there are no updates, so the
filter runs on prediction alone and its uncertainty grows.

**Why the bias becomes observable:** a biased heading makes predicted positions drift *sideways* from GPS in
a consistent way. Through the cross-covariance in $P$, the filter attributes that drift to $b_\omega$.
"""
    )

c1, c2 = st.columns([1, 2.4])
with c1:
    gps_hz = st.select_slider("GPS rate (Hz)", [0.5, 1.0, 2.0, 5.0, 10.0], 1.0)
    gps_std = st.slider("GPS noise σ (m)", 0.5, 10.0, 3.0, 0.5)
    t_start = st.slider("Tunnel entry (s)", 10, 100, 50, 5)
    t_len = st.slider("Tunnel length (s)", 0, 60, 25, 5)
    bias_dps = st.slider("True gyro bias (deg/s)", 0.0, 3.0, 0.6, 0.1)
    est_bias = st.toggle("EKF estimates the gyro bias", True)
    seed = st.number_input("Random seed", 0, 9999, 7)

cfg = veh.VehicleConfig(gps_hz=gps_hz, gps_std=gps_std, tunnel_start=float(t_start), tunnel_len=float(t_len),
                        gyro_bias=np.deg2rad(bias_dps), estimate_bias=est_bias, seed=int(seed))


@st.cache_data(show_spinner="Driving the car…")
def _sim(**kw):
    return veh.simulate_vehicle(veh.VehicleConfig(**kw))


r = _sim(**cfg.__dict__)
t, X, E, P, DR = r["t"], r["truth"], r["ekf"], r["P"], r["dr"]
tin = (t >= r["tunnel"][0]) & (t < r["tunnel"][1])

# ---------------------------------------------------------------- animated map
pad = 40
xr = [X[:, 0].min() - pad, X[:, 0].max() + pad]
yr = [X[:, 1].min() - pad, X[:, 1].max() + pad]
step = 10                     # stored at 10 Hz -> one frame per second of driving
idx = list(range(0, len(t), step))

fig = go.Figure()
fig.add_scatter(x=X[:, 0], y=X[:, 1], mode="lines", name="true road", line=dict(color=ui.C["truth"], width=2))
fig.add_scatter(x=np.where(tin, X[:, 0], np.nan), y=np.where(tin, X[:, 1], np.nan), mode="lines",
                name="tunnel (no GPS)", line=dict(color=ui.rgba(ui.C["truth"], 0.35), width=14))
fig.add_scatter(x=r["gps"][:, 0], y=r["gps"][:, 1], mode="markers", name="GPS fixes",
                marker=dict(size=5, color=ui.rgba(ui.C[2], 0.55)))
# dynamic traces (indices 3..7)
fig.add_scatter(x=DR[:1, 0], y=DR[:1, 1], mode="lines", name="IMU dead reckoning", line=dict(color=ui.C[8]))
fig.add_scatter(x=E[:1, 0], y=E[:1, 1], mode="lines", name="EKF estimate", line=dict(color=ui.C[1], width=3))
ex, ey = cov_ellipse(E[0, :2], P[0][:2, :2], 2.45)
fig.add_scatter(x=ex, y=ey, mode="lines", name="EKF 95% region", line=dict(color=ui.C[1], width=1),
                fill="toself", fillcolor=ui.rgba(ui.C[1], 0.18))
fig.add_scatter(x=X[:1, 0], y=X[:1, 1], mode="markers", name="true car", marker=dict(size=12, color=ui.C["truth"],
                symbol="circle-open", line=dict(width=3)))
fig.add_scatter(x=E[:1, 0], y=E[:1, 1], mode="markers", showlegend=False, marker=dict(size=9, color=ui.C[1]))
frames = []
for i in idx:
    ex, ey = cov_ellipse(E[i, :2], P[i][:2, :2], 2.45)
    frames.append(go.Frame(name=f"{t[i]:.0f}s", traces=[3, 4, 5, 6, 7], data=[
        go.Scatter(x=DR[: i + 1, 0], y=DR[: i + 1, 1]),
        go.Scatter(x=E[: i + 1, 0], y=E[: i + 1, 1]),
        go.Scatter(x=ex, y=ey),
        go.Scatter(x=X[i:i + 1, 0], y=X[i:i + 1, 1]),
        go.Scatter(x=E[i:i + 1, 0], y=E[i:i + 1, 1]),
    ]))
fig.frames = frames
fig.update_xaxes(range=xr)
fig.update_yaxes(range=yr, scaleanchor="x", scaleratio=1)
fig.update_layout(hovermode="closest")
ui.play_controls(fig, [f.name for f in frames], 60)
with c2:
    ui.show(ui.style(fig, "Press ▶ to drive (1 frame = 1 s). Watch the 95% region grow inside the tunnel.",
                     "east (m)", "north (m)", height=560), key="veh_map")

# ---------------------------------------------------------------- metrics
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
m = st.columns(5)
m[0].metric("Raw GPS RMSE", f"{rmse(r['gps_err']):.2f} m")
m[1].metric("EKF RMSE", f"{rmse(r['err_ekf']):.2f} m")
m[2].metric("Dead reckoning RMSE", f"{rmse(r['err_dr']):.0f} m")
m[3].metric("Worst EKF error in tunnel", f"{r['err_ekf'][tin].max():.1f} m" if tin.any() else "no tunnel")
m[4].metric("Truth inside 95% region", f"{np.mean(r['err_ekf'] < r['bound95']):.0%}",
            help="Should be about 95% or more. Much lower means the filter is overconfident.")

c3, c4 = st.columns(2)
with c3:
    f2 = go.Figure()
    if tin.any():
        f2.add_vrect(x0=r["tunnel"][0], x1=r["tunnel"][1], fillcolor=ui.rgba(ui.C["truth"], 0.15), line_width=0,
                     annotation_text="tunnel", annotation_position="top left")
    f2.add_scatter(x=t, y=r["bound95"], name="EKF claimed 95% bound", line=dict(color=ui.C[1], dash="dot"))
    f2.add_scatter(x=t, y=r["err_ekf"], name="EKF error", line=dict(color=ui.C[1]))
    f2.add_scatter(x=r["gps_t"], y=r["gps_err"], mode="markers", name="GPS error",
                   marker=dict(size=4, color=ui.C[2]))
    f2.add_scatter(x=t, y=r["err_dr"], name="dead-reckoning error", line=dict(color=ui.C[8]))
    f2.update_yaxes(type="log", range=[-1, np.log10(max(r["err_dr"].max(), 10)) + 0.2])
    ui.show(ui.style(f2, "Position error over time (log scale)", "time (s)", "error (m)", height=360))
with c4:
    f3 = go.Figure()
    b_sd = np.sqrt(np.clip(P[:, 4, 4], 0, None))
    f3.add_scatter(x=np.r_[t, t[::-1]], y=np.rad2deg(np.r_[E[:, 4] + 2 * b_sd, (E[:, 4] - 2 * b_sd)[::-1]]),
                   fill="toself", fillcolor=ui.rgba(ui.C[1], 0.15), line=dict(width=0), name="±2σ")
    f3.add_scatter(x=t, y=np.rad2deg(E[:, 4]), name="EKF bias estimate", line=dict(color=ui.C[1]))
    f3.update_yaxes(range=[-1.5, max(2.5, 2 * bias_dps)])
    f3.add_hline(y=bias_dps, line=dict(color=ui.C["truth"], dash="dash"), annotation_text="true bias")
    ui.show(ui.style(f3, "Gyro bias: learned without ever being measured", "time (s)", "deg/s", height=360))

ui.takeaway("Raw GPS is unbiased but jumpy, and dead reckoning is smooth but drifts off the map within a minute. "
            "The EKF uses the IMU between fixes and GPS to stop the drift, so it beats both. In the tunnel the "
            "learned bias keeps dead reckoning accurate, and the growing 95% region tells you how far to trust it.")
ui.break_it("Turn off bias estimation. In the tunnel the error explodes past the claimed bound: an overconfident "
            "filter that's wrong about how wrong it is. In a real car this is what makes navigation 'jump' when "
            "GPS returns. Then try GPS at 0.5 Hz with σ = 10 m: the IMU is doing most of the work.")
ui.in_practice("Every phone and car navigation system fuses GNSS with inertial (and often wheel-speed) sensors "
               "using Kalman-type filters. The same structure runs drone autopilots and AR/VR headset tracking.")

with st.expander("Code: the EKF model functions"):
    ui.source(veh._f, veh._F, veh.simulate_vehicle)
