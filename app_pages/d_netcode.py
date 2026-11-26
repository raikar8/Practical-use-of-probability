import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ssp import ui
from ssp.core.kalman import cov_ellipse
from ssp.sims import netcode as nc

ui.header("Domain demo · Real-time games", "Where is the other player *right now*?",
          "Network snapshots arrive late, jittery and sometimes not at all. What should the client draw this frame?")

st.markdown(
    r"""
The server sends the remote player's position $\mathbf z_k$ (time-stamped $t_k$) at a fixed rate. Each packet
takes $\tfrac{\text{ping}}{2}+\text{jitter}$ to arrive, or is lost. At every 60 Hz frame the client must draw an
estimate of the **present** position. Four strategies compete:

| Strategy | What it draws | Weakness |
|---|---|---|
| **Snap** | newest snapshot as-is | always behind by the network delay |
| **Extrapolate** | newest snapshot + velocity from the last two × elapsed time | amplifies snapshot noise, overshoots on turns |
| **Kalman** | constant-velocity KF, updated per packet, then *predicted* to now: $\hat{\mathbf x}(t)=F(t-t_k)\hat{\mathbf x}_k$ | needs a manoeuvre-noise $q$; still can't foresee a juke |
| **Interpolate** | blend two snapshots, rendered $\tfrac{\text{ping}}{2}$ + buffer in the past | smooth and correct about the past, but late |

The Kalman filter also reports an **uncertainty radius** that grows while packets are missing. A game can
use it to decide when to stop extrapolating, or a server can use it to judge whether a hit was plausible.
"""
)

c1, c2 = st.columns([1, 2.4])
with c1:
    ping = st.slider("Ping (round trip, ms)", 0, 400, 120, 10)
    jitter = st.slider("Jitter σ (ms)", 0, 80, 20, 5)
    loss = st.slider("Packet loss (%)", 0, 40, 5) / 100
    send_hz = st.select_slider("Snapshot rate (Hz)", [10, 15, 20, 30, 60], 20)
    quant = st.slider("Snapshot position noise σ (m)", 0.0, 0.5, 0.10, 0.01,
                      help="Quantisation / compression of positions in the packet")
    agility = st.slider("Player agility (direction changes per s)", 0.1, 4.0, 1.0, 0.1)
    interp = st.slider("Interpolation buffer (ms)", 0, 200, 100, 10)
    q = st.select_slider("Kalman manoeuvre noise q", [1, 5, 20, 60, 150, 400, 1000], 60)
    seed = st.number_input("Random seed", 0, 9999, 3)

cfg = nc.NetConfig(ping_ms=float(ping), jitter_ms=float(jitter), loss=float(loss), send_hz=int(send_hz),
                   quant_std=float(quant), agility=float(agility), interp_delay_ms=float(interp), kf_q=float(q),
                   seed=int(seed))


@st.cache_data(show_spinner="Simulating the match…")
def _sim(**kw):
    return nc.simulate_netcode(nc.NetConfig(**kw))


@st.cache_data(show_spinner="Sweeping ping…")
def _sweep(**kw):
    return nc.sweep_ping(nc.NetConfig(**kw), list(range(0, 401, 50)))


r = _sim(**cfg.__dict__)
t, truth, est, err = r["t"], r["truth"], r["est"], r["err"]
methods = {"snap": ("Snap", ui.C[8]), "extrapolate": ("Extrapolate", ui.C[2]),
           "kalman": ("Kalman", ui.C[1]), "interpolate": ("Interpolate", ui.C[7])}

# ---------------------------------------------------------------- animated arena (10 s window)
with c2:
    w0 = st.slider("Animation window start (s)", 0, int(cfg.duration) - 10, 5)
    zoom = st.select_slider("Zoom panel half-width (m)", [0.5, 1.0, 1.5, 2.0, 3.0], 1.5)
    i0, i1 = w0 * cfg.frame_hz, (w0 + 10) * cfg.frame_hz
    every = 2                                     # 30 fps animation
    trail = cfg.frame_hz // 2                     # 0.5 s of true trail
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.08)
    xmin, xmax, ymin, ymax = nc.ARENA
    fig.add_shape(type="rect", x0=xmin, y0=ymin, x1=xmax, y1=ymax, line=dict(color=ui.C["truth"]), row=1, col=1)
    # dynamic traces 0..6
    fig.add_scatter(x=truth[i0:i0 + 1, 0], y=truth[i0:i0 + 1, 1], mode="lines", name="true path (last 0.5 s)",
                    line=dict(color=ui.rgba(ui.C["truth"], 0.6), width=3), row=1, col=1)
    fig.add_scatter(x=truth[i0:i0 + 1, 0], y=truth[i0:i0 + 1, 1], mode="markers", name="true player",
                    marker=dict(size=14, color=ui.C["truth"], symbol="circle-open", line=dict(width=3)),
                    row=1, col=1)
    for m, (nm, col) in methods.items():
        d = est[m][i0] - truth[i0]
        fig.add_scatter(x=[d[0]], y=[d[1]], mode="markers", name=nm,
                        marker=dict(size=13, color=col, line=dict(width=2, color="white")), row=1, col=2)
    fig.add_scatter(x=[np.nan], y=[np.nan], mode="lines", name="Kalman 95% region", fill="toself",
                    fillcolor=ui.rgba(ui.C[1], 0.15), line=dict(color=ui.C[1], width=1), row=1, col=2)
    # static: the true player's body at the centre of the zoom panel
    th = np.linspace(0, 2 * np.pi, 40)
    fig.add_scatter(x=0.3 * np.cos(th), y=0.3 * np.sin(th), mode="lines", name="player body (0.3 m)",
                    line=dict(color=ui.C["truth"], width=2), fill="toself",
                    fillcolor=ui.rgba(ui.C["truth"], 0.15), row=1, col=2)
    frames = []
    for i in range(i0, i1, every):
        cov = r["kf_cov"][i]
        if np.isfinite(cov).all():
            ex, ey = cov_ellipse(est["kalman"][i] - truth[i], cov, 2.45, 32)
        else:
            ex, ey = [np.nan], [np.nan]
        data = [go.Scatter(x=truth[max(i - trail, 0):i + 1, 0], y=truth[max(i - trail, 0):i + 1, 1]),
                go.Scatter(x=truth[i:i + 1, 0], y=truth[i:i + 1, 1])]
        for m in methods:
            d = est[m][i] - truth[i]
            data.append(go.Scatter(x=[d[0]], y=[d[1]]))
        data.append(go.Scatter(x=ex, y=ey))
        frames.append(go.Frame(name=f"{t[i]:.2f}s", traces=list(range(7)), data=data))
    fig.frames = frames
    fig.update_xaxes(range=[xmin - 1, xmax + 1], showgrid=False, row=1, col=1, title="arena: east (m)")
    fig.update_yaxes(range=[ymin - 1, ymax + 1], scaleanchor="x", scaleratio=1, showgrid=False, row=1, col=1)
    fig.update_xaxes(range=[-zoom, zoom], row=1, col=2, title="zoom: error east (m)")
    fig.update_yaxes(range=[-zoom, zoom], scaleanchor="x2", scaleratio=1, row=1, col=2, title="error north (m)")
    fig.update_layout(hovermode="closest")
    ui.play_controls(fig, [f.name for f in frames], int(1000 * every / cfg.frame_hz))
    ui.show(ui.style(fig, "Press ▶: 10 seconds of the match in real time", None, None, height=470), key="arena")
    st.caption("Right panel: the true player is always at the centre, and each dot shows where a method draws "
               "them. Dots outside the grey body circle are visibly wrong on screen.")

# ---------------------------------------------------------------- results
summary = nc.summarize(err)
best = min(summary, key=lambda m: summary[m]["mean"])
df = pd.DataFrame({methods[m][0]: {"mean error (m)": v["mean"], "95th percentile (m)": v["p95"]}
                   for m, v in summary.items()}).T
cA, cB = st.columns([1, 1.6])
with cA:
    st.markdown(f"**Best this run: {methods[best][0]}**")
    st.dataframe(df.style.format("{:.2f}"), width="stretch")
    lost = r["packets"]["lost"]
    st.caption(f"{len(lost)} snapshots sent, {int(lost.sum())} lost. One-way delay ≈ {ping / 2:.0f} ms + jitter.")
with cB:
    f2 = go.Figure()
    smooth = lambda v: pd.Series(v).rolling(cfg.frame_hz // 2, min_periods=1, center=True).mean().to_numpy()
    for m, (nm, col) in methods.items():
        f2.add_scatter(x=t, y=smooth(err[m]), name=nm, line=dict(color=col))
    f2.add_scatter(x=t, y=smooth(r["kf_radius"]), name="Kalman claimed 95% radius",
                   line=dict(color=ui.C[1], dash="dot"))
    f2.add_vrect(x0=w0, x1=w0 + 10, fillcolor=ui.rgba(ui.C["truth"], 0.08), line_width=0,
                 annotation_text="animated window", annotation_position="top left")
    ui.show(ui.style(f2, "Distance from the true present position (0.5 s rolling mean)", "match time (s)", "error (m)",
                     height=340))

sw = _sweep(**cfg.__dict__)
f3 = go.Figure()
for m, (nm, col) in methods.items():
    f3.add_scatter(x=sw["ping"], y=sw[m], name=nm, mode="lines+markers", line=dict(color=col))
f3.add_vline(x=ping, line=dict(color=ui.C["truth"], dash="dash"), annotation_text="current ping")
ui.show(ui.style(f3, "Mean error as ping grows (same player, everything else as set)", "ping (ms)",
                 "mean error (m)", height=320))

ui.takeaway("Prediction is the only way to be close to the present. The Kalman filter wins because it does two "
            "things the heuristics don't: it averages snapshot noise into a clean velocity estimate, and it weighs "
            "each new packet against its prediction. Its error still grows with ping, because nobody can predict a "
            "juke that hasn't been sent yet.")
ui.break_it("(1) Raise snapshot noise to 0.3 m: naive extrapolation goes wild, Kalman stays calm. (2) Set q = 1: "
            "the filter trusts its constant-velocity model too much, overshoots every turn, and its claimed radius "
            "stops covering the truth. (3) Set q = 1000: it barely filters at all. (4) Agility 4 /s: every method "
            "suffers, and an IMM filter (Level 5) mixing 'running' and 'turning' models is the next step.")
ui.in_practice("Many engines render remote players slightly in the past with interpolation (Valve's Source "
               "engine is the well-known example) and switch to extrapolation when packets stop. Kalman-style "
               "predict-and-correct is the principled version of that trade-off. The same maths runs radar and "
               "air-traffic tracking.")

with st.expander("Code: the client-side estimators"):
    ui.source(nc.run_client)
