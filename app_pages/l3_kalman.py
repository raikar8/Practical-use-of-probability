import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import kalman as kf

ui.header("Level 3", "State-space models and the Kalman filter",
          "How do we combine a motion model with noisy sensors, in real time, and know how wrong we might be?")

t1, t2, t3, t4 = st.tabs(["1D: recursive Bayes", "2D tracking & smoothing", "Is my filter honest?", "Code"])

# --------------------------------------------------------------------------- 1D
with t1:
    st.markdown(
        r"""
**Model:** a slowly drifting quantity (a battery's state of charge, a room temperature, a sensor offset)
$x_k = x_{k-1} + w_k,\; w\sim\mathcal N(0,q)$, read by a noisy sensor $z_k = x_k + v_k,\; v\sim\mathcal N(0,r)$.

| Step | Equations | What happens |
|---|---|---|
| **Predict** | $\hat x^- = \hat x,\quad P^- = P + q$ | the world may have moved, so uncertainty **grows** |
| **Gain** | $K = P^-/(P^- + r)$ | how much to trust the sensor vs the prediction |
| **Update** | $\hat x = \hat x^- + K(z - \hat x^-),\quad P = (1-K)P^-$ | correct by the *innovation*; uncertainty **shrinks** |

This is the Gaussian conditioning from Level 0, applied once per time step.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        q = st.select_slider("True process noise q", [0.0001, 0.001, 0.01, 0.1, 1.0], 0.01)
        r = st.select_slider("True sensor noise r", [0.01, 0.1, 0.5, 1.0, 4.0], 1.0)
        qf = st.select_slider("Filter's assumed q", [0.0001, 0.001, 0.01, 0.1, 1.0], 0.01)
        rf = st.select_slider("Filter's assumed r", [0.01, 0.1, 0.5, 1.0, 4.0], 1.0)
        n = st.slider("Time steps", 50, 500, 200, key="k1n")
        seed = st.number_input("Random seed", 0, 9999, 1, key="k1s")
    x, z = kf.simulate_random_walk(n, q, r, 20.0, np.random.default_rng(int(seed)))
    f = kf.KalmanFilter(F=[[1.0]], H=[[1.0]], Q=[[qf]], R=[[rf]], x=[z[0]], P=[[rf]])
    out = kf.run_kalman(f, list(z))
    xe, sd = out["x"][:, 0], np.sqrt(out["P"][:, 0, 0])
    gains = np.array([g[0, 0] for g in out["K"]])
    Pm = (qf + np.sqrt(qf**2 + 4 * qf * rf)) / 2
    K_inf = Pm / (Pm + rf)
    with c2:
        k = np.arange(n)
        fig = go.Figure()
        fig.add_scatter(x=np.r_[k, k[::-1]], y=np.r_[xe + 2 * sd, (xe - 2 * sd)[::-1]], fill="toself",
                        fillcolor=ui.rgba(ui.C[1], 0.15), line=dict(width=0), name="±2σ belief", hoverinfo="skip")
        fig.add_scatter(x=k, y=z, mode="markers", name="measurements", marker=dict(size=4, color=ui.C[2]))
        fig.add_scatter(x=k, y=x, name="truth", line=dict(color=ui.C["truth"]))
        fig.add_scatter(x=k, y=xe, name="Kalman estimate", line=dict(color=ui.C[1]))
        ui.show(ui.style(fig, "Estimate with its own uncertainty band", "time step", "value"))
        fig2 = go.Figure()
        fig2.add_scatter(x=k, y=gains, name="K", line=dict(color=ui.C[1]))
        fig2.add_hline(y=K_inf, line=dict(color=ui.C["truth"], dash="dot"),
                       annotation_text=f"steady state K∞ = {K_inf:.3f}")
        ui.show(ui.style(fig2, "Kalman gain settles to a constant", "time step", "gain", height=240,
                         legend=False))
        m1, m2, m3 = st.columns(3)
        m1.metric("Sensor RMSE", f"{np.sqrt(np.mean((z - x) ** 2)):.3f}")
        m2.metric("Kalman RMSE", f"{np.sqrt(np.mean((xe - x) ** 2)):.3f}")
        m3.metric("Truth inside ±2σ", f"{np.mean(np.abs(xe - x) < 2 * sd):.0%}", help="≈95% if the filter is honest")
    ui.takeaway("Only the ratio q/r matters for the gain. A large q/r means 'the world moves a lot, trust the sensor' "
                "(K → 1). A small q/r means 'the world is steady, average many readings' (K → 0). A constant K∞ is "
                "an exponential moving average whose α is chosen optimally.")
    ui.break_it("Set the filter's q 100× too small: it becomes smooth but lags the truth, and the ±2σ band stops "
                "covering it (overconfident). Set it 100× too large: it just follows the noisy sensor.")

# --------------------------------------------------------------------------- 2D
with t2:
    st.markdown(
        r"""
**Model:** a target moving in a plane with a *constant-velocity* model, state $[p_x, v_x, p_y, v_y]$:

$$F=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}\otimes I_2,\qquad
Q=q\begin{bmatrix}\Delta t^3/3&\Delta t^2/2\\ \Delta t^2/2&\Delta t\end{bmatrix}\otimes I_2,\qquad
H \text{ picks out the positions.}$$

Velocity is **never measured**, but the filter infers it from how positions change. The **RTS smoother** runs
backwards after the fact and uses future measurements too. It's useful for offline analysis, not live
prediction.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        q2 = st.select_slider("True manoeuvre intensity q", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], 0.5)
        r2 = st.select_slider("Position noise variance r (m²)", [0.1, 0.5, 1.0, 2.0, 5.0], 1.0)
        qmul = st.select_slider("Filter q ÷ true q", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0], 1.0)
        drop = st.toggle("Sensor blackout (steps 80–110)", True)
        show_s = st.toggle("Show RTS smoother", True)
        seed2 = st.number_input("Random seed", 0, 9999, 3, key="k2s")
    dt, n2 = 0.1, 200
    X, Z = kf.simulate_cv_2d(n2, dt, q2, r2, np.random.default_rng(int(seed2)), x0=(0, 2, 0, 1))
    zs = [None if (drop and 80 <= i < 110) else Z[i] for i in range(n2)]
    F, Qf = kf.constant_velocity(dt, q2 * qmul, dims=2)
    H = kf.position_observation(2)
    filt = kf.KalmanFilter(F, H, Qf, np.eye(2) * r2, np.array([Z[0, 0], 0, Z[0, 1], 0]), np.diag([r2, 10, r2, 10]))
    o = kf.run_kalman(filt, zs)
    xs_s, Ps_s = kf.rts_smoother(o["x"], o["P"], o["x_pred"], o["P_pred"], F)
    pos_idx = [0, 2]
    with c2:
        fig = go.Figure()
        meas = np.array([z for z in zs if z is not None])
        fig.add_scatter(x=meas[:, 0], y=meas[:, 1], mode="markers", name="measurements",
                        marker=dict(size=4, color=ui.rgba(ui.C[2], 0.6)))
        fig.add_scatter(x=X[:, 0], y=X[:, 2], name="truth", mode="lines", line=dict(color=ui.C["truth"]))
        fig.add_scatter(x=o["x"][:, 0], y=o["x"][:, 2], name="Kalman (live)", mode="lines",
                        line=dict(color=ui.C[1]))
        if show_s:
            fig.add_scatter(x=xs_s[:, 0], y=xs_s[:, 2], name="RTS smoother (offline)", mode="lines",
                            line=dict(color=ui.C[3], dash="dash"))
        for i in range(0, n2, 10):
            ex, ey = kf.cov_ellipse(o["x"][i, pos_idx], o["P"][i][np.ix_(pos_idx, pos_idx)], 2.45)
            fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=ui.rgba(ui.C[1], 0.45), width=1),
                            showlegend=(i == 0), name="95% ellipse (every 10 steps)", hoverinfo="skip")
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        ui.show(ui.style(fig, "Path, estimates and uncertainty ellipses", "x (m)", "y (m)", height=460))
        err_f = np.linalg.norm(o["x"][:, pos_idx] - X[:, [0, 2]], axis=1)
        err_s = np.linalg.norm(xs_s[:, pos_idx] - X[:, [0, 2]], axis=1)
        err_z = np.linalg.norm(Z - X[:, [0, 2]], axis=1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw sensor RMSE", f"{np.sqrt(np.mean(err_z ** 2)):.2f} m")
        m2.metric("Kalman RMSE", f"{np.sqrt(np.mean(err_f ** 2)):.2f} m")
        m3.metric("Smoother RMSE", f"{np.sqrt(np.mean(err_s ** 2)):.2f} m")
        m4.metric("Velocity RMSE (never measured)",
                  f"{np.sqrt(np.mean(np.sum((o['x'][:, [1, 3]] - X[:, [1, 3]]) ** 2, axis=1))):.2f} m/s")
    ui.takeaway("During the blackout the filter coasts on its motion model and the ellipses balloon. That is "
                "honest uncertainty. When data returns, one update snaps them back. The smoother bridges the gap "
                "because it knows where the target reappeared.")

# --------------------------------------------------------------------------- consistency
with t3:
    st.markdown(
        r"""
A filter reports $P$, its own claim about its error. **NEES** $=\mathbf e^T P^{-1}\mathbf e$ (needs the truth,
so simulation only) and **NIS** $=\boldsymbol\nu^T S^{-1}\boldsymbol\nu$ (innovations only, so usable on
**real data**) should follow $\chi^2$ distributions with mean = dimension. Averaged over Monte Carlo runs, they
must stay inside the $\chi^2$ band. This is how engineers tune Q and R on real systems.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        qm = st.select_slider("Filter q ÷ true q", [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0], 1.0, key="nq")
        rm = st.select_slider("Filter r ÷ true r", [0.1, 0.3, 1.0, 3.0, 10.0], 1.0, key="nr")
        runs = st.select_slider("Monte Carlo runs", [10, 25, 50, 100], 50)

    @st.cache_data(show_spinner="Running Monte Carlo…")
    def _consistency(qm, rm, runs):
        n, dt, q, r = 150, 0.1, 0.5, 0.5
        F, Q = kf.constant_velocity(dt, q, dims=2)
        _, Qf = kf.constant_velocity(dt, q * qm, dims=2)
        H = kf.position_observation(2)
        NE, NI = np.zeros(n), np.zeros(n)
        for i in range(runs):
            X, Z = kf.simulate_cv_2d(n, dt, q, r, np.random.default_rng(500 + i), x0=(0, 0, 0, 0))
            f = kf.KalmanFilter(F, H, Qf, np.eye(2) * r * rm, np.zeros(4), np.eye(4) * 1e-6)
            o = kf.run_kalman(f, list(Z))
            NE += kf.nees(X, o["x"], o["P"])
            NI += o["nis"]
        return NE / runs, NI / runs

    ne, ni = _consistency(qm, rm, runs)
    lo4, hi4 = kf.chi2_band(4, runs)
    lo2, hi2 = kf.chi2_band(2, runs)
    with c2:
        k = np.arange(len(ne))
        for vals, lo, hi, dof, title in ((ne, lo4, hi4, 4, "Average NEES (state error vs claimed P)"),
                                        (ni, lo2, hi2, 2, "Average NIS (innovation vs claimed S)")):
            fig = go.Figure()
            fig.add_scatter(x=np.r_[k, k[::-1]], y=np.r_[np.full(len(k), hi), np.full(len(k), lo)], fill="toself",
                            fillcolor=ui.rgba(ui.C[3], 0.15), line=dict(width=0), name="95% χ² band")
            fig.add_scatter(x=k, y=vals, name="average", line=dict(color=ui.C[1]))
            fig.add_hline(y=dof, line=dict(color=ui.C["truth"], dash="dot"))
            fig.update_yaxes(type="log")
            ui.show(ui.style(fig, title, "time step", "value (log)", height=270))
        inside = np.mean((ne > lo4) & (ne < hi4))
        verdict = ("consistent: the filter is honest about its uncertainty" if inside > 0.85 else
                   "overconfident: real errors are bigger than P claims" if np.median(ne) > hi4 else
                   "conservative: P is too pessimistic, so the filter wastes information")
        st.metric("NEES inside band", f"{inside:.0%}", help=verdict)
        st.markdown(f"**Verdict:** {verdict}.")
    ui.break_it("Shrink filter q to 0.01×: NEES shoots above the band (overconfident, the classic cause of "
                "filter divergence). Inflate r to 10×: NEES drops below (sluggish). Only the matched model lives "
                "in the band.")

with t4:
    ui.source(kf.KalmanFilter, kf.run_kalman, kf.rts_smoother, kf.constant_velocity, kf.nees, kf.chi2_band)
