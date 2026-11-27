import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ssp import ui
from ssp.core import kalman as kf

ui.lab_intro(
    "l3",
    goal="Run a Kalman filter on simulated data where you control both the **real** world (how much it moves, "
         "how noisy the sensor is) and what the filter **believes**. See what good tuning, bad tuning, missing "
         "data and smoothing look like, and learn to check whether a filter is honest.",
    on_screen=[
        "**1D: recursive Bayes:** a drifting value, noisy readings, the estimate with its ±2σ band, and the gain "
        "K over time (Theory §2–3)",
        "**2D tracking & smoothing:** a moving target, uncertainty ellipses, a sensor blackout, the offline "
        "smoother (Theory §4)",
        "**Is my filter honest?:** NEES/NIS consistency tests over many runs (Theory §5)",
    ],
    start_here="In the 1D tab, keep the defaults and compare the orange dots (sensor) with the blue line "
               "(estimate): the filter cuts the error by about 3×. Then set *Filter's assumed q* to 0.0001 and "
               "watch the band stop covering the truth.",
)

t1, t2, t3, t4 = st.tabs(["1D: recursive Bayes", "2D tracking & smoothing", "Is my filter honest?", "Code"])

# --------------------------------------------------------------------------- 1D
with t1:
    st.markdown(
        r"""
**The situation.** A slowly drifting value (a room temperature, a battery's charge) is read by a noisy sensor
(Theory §1, the 1-D column). Every step the filter **predicts** (uncertainty grows by $q$), then **updates** with the
reading using gain $K = P^-/(P^- + r)$. The left two sliders set the *real* world; the next two set what the
filter *assumes*. When they match, the filter is optimal.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown("**The real world**")
        q = st.select_slider("True process noise q", [0.0001, 0.001, 0.01, 0.1, 1.0], 0.01,
                             help="Variance of the true value's random step each time step: how much it really drifts.")
        r = st.select_slider("True sensor noise r", [0.01, 0.1, 0.5, 1.0, 4.0], 1.0,
                             help="Variance of the sensor's error.")
        st.markdown("**What the filter assumes**")
        qf = st.select_slider("Filter's assumed q", [0.0001, 0.001, 0.01, 0.1, 1.0], 0.01,
                              help="The Q the filter uses. Set it different from the true q to see mis-tuning.")
        rf = st.select_slider("Filter's assumed r", [0.01, 0.1, 0.5, 1.0, 4.0], 1.0,
                              help="The R the filter uses.")
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
        ui.show(ui.style(fig2, "Kalman gain settles to a constant (Theory eq. 7)", "time step", "gain", height=240,
                         legend=False))
        m1, m2, m3 = st.columns(3)
        m1.metric("Sensor RMSE", f"{np.sqrt(np.mean((z - x) ** 2)):.3f}", help="Error if you just used the readings")
        m2.metric("Kalman RMSE", f"{np.sqrt(np.mean((xe - x) ** 2)):.3f}")
        m3.metric("Truth inside ±2σ", f"{np.mean(np.abs(xe - x) < 2 * sd):.0%}",
                  help="Should be about 95% if the filter's uncertainty is honest")
    ui.controls([
        ("True process noise q", r"$Q$ (real)", "how much the truth really drifts", "a wigglier grey line"),
        ("True sensor noise r", r"$R$ (real)", "sensor noise", "more scattered orange dots"),
        ("Filter's assumed q", r"$Q$ (filter)", "how much drift the filter expects", "larger K: follows readings more"),
        ("Filter's assumed r", r"$R$ (filter)", "how noisy the filter thinks the sensor is", "smaller K: smoother, slower"),
        ("Time steps", "–", "simulation length", "–"),
    ])
    ui.experiments([
        ("Keep the defaults.", "Sensor RMSE ≈ 0.90, Kalman RMSE ≈ 0.28, and ~98% of the truth inside the band.",
         "Matched Q and R: the optimal filter averages about 1/K∞ ≈ 10 readings' worth of information."),
        ("Set Filter's assumed q = 0.0001 (100× too small).",
         "The estimate becomes very smooth but lags the truth; coverage falls to about 50%.",
         "The filter thinks the value barely moves, so it ignores readings. It's **overconfident**: its band is too "
         "narrow for its real error."),
        ("Set Filter's assumed q = 1 (100× too large).", "The estimate jitters like the sensor; RMSE ≈ 0.57.",
         "K is close to 1: the filter trusts each reading, so it averages almost nothing."),
        ("Set the true q = 1 and the filter's q = 1.", "The gain plot settles much higher, at K∞ ≈ 0.6.",
         "Only q/r matters for K. When the world moves a lot, recent readings are worth more."),
    ])
    ui.takeaway("Only the ratio q/r matters for the gain. A large q/r means 'the world moves a lot, trust the sensor' "
                "(K → 1). A small q/r means 'the world is steady, average many readings' (K → 0). A constant K∞ is "
                "an exponential moving average whose α is chosen optimally.")

# --------------------------------------------------------------------------- 2D
with t2:
    st.markdown(
        r"""
**The situation.** A target moves in a plane; a sensor measures only its **position**, with noise. The filter
uses the constant-velocity model (Theory eq. 8): state = position and velocity on each axis. Ellipses show where
the filter believes the target is with 95% probability. During a **blackout** there are no measurements, so the
filter only predicts. The **RTS smoother** re-processes everything backwards after the fact.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        q2 = st.select_slider("True manoeuvre intensity q", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], 0.5,
                              help="How violently the true target changes velocity (acceleration noise density).")
        r2 = st.select_slider("Position noise variance r (m²)", [0.1, 0.5, 1.0, 2.0, 5.0], 1.0,
                              help="Variance of each position measurement.")
        qmul = st.select_slider("Filter q ÷ true q", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0], 1.0,
                                help="1 = the filter knows the true q. Smaller = it expects smoother motion than reality.")
        drop = st.toggle("Sensor blackout (steps 80–110)", True,
                         help="Remove 30 measurements in the middle, like a tunnel.")
        show_s = st.toggle("Show RTS smoother", True, help="The offline smoother that also uses future data.")
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
    ui.controls([
        ("True manoeuvre intensity q", r"$q$ in eq. 8 (real)", "how much the target's velocity wanders", "a curvier path"),
        ("Position noise variance r", r"$R$", "sensor noise", "more scatter; the filter smooths more"),
        ("Filter q ÷ true q", "–", "deliberate mis-tuning of Q", "<1: lags turns; >1: noisy estimate"),
        ("Sensor blackout", "–", "30 missing measurements", "ellipses grow during the gap"),
        ("Show RTS smoother", "–", "offline backward pass", "dashed line through the gap"),
    ])
    ui.experiments([
        ("Look at the ellipses with the blackout on.",
         "They grow steadily through the gap, then shrink right back after the first new measurement.",
         "No update means P only grows: FPFᵀ + Q at every step. One measurement removes most of it."),
        ("Compare the blue (live) and dashed (smoother) lines inside the gap.",
         "The live filter coasts in a straight line; the smoother bends to meet where the target reappeared.",
         "The smoother uses future measurements, so it's better, but only available afterwards."),
        ("Set Filter q ÷ true q = 0.01.", "The blue line lags and overshoots every curve; ellipses become tiny.",
         "The filter assumes near-straight motion and is overconfident about it."),
        ("Read the 'Velocity RMSE' metric.", "It's small even though velocity is never measured.",
         "The model ties position changes to velocity, and P carries the correlation (Theory §4)."),
    ])
    ui.takeaway("During the blackout the filter coasts on its motion model and the ellipses balloon. That is "
                "honest uncertainty. When data returns, one update snaps them back. The smoother bridges the gap "
                "because it knows where the target reappeared.")

# --------------------------------------------------------------------------- consistency
with t3:
    st.markdown(
        r"""
**The situation.** Run the 2-D tracker many times and average two statistics that test the filter's own error
claims (Theory §5). **NEES** compares the real error with $P$; its expected value is 4 (the state dimension).
**NIS** compares each innovation with $S$; its expected value is 2. If the filter is honest, the blue line stays
inside the green 95% band. Both plots use a log scale.
"""
    )
    c1, c2 = st.columns([1, 2.2])
    with c1:
        qm = st.select_slider("Filter q ÷ true q", [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0], 1.0, key="nq",
                              help="Mis-tune the filter's process noise.")
        rm = st.select_slider("Filter r ÷ true r", [0.1, 0.3, 1.0, 3.0, 10.0], 1.0, key="nr",
                              help="Mis-tune the filter's measurement noise.")
        runs = st.select_slider("Monte Carlo runs", [10, 25, 50, 100], 50,
                                help="Independent runs averaged. More runs give a narrower band.")

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
    ui.controls([
        ("Filter q ÷ true q", "–", "process-noise mismatch", "<1 overconfident (above band); >1 pessimistic"),
        ("Filter r ÷ true r", "–", "measurement-noise mismatch", "<1 overconfident; >1 pessimistic (below band)"),
        ("Monte Carlo runs", "–", "runs averaged", "a narrower band, a stricter test"),
    ])
    ui.experiments([
        ("Defaults (both ratios 1).", "Average NEES sits near 4, inside the band about 95% of the time.",
         "A correctly modelled filter's errors match its claimed covariance."),
        ("Set Filter q ÷ true q = 0.01.", "NEES jumps to around 200 (log scale!), far above the band.",
         "Real errors are ~50× bigger than P claims. That's how filters diverge in practice."),
        ("Set Filter q ÷ true q = 100.", "NEES drops below the band (≈ 2).",
         "P claims far more uncertainty than there is. Safe, but it throws information away."),
        ("Reset q, set Filter r ÷ true r = 0.1, and look at NIS.", "NIS rises above its band too.",
         "NIS needs no ground truth, so this is the test engineers can run on real recorded data."),
    ])

with t4:
    st.markdown("The exact functions this page runs (from `ssp/core/kalman.py`):")
    ui.source(kf.KalmanFilter, kf.run_kalman, kf.rts_smoother, kf.constant_velocity, kf.nees, kf.chi2_band)
