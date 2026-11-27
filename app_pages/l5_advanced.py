import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import changepoint as cp
from ssp.core import imm, kalman
from ssp.core import particle as pf

ui.lab_intro(
    "l5",
    goal="See where a single Gaussian breaks, and what replaces it: a particle cloud that can hold two answers at "
         "once, a bank of Kalman filters that switches between motion models, and a detector for sudden changes.",
    on_screen=[
        "**Particle filter vs EKF:** the exact posterior as a heat map, with the particle filter and the EKF on top "
        "(Theory A)",
        "**Where am I on this road?:** an animated particle cloud localising a car from altitude alone (Theory A)",
        "**IMM:** two Kalman filters and their probability-weighted mix tracking a turning target (Theory B)",
        "**Change-point detection:** the run-length posterior finding jumps in a sensor signal (Theory C)",
    ],
    start_here="In the first tab, find a time where the grey heat map shows **two** dark bands. The red EKF line "
               "follows one of them; check whether it's the one with the yellow truth dots.",
)

t_pf, t_road, t_imm, t_cp, t_code = st.tabs(
    ["Particle filter vs EKF", "Where am I on this road?", "IMM: manoeuvring target", "Change-point detection",
     "Code"])

# --------------------------------------------------------------------------- PF benchmark
with t_pf:
    st.markdown(
        r"""
**The situation.** The classic nonlinear test model (Theory A): the sensor measures $x^2/20$, so it can't tell
$+x$ from $-x$ and the true belief often has **two peaks**. The grey heat map is the *exact* posterior from a
fine grid (darker = more probable at that time step). The yellow dots are the truth, red is the EKF (one Gaussian),
and blue is the particle filter's mean.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        N = st.select_slider("Particles", [50, 100, 250, 500, 1000, 2500, 5000], 1000,
                             help="Number of samples representing the belief. More = closer to the exact posterior.")
        scheme = st.selectbox("Resampling", ["systematic", "stratified", "multinomial", "none"],
                              help="How to clone heavy particles and drop light ones when weights become uneven "
                                   "(Theory A). 'none' shows what goes wrong without it.")
        ess_frac = st.slider("Resample when ESS below (fraction of N)", 0.1, 1.0, 0.5, 0.05,
                             disabled=scheme == "none")
        seed = st.number_input("Random seed", 0, 9999, 0, key="pf_seed", help="A different simulated run.")

    @st.cache_data(show_spinner="Filtering…")
    def _bench(N, scheme, ess_frac, seed):
        x, z = pf.ungm_simulate(100, np.random.default_rng(seed))
        res = pf.bootstrap_pf(list(z), N, lambda n, g: 2 * g.standard_normal(n),
                              lambda p, t, g: pf.ungm_f(p, t) + np.sqrt(10) * g.standard_normal(p.size),
                              lambda zz, p, t: -0.5 * (zz - pf.ungm_h(p)) ** 2,
                              np.random.default_rng(seed + 1), scheme, ess_frac)
        grid = np.linspace(-30, 30, 601)
        gr = pf.grid_filter(list(z), grid, np.exp(-0.5 * grid**2 / 4), pf.ungm_f, 10.0,
                            lambda zz, g, t: -0.5 * (zz - pf.ungm_h(g)) ** 2)
        return x, z, res, gr, grid, pf.ungm_ekf(z, 10.0, 1.0)

    x, z, res, gr, grid, ekf = _bench(N, scheme, ess_frac, int(seed))
    t = np.arange(len(x))
    with c2:
        fig = go.Figure()
        post_n = gr["post"] / gr["post"].max(axis=1, keepdims=True)       # each time step scaled to peak at 1
        fig.add_heatmap(x=t, y=grid, z=post_n.T, colorscale="Greys", zmin=0, zmax=1.2, showscale=False,
                        name="exact posterior", hoverinfo="skip")
        fig.add_scatter(x=t, y=x, name="truth", mode="markers", marker=dict(size=7, color=ui.C[4],
                        line=dict(width=1, color="black")))
        fig.add_scatter(x=t, y=ekf["mean"], name="EKF", line=dict(color=ui.C[8]))
        fig.add_scatter(x=t, y=res["mean"], name=f"particle filter (N={N})", line=dict(color=ui.C[1]))
        fig.update_yaxes(range=[-30, 30])
        ui.show(ui.style(fig, "Exact posterior (grey, darker = more probable) with estimates", "time step",
                         "state x", height=400))
        rm = lambda e: float(np.sqrt(np.mean(e**2)))
        m = st.columns(4)
        m[0].metric("EKF RMSE", f"{rm(ekf['mean'] - x):.1f}")
        m[1].metric("Particle filter RMSE", f"{rm(res['mean'] - x):.1f}")
        m[2].metric("Exact posterior mean RMSE", f"{rm(gr['mean'] - x):.1f}", help="The best any filter can do")
        m[3].metric("Steps that resampled", f"{res['resampled'].mean():.0%}")
        f2 = go.Figure()
        f2.add_scatter(x=t, y=res["ess"], name="effective sample size", line=dict(color=ui.C[1]))
        rs = res["resampled"]
        f2.add_scatter(x=t[rs], y=res["ess"][rs], mode="markers", name="resampled here",
                       marker=dict(color=ui.C[2], size=7))
        f2.update_yaxes(type="log")
        ui.show(ui.style(f2, "Weight degeneracy: ESS collapses whenever a measurement is surprising",
                         "time step", "ESS (log)", height=250))
    ui.takeaway("Where the grey shows two bands, the EKF follows one and is often on the wrong side, with errors of "
                "~20. The particle filter tracks the exact posterior mean closely and is limited only by what the "
                "measurements can tell you.")
    ui.controls([
        ("Particles", r"$N$", "samples in the cloud", "closer to the exact posterior; slower"),
        ("Resampling", "–", "scheme for cloning/dropping particles", "'none' → weight collapse"),
        ("Resample when ESS below", "threshold on eq. 2", "how degenerate weights may get", "resamples more often"),
    ])
    ui.experiments([
        ("Defaults: compare the RMSE metrics.", "EKF ≈ 13.6, particle filter ≈ 4.8, exact posterior about the same.",
         "Where the posterior has two peaks the EKF commits to one, often the wrong one. Particles keep both."),
        ("Set Resampling = none.", "The ESS plot drops to 1 and stays there; the particle RMSE roughly doubles.",
         "Without resampling, one particle ends up with all the weight: a single lost guess (degeneracy)."),
        ("Set Particles = 50 and try a few seeds.", "Results vary much more between seeds; ESS often hits 1.",
         "Too few samples to cover both peaks reliably; Monte Carlo noise dominates."),
        ("Compare systematic and multinomial.", "Nearly the same RMSE.",
         "Both are unbiased; multinomial just adds a little extra randomness at each resample."),
    ])

# --------------------------------------------------------------------------- road
with t_road:
    st.markdown(
        r"""
**The situation.** A car starts with GPS unavailable (an underground car park, a cold start). It knows the road's
**elevation map** (grey line) and has a barometric altimeter and wheel odometry. Particles (blue) start spread
**uniformly over the whole 6 km road**, which a Kalman filter can't represent. Each step they move with the
odometer and are weighted by how well the map altitude at their position matches the barometer (Theory A).
Press ▶ and watch the cloud collapse onto the orange car.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        Nr = st.select_slider("Particles", [300, 1000, 2000, 5000], 2000, key="road_N",
                              help="Particles covering 6 km of road. Too few and the right spot may get no particle.")
        baro = st.slider("Barometer noise σ (m)", 0.2, 5.0, 1.5, 0.1,
                         help="Altitude measurement noise. Larger = each reading eliminates fewer candidate places.")
        odo = st.slider("Odometer noise σ (m/s)", 0.05, 2.0, 0.3, 0.05,
                        help="Wheel-speed error. Larger = particles spread out more between readings.")
        start = st.slider("True start position (m)", 0, 4500, 1200, 100,
                          help="Where the car really starts (hidden from the filter).")
        rseed = st.number_input("Random seed", 0, 9999, 2, key="road_seed")

    @st.cache_data(show_spinner="Localising…")
    def _road(Nr, baro, odo, start, seed):
        T, dt = 400, 0.1
        d = pf.road_simulate(T, dt, float(start), np.random.default_rng(seed), baro, odo)
        res = pf.bootstrap_pf(list(d["z"]), Nr, lambda n, g: g.uniform(0, 6000, n),
                              lambda s, t, g: s + (d["v_meas"][t] + odo * g.standard_normal(s.size)) * dt if t else s,
                              lambda zz, s, t: -0.5 * (zz - pf.road_profile(s)) ** 2 / baro**2,
                              np.random.default_rng(seed + 1), snapshot_every=4)
        return d, res

    d, rres = _road(Nr, baro, odo, start, int(rseed))
    road_s = np.linspace(0, 6000, 1200)
    road_h = pf.road_profile(road_s)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=road_s, y=road_h, name="road elevation map", line=dict(color=ui.C["truth"], width=2))
        t0, X0, W0 = rres["snapshots"][0]
        rng_draw = np.random.default_rng(0)
        def _draw(X, W, n=400):
            idx = pf.systematic_resample(W, rng_draw)[:: max(1, len(W) // n)]
            s = X[idx, 0]
            return s, pf.road_profile(s) + 1.0
        sx, sy = _draw(X0, W0)
        fig.add_scatter(x=sx, y=sy, mode="markers", name="particles (car could be here)",
                        marker=dict(size=5, color=ui.rgba(ui.C[1], 0.35)))
        fig.add_scatter(x=[d["s"][0]], y=[pf.road_profile(np.array([d["s"][0]]))[0] + 4], mode="markers",
                        name="true car", marker=dict(size=14, color=ui.C[2], symbol="triangle-down"))
        frames = []
        for (ti, X, W) in rres["snapshots"]:
            sx, sy = _draw(X, W)
            frames.append(go.Frame(name=f"{ti * 0.1:.1f}s", traces=[1, 2], data=[
                go.Scatter(x=sx, y=sy),
                go.Scatter(x=[d["s"][ti]], y=[pf.road_profile(np.array([d["s"][ti]]))[0] + 4])]))
        fig.frames = frames
        fig.update_xaxes(range=[0, 6000])
        fig.update_layout(hovermode="closest")
        ui.play_controls(fig, [f.name for f in frames], 120)
        ui.show(ui.style(fig, "Press ▶: particles collapse onto the true position as the hills go by",
                         "distance along road (m)", "elevation (m)", height=420), key="road_anim")
        tt = np.arange(len(d["s"])) * 0.1
        f2 = go.Figure()
        f2.add_scatter(x=tt, y=np.abs(rres["mean"] - d["s"]), name="|error| of the particle mean",
                       line=dict(color=ui.C[1]))
        f2.add_scatter(x=tt, y=rres["std"], name="particle spread (std)", line=dict(color=ui.C[2], dash="dot"))
        f2.update_yaxes(type="log")
        ui.show(ui.style(f2, "From 'anywhere on 6 km' to a few metres", "time (s)", "metres (log)", height=260))
    ui.takeaway("Early on the mean is meaningless: it sits *between* the candidate hills, where the car certainly is "
                "not. With a multi-modal belief, report the modes (or the whole cloud), not the mean. Each "
                "distinctive climb or dip eliminates the wrong candidates.")
    ui.controls([
        ("Particles", r"$N$", "samples along the road", "more reliable, slower"),
        ("Barometer noise σ", r"$\sigma$ in $p(z\mid s)$", "altitude sensor noise", "takes longer to rule places out"),
        ("Odometer noise σ", "motion noise", "speed error", "the cloud spreads more between readings"),
        ("True start position", r"$s_0$", "hidden starting point", "different terrain, different difficulty"),
    ])
    ui.experiments([
        ("Press ▶ with the defaults.", "Several clusters survive for the first seconds; by ~10 s one remains.",
         "Each altitude reading rules out places whose map altitude doesn't match. Distinctive climbs decide it."),
        ("In the bottom plot, compare the error and the spread early on.",
         "While several clusters survive, the mean's error is huge, and so is the spread.",
         "The mean of a two-peaked belief lies between the peaks, where the car certainly isn't."),
        ("Set Barometer noise σ = 5 m.", "Convergence takes about twice as long (~20 s).",
         "Noisier readings carry less information, so more terrain must pass before one place wins."),
    ])
    ui.in_practice("Map-matching and terrain-aided navigation, robot localisation in a known map (Monte Carlo "
                   "localisation), indoor positioning from Wi-Fi/magnetic fingerprints, and visual tracking in "
                   "clutter all use particle filters.")

# --------------------------------------------------------------------------- IMM
with t_imm:
    st.markdown(
        r"""
**The situation.** A radar tracks a target that cruises straight, turns hard, cruises, then turns the other way.
Two Kalman filters differ only in process noise $q$: **cruise** (small $q$) and **manoeuvre** (large $q$). The
**IMM** runs both and keeps a probability for each, updated from how well each one predicts the radar plots
(Theory B). The top plot is one run; the bottom plots average many runs.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        rate = st.slider("Turn rate (deg/s)", 2, 20, 9,
                         help="How hard the target turns (the second turn is 1.3× harder). At 100 m/s, 9°/s ≈ 1.6 g.")
        sig = st.slider("Radar position noise σ (m)", 5, 60, 20, help="Measurement noise of each radar plot.")
        ql = st.select_slider("Cruise model q", [0.01, 0.1, 1.0, 5.0], 0.1,
                              help="Process noise of the 'cruise' filter: expects nearly straight flight.")
        qh = st.select_slider("Manoeuvre model q", [20.0, 50.0, 100.0, 200.0, 500.0], 200.0,
                              help="Process noise of the 'manoeuvre' filter: expects hard accelerations.")
        pstay = st.select_slider("P(stay in the same model) per step", [0.8, 0.9, 0.95, 0.98, 0.99], 0.95,
                                 help="IMM's switching prior. 0.95 = expect a mode change about every 20 scans.")
        runs = st.select_slider("Monte Carlo runs", [5, 20, 50], 20, help="Runs averaged in the RMSE plot and table.")

    @st.cache_data(show_spinner="Tracking…")
    def _imm(rate, sig, ql, qh, pstay, runs):
        X = imm.coordinated_turn_truth([(30, 0), (20, rate), (30, 0), (15, -rate * 1.3), (25, 0)], 1.0, 100)
        H = kalman.position_observation(2)
        R = np.eye(2) * sig**2
        F, Q1 = kalman.constant_velocity(1.0, ql, 2)
        _, Q2 = kalman.constant_velocity(1.0, qh, 2)
        Pt = np.array([[pstay, 1 - pstay], [1 - pstay, pstay]])
        errs = {"cruise KF": [], "manoeuvre KF": [], "IMM": []}
        example = None
        for i in range(runs):
            Z = X[:, [0, 2]] + sig * np.random.default_rng(i).standard_normal((len(X), 2))
            x0 = np.array([Z[0, 0], 100, Z[0, 1], 0.0])
            P0 = np.diag([sig**2, 100, sig**2, 100])
            lo = kalman.run_kalman(kalman.KalmanFilter(F, H, Q1, R, x0, P0), list(Z))["x"]
            hi = kalman.run_kalman(kalman.KalmanFilter(F, H, Q2, R, x0, P0), list(Z))["x"]
            o = imm.imm_filter(Z, [F, F], [Q1, Q2], H, R, Pt, x0, P0, [0.5, 0.5])
            for nm, est in (("cruise KF", lo), ("manoeuvre KF", hi), ("IMM", o["x"])):
                errs[nm].append(np.linalg.norm(est[:, [0, 2]] - X[:, [0, 2]], axis=1))
            if i == 0:
                example = (Z, lo, hi, o)
        return X, {k: np.array(v) for k, v in errs.items()}, example

    X, errs, (Z, lo, hi, o) = _imm(rate, sig, ql, qh, pstay, runs)
    turning = np.zeros(len(X), bool)
    turning[30:50] = turning[80:95] = True
    cols = {"cruise KF": ui.C[2], "manoeuvre KF": ui.C[7], "IMM": ui.C[1]}
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", name="radar plots",
                        marker=dict(size=4, color=ui.rgba(ui.C["truth"], 0.5)))
        fig.add_scatter(x=X[:, 0], y=X[:, 2], name="truth", line=dict(color=ui.C["truth"], width=4))
        for nm, est in (("cruise KF", lo), ("manoeuvre KF", hi), ("IMM", o["x"])):
            fig.add_scatter(x=est[:, 0], y=est[:, 2], name=nm, line=dict(color=cols[nm]))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(hovermode="closest")
        ui.show(ui.style(fig, "One run: track in the plane", "x (m)", "y (m)", height=380))
        f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.07)
        k = np.arange(len(X))
        for nm, e in errs.items():
            f2.add_scatter(x=k, y=np.sqrt(np.mean(e**2, axis=0)), name=nm, line=dict(color=cols[nm]), row=1, col=1)
        f2.add_scatter(x=k, y=o["mu"][:, 1], name="IMM P(manoeuvre)", line=dict(color=ui.C[1], dash="dot"),
                       row=2, col=1)
        f2.add_scatter(x=k, y=turning.astype(float), name="truly turning", fill="tozeroy",
                       fillcolor=ui.rgba(ui.C["truth"], 0.2), line=dict(width=0), row=2, col=1)
        f2.update_yaxes(title="RMSE (m)", type="log", row=1, col=1)
        f2.update_yaxes(title="probability", range=[0, 1.05], row=2, col=1)
        f2.update_xaxes(title="time (s)", row=2, col=1)
        ui.show(ui.style(f2, f"Position RMSE over {runs} runs, and the IMM's belief about the mode", height=420))
        tbl = pd.DataFrame({nm: {"straight RMSE (m)": np.sqrt(np.mean(e[:, ~turning] ** 2)),
                                 "turning RMSE (m)": np.sqrt(np.mean(e[:, turning] ** 2))}
                            for nm, e in errs.items()}).T
        st.dataframe(tbl.style.format("{:.1f}"), width="stretch")
    ui.takeaway("The cruise filter is excellent until the first turn, then lags badly. The manoeuvre filter never "
                "loses the target but stays noisy. The IMM switches its belief to 'manoeuvre' within a few scans of "
                "each turn and back afterwards, so it is close to the better filter in every phase.")
    ui.controls([
        ("Turn rate", "–", "true manoeuvre severity", "the cruise filter falls further behind"),
        ("Radar noise σ", r"$\sqrt R$", "measurement noise", "everything noisier; mode decisions slower"),
        ("Cruise / Manoeuvre q", r"$q$ of each model", "what each filter expects", "more agile but noisier filter"),
        ("P(stay)", r"diagonal of $\Pi$", "switching prior", "fewer, more confident mode switches"),
    ])
    ui.experiments([
        ("Defaults: read the table.",
         "Straight segments: IMM ≈ 19 m vs manoeuvre filter ≈ 23 m. Turns: about equal (≈ 27 m). Cruise: hundreds.",
         "IMM uses the precise model on straights and the agile one in turns."),
        ("Watch the dotted P(manoeuvre) line against the shaded turning periods.",
         "It rises within a few scans of each turn and falls afterwards.",
         "The cruise filter's innovations become too large for its small S, so its likelihood collapses."),
        ("Set P(stay) = 0.8.", "The IMM's straight-segment error rises toward the manoeuvre filter's (≈ 21 m).",
         "A weak switching prior lets noise flip the mode, mixing in the noisy filter."),
        ("Set the turn rate to 20°/s.", "The cruise filter diverges completely; the IMM still holds on.",
         "The manoeuvre model covers the turn, and the IMM gives it the weight."),
    ])
    ui.in_practice("IMM is the workhorse of air-traffic control and automotive radar tracking (adaptive cruise, "
                   "collision avoidance). The same idea, mode probabilities over a bank of filters, is used in "
                   "fault-tolerant navigation.")

# --------------------------------------------------------------------------- BOCPD
with t_cp:
    st.markdown(
        r"""
**The situation.** A sensor signal sits at one level, then jumps (in level, noise, or both). BOCPD tracks the
hidden **run length**: how many samples since the last change (Theory C). In the top plot, yellow dashed lines are
the true changes and orange solid lines are detections. In the bottom heat map, darker = more probable run
length; each diagonal ridge is one segment growing, and a new ridge starting at zero is a change.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        src = st.radio("Signal", ["Machine sensor: level & noise shifts", "Daily returns from the market simulator"],
                       help="A synthetic sensor with level/noise jumps, or market returns whose volatility switches.")
        if src.startswith("Machine"):
            nchg = st.slider("Number of changes", 0, 8, 4, help="How many jumps the true signal has.")
            jump = st.slider("Typical level jump (× noise σ)", 0.5, 5.0, 2.0, 0.25,
                             help="Size of level shifts relative to the noise. Below ~1 they're hard to see.")
            vr = st.slider("Noise-level change ratio", 1.0, 3.0, 1.5, 0.1,
                           help="How much the noise level can change at a jump (1 = never).")
            T = 600
        else:
            T = st.slider("Days", 300, 1500, 800, 100, help="Length of the simulated return series.")
        lam = st.select_slider("Expected run length λ (prior)", [25, 50, 100, 250, 1000], 100,
                               help="Prior belief about segment length; hazard H = 1/λ per sample.")
        confirm = st.slider("Confirmation delay (samples)", 0, 20, 5,
                            help="Wait this long before declaring a change. Fewer false alarms, slower alarms.")
        cseed = st.number_input("Random seed", 0, 9999, 3, key="cp_seed", help="A different random signal.")
    if src.startswith("Machine"):
        x, true_cps, levels, _ = cp.piecewise_signal(T, nchg, jump, vr, np.random.default_rng(int(cseed)))
        b = cp.bocpd(x, lam, 0.0, 0.1, 1.0, 1.0)
    else:
        from ssp.sims.market import regime_returns
        m = regime_returns(T, 2, 0.99, np.random.default_rng(int(cseed)))
        x = m["r"]
        true_cps = (np.flatnonzero(np.diff(m["states"])) + 1).tolist()
        levels = None
        b = cp.bocpd(x, lam, 0.0, 1.0, 2.0, 1.0)
    det = cp.detect_changes_detail(b["map"], confirm=confirm)
    found = [s for s, _ in det]
    with c2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06)
        tt = np.arange(T)
        fig.add_scatter(x=tt, y=x, name="signal", line=dict(color=ui.C["truth"], width=1), row=1, col=1)
        if levels is not None:
            fig.add_scatter(x=tt, y=levels, name="true level", line=dict(color=ui.C[4], width=2), row=1, col=1)
        for c in true_cps:
            fig.add_vline(x=c, line=dict(color=ui.C[4], dash="dash", width=1), row=1, col=1)
        for s_, dcl in det:
            fig.add_vline(x=s_, line=dict(color=ui.C[2], width=2), row=1, col=1)
        Rm = b["R"][:, :min(b["R"].shape[1], T)]
        rmax_show = int(min(Rm.shape[1], max(60, b["map"].max() + 20)))
        fig.add_heatmap(x=tt, y=np.arange(rmax_show), z=np.log10(Rm[:, :rmax_show].T + 1e-6), colorscale="Blues",
                        zmin=-4, zmax=0, showscale=False, hoverinfo="skip", row=2, col=1)
        fig.add_scatter(x=tt, y=b["map"], name="most probable run length", line=dict(color=ui.C[2], width=1.5),
                        row=2, col=1)
        fig.update_yaxes(title="value", row=1, col=1)
        fig.update_yaxes(title="run length", row=2, col=1)
        fig.update_xaxes(title="time", row=2, col=1)
        ui.show(ui.style(fig, "Signal (dashed = true changes, solid = detected) and the run-length posterior",
                         height=520))
        tol = 10
        hits = [c for c in true_cps if any(abs(c - f) <= tol for f in found)]
        fa = [f for f in found if not any(abs(c - f) <= tol for c in true_cps)]
        delays = [min(dcl - c for (f, dcl) in det if abs(c - f) <= tol) for c in hits]
        mm = st.columns(3)
        mm[0].metric("Changes found", f"{len(hits)} / {len(true_cps)}")
        mm[1].metric("False alarms", f"{len(fa)}")
        mm[2].metric("Mean alarm delay", f"{np.mean(delays):.0f} samples" if delays else "–")
    ui.takeaway("The posterior's diagonal ridges are runs growing one sample at a time. A change shows up as a new "
                "ridge starting at zero. The detector is fully probabilistic, so the trade-off between false alarms "
                "and delay is explicit: the confirmation delay and the hazard λ.")
    ui.controls([
        ("Number of changes, level jump, noise ratio", "–", "the true signal", "bigger jumps are easier to detect"),
        ("Expected run length λ", r"$1/H$ in eq. 4–5", "prior on how often changes happen", "fewer false alarms, slower"),
        ("Confirmation delay", "–", "samples to wait before declaring", "fewer false alarms, longer delay"),
    ])
    ui.experiments([
        ("Defaults: read the three metrics.", "4/4 changes found, 0 false alarms, about 9 samples of delay.",
         "Clear jumps (2σ) are easy; the delay is mostly the 5-sample confirmation."),
        ("Set the confirmation delay to 0, then 20.",
         "0: delay ≈ 4 samples but a false alarm appears. 20: no false alarms, delay ≈ 24.",
         "Every detector trades false alarms against delay. The confirmation delay makes that trade explicit."),
        ("Set the typical level jump to 0.75.", "Only about half the changes are found.",
         "Shifts smaller than the noise need many samples to become convincing."),
        ("With the jump at 0.75, set λ = 25.", "More false alarms appear.",
         "Telling the model that changes are frequent makes it quicker to believe noise is a change."),
        ("Switch the signal to market returns.", "Most switches are found even though the mean barely changes.",
         "The Student-t predictive notices changes in *variance*, not just in level."),
    ])
    ui.in_practice("Predictive maintenance (vibration and temperature shifts), network and fraud monitoring, "
                   "detecting regime breaks in finance, and segmenting audio or sensor streams.")

with t_code:
    st.markdown("The exact functions this page runs (from `ssp/core/particle.py`, `imm.py`, `changepoint.py`):")
    ui.source(pf.systematic_resample, pf.bootstrap_pf, pf.grid_filter, imm.imm_filter, cp.bocpd)
