import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import finance, hmm
from ssp.sims import market as mk

ui.lab_intro(
    "markets",
    goal="Estimate three hidden quantities from prices: the market's regime (calm or turbulent), its current "
         "volatility, and a drifting relationship between two assets. Data is simulated so you can check each "
         "estimate against the truth; you can also upload real prices.",
    on_screen=[
        "**Regimes (HMM):** price coloured by decoded regime; filtered vs smoothed probability of the turbulent "
        "regime (Theory §2)",
        "**Volatility (GARCH by MLE):** returns with a ±2σ band, fitted vs true volatility, forecasts, and the "
        "autocorrelation evidence for volatility clustering (Theory §3)",
        "**Hedge ratio (Kalman):** a drifting coefficient tracked by a Kalman filter vs a rolling regression, plus "
        "the spread's z-score (Theory §4)",
    ],
    start_here="In the Regimes tab, compare the orange (filtered, knowable that day) and violet (smoothed, "
               "hindsight) curves at the start of a shaded turbulent period. Which one switches first?",
)
st.caption("For learning statistical estimation only. Nothing on this page is investment advice, and none of "
           "these models predicts prices.")

t_reg, t_vol, t_hedge = st.tabs(["Regimes (HMM)", "Volatility (GARCH by MLE)", "Hedge ratio (Kalman)"])


def _data_source(key: str):
    """Synthetic regime data by default; optional CSV upload of daily prices."""
    src = st.radio("Data", ["Simulated (truth known)", "Upload a CSV of daily prices"], key=f"{key}_src",
                   help="Simulated data has a known truth to compare with; uploaded prices don't.")
    if src.startswith("Upload"):
        up = st.file_uploader("CSV with a Date column and a Close / Adj Close / Price column", type="csv",
                              key=f"{key}_up")
        if up is None:
            st.info("Upload a file to continue. Most finance sites can export daily prices as CSV.")
            return None
        r, col = mk.returns_from_csv(up.getvalue())
        st.caption(f"Using column **{col}**: {len(r)} daily log-returns.")
        return {"r": r.to_numpy(), "dates": r.index, "truth": None}
    return "sim"


# --------------------------------------------------------------------------- regimes
with t_reg:
    st.markdown(
        r"""
**The situation.** Daily returns come from hidden **regimes** with different average return and volatility,
switching like a Markov chain. A Gaussian HMM learns the regimes from the returns alone (Baum–Welch), then
reports two probabilities: **filtered** (what was knowable on that day) and **smoothed** (hindsight). The top
plot colours the price by the decoded (Viterbi) regime; the shaded area in the bottom plot is the true turbulent
regime.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        data = _data_source("reg")
        if data == "sim":
            K_true = st.radio("True number of regimes", [2, 3], horizontal=True,
                              help="How many regimes the simulator really uses.")
            stay = st.slider("Regime persistence (P stay per day)", 0.9, 0.998, 0.985, 0.001, format="%.3f",
                             help="Daily probability of staying in the same regime. 0.985 ≈ 67-day regimes.")
            T = st.slider("Trading days", 500, 5000, 2500, 250, help="Length of the history (~250 per year).")
            seed = st.number_input("Random seed", 0, 9999, 2, key="reg_seed")
            sim = mk.regime_returns(T, K_true, stay, np.random.default_rng(int(seed)))
            data = {"r": sim["r"], "dates": np.arange(T), "truth": sim}
        K = st.radio("Regimes in the fitted model", [2, 3], horizontal=True,
                     help="How many regimes the HMM looks for. It may differ from the truth, as it would with real data.")
    if data is not None:
        r = data["r"]

        @st.cache_data(show_spinner="Fitting the HMM…")
        def _fit_regimes(r_bytes, K):
            r = np.frombuffer(r_bytes)
            return hmm.baum_welch(r, hmm.init_gaussian(r, K, stay=0.97), n_iter=300, var_floor=1e-4)

        fit = _fit_regimes(r.astype(float).tobytes(), K)
        p = fit["params"]
        order = np.argsort(np.sqrt(p.variances.ravel()))           # rank states: low vol -> high vol
        rank = np.argsort(order)
        logB = p.loglik_matrix(r)
        filt = hmm.forward(p.pi, p.A, logB)[0][:, order]
        smooth = fit["gamma"][:, order]
        path = rank[hmm.viterbi(p.pi, p.A, logB)[0]]
        names = (["low vol", "high vol"] if K == 2 else ["low vol", "mid vol", "high vol"])
        cols = [ui.C[1], ui.C[4], ui.C[8]][:K] if K == 3 else [ui.C[1], ui.C[8]]
        price = 100 * np.exp(np.cumsum(r / 100))
        x = data["dates"]
        with c2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05)
            for k in range(K):
                fig.add_scatter(x=x, y=np.where(path == k, price, np.nan), mode="lines", name=f"decoded: {names[k]}",
                                line=dict(color=cols[k], width=2), connectgaps=False, row=1, col=1)
            fig.add_scatter(x=x, y=filt[:, -1], name="filtered P(high vol) — known that day",
                            line=dict(color=ui.C[2], width=1.5), row=2, col=1)
            fig.add_scatter(x=x, y=smooth[:, -1], name="smoothed P(high vol) — hindsight",
                            line=dict(color=ui.C[7], width=1.5), row=2, col=1)
            if data["truth"] is not None:
                tr = data["truth"]["states"] == data["truth"]["states"].max()
                fig.add_scatter(x=x, y=tr.astype(float), name="true high-vol regime", fill="tozeroy",
                                fillcolor=ui.rgba(ui.C["truth"], 0.2), line=dict(width=0), row=2, col=1)
            fig.update_yaxes(title="price (index)", type="log", row=1, col=1)
            fig.update_yaxes(title="probability", range=[0, 1.02], row=2, col=1)
            ui.show(ui.style(fig, "Price coloured by decoded regime, and the regime probability", height=520))
            dur = 1 / (1 - np.clip(np.diag(p.A)[order], 0, 0.99999))
            tbl = pd.DataFrame({"mean return (%/day)": p.means.ravel()[order],
                                "volatility (%/day)": np.sqrt(p.variances.ravel()[order]),
                                "expected duration (days)": dur,
                                "time spent": np.bincount(path, minlength=K) / len(path)}, index=names)
            st.dataframe(tbl.style.format({"mean return (%/day)": "{:+.3f}", "volatility (%/day)": "{:.2f}",
                                           "expected duration (days)": "{:.0f}", "time spent": "{:.0%}"}),
                         width="stretch")
            if data["truth"] is not None and K == len(data["truth"]["names"]):
                st.metric("Viterbi agreement with the true regimes", f"{np.mean(path == data['truth']['states']):.1%}")
                st.caption("True regimes: " + ", ".join(
                    f"{n} (μ={m:+.2f}%, σ={s:.1f}%)" for n, m, s in
                    zip(data["truth"]["names"], data["truth"]["mu"], data["truth"]["sigma"])))
        ui.takeaway("Volatility regimes are easy to detect because variance shows up immediately in |returns|. "
                    "Mean-return differences are tiny next to daily noise, so the model can tell you *how risky* "
                    "the market is now far better than *which way* it will go.")
        ui.controls([
            ("True number of regimes", "–", "the simulator's truth", "–"),
            ("Regime persistence", r"$A_{kk}$", "daily P(stay)", "longer, easier-to-detect regimes"),
            ("Trading days", r"$T$", "history length", "better-estimated parameters"),
            ("Regimes in the fitted model", r"$K$", "HMM size", "more regimes can split real ones"),
        ])
        ui.experiments([
            ("Defaults: read the agreement and the table.",
             "Viterbi agrees with the truth ≈ 98%; learned volatilities ≈ 0.7% and 2.2% (true 0.7 and 2.2); expected "
             "regime length ≈ 65 days.",
             "Volatility regimes are well separated, so Baum–Welch finds them without labels."),
            ("Compare the orange and violet curves at regime switches.",
             "The violet (smoothed) curve switches earlier and more cleanly.",
             "It uses later returns. Trading on it in a backtest is look-ahead bias; only the orange curve is honest."),
            ("Set persistence = 0.9 (≈ 10-day regimes).", "Agreement drops to ≈ 87%.",
             "Short regimes give little evidence before they end, the same lesson as the dice in Level 4."),
            ("Fit 3 regimes to 2-regime data.", "One true regime gets split into two similar states.",
             "Model selection matters; with real data you don't know K."),
        ])
        ui.in_practice("Regime-switching models are used for risk management, volatility targeting, stress "
                       "testing and macro nowcasting (recession probabilities). The filtered probability is the "
                       "number a live system would actually act on.")

# --------------------------------------------------------------------------- volatility
with t_vol:
    st.markdown(
        r"""
**The situation.** Returns are simulated from a GARCH(1,1) model (Theory eq. 3): today's variance depends on
yesterday's shock (α) and yesterday's variance (β). We then forget the true α, β and recover them by **maximum
likelihood** (Theory eq. 4), exactly the Level 1 recipe applied to a recursive model. The fitted model gives a
time-varying volatility and forecasts that decay toward the long-run level.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        vdata = _data_source("vol")
        if vdata == "sim":
            a_t = st.slider("True α (reaction to shocks)", 0.0, 0.3, 0.08, 0.01,
                            help="How much a big move today raises tomorrow's variance.")
            b_t = st.slider("True β (memory)", 0.5, 0.98, 0.90, 0.01,
                            help="How much of today's variance carries over to tomorrow. α + β must stay below 1.")
            if a_t + b_t >= 0.999:
                st.warning("α + β must be below 1 for a stationary process; β was reduced.")
                b_t = 0.998 - a_t
            Tv = st.slider("Trading days", 500, 5000, 2500, 250, key="vol_T", help="History length for the fit.")
            vseed = st.number_input("Random seed", 0, 9999, 4, key="vol_seed")
            rv, s2_true = finance.garch11_simulate(Tv, 0.03, 0.02, a_t, b_t, np.random.default_rng(int(vseed)))
            vdata = {"r": rv, "dates": np.arange(Tv), "truth": {"sigma2": s2_true, "alpha": a_t, "beta": b_t}}
        horizon = st.slider("Forecast horizon (days)", 10, 250, 60, help="How far ahead to forecast volatility.")
    if vdata is not None:
        rv = vdata["r"]

        @st.cache_data(show_spinner="Maximising the likelihood…")
        def _fit_garch(r_bytes):
            return finance.garch11_fit(np.frombuffer(r_bytes))

        g = _fit_garch(rv.astype(float).tobytes())
        sig = np.sqrt(g["sigma2"][:-1])
        xv = np.arange(len(rv))
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=xv, y=rv, mode="lines", name="daily return (%)", line=dict(color=ui.C["truth"], width=1))
            fig.add_scatter(x=xv, y=2 * sig, name="±2σₜ (GARCH)", line=dict(color=ui.C[1], width=1.5))
            fig.add_scatter(x=xv, y=-2 * sig, showlegend=False, line=dict(color=ui.C[1], width=1.5))
            ui.show(ui.style(fig, "Returns and the fitted conditional volatility band", "day", "%", height=320))
            fig2 = go.Figure()
            if vdata["truth"] is not None:
                fig2.add_scatter(x=xv, y=np.sqrt(vdata["truth"]["sigma2"]), name="true σₜ",
                                 line=dict(color=ui.rgba(ui.C["truth"], 0.6), width=5))
            fig2.add_scatter(x=xv, y=sig, name="fitted σₜ", line=dict(color=ui.C[1], width=1.5))
            fc = np.sqrt(finance.garch11_forecast(g, horizon))
            xf = np.arange(len(rv), len(rv) + horizon)
            fig2.add_scatter(x=xf, y=fc, name="forecast σ", line=dict(color=ui.C[2], width=2.5))
            fig2.add_hline(y=np.sqrt(g["long_run_var"]), line=dict(color=ui.C[2], dash="dot"),
                           annotation_text="long-run σ")
            fig2.update_xaxes(range=[max(0, len(rv) - 500), len(rv) + horizon])
            ui.show(ui.style(fig2, "Volatility: last 500 days and the forecast", "day", "σ (%/day)", height=300))
            lags = np.arange(1, 31)
            ac = lambda v: [np.corrcoef(v[:-k], v[k:])[0, 1] for k in lags]
            fig3 = go.Figure()
            fig3.add_bar(x=lags, y=ac(rv - rv.mean()), name="returns", marker_color=ui.C[1])
            fig3.add_bar(x=lags, y=ac((rv - rv.mean()) ** 2), name="squared returns", marker_color=ui.C[2])
            ui.show(ui.style(fig3, "Autocorrelation: direction is unpredictable, size is not", "lag (days)",
                             "correlation", height=260))
            rows = {"α": g["alpha"], "β": g["beta"], "persistence α+β": g["persistence"],
                    "volatility half-life (days)": np.log(0.5) / np.log(g["persistence"])}
            tbl = pd.DataFrame({"fitted": rows})
            if vdata["truth"] is not None:
                at, bt = vdata["truth"]["alpha"], vdata["truth"]["beta"]
                tbl["true"] = [at, bt, at + bt, np.log(0.5) / np.log(at + bt) if at + bt > 0 else np.nan]
            st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
        ui.takeaway("The MLE recovers α and β from a few thousand days. The fitted band widens right after big "
                    "moves and decays slowly: that is volatility clustering, and it's why risk models use a "
                    "conditional σₜ rather than one long-run number.")
        ui.controls([
            ("True α", r"$\alpha$", "reaction to shocks", "spikier volatility"),
            ("True β", r"$\beta$", "volatility memory", "longer calm and turbulent spells"),
            ("Trading days", "–", "fit sample size", "more precise estimates (as in Level 1)"),
            ("Forecast horizon", r"$h$", "days ahead", "the forecast approaches the long-run level"),
        ])
        ui.experiments([
            ("Defaults: compare 'fitted' and 'true' in the table.", "α ≈ 0.07 vs 0.08, β ≈ 0.91 vs 0.90.",
             "Maximum likelihood on a few thousand days recovers the hidden dynamics."),
            ("Look at the autocorrelation bars.", "Blue (returns) ≈ 0 at all lags; orange (squared returns) is "
             "clearly positive.", "Direction is unpredictable, size is predictable: volatility clustering."),
            ("Set α = 0.", "The orange bars drop to ≈ 0 and the fitted band becomes flat.",
             "With no shocks feeding variance, there's nothing to cluster. β can't be learned (no information)."),
            ("Set α = 0.2, β = 0.75.", "Spiky volatility that dies out quickly; the half-life in the table falls.",
             "Persistence α + β = 0.95: volatility half-life = ln 0.5 / ln 0.95 ≈ 14 days."),
        ])

# --------------------------------------------------------------------------- hedge ratio
with t_hedge:
    st.markdown(
        r"""
**The situation.** Two related prices follow $y_t \approx \beta_t x_t + \alpha_t$, where the **hedge ratio**
$\beta_t$ drifts and, at some point, jumps. A Kalman filter treats $[\beta_t, \alpha_t]$ as a hidden random-walk
state (Theory eq. 5). We compare it with a rolling regression over the last W days. The bottom plot is the filter's
**innovation z-score**: the surprise each day relative to the filter's own uncertainty.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        Th = st.slider("Days", 300, 3000, 1200, 100, help="Length of the simulated price history.")
        b0 = st.slider("Initial true β", 0.5, 3.0, 1.5, 0.1, help="Starting hedge ratio (it also drifts slowly).")
        jump = st.slider("Sudden change in β", -1.0, 1.0, 0.5, 0.1,
                         help="A structural break: β jumps by this amount (0 = no break).")
        at = st.slider("Change happens at (fraction of sample)", 0.1, 0.9, 0.6, 0.05, help="When the break happens.")
        svol = st.slider("Spread noise", 0.1, 2.0, 0.5, 0.1,
                         help="Size of the day-to-day deviation of y from βx + α (mean-reverting).")
        use_mle = st.toggle("Fit δ and R by maximum likelihood", True,
                            help="Maximise the likelihood of the innovations (the L1 recipe applied to a filter)")
        delta = st.select_slider("Kalman δ (how fast β may drift)", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 1e-4,
                                 disabled=use_mle)
        win = st.slider("Rolling-OLS window (days)", 20, 250, 60, 10,
                        help="Window of the comparison method: a regression over the last W days.")
        hseed = st.number_input("Random seed", 0, 9999, 5, key="h_seed")
    pr = mk.pairs(Th, b0, jump, at, svol, np.random.default_rng(int(hseed)))

    @st.cache_data(show_spinner="Fitting δ and R by maximum likelihood…")
    def _hedge_mle(y_bytes, x_bytes):
        return finance.kalman_regression_mle(np.frombuffer(y_bytes), np.frombuffer(x_bytes))

    r_meas = None
    if use_mle:
        mle = _hedge_mle(pr["y"].tobytes(), pr["x"].tobytes())
        delta, r_meas = mle["delta"], mle["r"]
    kr = finance.kalman_regression(pr["y"], pr["x"], delta=delta, r=r_meas)
    ols = finance.rolling_ols_beta(pr["y"], pr["x"], win)
    xd = np.arange(Th)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=xd, y=pr["x"], name="asset x", line=dict(color=ui.C[1]))
        fig.add_scatter(x=xd, y=pr["y"], name="asset y", line=dict(color=ui.C[2]))
        ui.show(ui.style(fig, "Two co-moving prices", "day", "price", height=240))
        fig2 = go.Figure()
        sd = kr["sd"][:, 0]
        fig2.add_scatter(x=np.r_[xd, xd[::-1]], y=np.r_[kr["beta"] + 2 * sd, (kr["beta"] - 2 * sd)[::-1]],
                         fill="toself", fillcolor=ui.rgba(ui.C[1], 0.15), line=dict(width=0), name="Kalman ±2σ")
        fig2.add_scatter(x=xd, y=pr["beta"], name="true β", line=dict(color=ui.C["truth"], width=3))
        fig2.add_scatter(x=xd, y=kr["beta"], name="Kalman β", line=dict(color=ui.C[1]))
        fig2.add_scatter(x=xd, y=ols, name=f"rolling OLS ({win} d)", line=dict(color=ui.C[2]))
        fig2.update_yaxes(range=[min(pr["beta"].min(), b0) - 1, max(pr["beta"].max(), b0) + 1])
        ui.show(ui.style(fig2, "Hedge ratio: tracking a coefficient that drifts and jumps", "day", "β", height=320))
        z = kr["innov"] / kr["innov_sd"]
        fig3 = go.Figure()
        fig3.add_scatter(x=xd[20:], y=z[20:], name="spread z-score", line=dict(color=ui.C[7], width=1))
        for lvl in (-2, 2):
            fig3.add_hline(y=lvl, line=dict(color=ui.C["truth"], dash="dot"))
        ui.show(ui.style(fig3, "Innovation z-score (|z| > 2 is an unusual spread)", "day", "z", height=240,
                         legend=False))
        sl = slice(100, None)
        m = st.columns(4)
        m[3].metric("δ used", f"{delta:.1e}", help="Fitted by MLE" if use_mle else "Set by hand")
        m[0].metric("Kalman β RMSE", f"{np.sqrt(np.mean((kr['beta'][sl] - pr['beta'][sl]) ** 2)):.3f}")
        m[1].metric("Rolling OLS β RMSE", f"{np.sqrt(np.nanmean((ols[sl] - pr['beta'][sl]) ** 2)):.3f}")
        m[2].metric("Days with |z| > 2", f"{np.mean(np.abs(z[20:]) > 2):.1%}", help="≈4.6% if the filter is "
                    "well calibrated (Gaussian tails)")
    ui.takeaway("Rolling OLS forces a hard choice of window: short is noisy, long is slow. The Kalman filter makes "
                "the same trade-off with one interpretable knob (δ = how fast β may drift), reacts to the jump, and "
                "tells you its uncertainty. It's the L3 random-walk filter with a time-varying H. The jump day appears as "
                "one huge innovation: the filter's own way of flagging a structural break.")
    ui.controls([
        ("Initial β, Sudden change, Change at", r"$\beta_t$", "the hidden truth", "–"),
        ("Spread noise", r"$v_t$", "deviation of y from βx + α", "a noisier β estimate"),
        ("Fit δ and R by MLE", "–", "choose the noise levels by maximum likelihood", "–"),
        ("Kalman δ", r"$\delta$ (eq. 5)", "how fast β may drift", "faster tracking, noisier β"),
        ("Rolling-OLS window", r"$W$", "the baseline's memory", "smoother but slower baseline"),
    ])
    ui.experiments([
        ("Defaults: compare the RMSE metrics.", "Kalman (MLE-tuned) ≈ 0.05; rolling OLS (60 days) ≈ 1.0.",
         "The filter models the drift explicitly and reacts to the jump; the short regression is swamped by noise."),
        ("Set the rolling window to 250.", "OLS improves to ≈ 0.6 but reacts to the jump months late.",
         "A long window averages noise but remembers the old β: the window trade-off the filter avoids."),
        ("Turn off the MLE fit and set δ = 1e-7.", "The Kalman β takes hundreds of days to accept the jump (RMSE ≈ 0.65).",
         "The filter is told β almost never changes, so it explains the break as noise."),
        ("Back to MLE; look at the z-score plot.", "One enormous spike (≈ 27) on the break day; '|z| > 2' is well "
         "under the 4.6% you'd expect.",
         "Under the Gaussian model the break is a 27σ event. The MLE stretches the noise levels to accommodate it, "
         "so ordinary days look too calm. Same lesson as Level 1's outliers."),
        ("Set the sudden change to 0.", "Now '|z| > 2' ≈ 4.5% and the z-scores look like N(0, 1).",
         "Without the break the model is right, and MLE-tuned innovations are properly calibrated."),
    ])
    ui.in_practice("Time-varying regressions with Kalman filters are used for dynamic hedging, factor exposures "
                   "(time-varying CAPM β), and statistical arbitrage. The same structure estimates channel taps in "
                   "communications. It's RLS from L2 with explicit process noise.")
