import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import finance, hmm
from ssp.sims import market as mk

ui.header("Domain demo · Markets", "Regimes, volatility and hedge ratios",
          "Prices are noisy observations of hidden things: the market's mood, its current risk, and how two "
          "assets move together. Can we estimate them as they change?")
st.caption("For learning statistical estimation only. Nothing on this page is investment advice, and none of "
           "these models predicts prices.")

t_reg, t_vol, t_hedge = st.tabs(["Regimes (HMM)", "Volatility (GARCH by MLE)", "Hedge ratio (Kalman)"])


def _data_source(key: str):
    """Synthetic regime data by default; optional CSV upload of daily prices."""
    src = st.radio("Data", ["Simulated (truth known)", "Upload a CSV of daily prices"], key=f"{key}_src")
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
**Model:** daily returns come from one of $K$ hidden regimes, each with its own mean and volatility, and the
regime follows a Markov chain (Hamilton's regime-switching model). Baum–Welch learns everything from the returns;
then we ask two different questions:
**filtered** $P(\text{regime}_t\mid r_{1:t})$, which you could have known *on that day*, and
**smoothed** $P(\text{regime}_t\mid r_{1:T})$, which is only available with hindsight.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        data = _data_source("reg")
        if data == "sim":
            K_true = st.radio("True number of regimes", [2, 3], horizontal=True)
            stay = st.slider("Regime persistence (P stay per day)", 0.9, 0.998, 0.985, 0.001, format="%.3f")
            T = st.slider("Trading days", 500, 5000, 2500, 250)
            seed = st.number_input("Random seed", 0, 9999, 2, key="reg_seed")
            sim = mk.regime_returns(T, K_true, stay, np.random.default_rng(int(seed)))
            data = {"r": sim["r"], "dates": np.arange(T), "truth": sim}
        K = st.radio("Regimes in the fitted model", [2, 3], horizontal=True)
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
        ui.break_it("Compare the two probability curves: the smoothed one switches *before* the filtered one could "
                    "have. A backtest that trades on smoothed (or Viterbi) regimes is using the future, which is the "
                    "classic look-ahead bias. Only the filtered curve is honest.")
        ui.in_practice("Regime-switching models are used for risk management, volatility targeting, stress "
                       "testing and macro nowcasting (recession probabilities). The filtered probability is the "
                       "number a live system would actually act on.")

# --------------------------------------------------------------------------- volatility
with t_vol:
    st.markdown(
        r"""
Returns themselves are nearly uncorrelated, but their **size** is not: calm days follow calm days and wild days
follow wild days. GARCH(1,1) models the conditional variance

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,$$

and its parameters are found by **maximum likelihood**: maximise $\sum_t \log\mathcal N(r_t;\mu,\sigma_t^2)$,
exactly the L1 recipe applied to a recursive model. $\alpha+\beta$ is the *persistence*. Forecasts revert to the
long-run variance $\omega/(1-\alpha-\beta)$ at that rate.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        vdata = _data_source("vol")
        if vdata == "sim":
            a_t = st.slider("True α (reaction to shocks)", 0.0, 0.3, 0.08, 0.01)
            b_t = st.slider("True β (memory)", 0.5, 0.98, 0.90, 0.01)
            if a_t + b_t >= 0.999:
                st.warning("α + β must be below 1 for a stationary process; β was reduced.")
                b_t = 0.998 - a_t
            Tv = st.slider("Trading days", 500, 5000, 2500, 250, key="vol_T")
            vseed = st.number_input("Random seed", 0, 9999, 4, key="vol_seed")
            rv, s2_true = finance.garch11_simulate(Tv, 0.03, 0.02, a_t, b_t, np.random.default_rng(int(vseed)))
            vdata = {"r": rv, "dates": np.arange(Tv), "truth": {"sigma2": s2_true, "alpha": a_t, "beta": b_t}}
        horizon = st.slider("Forecast horizon (days)", 10, 250, 60)
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
        ui.break_it("Set α = 0 (no clustering): squared returns stop being autocorrelated and β becomes poorly "
                    "determined. The likelihood surface is flat in β, which is the CRLB idea from L1 again: no "
                    "information, no precision.")

# --------------------------------------------------------------------------- hedge ratio
with t_hedge:
    st.markdown(
        r"""
Two related assets (say two companies in the same industry) satisfy $y_t \approx \beta_t x_t + \alpha_t$. A
**hedge ratio** $\beta_t$ says how many units of $x$ offset one unit of $y$, but $\beta_t$ drifts. Treat
$[\beta_t,\alpha_t]$ as the **hidden state** of a random walk and each day's prices as a measurement with
$H_t = [x_t,\;1]$. That's a Kalman filter whose observation matrix changes every step. The innovation
$y_t - H_t\hat\theta_t^-$ is the **spread**, and dividing by $\sqrt{S_t}$ gives its z-score.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        Th = st.slider("Days", 300, 3000, 1200, 100)
        b0 = st.slider("Initial true β", 0.5, 3.0, 1.5, 0.1)
        jump = st.slider("Sudden change in β", -1.0, 1.0, 0.5, 0.1)
        at = st.slider("Change happens at (fraction of sample)", 0.1, 0.9, 0.6, 0.05)
        svol = st.slider("Spread noise", 0.1, 2.0, 0.5, 0.1)
        use_mle = st.toggle("Fit δ and R by maximum likelihood", True,
                            help="Maximise the likelihood of the innovations (the L1 recipe applied to a filter)")
        delta = st.select_slider("Kalman δ (how fast β may drift)", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 1e-4,
                                 disabled=use_mle)
        win = st.slider("Rolling-OLS window (days)", 20, 250, 60, 10)
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
    ui.break_it("Turn off the MLE fit and set δ = 1e-7: β is assumed nearly constant, so the filter takes hundreds "
                "of days to accept the jump. δ = 1e-2: β chases noise. Also watch '|z| > 2': with hand-set noise "
                "levels the z-scores are miscalibrated, and with the MLE fit they come out close to N(0, 1), which "
                "is how you check a filter on real data where the truth is unknown.")
    ui.in_practice("Time-varying regressions with Kalman filters are used for dynamic hedging, factor exposures "
                   "(time-varying CAPM β), and statistical arbitrage. The same structure estimates channel taps in "
                   "communications. It's RLS from L2 with explicit process noise.")
