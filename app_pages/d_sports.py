import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui
from ssp.sims import sports as sp

ui.header("Domain demo · Sports", "Skill ratings and live win probability",
          "A result is a noisy measurement of a skill difference. How should ratings react, and how should a live "
          "win probability move as goals go in?")

t_rate, t_live = st.tabs(["Skill ratings: Elo vs Kalman", "Live win probability"])

# --------------------------------------------------------------------------- ratings
with t_rate:
    st.markdown(
        r"""
Each team has a hidden skill $\theta_i$ that drifts (form, injuries, transfers). A game between home $h$ and away $a$
ends in a home win with probability $\sigma(\theta_h-\theta_a+\text{home advantage})$ (no draws, as in
basketball, tennis or esports).

* **Elo:** $\theta \mathrel{+}= K(\text{result}-\text{expected})$. A filter with a **fixed** gain $K$.
* **Kalman (Glicko/TrueSkill idea):** each team has a belief $\mathcal N(\mu_i, v_i)$. Between rounds $v_i$ grows by
  $q$ (skills drift). After a game, an EKF update on the logistic measurement gives a **per-team gain**
  $K_i \approx \frac{v_i\,g}{g^2(v_h+v_a)+g}$ with $g=p(1-p)$. Uncertain teams move a lot; well-known teams barely move.

At the start of season 2 a strong **newcomer** replaces team 0, and both systems start it at the average.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        n_teams = st.select_slider("Teams", [8, 12, 16, 20, 24], 16)
        seasons = st.slider("Seasons", 1, 5, 3)
        drift = st.slider("Skill drift per round (σ)", 0.0, 0.15, 0.04, 0.01)
        newc = st.slider("Newcomer's true skill (league σ = 1)", -1.0, 3.0, 1.5, 0.1)
        K = st.select_slider("Elo K", [0.02, 0.05, 0.1, 0.2, 0.3, 0.5], 0.2)
        q = st.select_slider("Kalman q (skill drift variance per round)", [0.0001, 0.001, 0.002, 0.005, 0.02], 0.002)
        lseed = st.number_input("Random seed", 0, 9999, 4, key="lg_seed")
    cfg = sp.LeagueConfig(n_teams=n_teams, seasons=seasons, drift=drift, new_team_skill=newc, seed=int(lseed))
    lg = sp.simulate_league(cfg)
    o = lg["outcome"]
    e = sp.elo(lg, n_teams, K, cfg.home_adv)
    kr = sp.kalman_ratings(lg, n_teams, q, cfg.home_adv)
    preds = {"Elo": e["pred"], "Kalman": kr["pred"], "Oracle (true skills)": lg["true_p"]}
    cols = {"Elo": ui.C[2], "Kalman": ui.C[1], "Oracle (true skills)": ui.C["truth"]}
    with c2:
        win = max(n_teams * 3, 40)
        fig = go.Figure()
        for nm, p in preds.items():
            ll = pd.Series(sp.log_loss(p, o)).rolling(win, min_periods=win // 2).mean()
            fig.add_scatter(x=np.arange(o.size), y=ll, name=nm, line=dict(color=cols[nm]))
        fig.add_hline(y=np.log(2), line=dict(color=ui.C["truth"], dash="dot"), annotation_text="coin flip")
        if lg["new_team_round"] is not None:
            g0 = int(np.argmax(lg["round"] >= lg["new_team_round"]))
            fig.add_vline(x=g0, line=dict(color=ui.C["truth"], dash="dash"), annotation_text="newcomer arrives")
        ui.show(ui.style(fig, f"Prediction quality: rolling log-loss over {win} games (lower is better)",
                         "game number", "log-loss", height=320))
        team = st.selectbox("Team to inspect", list(range(n_teams)), format_func=lambda i: f"Team {i}" +
                            (" (newcomer from season 2)" if i == 0 and seasons > 1 else ""))
        S = lg["skills"] - lg["skills"].mean(axis=1, keepdims=True)
        E = e["ratings"] - e["ratings"].mean(axis=1, keepdims=True)
        M = kr["ratings"] - kr["ratings"].mean(axis=1, keepdims=True)
        rr = np.arange(S.shape[0])
        f2 = go.Figure()
        sd = kr["sd"][:, team]
        f2.add_scatter(x=np.r_[rr, rr[::-1]], y=np.r_[M[:, team] + 2 * sd, (M[:, team] - 2 * sd)[::-1]], fill="toself",
                       fillcolor=ui.rgba(ui.C[1], 0.15), line=dict(width=0), name="Kalman ±2σ")
        f2.add_scatter(x=rr, y=S[:, team], name="true skill", line=dict(color=ui.C["truth"], width=4))
        f2.add_scatter(x=rr, y=E[:, team], name="Elo", line=dict(color=ui.C[2]))
        f2.add_scatter(x=rr, y=M[:, team], name="Kalman", line=dict(color=ui.C[1]))
        ui.show(ui.style(f2, f"Team {team}: rating vs hidden skill (relative to the league average)", "round",
                         "skill", height=320))
        mine = (lg["home"] == team) | (lg["away"] == team)
        f3 = go.Figure()
        kk = np.where(lg["home"][mine] == team, kr["gain_h"][mine], kr["gain_a"][mine])
        f3.add_scatter(x=lg["round"][mine], y=kk, mode="markers", name=f"Kalman gain for team {team}",
                       marker=dict(color=ui.C[1], size=6))
        f3.add_hline(y=K, line=dict(color=ui.C[2]), annotation_text=f"Elo K = {K}")
        ui.show(ui.style(f3, "Elo uses one fixed gain; the Kalman gain shrinks as evidence accumulates",
                         "round", "gain", height=240))
    cA, cB = st.columns([1.3, 1])
    with cA:
        last = S.shape[0] - 1
        tbl = pd.DataFrame({
            nm: {"log-loss": sp.log_loss(p, o).mean(), "Brier": np.mean((p - o) ** 2)} for nm, p in preds.items()}).T
        tbl["rating RMSE vs truth"] = [np.sqrt(np.mean((E - S)[5:] ** 2)), np.sqrt(np.mean((M - S)[5:] ** 2)), 0.0]
        tbl["final rank correlation"] = [stats.spearmanr(E[last], S[last])[0], stats.spearmanr(M[last], S[last])[0], 1.0]
        st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    with cB:
        f4 = go.Figure()
        f4.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.C["truth"], dash="dot"),
                       name="perfect calibration")
        for nm in ("Elo", "Kalman"):
            mp, fr, cnt = sp.calibration_bins(preds[nm], o, 8)
            f4.add_scatter(x=mp, y=fr, mode="lines+markers", name=nm, line=dict(color=cols[nm]),
                           marker=dict(size=np.clip(cnt / cnt.max() * 16, 6, 16)))
        f4.update_layout(hovermode="closest")
        ui.show(ui.style(f4, "Calibration: when we say 70%, do they win 70%?", "predicted P(home win)",
                         "observed frequency", height=300))
    ui.takeaway("Elo with a small K is stable but takes a whole season to believe the newcomer. A large K reacts "
                "fast but makes every established team's rating noisy. The Kalman filter gets both, because its "
                "gain comes from its uncertainty. Averaged over many simulated leagues, its log-loss and rating "
                "error are lower than the best fixed-K Elo.")
    ui.break_it("Kalman q = 0.0001 with drift 0.1: the filter thinks skills are frozen, so its gains shrink to "
                "nothing and it stops tracking form. That's the 'Q too small' failure from L3, in sport.")
    ui.in_practice("Elo is used in chess and many sports rankings. Glicko adds a rating deviation (exactly v here). "
                   "TrueSkill (Xbox matchmaking) extends the Gaussian belief to teams and multiplayer games. Win "
                   "probabilities from such ratings drive matchmaking and seeding.")

# --------------------------------------------------------------------------- live
with t_live:
    st.markdown(
        r"""
**Model (football):** goals are Poisson arrivals with pre-match rates $\lambda_H, \lambda_A$ per 90 minutes. At minute
$m$ with score $(g_H, g_A)$, the remaining goals are $\text{Poisson}(\lambda\,\tfrac{90-m}{90})$ for each side, and
$P(\text{home win}) = P(g_H + X > g_A + Y)$.

**Learning during the match (optional):** put a Gamma prior on each rate, worth a few games of evidence.
Gamma is conjugate to Poisson, so after $k$ goals in $m$ minutes the posterior is still Gamma and the remaining
goals become **negative binomial**. A team that keeps scoring is believed to be stronger than expected.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        lh = st.slider("Home expected goals (pre-match)", 0.3, 3.5, 1.6, 0.1)
        la = st.slider("Away expected goals (pre-match)", 0.3, 3.5, 1.1, 0.1)
        learn = st.toggle("Learn the scoring rates during the match", False)
        prior_min = st.select_slider("Prior strength (minutes of evidence)", [90, 180, 270, 540, 900], 270,
                                     disabled=not learn)
        rc = st.slider("Red card minute (0 = none)", 0, 89, 0)
        rc_side = st.radio("Red card for", ["away", "home"], horizontal=True, disabled=rc == 0)
        mseed = st.number_input("Match seed", 0, 9999, 4)
    rcm = float(rc) if rc else None
    g = sp.simulate_match(lh, la, np.random.default_rng(int(mseed)), rcm, rc_side)
    minutes, curve = sp.live_curve(g, lh, la, learn, prior_min, rcm, rc_side)
    with c2:
        fig = go.Figure()
        for j, (nm, col) in enumerate((("home win", ui.C[1]), ("draw", ui.C["truth"]), ("away win", ui.C[2]))):
            fig.add_scatter(x=minutes, y=curve[:, j], name=nm, line=dict(color=col, width=2.5))
        for side, col in (("home", ui.C[1]), ("away", ui.C[2])):
            for tg in g[side]:
                gh = sum(x <= tg for x in g["home"])
                ga = sum(x <= tg for x in g["away"])
                fig.add_vline(x=tg, line=dict(color=col, dash="dot"))
                fig.add_annotation(x=tg, y=1.06, text=f"{gh}–{ga}", showarrow=False, font=dict(size=12))
        if rcm is not None:
            fig.add_vline(x=rcm, line=dict(color=ui.C[8], width=3), annotation_text=f"red card ({rc_side})",
                          annotation_position="bottom right")
        fig.update_yaxes(range=[0, 1.1], tickformat=".0%", tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        fig.update_xaxes(range=[0, 90])
        ui.show(ui.style(fig, f"Live probabilities (final score {len(g['home'])}–{len(g['away'])})", "minute",
                         "probability", height=380))
        st.markdown("**Are these probabilities honest?** Simulate many matches and check calibration at one minute.")
        cc1, cc2, cc3 = st.columns(3)
        at_min = cc1.slider("Check at minute", 0, 85, 60, 5)
        n_sim = cc2.select_slider("Simulated matches", [500, 1000, 3000], 1000)
        spread = cc3.slider("Pre-match rate error (log σ)", 0.0, 0.8, 0.0, 0.1,
                            help="0 = the pre-match rates are exactly right. Larger = real rates differ, as in reality.")

        @st.cache_data(show_spinner="Simulating matches…")
        def _calib(lh, la, at_min, n_sim, spread):
            out = {}
            for lrn in (False, True):
                p, y = sp.calibration_experiment(lh, la, at_min, n_sim, np.random.default_rng(7), lrn, spread)
                out[lrn] = (p, y)
            return out

        cal = _calib(lh, la, at_min, n_sim, spread)
        f2 = go.Figure()
        f2.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.C["truth"], dash="dot"),
                       name="perfect calibration")
        rows = {}
        for lrn, col, nm in ((False, ui.C[2], "fixed rates"), (True, ui.C[1], "learned rates")):
            p, y = cal[lrn]
            mp, fr, cnt = sp.calibration_bins(p, y, 8)
            f2.add_scatter(x=mp, y=fr, mode="lines+markers", name=nm, line=dict(color=col))
            rows[nm] = {"Brier score": np.mean((p - y) ** 2), "log-loss": sp.log_loss(p, y).mean()}
        f2.update_layout(hovermode="closest")
        ui.show(ui.style(f2, f"Calibration of P(home win) at minute {at_min}", "predicted", "observed", height=300))
        st.dataframe(pd.DataFrame(rows).T.style.format("{:.4f}"), width="stretch")
    ui.takeaway("The curve jumps at goals and drifts the rest of the time: every scoreless minute favours the team "
                "that's ahead, because there's less time to come back. When the pre-match rates are right, the plain "
                "Poisson model is already calibrated. When they're off, learning the rates in-game helps only "
                "slightly: goals are rare, so the prior still dominates.")
    ui.break_it("Prior strength 90 minutes with learning on: one early goal makes the model think that team is "
                "unstoppable. It overreacts, like a small-sample MLE (L1). A red card shifts the rates for the rest "
                "of the match, and the curve bends immediately even without a goal.")
    ui.in_practice("Broadcast win-probability graphics, sports analytics and in-play pricing are built on models of "
                   "this shape, usually with richer inputs (expected-goals from shots, possession, cards).")
