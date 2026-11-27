import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ssp import ui
from ssp.sims import sports as sp

ui.lab_intro(
    "sports",
    goal="Treat sport as an estimation problem: rate hidden team strengths from noisy results (Elo vs a Kalman "
         "rating), and compute live win probabilities that update minute by minute, then check that they're honest.",
    on_screen=[
        "**Skill ratings:** prediction quality over a simulated league, one team's rating vs its true skill, and how "
        "big each rating update is (Theory §1–3)",
        "**Live win probability:** home/draw/away probabilities through a simulated football match, and a calibration "
        "check over thousands of matches (Theory §4)",
    ],
    start_here="In Skill ratings, pick *Team 0*: it's replaced by a strong newcomer at the start of season 2. "
               "Watch how fast Elo (orange) and Kalman (blue) climb toward its true skill (grey).",
)

t_rate, t_live = st.tabs(["Skill ratings: Elo vs Kalman", "Live win probability"])

# --------------------------------------------------------------------------- ratings
with t_rate:
    st.markdown(
        r"""
**The situation.** A league of teams with hidden skills that drift over the seasons. Every game is a noisy
yes/no measurement of a skill *difference* (no draws, as in basketball or esports). Two rating systems watch
the results:

* **Elo:** a fixed-size step toward the surprise, $\theta \mathrel{+}= K(\text{result} - \text{expected})$ (Theory eq. 2)
* **Kalman:** a mean *and* an uncertainty per team; the step size comes from the uncertainty (Theory eq. 3)

Both predict each game *before* seeing it, and we score those predictions. At the start of season 2 a strong
**newcomer** replaces team 0, and both systems start it at the league average.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        st.markdown("**The league**")
        n_teams = st.select_slider("Teams", [8, 12, 16, 20, 24], 16, help="Teams playing a double round-robin each season.")
        seasons = st.slider("Seasons", 1, 5, 3, help="Number of seasons simulated.")
        drift = st.slider("Skill drift per round (σ)", 0.0, 0.15, 0.04, 0.01,
                          help="How much true skills change between rounds (form, injuries). League spread is 1.")
        newc = st.slider("Newcomer's true skill (league σ = 1)", -1.0, 3.0, 1.5, 0.1,
                         help="True skill of the team that replaces team 0 in season 2. 1.5 ≈ top of the league.")
        st.markdown("**The rating systems**")
        K = st.select_slider("Elo K", [0.02, 0.05, 0.1, 0.2, 0.3, 0.5], 0.2,
                             help="Elo's fixed step size: how much one result moves a rating.")
        q = st.select_slider("Kalman q (skill drift variance per round)", [0.0001, 0.001, 0.002, 0.005, 0.02], 0.002,
                             help="How much the Kalman rating expects skills to drift between rounds (its Q).")
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
        team = st.selectbox("Team to inspect", list(range(n_teams)), help="Whose rating history to plot.",
                            format_func=lambda i: f"Team {i}" +
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
    ui.controls([
        ("Teams, Seasons", "–", "league size and length", "more data per team"),
        ("Skill drift", "random walk of θ", "how much true skills change", "ratings must keep adapting"),
        ("Newcomer's true skill", "–", "the hidden strength of team 0 from season 2", "a bigger surprise to learn"),
        ("Elo K", r"$K$ (eq. 2)", "fixed step size", "faster but noisier ratings"),
        ("Kalman q", r"$q$ (§3)", "expected skill drift", "larger gains, faster adaptation"),
    ])
    ui.experiments([
        ("Defaults: read the table.", "Log-loss: Kalman ≈ 0.535, Elo ≈ 0.547, oracle (true skills) ≈ 0.493, coin ≈ 0.693.",
         "The Kalman rating adapts its step size per team, so it predicts better than any single fixed K."),
        ("Pick Team 0 and look at the gain plot.",
         "The Kalman gain is large for the newcomer's first games, then shrinks toward Elo's line.",
         "The newcomer starts uncertain, so each result is informative; as evidence accumulates, v (and the gain) falls."),
        ("Set Elo K = 0.05.", "Elo's log-loss worsens to ≈ 0.60, and its newcomer line crawls upward all season.",
         "A small fixed gain is stable but can't catch up with large changes."),
        ("Set Elo K = 0.5 and inspect an established team.", "Elo's line jumps up and down every round.",
         "A large fixed gain chases every lucky or unlucky result."),
    ])
    ui.in_practice("Elo is used in chess and many sports rankings. Glicko adds a rating deviation (exactly v here). "
                   "TrueSkill (Xbox matchmaking) extends the Gaussian belief to teams and multiplayer games. Win "
                   "probabilities from such ratings drive matchmaking and seeding.")

# --------------------------------------------------------------------------- live
with t_live:
    st.markdown(
        r"""
**The situation.** A football match between teams expected to score $\lambda_H$ and $\lambda_A$ goals per 90
minutes. Goals arrive at random (a Poisson process). At each minute we compute the probability of a home win, a
draw and an away win from the current score and the time left (Theory eq. 4). Optionally the model **learns**
each team's scoring rate during the match (Theory eq. 5), and a red card changes both rates.
"""
    )
    c1, c2 = st.columns([1, 2.4])
    with c1:
        lh = st.slider("Home expected goals (pre-match)", 0.3, 3.5, 1.6, 0.1,
                       help="Pre-match estimate of the home team's goals per 90 minutes (λ_H).")
        la = st.slider("Away expected goals (pre-match)", 0.3, 3.5, 1.1, 0.1, help="Same for the away team (λ_A).")
        learn = st.toggle("Learn the scoring rates during the match", False,
                          help="Update λ with a Gamma prior as goals happen, instead of keeping the pre-match values.")
        prior_min = st.select_slider("Prior strength (minutes of evidence)", [90, 180, 270, 540, 900], 270,
                                     disabled=not learn)
        rc = st.slider("Red card minute (0 = none)", 0, 89, 0,
                       help="From this minute, the penalised team's rate × 0.65 and the opponent's × 1.3.")
        rc_side = st.radio("Red card for", ["away", "home"], horizontal=True, disabled=rc == 0)
        mseed = st.number_input("Match seed", 0, 9999, 4, help="A different simulated match.")
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
        at_min = cc1.slider("Check at minute", 0, 85, 60, 5,
                            help="Record the predicted P(home win) at this minute in every simulated match.")
        n_sim = cc2.select_slider("Simulated matches", [500, 1000, 3000], 1000,
                                  help="More matches = a less noisy calibration curve.")
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
    ui.controls([
        ("Expected goals", r"$\lambda_H, \lambda_A$", "pre-match scoring rates", "that team's win probability rises"),
        ("Learn the scoring rates", "Gamma–Poisson (eq. 5)", "update rates from goals scored", "reacts to surprises"),
        ("Prior strength", r"$m_0$ minutes", "how much the pre-match estimate counts", "less reaction to early goals"),
        ("Red card minute / side", "rate × 0.65 / × 1.3", "a sending-off", "the curve bends without any goal"),
        ("Check at minute, Simulated matches, Pre-match rate error", "–", "calibration experiment",
         "tests whether the probabilities are honest"),
    ])
    ui.experiments([
        ("Defaults: follow the curves.", "Before the first goal, home ≈ 49%, draw ≈ 25%, away ≈ 26%; each home goal "
         "makes a jump, and the curves drift between goals.",
         "Every scoreless minute leaves less time to change the result, so the current leader gains."),
        ("Read the curves around minute 80 at 0–0 (try other match seeds).", "A draw becomes the most likely outcome "
         "(≈ 76% at 0–0 with 10 minutes left).",
         "Few goals are expected in 10 minutes; 'no more goals' is the single most likely future."),
        ("Turn learning on with prior strength 90.", "After an early goal the scorer's win probability rises more "
         "than without learning.",
         "A weak prior plus one goal = a big upward revision of that team's rate: small-sample overreaction (Level 1)."),
        ("Calibration: set 'Pre-match rate error' to 0.", "Both curves lie near the diagonal.",
         "If the model's assumptions hold, its probabilities are honest: 70% predictions win ~70% of the time."),
    ])
    ui.in_practice("Broadcast win-probability graphics, sports analytics and in-play pricing are built on models of "
                   "this shape, usually with richer inputs (expected-goals from shots, possession, cards).")
