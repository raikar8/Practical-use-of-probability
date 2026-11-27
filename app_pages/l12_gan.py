import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import generative as gen

ui.lab_intro(
    "l12",
    goal="Watch a generator and a discriminator train against each other, first in 1-D where every curve is "
         "visible, then in 2-D where you can make the classic failure, mode collapse, happen and fix it.",
    on_screen=[
        "**The duel in 1-D:** the data, the generator's distribution and the discriminator's opinion D(x), "
        "frame by frame (Theory §1–4)",
        "**2-D: modes & collapse:** generated points over the discriminator's map of the plane, with a mode "
        "counter (Theory §5)",
    ],
    start_here="In *The duel in 1-D*, press ▶ and watch the orange D(x) curve: it starts confident and ends flat "
               "near 0.5, when it can no longer tell real from fake.",
)

t_1d, t_2d, t_code = st.tabs(["The duel in 1-D", "2-D: modes & collapse", "Code"])

# --------------------------------------------------------------------------- 1-D
with t_1d:
    st.markdown(
        r"""
**The situation.** Real data: a two-peaked 1-D distribution (grey). The generator $G$ (a small network, 1 noise
number in, 1 number out) starts out producing a narrow blob near zero (blue). The discriminator $D$ (orange)
outputs the probability that a value is real. They take turns improving (Theory §4).
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        steps = st.select_slider("Training steps", [300, 1000, 2000, 4000], 2000, help="Rounds of D step + G step.")
        lr_g = st.select_slider("Generator learning rate", [1e-4, 3e-4, 1e-3, 3e-3], 1e-3, key="lrg1",
                                format_func=lambda v: f"{v:g}", help="How fast the forger adapts.")
        lr_d = st.select_slider("Discriminator learning rate", [1e-4, 3e-4, 1e-3, 3e-3], 1e-3, key="lrd1",
                                format_func=lambda v: f"{v:g}", help="How fast the detective adapts.")
        sep = st.slider("Distance between the peaks", 1.0, 8.0, 4.0, 0.5, help="Further apart = harder to cover "
                                                                               "both.")
        gseed = st.number_input("Random seed", 0, 9999, 0, key="g1_seed", help="Different initial weights.")
    means = (-sep / 2, sep / 2)
    xg = np.linspace(-sep / 2 - 3, sep / 2 + 3, 200)

    @st.cache_data(show_spinner="Training the GAN…")
    def _gan1(steps, lr_g, lr_d, sep, seed):
        r = gen.train_gan(lambda n, rng: gen.mog_1d(n, rng, means, 0.4), 1, steps=steps, z_dim=1, hidden=32,
                          lr_g=lr_g, lr_d=lr_d, seed=seed, snapshots=40, grid=xg, n_show=3000)
        bins = np.linspace(xg[0], xg[-1], 81)
        hists = [np.histogram(s_[:, 0], bins, density=True)[0] for s_ in r["snaps"]]
        final = r["snaps"][-1][:, 0]
        ed = gen.energy_distance(gen.mog_1d(3000, np.random.default_rng(5), means, 0.4)[:, None], final[:, None])
        return {"hists": hists, "bins": bins, "d": r["dsnaps"], "k": r["snap_steps"], "dr": r["d_real"],
                "df": r["d_fake"], "ed": ed, "final": final}
    res = _gan1(steps, lr_g, lr_d, sep, int(gseed))
    with c2:
        ctr = 0.5 * (res["bins"][1:] + res["bins"][:-1])
        dens = 0.5 * (np.exp(-0.5 * ((xg - means[0]) / 0.4) ** 2) + np.exp(-0.5 * ((xg - means[1]) / 0.4) ** 2)) / (
            0.4 * np.sqrt(2 * np.pi))
        pd = 0.5 * (np.exp(-0.5 * ((ctr - means[0]) / 0.4) ** 2) + np.exp(-0.5 * ((ctr - means[1]) / 0.4) ** 2)) / (
            0.4 * np.sqrt(2 * np.pi))

        def fr(h, dd):
            dstar = pd / (pd + h + 1e-12)
            return [go.Scatter(x=ctr, y=h, line=dict(color=ui.C[1], width=3, shape="hvh"), name="generator p_g"),
                    go.Scatter(x=xg, y=dd, line=dict(color=ui.C[2], width=3), name="discriminator D(x)", yaxis="y2"),
                    go.Scatter(x=ctr, y=dstar, line=dict(color=ui.C[2], width=1.5, dash="dot"),
                               name="ideal D* for this p_g (eq. 2)", yaxis="y2")]
        fig = go.Figure()
        fig.add_scatter(x=xg, y=dens, fill="tozeroy", fillcolor=ui.rgba(ui.C["truth"], 0.3),
                        line=dict(color=ui.C["truth"]), name="real data p_data")
        for t_ in fr(res["hists"][-1], res["d"][-1]):
            fig.add_trace(t_)
        fig.frames = [go.Frame(data=fr(h, dd), traces=[1, 2, 3], name=str(k))
                      for h, dd, k in zip(res["hists"], res["d"], res["k"])]
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], title="D(x)", showgrid=False))
        fig.update_yaxes(range=[0, 1.6], title="density")
        ui.play_controls(fig, [str(k) for k in res["k"]], 150, "step ", start_at_end=True)
        ui.show(ui.style(fig, "Real data, generator and discriminator during training", "x", None, height=420))
        sm = np.ones(30) / 30
        f2 = go.Figure()
        f2.add_scatter(y=np.convolve(res["dr"], sm, "valid"), name="mean D(real)", line=dict(color=ui.C[3]))
        f2.add_scatter(y=np.convolve(res["df"], sm, "valid"), name="mean D(fake)", line=dict(color=ui.C[8]))
        f2.add_hline(y=0.5, line=dict(color=ui.C["truth"], dash="dot"))
        f2.update_yaxes(range=[0, 1])
        ui.show(ui.style(f2, "How sure the discriminator is", "step", "D", height=220))
        m = st.columns(3)
        m[0].metric("Final mean D(real)", f"{np.mean(res['dr'][-100:]):.2f}")
        m[1].metric("Final mean D(fake)", f"{np.mean(res['df'][-100:]):.2f}")
        m[2].metric("Energy distance to data", f"{res['ed']:.3f}", help="0 = identical distributions.")
    ui.takeaway("The discriminator's curve shows the generator where it's wrong: low D(x) where fakes pile up, high "
                "D(x) where real data is missing. The generator moves its samples up that slope until D is flat at "
                "0.5. Note the thin bridge of samples between the peaks: a continuous G can't jump from one peak to "
                "the other without passing through the gap.")
    ui.controls([
        ("Training steps", "–", "rounds of the game", "–"),
        ("Generator learning rate", "Adam step for G", "how fast the forger adapts", "faster, but can overshoot"),
        ("Discriminator learning rate", "Adam step for D", "how fast the detective adapts", "sharper D; too "
         "sharp starves G"),
        ("Distance between the peaks", "–", "how separated the two modes are", "a bigger gap to bridge"),
    ])
    ui.experiments([
        ("Defaults, press ▶.", "D(real) and D(fake) both end close to 0.5, and the blue histogram covers both "
         "peaks.", "The equilibrium of Theory §3: D* = 1/2 when p_g = p_data."),
        ("Compare the orange solid and dotted curves mid-training.", "The trained D roughly follows the ideal "
         "D* = p_data / (p_data + p_g).", "The discriminator approximates Bayes' rule between real and fake."),
        ("Set the distance to 8.", "Both peaks are still covered in the end (about half the samples each), but "
         "the bridge between them is longer.", "One continuous G must stretch across the whole gap."),
    ])

# --------------------------------------------------------------------------- 2-D
with t_2d:
    st.markdown(
        r"""
**The situation.** 2-D datasets, a generator with 2 hidden layers of 64 units, and a discriminator of the same
size. Each frame shows 600 generated points (the same noise inputs every frame, so you can follow them) on top of
the discriminator's opinion of the plane (darker = "looks real"). The **ring** of 8 Gaussians is the standard
test for **mode collapse**.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        ds = st.selectbox("Dataset", gen.DATASETS_2D, help="The real 2-D data.")
        st2 = st.select_slider("Training steps", [500, 1000, 2000, 3000], 2000, key="st2",
                               help="Rounds of D step + G step.")
        lg2 = st.select_slider("Generator learning rate", [2e-4, 1e-3, 5e-3], 1e-3, key="lrg2",
                               format_func=lambda v: f"{v:g}", help="How fast the forger adapts.")
        ld2 = st.select_slider("Discriminator learning rate", [2e-4, 1e-3, 5e-3], 1e-3, key="lrd2",
                               format_func=lambda v: f"{v:g}", help="How fast the detective adapts.")
        zd = st.select_slider("Noise dimensions", [1, 2, 4], 2, help="Size of z.")
        seed2 = st.number_input("Random seed", 0, 9999, 0, key="g2_seed", help="Different initial weights.")
    g1 = np.linspace(-3, 3, 50)
    GX, GY = np.meshgrid(g1, g1)
    grid2 = np.c_[GX.ravel(), GY.ravel()]

    @st.cache_data(show_spinner="Training the GAN (a few seconds)…")
    def _gan2(ds, steps, lg, ld, zd, seed):
        r = gen.train_gan(lambda n, rng: gen.toy_2d(ds, n, rng), 2, steps=steps, z_dim=zd, lr_g=lg, lr_d=ld,
                          seed=seed, snapshots=30, grid=grid2)
        S = r["G"](np.random.default_rng(9).standard_normal((2000, zd)))
        real = gen.toy_2d(ds, 2000, np.random.default_rng(5))
        cov = gen.ring_coverage(S) if ds == "ring" else None
        return {"snaps": r["snaps"], "d": r["dsnaps"], "k": r["snap_steps"], "dr": r["d_real"], "df": r["d_fake"],
                "ed": gen.energy_distance(real, S), "cov": cov, "real": real[:800]}
    res2 = _gan2(ds, st2, lg2, ld2, zd, int(seed2))
    with c2:
        def fr2(S, dd):
            return [go.Heatmap(x=g1, y=g1, z=dd.reshape(50, 50), colorscale=[[0, "#ffffff"], [1, "#c9c9c9"]],
                               zmin=0, zmax=1, showscale=False, hoverinfo="skip"),
                    go.Scatter(x=S[:, 0], y=S[:, 1], mode="markers", marker=dict(size=4, color=ui.C[1], opacity=0.7),
                               name="generated")]
        fig = go.Figure(data=fr2(res2["snaps"][-1], res2["d"][-1]))
        fig.add_scatter(x=res2["real"][:, 0], y=res2["real"][:, 1], mode="markers", name="real",
                        marker=dict(size=3, color=ui.C[2], opacity=0.35))
        fig.frames = [go.Frame(data=fr2(S, dd), traces=[0, 1], name=str(k))
                      for S, dd, k in zip(res2["snaps"], res2["d"], res2["k"])]
        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[-3, 3], scaleanchor="x")
        ui.play_controls(fig, [str(k) for k in res2["k"]], 200, "step ", start_at_end=True)
        ui.show(ui.style(fig, "Generated (blue) vs real (orange); shading = D(x)", height=520))
        m = st.columns(4)
        if res2["cov"] is not None:
            m[0].metric("Ring modes covered", f"{res2['cov'][0]} / 8", help="Modes receiving ≥ 2% of samples.")
            m[1].metric("Samples near a real mode", f"{res2['cov'][1]:.0%}", help="Within 3σ of one of the 8 centres.")
        m[2].metric("Energy distance to data", f"{res2['ed']:.3f}", help="0 = identical distributions.")
        m[3].metric("Final mean D(fake)", f"{np.mean(res2['df'][-100:]):.2f}")
    ui.takeaway("When the two players are balanced, the generator spreads over every mode and the shading fades "
                "to uniform. Let the generator learn much faster than the discriminator and it chases whatever "
                "the current discriminator likes: it collapses onto a few modes.")
    ui.controls([
        ("Dataset", r"$p_{\text{data}}$", "the real 2-D distribution", "–"),
        ("Generator / discriminator learning rate", "Adam steps", "relative speed of the players", "imbalance "
         "→ collapse or oscillation"),
        ("Noise dimensions", r"$\dim z$", "how many random numbers G starts from", "1 = G can only draw a curve"),
        ("Training steps", "–", "rounds of the game", "–"),
    ])
    ui.experiments([
        ("Ring, defaults, press ▶.", "All 8 modes covered, ≈ 90% of samples near a real mode.",
         "Balanced players: the discriminator keeps pointing at uncovered modes."),
        ("Ring: generator lr 5e-3, discriminator lr 2e-4.", "Only 2–3 modes covered, and fewer than 20% of samples "
         "near a real mode.", "Mode collapse: a fast generator over-reacts to a slow discriminator and chases a "
         "few modes."),
        ("Ring: noise dimensions = 1.", "Only about 4–6 modes, joined by streaks of bad samples; ≈ 35–55% near a "
         "mode (vs ≈ 90%).",
         "With 1-D noise, G draws a single continuous curve through the plane and must pass between clusters."),
        ("Try moons and spiral.", "Thin curved shapes are learned well (energy distance ≈ 0.01). Watch early "
         "frames: points first gather in the bulk, then spread along the arms.",
         "The discriminator first flags the biggest mismatch, then finer ones."),
    ])
    ui.in_practice("GAN-trained vocoders (HiFi-GAN and successors) generate the waveform in most text-to-speech "
                   "systems; adversarial losses sharpen speech enhancement and neural audio codecs; image GANs "
                   "power super-resolution and fast style transfer.")

with t_code:
    st.markdown("The exact code this page runs (from `ssp/core/generative.py`):")
    ui.source(gen.train_gan)
