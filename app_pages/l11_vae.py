import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import generative as gen

ui.lab_intro(
    "l11",
    goal="Take a variational autoencoder apart: first the ELBO for a single data point where everything can be "
         "computed exactly, then a real VAE trained live on handwritten digits, then use it to generate and morph "
         "digits.",
    on_screen=[
        "**The ELBO up close:** a one-number latent code, the true posterior, and a Gaussian guess you control; "
        "see log p(x) = ELBO + gap (Theory §3–4)",
        "**A VAE on digits:** the 2-D map of codes the encoder learns, what the decoder draws at every point of "
        "that map, and the two parts of the loss (Theory §5–6)",
        "**Generate & interpolate:** brand-new digits from random codes, and smooth morphs between two digits "
        "(Theory §6)",
    ],
    start_here="In *The ELBO up close*, move the *q mean* slider and watch the gap: it's smallest when the blue "
               "curve sits on the orange one. Then tick *Snap q to the best Gaussian*.",
)

DIGIT_COL = [ui.C[i] for i in (1, 2, 3, 4, 5, 6, 7, 8)] + ["#8a8a8a", "#3d3d3d"]
t_elbo, t_train, t_gen, t_code = st.tabs(["The ELBO up close", "A VAE on digits", "Generate & interpolate", "Code"])

# --------------------------------------------------------------------------- ELBO toy
with t_elbo:
    st.markdown(
        r"""
**The situation.** The smallest possible VAE: a **single** latent number $z \sim \mathcal N(0, 1)$, a fixed
"decoder" that maps it to a point on a curve in 2-D, $f(z) = (0.6z^2 - 1,\; 0.35z)$, and Gaussian noise
$x = f(z) + \text{noise}$. We observe **one** point $x$ (the red star). Because $z$ is 1-D, the true posterior
$p(z \mid x)$ and $\log p(x)$ can be computed exactly. You play the encoder: choose the Gaussian
$q(z) = \mathcal N(m, s^2)$ (Theory eq. 3) and see the ELBO (eq. 4).
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        z0 = st.slider("The code that produced x", -2.5, 2.5, 1.5, 0.1,
                       help="x is placed at f(z) for this z. The model doesn't know it.")
        noise = st.select_slider("Decoder noise σ", [0.1, 0.2, 0.3, 0.45, 0.6, 0.8], 0.3,
                                 help="How noisy p(x | z) is. Large = many codes could explain x.")
        snap = st.toggle("Snap q to the best Gaussian", False, help="Find the (m, s) with the highest ELBO.")
        m = st.slider("q mean m", -3.0, 3.0, 0.0, 0.05, disabled=snap, help="The encoder's best-guess code.")
        s = st.slider("q std s", 0.03, 2.0, 1.0, 0.01, disabled=snap, help="The encoder's uncertainty.")
    x = gen.toy_decoder(np.array(z0))
    if snap:
        m, s = gen.toy_best_q(x, noise)
    r = gen.toy_elbo(x, m, s, noise)
    with c2:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                            subplot_titles=["data space: where codes land", "code space: beliefs about z"])
        zc = np.linspace(-3, 3, 200)
        fc = gen.toy_decoder(zc)
        fig.add_scatter(x=fc[:, 0], y=fc[:, 1], mode="lines", line=dict(color=ui.C["truth"], width=3),
                        name="decoder curve f(z)", row=1, col=1)
        zs = m + s * np.random.default_rng(0).standard_normal(60)
        fs = gen.toy_decoder(zs)
        fig.add_scatter(x=fs[:, 0], y=fs[:, 1], mode="markers", marker=dict(color=ui.C[1], size=6, opacity=0.6),
                        name="f(z) for z drawn from q", row=1, col=1)
        th = np.linspace(0, 2 * np.pi, 60)
        fig.add_scatter(x=x[0] + 2 * noise * np.cos(th), y=x[1] + 2 * noise * np.sin(th), mode="lines",
                        line=dict(color=ui.C[8], dash="dot", width=1), name="±2σ noise around x", row=1, col=1)
        fig.add_scatter(x=[x[0]], y=[x[1]], mode="markers", marker=dict(symbol="star", size=16, color=ui.C[8]),
                        name="observed x", row=1, col=1)
        fig.add_scatter(x=r["z"], y=np.exp(-0.5 * r["z"] ** 2) / np.sqrt(2 * np.pi), name="prior p(z)",
                        line=dict(color=ui.C["truth"], dash="dash"), row=1, col=2)
        fig.add_scatter(x=r["z"], y=r["posterior"], name="true posterior p(z | x)",
                        line=dict(color=ui.C[2], width=3), row=1, col=2)
        fig.add_scatter(x=r["z"], y=r["q"], name="your q(z)", line=dict(color=ui.C[1], width=3), row=1, col=2)
        fig.update_xaxes(range=[-3.5, 3.5], title="z", row=1, col=2)
        fig.update_xaxes(range=[-1.5, 4.5], title="x₁", row=1, col=1)
        fig.update_yaxes(range=[-1.5, 1.5], title="x₂", scaleanchor="x", row=1, col=1)
        ui.show(ui.style(fig, None, height=380, legend_below=True))
        mm = st.columns(5)
        mm[0].metric("log p(x)", f"{r['log_px']:.3f}", help="Exact, by numerical integration over z.")
        mm[1].metric("ELBO", f"{r['elbo']:.3f}")
        mm[2].metric("Gap = KL(q ‖ posterior)", f"{r['gap']:.3f}")
        mm[3].metric("Reconstruction term", f"{r['rec']:.2f}", help="E_q[log p(x | z)]")
        mm[4].metric("KL(q ‖ prior)", f"{r['kl_prior']:.2f}", help="Theory eq. 5. ELBO = reconstruction − this.")
    ui.takeaway("log p(x) doesn't depend on q; the ELBO does. Moving q onto the true posterior closes the gap. "
                "But a Gaussian can only be one bell, so when the posterior has two peaks the best possible gap "
                "stays above zero, and that is the VAE's built-in approximation error.")
    ui.controls([
        ("The code that produced x", "–", "where the observed point sits on the curve", "–"),
        ("Decoder noise σ", r"noise in $p(x \mid z)$", "how precisely a code pins down x", "wider, possibly "
         "two-peaked posterior"),
        ("q mean m, q std s", r"$\mu_\phi(x)$, $\sigma_\phi(x)$", "the encoder's Gaussian for this x", "–"),
        ("Snap q to the best Gaussian", "max ELBO", "what a perfectly trained encoder would output", "–"),
    ])
    ui.experiments([
        ("Leave q at the prior (m = 0, s = 1).", "The gap is ≈ 6.5 nats: an ELBO far below log p(x).",
         "Codes from the prior mostly decode far from x, so the reconstruction term is terrible."),
        ("Tick *Snap q to the best Gaussian* (defaults).", "q sits on the posterior (m ≈ 1.43, s ≈ 0.17) and the "
         "gap is ≈ 0.015.", "With small noise the posterior is nearly Gaussian, so a Gaussian q fits it."),
        ("Keep the snap on and set noise σ = 0.6.", "The posterior grows a second bump at negative z (≈ 25% of "
         "its mass); the best gap rises to ≈ 0.34.", "Both z and −z could have produced x. One bell can't cover "
         "two peaks, so the bound can't be tight."),
        ("Set s = 0.03 with m on the posterior peak.", "The KL-to-prior term grows; the ELBO drops.",
         "A very confident code is expensive (eq. 5) and here unjustified."),
    ])

# --------------------------------------------------------------------------- train on digits
X, ylab = gen.load_digits()


@st.cache_data(show_spinner="Training the VAE (a few seconds)…")
def _train(beta, steps, hidden, seed):
    track = np.arange(0, len(X), 3)
    r = gen.train_vae(X, 2, (hidden,), steps, 2e-3, beta, 128, seed=seed, snapshots=25, X_track=X[track])
    vae = r["vae"]
    mu, lv = vae.encode(X)
    grid = np.linspace(-2.2, 2.2, 13)
    G = np.array([[a, b] for b in grid[::-1] for a in grid])
    manifold = vae.decode(G)
    recon = vae.decode(mu)
    d = ((mu[:, None] - mu[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, 1)[:, :5]
    knn = float(np.mean(np.array([np.bincount(ylab[n], minlength=10).argmax() for n in nb]) == ylab))
    state = {"enc": vae.enc.state(), "dec": vae.dec.state()}
    return {"rec": r["rec"], "kl": r["kl"], "snaps": r["snaps"], "snap_steps": r["snap_steps"], "track": track,
            "mu": mu, "sd": np.exp(0.5 * lv), "manifold": manifold, "recon": recon, "knn": knn, "state": state}


def _model(res, hidden):
    v = gen.VAE(64, 2, [hidden], "bernoulli")
    v.enc.load(res["state"]["enc"])
    v.dec.load(res["state"]["dec"])
    return v


with t_train:
    st.markdown(
        r"""
**The situation.** 1,797 handwritten digits, 8 × 8 pixels each. A VAE with a **2-number code** is trained live:
encoder 64 → hidden → (μ, log σ²), decoder 2 → hidden → 64 pixel probabilities, loss = −ELBO with Bernoulli
pixels (Theory eq. 4–6). The labels are **never** shown to the model; they're only used to colour the map.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        beta = st.select_slider("KL weight β", [0.1, 0.3, 1.0, 2.0, 4.0], 1.0,
                                help="Theory eq. 7. 1 = the true ELBO. Larger = smoother map, blurrier digits.")
        steps = st.select_slider("Training steps", [300, 1000, 3000, 6000], 3000, help="Mini-batches of 128.")
        hidden = st.select_slider("Hidden units", [64, 256], 256, help="Width of the encoder and decoder layers.")
        vseed = st.number_input("Random seed", 0, 9999, 0, key="vae_seed", help="Different initial weights.")
    res = _train(beta, steps, hidden, int(vseed))
    with c2:
        tr = res["track"]

        def fr(mu_):
            return [go.Scatter(x=mu_[ylab[tr] == d, 0], y=mu_[ylab[tr] == d, 1], mode="text", text=[str(d)] *
                               int(np.sum(ylab[tr] == d)), textfont=dict(color=DIGIT_COL[d], size=11), name=str(d))
                    for d in range(10)]
        fig = go.Figure(data=fr(res["snaps"][-1]),
                        frames=[go.Frame(data=fr(s_), name=str(k)) for s_, k in zip(res["snaps"], res["snap_steps"])])
        lim = float(np.percentile(np.abs(res["mu"]), 99.5)) + 0.3
        fig.update_xaxes(range=[-lim, lim], title="z₁")
        fig.update_yaxes(range=[-lim, lim], title="z₂")
        ui.play_controls(fig, [str(k) for k in res["snap_steps"]], 200, "step ", start_at_end=True)
        ui.show(ui.style(fig, "The latent map: encoder mean μ(x) of each digit (a third of them shown)",
                         height=460, legend=False))
        cc = st.columns([1.1, 1])
        with cc[0]:
            st.plotly_chart(ui.image_grid(res["manifold"], 13, title="Decoder output across the map (z from −2.2 to 2.2)",
                                          height=330), width="stretch")
        with cc[1]:
            idx = [int(np.where(ylab == d)[0][0]) for d in range(10)]
            st.plotly_chart(ui.image_grid(np.r_[X[idx], res["recon"][idx]], 10,
                                          title="Originals (top) and reconstructions (bottom)", height=150),
                            width="stretch")
            sm = np.ones(50) / 50
            f2 = go.Figure()
            f2.add_scatter(y=np.convolve(res["rec"], sm, "valid"), name="reconstruction (−log p(x|z))",
                           line=dict(color=ui.C[1]))
            f2.add_scatter(y=np.convolve(res["kl"], sm, "valid"), name="KL(q ‖ prior)", line=dict(color=ui.C[2]))
            ui.show(ui.style(f2, "The two parts of −ELBO (nats per digit)", "step", None, height=200))
        m = st.columns(4)
        m[0].metric("Reconstruction (nats)", f"{res['rec'][-200:].mean():.1f}")
        m[1].metric("KL (nats)", f"{res['kl'][-200:].mean():.2f}")
        m[2].metric("Mean encoder σ", f"{res['sd'].mean():.2f}", help="1 = the code says nothing about the input.")
        m[3].metric("Map quality: 5-NN label accuracy", f"{res['knn']:.0%}",
                    help="How often a digit's 5 nearest codes have its label (chance = 10%). Uses labels only "
                         "for this score.")
    ui.takeaway("Without seeing a single label, the VAE arranges digits by appearance: same digits cluster, "
                "similar-looking ones sit side by side, and every point of the map decodes to something "
                "digit-like. β trades sharper reconstructions against a smoother, more regular map.")
    ui.controls([
        ("KL weight β", r"$\beta$, eq. (7)", "price of an informative code", "codes shrink toward the prior; "
         "blurrier; at β ≥ 4, collapse"),
        ("Training steps", "–", "how long to train", "sharper decoder, better-separated map"),
        ("Hidden units", "network width", "capacity of encoder and decoder", "better reconstructions"),
        ("Random seed", "–", "initial weights", "the map comes out rotated or reflected; same structure"),
    ])
    ui.experiments([
        ("Defaults (β = 1, 3,000 steps), press ▶.", "The cloud starts as one blob at the origin and separates into "
         "digit groups; reconstruction ≈ 21 nats, KL ≈ 2.8 nats, 5-NN accuracy ≈ 83%.",
         "The encoder learns which visual features distinguish digits."),
        ("Set β = 0.1.", "Reconstructions sharpen (≈ 20 nats), KL rises to ≈ 6 and the map separates digits "
         "better (≈ 88%), but encoder σ drops to ≈ 0.07: tight, precise codes.",
         "A cheap KL lets the encoder spend information on precise codes."),
        ("Set β = 4.", "Everything collapses: KL ≈ 0, encoder σ ≈ 1, every reconstruction is the same blurry "
         "average, accuracy ≈ 18%.", "Posterior collapse: ignoring the input became the cheapest option."),
        ("Change the seed.", "The map is rotated or mirrored, but the neighbourhoods are the same.",
         "The prior N(0, I) looks the same in every direction, so any rotation of the code is equally good."),
    ])

# --------------------------------------------------------------------------- generate & interpolate
with t_gen:
    st.markdown(
        r"""
**The situation.** Using the VAE trained in the previous tab (same settings), we **generate**: draw codes from
the prior $z \sim \mathcal N(0, I)$ and decode them. Then we **interpolate**: walk in a straight line from the code
of one digit to another and decode every step.
"""
    )
    model = _model(res, hidden)
    c1, c2 = st.columns([1, 2.3])
    with c1:
        gseed = st.number_input("Sampling seed", 0, 9999, 0, key="gen_seed", help="Different random codes.")
        temp = st.select_slider("Sampling spread (× prior std)", [0.5, 1.0, 1.5, 2.5], 1.0,
                                help="Scale the random codes: <1 stays near the centre of the map, >1 goes beyond it.")
        a_dig = st.selectbox("Interpolate from digit", list(range(10)), index=1, help="Start of the path.")
        b_dig = st.selectbox("…to digit", list(range(10)), index=7, help="End of the path.")
    zs = temp * np.random.default_rng(int(gseed)).standard_normal((40, 2))
    gen_imgs = model.decode(zs)
    za = res["mu"][ylab == a_dig].mean(0)
    zb = res["mu"][ylab == b_dig].mean(0)
    path = za[None] + np.linspace(0, 1, 10)[:, None] * (zb - za)[None]
    with c2:
        st.plotly_chart(ui.image_grid(gen_imgs, 10, title=f"40 new digits from random codes (spread × {temp})",
                                      height=300), width="stretch")
        st.plotly_chart(ui.image_grid(model.decode(path), 10, title=f"Walking in a straight line from the average "
                                      f"code of '{a_dig}' to '{b_dig}'", height=120), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=res["mu"][:, 0], y=res["mu"][:, 1], mode="markers", name="codes of real digits",
                        marker=dict(size=4, color=ui.C["truth"], opacity=0.25))
        fig.add_scatter(x=zs[:, 0], y=zs[:, 1], mode="markers", name="random codes used above",
                        marker=dict(size=8, color=ui.C[1]))
        fig.add_scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", name="interpolation path",
                        line=dict(color=ui.C[2], width=3))
        fig.update_yaxes(scaleanchor="x")
        ui.show(ui.style(fig, "Where the generated codes sit on the latent map", "z₁", "z₂", height=360))
    ui.takeaway("Because the KL term packs all codes into one Gaussian cloud, random codes land among real ones and "
                "decode to digits, and a path between two codes passes through plausible in-between shapes. "
                "Codes far outside the cloud (spread × 2.5) decode to odd, over-thick strokes.")
    ui.experiments([
        ("Look at the 40 samples.", "Most are recognisable digits; some are ambiguous blends.",
         "Some random codes fall between clusters, where the decoder blends neighbouring digits."),
        ("Set spread × 0.5.", "Samples look cleaner but less varied, often the 'average' style.",
         "Codes near the centre of the prior are typical; this is the VAE version of 'lowering the temperature'."),
        ("Set spread × 2.5.", "More odd shapes.", "The decoder was never trained on codes that far out."),
        ("Interpolate 1 → 7, then 3 → 8.", "One digit morphs into the other through in-between shapes.",
         "The decoder is a smooth function of z, and the map places similar shapes nearby."),
    ])
    ui.in_practice("Latent diffusion image generators (Stable Diffusion and successors) use a VAE to compress "
                   "images into a small latent grid and run diffusion there. VAE-style codecs compress speech and "
                   "music for neural audio coding; low ELBO flags anomalies in sensor data; chemists search VAE "
                   "latent spaces for new molecules.")

with t_code:
    st.markdown("The exact code this page runs (from `ssp/core/generative.py`):")
    ui.source(gen.VAE, gen.train_vae, gen.toy_elbo)
