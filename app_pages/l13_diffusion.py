from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ssp import ui
from ssp.core import generative as gen

ui.lab_intro(
    "l13",
    goal="See every piece of a diffusion model: data dissolving into noise, a perfect denoiser pulling samples "
         "back out, a denoiser network trained live, and a pre-trained digit model where you can compare the MMSE "
         "estimate with a true posterior sample.",
    on_screen=[
        "**Forward: dissolving data:** the fixed noising process and its SNR, on 2-D points and on digits "
        "(Theory §2)",
        "**Reverse with a perfect denoiser:** for data whose ideal denoiser is known exactly, arrows from noisy "
        "points to their MMSE estimates, and samples flowing from noise to data (Theory §4–5)",
        "**Learn the denoiser:** train the noise-predicting network yourself and sample with DDPM or DDIM "
        "(Theory §4–5)",
        "**Digits:** a pre-trained model generating digits, and the MMSE-vs-sample comparison (Theory §7)",
    ],
    start_here="In *Reverse with a perfect denoiser*, move the *noise level t* slider from high to low and watch "
               "where the arrows point: first all to the centre, then each to its nearest cluster.",
)

t_fwd, t_exact, t_learn, t_dig, t_code = st.tabs(["Forward: dissolving data", "Reverse with a perfect denoiser",
                                                  "Learn the denoiser", "Digits", "Code"])
X_dig, y_dig = gen.load_digits()

# --------------------------------------------------------------------------- forward
with t_fwd:
    st.markdown(
        r"""
**The situation.** 1,000 points from a 2-D dataset, each blended with Gaussian noise following Theory eq. (2):
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$. Press ▶ to move from $t = 0$ (data) to
$t = T$ (noise). Nothing is learned here; this is the fixed process the model will learn to reverse.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        ds = st.selectbox("Dataset", gen.DATASETS_2D, index=2, help="The clean data x₀.")
        kind = st.radio("Noise schedule", ["cosine", "linear"], horizontal=True, help="How fast the SNR falls "
                                                                                       "(Theory §2).")
        T = st.select_slider("Steps T", [50, 100, 200, 1000], 200, help="Number of noising steps.")
    sch = gen.schedule(T, kind)
    x0 = gen.toy_2d(ds, 1000, np.random.default_rng(0))
    eps = np.random.default_rng(1).standard_normal(x0.shape)
    ts = np.unique(np.linspace(0, T - 1, 21).astype(int))
    with c2:
        def fr(t):
            xt = gen.q_sample(x0, np.full(len(x0), t), sch, eps)
            return [go.Scatter(x=xt[:, 0], y=xt[:, 1], mode="markers", marker=dict(size=4, color=ui.C[1],
                                                                                    opacity=0.6), name="x_t")]
        fig = go.Figure(data=fr(0), frames=[go.Frame(data=fr(t), name=f"{t + 1} (SNR "
                                                     f"{10 * np.log10(sch['abar'][t] / (1 - sch['abar'][t])):.0f} dB)")
                                            for t in ts])
        fig.update_xaxes(range=[-4, 4])
        fig.update_yaxes(range=[-4, 4], scaleanchor="x")
        ui.play_controls(fig, [f.name for f in fig.frames], 200, "t = ")
        ui.show(ui.style(fig, "Data dissolving into Gaussian noise", height=430, legend=False))
        snr = 10 * np.log10(sch["abar"] / (1 - sch["abar"]))
        sd = gen.schedule(200, kind)
        cols = [0, 20, 40, 60, 80, 100, 120, 140, 160, 199]
        imgs = []
        rng = np.random.default_rng(2)
        for i in (0, 1, 2):
            e_ = rng.standard_normal(64)
            for t in cols:
                imgs.append((gen.q_sample((X_dig[i] * 2 - 1)[None], np.array([t]), sd, e_[None])[0] + 1) / 2)
        cc = st.columns([1.2, 1])
        with cc[0]:
            st.plotly_chart(ui.image_grid(imgs, len(cols), title=f"Digits at t = {', '.join(str(c) for c in cols)} "
                                          f"(T = 200, {kind})", height=170), width="stretch")
        with cc[1]:
            f2 = go.Figure(go.Scatter(x=np.arange(1, T + 1), y=snr, line=dict(color=ui.C[1]), name="SNR"))
            f2.add_hline(y=0, line=dict(color=ui.C["truth"], dash="dot"))
            ui.show(ui.style(f2, "SNR of x_t", "t", "dB", height=200, legend=False))
    ui.takeaway("Structure disappears in a fixed order: fine detail (the thin spiral arms, the strokes) goes first, "
                "coarse layout (where the mass is) last. Generation will run this in reverse: coarse layout first, "
                "fine detail last.")
    ui.controls([
        ("Dataset", "$x_0$", "the clean data", "–"),
        ("Noise schedule", r"$\beta_t$", "how the SNR falls with t", "–"),
        ("Steps T", "$T$", "number of steps", "smaller changes per step"),
    ])
    ui.experiments([
        ("Press ▶ with the spiral.", "Around 10 dB SNR the arms blur together; by 0 dB only a blob remains.",
         "Noise wipes out small-scale structure first."),
        ("Switch to the linear schedule.", "The digits are unreadable by t ≈ 60 and the SNR plunges early.",
         "The linear schedule reaches 0 dB after a quarter of the steps (Theory §2)."),
    ])

# --------------------------------------------------------------------------- exact score
M = np.array([[-1.5, 1.0], [1.5, 1.0], [0.0, -1.5]])
SD = np.array([0.3, 0.3, 0.45])
W = np.array([0.3, 0.3, 0.4])


def sample_mix(n, rng):
    k = rng.choice(3, n, p=W)
    return M[k] + SD[k, None] * rng.standard_normal((n, 2))


with t_exact:
    st.markdown(
        r"""
**The situation.** If the data is a mixture of Gaussian blobs, the ideal denoiser $\mathbb E[x_0 \mid x_t]$
has an exact formula (every blob stays Gaussian under noise). So we can run a diffusion model with a **perfect**
denoiser and no network at all. Left: arrows from noisy points $x_t$ to the denoiser's answer (Theory eq. 5–6).
Right: 600 samples starting from pure noise and following the reverse chain (eq. 7).
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        sch2 = gen.schedule(100)
        t_ar = st.slider("Noise level t for the arrows", 1, 100, 60, help="Step of the noisy points. Higher = noisier.")
        sampler = st.radio("Sampler", ["DDPM (stochastic, 100 steps)", "DDIM (deterministic)"], help="Theory §5.")
        n_ddim = st.select_slider("DDIM steps", [3, 5, 10, 20, 50], 10, disabled=sampler.startswith("DDPM"),
                                  help="Network calls used by DDIM.")
        sseed = st.number_input("Random seed", 0, 9999, 0, key="ex_seed", help="Different starting noise.")

    def eps_exact(x, t):
        return gen.mixture_eps(x, t, sch2, M, SD, W)
    g = np.linspace(-3, 3, 13)
    P = np.array([[a, b] for a in g for b in g])
    tvec = np.full(len(P), t_ar - 1)
    x0h = gen.tweedie_x0(P, tvec, eps_exact(P, tvec), sch2)
    rng = np.random.default_rng(int(sseed))
    if sampler.startswith("DDPM"):
        out = gen.sample_ddpm(eps_exact, sch2, 600, 2, rng, keep=25)
        frames, labels = out["traj"], [str(t) for t in out["traj_t"]]
        calls = 100
    else:
        out = gen.sample_ddim(eps_exact, sch2, 600, 2, rng, n_steps=n_ddim, clip=3.0)
        frames = list(out["path"])
        labels = ["100"] + [str(int(t)) for t in out["ts"][1:]] + ["0"]
        calls = n_ddim
    real = sample_mix(3000, np.random.default_rng(5))
    with c2:
        snr_ar = 10 * np.log10(sch2["abar"][t_ar - 1] / (1 - sch2["abar"][t_ar - 1]))
        st.markdown(f"**Left:** denoiser arrows at t = {t_ar} (SNR {snr_ar:.0f} dB). "
                    "**Right:** samples flowing from noise to data (press ▶).")
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06)
        xs_, ys_ = [], []
        for p, q in zip(P, x0h):
            xs_ += [p[0], q[0], None]
            ys_ += [p[1], q[1], None]
        fig.add_scatter(x=real[:600, 0], y=real[:600, 1], mode="markers", marker=dict(size=3, color=ui.C[2],
                        opacity=0.3), name="real data", row=1, col=1)
        fig.add_scatter(x=xs_, y=ys_, mode="lines", line=dict(color=ui.C[1], width=1.2), name="x_t → E[x₀ | x_t]",
                        row=1, col=1)
        fig.add_scatter(x=x0h[:, 0], y=x0h[:, 1], mode="markers", marker=dict(size=5, color=ui.C[1]),
                        showlegend=False, row=1, col=1)
        fig.add_scatter(x=real[:600, 0], y=real[:600, 1], mode="markers", marker=dict(size=3, color=ui.C[2],
                        opacity=0.3), showlegend=False, row=1, col=2)
        fig.add_scatter(x=frames[-1][:, 0], y=frames[-1][:, 1], mode="markers", name="samples",
                        marker=dict(size=4, color=ui.C[7]), row=1, col=2)
        fig.frames = [go.Frame(data=[go.Scatter(x=f[:, 0], y=f[:, 1], mode="markers",
                                                marker=dict(size=4, color=ui.C[7]))], traces=[4], name=lab)
                      for f, lab in zip(frames, labels)]
        for c in (1, 2):
            fig.update_xaxes(range=[-3.2, 3.2], row=1, col=c)
            fig.update_yaxes(range=[-3.2, 3.2], row=1, col=c)
        ui.play_controls(fig, labels, 120, "t = ", start_at_end=True)
        ui.show(ui.style(fig, None, height=470))
        m = st.columns(3)
        m[0].metric("Energy distance: samples vs data", f"{gen.energy_distance(real, frames[-1]):.4f}",
                    help="0 = same distribution. Pure noise scores ≈ 0.17 here.")
        m[1].metric("Denoiser calls per sample", calls)
        m[2].metric("Arrow targets inside a blob", f"{np.mean(np.min(np.sqrt(((x0h[:, None] - M[None]) ** 2).sum(-1)), 1) < 0.6):.0%}",
                    help="Share of the arrow tips within 0.6 of a blob centre.")
    ui.takeaway("At high noise the best guess of the clean point is the overall average, so all arrows point to the "
                "middle. As noise drops, the guesses commit to the nearest blob. Chaining many small steps of "
                "'move toward the current best guess, add a little noise' turns pure noise into the data distribution.")
    ui.controls([
        ("Noise level t for the arrows", "$t$", "noise of the points whose denoised guess is drawn", "arrows "
         "converge to the overall mean"),
        ("Sampler", "DDPM / DDIM, §5", "stochastic many-step vs deterministic few-step", "–"),
        ("DDIM steps", "–", "number of denoiser calls", "closer to the DDPM result"),
    ])
    ui.experiments([
        ("Move t from 100 down to 1.", "Arrows first all point to one spot in the middle, then split toward the three "
         "blobs, then become tiny.", "E[x₀ | x_t] is the posterior mean: an average over everything plausible, "
         "which narrows as the noise drops."),
        ("DDPM, press ▶.", "Points first drift as one cloud, decide on a blob around the middle steps, then "
         "tighten; energy distance ≈ 0.004 (pure noise: ≈ 0.17).", "Coarse decisions happen at high noise, fine detail at low noise."),
        ("DDIM with 3 steps, then 10.", "3 steps: energy distance ≈ 0.20, no better than noise. 10 steps: ≈ 0.01, "
         "close to DDPM with a tenth of the calls.", "Each DDIM jump trusts the current x̂₀ estimate; with too few jumps, early "
         "rough guesses aren't corrected."),
    ])

# --------------------------------------------------------------------------- learned 2-D
with t_learn:
    st.markdown(
        r"""
**The situation.** Real data has no formula, so train a network $\hat\varepsilon_\theta(x_t, t)$ (3 hidden layers,
sinusoidal features of $t$ as extra inputs) with the regression loss of Theory eq. (4) on 5,000 points, then
sample with it.
"""
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        ds3 = st.selectbox("Dataset", gen.DATASETS_2D, index=2, key="d3", help="The training data.")
        st3 = st.select_slider("Training steps", [500, 1000, 3000, 6000], 3000, help="Mini-batches of 256.")
        h3 = st.select_slider("Hidden units", [32, 64, 128], 128, help="Width of each of the 3 hidden layers.")
        n3 = st.select_slider("DDIM steps", [5, 10, 20, 50, 100], 20, key="n3", help="For the DDIM samples.")

    @st.cache_data(show_spinner="Training the denoiser (a few seconds)…")
    def _train(ds, steps, h):
        X = gen.toy_2d(ds, 5000, np.random.default_rng(0))
        D = gen.train_diffusion(X, T=100, hidden=(h, h, h), steps=steps, seed=0)
        return D["model"].net.state(), D["loss"]
    state, loss = _train(ds3, st3, h3)
    model = gen.Denoiser(2, 100, (h3, h3, h3))
    model.net.load(state)
    sch3 = gen.schedule(100)
    real3 = gen.toy_2d(ds3, 3000, np.random.default_rng(5))
    Sp = gen.sample_ddpm(model, sch3, 1500, 2, np.random.default_rng(1))["x"]
    Si = gen.sample_ddim(model, sch3, 1500, 2, np.random.default_rng(1), n_steps=n3, clip=3.0)
    with c2:
        fig = make_subplots(rows=1, cols=3, subplot_titles=["training data", "DDPM samples (100 calls)",
                                                            f"DDIM samples ({len(Si['ts'])} calls) + paths"])
        fig.add_scatter(x=real3[:1500, 0], y=real3[:1500, 1], mode="markers", marker=dict(size=3, color=ui.C[2],
                        opacity=0.5), showlegend=False, row=1, col=1)
        fig.add_scatter(x=Sp[:, 0], y=Sp[:, 1], mode="markers", marker=dict(size=3, color=ui.C[1], opacity=0.5),
                        showlegend=False, row=1, col=2)
        fig.add_scatter(x=Si["x"][:, 0], y=Si["x"][:, 1], mode="markers", marker=dict(size=3, color=ui.C[7],
                        opacity=0.5), showlegend=False, row=1, col=3)
        for j in range(25):
            pth = Si["path"][:, j]
            fig.add_scatter(x=pth[:, 0], y=pth[:, 1], mode="lines", line=dict(color=ui.C[4], width=1),
                            showlegend=False, row=1, col=3)
        for c in (1, 2, 3):
            fig.update_xaxes(range=[-3, 3], row=1, col=c)
            fig.update_yaxes(range=[-3, 3], row=1, col=c)
        ui.show(ui.style(fig, None, height=360, legend=False, legend_below=True))
        f2 = go.Figure(go.Scatter(y=np.convolve(loss, np.ones(50) / 50, "valid"), line=dict(color=ui.C[1])))
        f2.add_hline(y=1.0, line=dict(color=ui.C["truth"], dash="dot"), annotation_text="predicting 0 (no skill)")
        ui.show(ui.style(f2, "Training loss: mean squared error of the predicted noise", "step", "loss", height=220,
                         legend=False))
        m = st.columns(3)
        m[0].metric("Energy distance, DDPM", f"{gen.energy_distance(real3, Sp):.4f}")
        m[1].metric(f"Energy distance, DDIM ({n3} steps)", f"{gen.energy_distance(real3, Si['x']):.4f}")
        m[2].metric("Energy distance, pure noise", f"{gen.energy_distance(real3, np.random.default_rng(2).standard_normal((1500, 2))):.3f}",
                    help="A reference point for 'no skill'.")
    ui.takeaway("The loss never gets near zero: at high noise, the noise can't be predicted from x_t (that's the "
                "point). It only has to be predicted as well as possible on average, which is exactly the MMSE "
                "denoiser. Yet chaining those imperfect denoisers reproduces the thin spiral faithfully.")
    ui.controls([
        ("Dataset", "$x_0$", "training data", "–"),
        ("Training steps", "–", "how long the denoiser trains", "sharper samples"),
        ("Hidden units", "network width", "denoiser capacity", "finer detail"),
        ("DDIM steps", "–", "network calls for DDIM", "closer to DDPM quality"),
    ])
    ui.experiments([
        ("Defaults (spiral, 3,000 steps).", "DDPM samples trace the spiral (energy distance ≈ 0.006, vs ≈ 0.11 for "
         "pure noise); DDIM with 20 steps is close (≈ 0.02).", "A good denoiser at every noise level is all it "
         "takes."),
        ("Set training steps to 500.", "Samples are a fuzzier spiral with more points between the arms (energy "
         "distance roughly doubles, to ≈ 0.012).",
         "An undertrained denoiser is imprecise at low noise, where the fine detail is decided."),
        ("Set DDIM steps to 5.", "The DDIM samples degrade sharply (for moons, worse than noise); the orange paths "
         "take big, straight jumps.", "Few large jumps amplify the denoiser's errors at high noise."),
    ])

# --------------------------------------------------------------------------- digits
with t_dig:
    st.markdown(
        r"""
**The situation.** A denoiser network (64 pixels + time features → 3 × 256 units → 64), trained for 25,000 steps on
the 1,797 digits with exactly this app's code (`scripts/train_digits_diffusion.py`, about 3 minutes on a laptop).
Generation: start from 64 random numbers and run the reverse chain. Below that, the key comparison for
enhancement (Theory §7): from a noisy real digit, compute the **one-step MMSE estimate** vs a **sample from the
reverse chain**.
"""
    )
    wpath = Path(__file__).resolve().parents[1] / "ssp" / "data" / "digits_diffusion.npz"
    d = np.load(wpath)
    Td = int(d["T"])
    dm = gen.Denoiser(64, Td, tuple(int(h) for h in d["hidden"]))
    dm.net.load({k: d[k].astype(float) for k in d.files if k[0] in "Wb"})
    schd = gen.schedule(Td)
    c1, c2 = st.columns([1, 2.3])
    with c1:
        smp = st.radio("Sampler", ["DDPM (200 steps)", "DDIM"], help="Theory §5.")
        nd = st.select_slider("DDIM steps", [5, 10, 20, 50], 20, disabled=smp.startswith("DDPM"),
                              help="Network calls for DDIM.")
        dseed = st.number_input("Sampling seed", 0, 9999, 0, key="dig_seed", help="Different starting noise.")
    rng = np.random.default_rng(int(dseed))
    if smp.startswith("DDPM"):
        out = gen.sample_ddpm(dm, schd, 40, 64, rng, keep=6, clip=1.0)
        S = out["x"]
        prog_x = out["traj"][1:]
        prog_x0 = out["x0"]
        prog_t = out["traj_t"][1:]
    else:
        out = gen.sample_ddim(dm, schd, 40, 64, rng, n_steps=nd, clip=1.0)
        S = out["x"]
        prog_x, prog_x0, prog_t = [], [], []
    with c2:
        st.plotly_chart(ui.image_grid((S + 1) / 2, 10, title="40 generated digits", height=300), width="stretch")
        if prog_x:
            rows = [(prog_x[k][j] + 1) / 2 for j in range(6) for k in range(len(prog_x))]
            rows0 = [(np.clip(prog_x0[k][j], -1, 1) + 1) / 2 for j in range(6) for k in range(len(prog_x0))]
            cc = st.columns(2)
            cc[0].plotly_chart(ui.image_grid(rows, len(prog_x), title=f"x_t along the chain (t = {prog_t[0]} → 0)",
                                             height=330),
                               width="stretch")
            cc[1].plotly_chart(ui.image_grid(rows0, len(prog_x0), title="…and the denoiser's guess x̂₀",
                                             height=330), width="stretch")
    st.divider()
    st.markdown("**MMSE estimate vs posterior sample.** Eight real digits, noised to level *t*, then cleaned two ways.")
    c3, c4 = st.columns([1, 2.3])
    with c3:
        tn = st.slider("Noise level t", 20, 190, 120, 10, help="Higher = noisier input.")
        st.caption(f"SNR at t = {tn}: {10 * np.log10(schd['abar'][tn] / (1 - schd['abar'][tn])):.1f} dB")
        nseed = st.number_input("Noise seed", 0, 9999, 0, key="mmse_seed", help="Different noise and sampling.")
    idx = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * 10 + 3
    x0r = X_dig[idx] * 2 - 1
    r2 = np.random.default_rng(int(nseed))
    xt = gen.q_sample(x0r, np.full(8, tn), schd, r2.standard_normal(x0r.shape))
    x0_mmse = np.clip(gen.tweedie_x0(xt, np.full(8, tn), dm(xt, np.full(8, tn)), schd), -1, 1)
    s1 = gen.sample_ddpm(dm, schd, 8, 64, r2, clip=1.0, x_init=xt, t_start=tn)["x"]
    s2 = gen.sample_ddpm(dm, schd, 8, 64, r2, clip=1.0, x_init=xt, t_start=tn)["x"]
    with c4:
        grid_imgs = [(v + 1) / 2 for v in np.r_[x0r, xt, x0_mmse, s1, s2]]
        st.plotly_chart(ui.image_grid(grid_imgs, 8, title="Rows: clean · noisy input · one-step MMSE estimate · "
                                      "reverse-chain sample 1 · sample 2", height=420), width="stretch")
        m = st.columns(4)
        m[0].metric("Mean squared error: MMSE", f"{np.mean((x0_mmse - x0r) ** 2):.3f}")
        m[1].metric("Mean squared error: samples", f"{0.5 * (np.mean((s1 - x0r) ** 2) + np.mean((s2 - x0r) ** 2)):.3f}")
        m[2].metric("Contrast: MMSE", f"{np.mean(np.abs(x0_mmse)):.2f}",
                    help="Mean |pixel| on the −1…1 scale. Real digits: shown on the right. Lower = greyer, blurrier.")
        m[3].metric("Contrast: samples / real", f"{0.5 * (np.mean(np.abs(s1)) + np.mean(np.abs(s2))):.2f} / "
                                                f"{np.mean(np.abs(x0r)):.2f}")
    ui.takeaway("Generation goes coarse to fine: the denoiser's guess settles on 'which digit' while x_t is still "
                "mostly noise. In the comparison, the MMSE estimate wins on squared error but is grey and blurry; "
                "the reverse-chain samples are crisp and realistic but may be a different plausible digit, and two "
                "samples differ. That's the perception–distortion trade-off behind diffusion-based enhancement.")
    ui.controls([
        ("Sampler, DDIM steps", "§5", "how many denoiser calls", "–"),
        ("Noise level t", "$t$", "how noisy the input is", "MMSE gets blurrier; samples diverge from the truth"),
    ])
    ui.experiments([
        ("Generate with DDPM.", "Many of the 40 samples are recognisable digits of several classes (some are "
         "malformed: it's a small network trained on 1,797 images); the x̂₀ grid shows each digit's identity fixed "
         "well before the last steps.", "Coarse structure is decided at high noise (Theory §5)."),
        ("Compare t = 40 and t = 160 in the MMSE comparison.", "At t = 40 both recover the digits (MSE ≈ 0.03 vs "
         "0.06). At t = 160 the MMSE estimate is greyer and blurrier (contrast ≈ 0.63 vs 0.78 for real digits) "
         "while samples stay crisp (≈ 0.72), with a higher MSE (≈ 0.42 vs 0.33), and can switch identity.", "At high noise many digits fit the input; the mean "
         "averages them, a sample picks one."),
        ("DDIM with 5 steps.", "Digits are rougher but still mostly recognisable, with 40× fewer network calls than "
         "DDPM.", "Fewer, larger jumps trade quality for speed."),
    ])
    ui.in_practice("Diffusion models generate images and video (latent diffusion), speech and music (diffusion "
                   "vocoders, text-to-speech, text-to-music), and enhance audio: score-based speech enhancement "
                   "and dereverberation sample clean speech conditioned on the noisy recording, trading a little "
                   "squared error for much more natural sound.")

with t_code:
    st.markdown("The exact code this page runs (from `ssp/core/generative.py`):")
    ui.source(gen.schedule, gen.q_sample, gen.Denoiser, gen.train_diffusion, gen.sample_ddpm, gen.sample_ddim,
              gen.mixture_eps, gen.tweedie_x0)
