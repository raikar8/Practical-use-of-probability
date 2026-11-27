"""Deep generative models from scratch: VAE, GAN and diffusion (DDPM / DDIM), on 2-D toy data and 8x8 digits.

All networks are `ssp.core.nn.MLP`, trained with hand-written gradients. Everything is small enough to train in
seconds on a laptop CPU, which is the point: you can watch every model learn, fail, and recover.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ssp.core.nn import MLP, Adam, sigmoid, softplus

# ============================================================================================== data
RING_MODES = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4]) * 2.0
RING_STD = 0.12


def toy_2d(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """2-D datasets, roughly zero mean and unit spread."""
    if name == "ring":                                   # 8 Gaussians on a circle: the classic mode-collapse test
        k = rng.integers(0, 8, n)
        return RING_MODES[k] + RING_STD * rng.standard_normal((n, 2))
    if name == "moons":
        a = rng.uniform(0, np.pi, n)
        up = rng.random(n) < 0.5
        x = np.where(up, np.cos(a), 1 - np.cos(a))
        y = np.where(up, np.sin(a), 0.5 - np.sin(a))
        X = np.c_[x - 0.5, y - 0.25] * 1.6
        return X + 0.06 * rng.standard_normal((n, 2))
    if name == "spiral":
        t = np.sqrt(rng.random(n)) * 3 * np.pi
        X = np.c_[t * np.cos(t), t * np.sin(t)] / (1.5 * np.pi)
        return X + 0.05 * rng.standard_normal((n, 2))
    if name == "two blobs":
        k = rng.random(n) < 0.3
        return np.where(k[:, None], [-1.5, 0.8], [1.0, -0.5]) + 0.3 * rng.standard_normal((n, 2))
    raise ValueError(name)


DATASETS_2D = ["ring", "moons", "spiral", "two blobs"]


def load_digits() -> tuple[np.ndarray, np.ndarray]:
    """1,797 handwritten digits, 8x8 pixels in [0, 1] (UCI optdigits test set, as shipped with scikit-learn)."""
    d = np.load(Path(__file__).resolve().parents[1] / "data" / "digits8x8.npz")
    return d["images"].reshape(-1, 64).astype(float) / 16.0, d["labels"].astype(int)


def mog_1d(n: int, rng: np.random.Generator, means=(-2.0, 2.0), std: float = 0.5, w: float = 0.5) -> np.ndarray:
    k = rng.random(n) < w
    return np.where(k, means[0], means[1]) + std * rng.standard_normal(n)


def energy_distance(X: np.ndarray, Y: np.ndarray, n: int = 800, seed: int = 0) -> float:
    """Energy distance between two samples: 0 iff the distributions are equal (a simple sample-quality score)."""
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)].reshape(-1, X.shape[-1] if X.ndim > 1 else 1)
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)].reshape(-1, Y.shape[-1] if Y.ndim > 1 else 1)

    def md(A, B):
        return np.mean(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)))
    return float(2 * md(X, Y) - md(X, X) - md(Y, Y))


def ring_coverage(S: np.ndarray, min_frac: float = 0.02) -> tuple[int, float]:
    """(modes covered, fraction of 'good' samples) for the 8-Gaussian ring: good = within 3σ of some mode."""
    d = np.sqrt(((S[:, None, :] - RING_MODES[None]) ** 2).sum(-1))
    near = d.min(1) < 3 * RING_STD
    counts = np.bincount(d.argmin(1)[near], minlength=8)
    return int(np.sum(counts >= min_frac * len(S))), float(near.mean())


def two_sample_classifier(X: np.ndarray, Y: np.ndarray, steps: int = 600, seed: int = 0) -> dict:
    """Train a small classifier to tell real (X) from generated (Y). Held-out accuracy ≈ 50% means 'can't tell'."""
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y))
    X = X[rng.permutation(len(X))[:n]].reshape(n, -1)
    Y = Y[rng.permutation(len(Y))[:n]].reshape(n, -1)
    Z = np.r_[X, Y]
    lab = np.r_[np.ones(n), np.zeros(n)]
    p = rng.permutation(2 * n)
    Z, lab = Z[p], lab[p]
    cut = int(0.7 * len(Z))
    net = MLP([Z.shape[1], 32, 32, 1], act="relu", rng=rng)
    opt = Adam(net.params(), lr=0.01)
    for _ in range(steps):
        i = rng.choice(cut, min(256, cut), replace=False)
        lg = net(Z[i])[:, 0]
        net.backward(((sigmoid(lg) - lab[i]) / len(i))[:, None])
        opt.step(net.params(), net.grads())
    pred = net(Z[cut:])[:, 0] > 0
    return {"accuracy": float(np.mean(pred == (lab[cut:] > 0.5))), "net": net}


# ============================================================================================== 1-D flow
class Flow1D:
    """A tiny normalising flow in 1-D: z = f(x) = e^α x + Σ_i e^{β_i} tanh(e^{γ_i} x + d_i), with z ~ N(0, 1).

    Every term is increasing in x, so f is invertible and the change-of-variables formula gives the exact density
        log p(x) = log N(f(x); 0, 1) + log f'(x).
    Sampling runs it backwards: draw z ~ N(0, 1) and solve f(x) = z (by bisection, since f is monotone).
    """

    def __init__(self, n_units: int = 6, seed: int = 0, scale: float = 1.0):
        rng = np.random.default_rng(seed)
        self.a = np.array([np.log(0.3 / scale)])
        self.b = np.log(np.full(n_units, 0.3))
        self.g = np.log(np.full(n_units, 1.0 / scale))
        self.d = rng.uniform(-2, 2, n_units)

    def params(self):
        return [self.a, self.b, self.g, self.d]

    def _parts(self, x):
        A, B, G = np.exp(self.a[0]), np.exp(self.b), np.exp(self.g)
        u = G[None] * x[:, None] + self.d[None]
        th = np.tanh(u)
        s2 = 1 - th**2
        f = A * x + th @ B
        fp = A + (s2 * (B * G)[None]).sum(1)
        return A, B, G, u, th, s2, f, fp

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        *_, f, fp = self._parts(np.asarray(x, float))
        return -0.5 * f**2 - 0.5 * np.log(2 * np.pi) + np.log(fp)

    def loss_and_grads(self, x: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Negative mean log-likelihood and its gradient (written out by hand)."""
        A, B, G, u, th, s2, f, fp = self._parts(x)
        n = len(x)
        w = 1.0 / fp                                     # d log f' / d f'
        # d(−0.5 f²)/dθ = −f df/dθ ;  d(log f')/dθ = w df'/dθ
        ga = np.sum(-f * A * x + w * A)
        gb = np.sum((-f[:, None] * B[None] * th) + w[:, None] * (B * G)[None] * s2, 0)
        dfp_dg = (B * G)[None] * s2 * (1 - 2 * th * G[None] * x[:, None])
        gg = np.sum(-f[:, None] * B[None] * s2 * G[None] * x[:, None] + w[:, None] * dfp_dg, 0)
        gd = np.sum(-f[:, None] * B[None] * s2 + w[:, None] * (B * G)[None] * (-2 * s2 * th), 0)
        loss = -float(np.mean(-0.5 * f**2 - 0.5 * np.log(2 * np.pi) + np.log(fp)))
        return loss, [-np.array([ga]) / n, -gb / n, -gg / n, -gd / n]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._parts(np.asarray(x, float))[6]

    def sample(self, n: int, rng: np.random.Generator, lo: float = -50, hi: float = 50) -> np.ndarray:
        z = rng.standard_normal(n)
        a, b = np.full(n, lo), np.full(n, hi)
        for _ in range(60):                               # bisection on the monotone f
            mid = 0.5 * (a + b)
            big = self.forward(mid) > z
            b = np.where(big, mid, b)
            a = np.where(big, a, mid)
        return 0.5 * (a + b)


def train_flow1d(x: np.ndarray, n_units: int = 6, steps: int = 1500, lr: float = 0.05, seed: int = 0,
                 snapshots: int = 0, grid: np.ndarray | None = None) -> dict:
    """Maximum likelihood for the 1-D flow with Adam. Records the learned density on `grid` during training."""
    flow = Flow1D(n_units, seed, scale=float(np.std(x)))
    opt = Adam(flow.params(), lr=lr)
    losses, snaps, snap_steps = [], [], []
    every = max(1, steps // snapshots) if snapshots else None
    for k in range(steps):
        loss, g = flow.loss_and_grads(x)
        losses.append(loss)
        opt.step(flow.params(), g, clip=10.0)
        if every and grid is not None and (k % every == 0 or k == steps - 1):
            snaps.append(np.exp(flow.log_prob(grid)))
            snap_steps.append(k + 1)
    return {"flow": flow, "loss": np.array(losses), "snaps": snaps, "snap_steps": snap_steps}


def data_1d(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "two peaks":
        return mog_1d(n, rng, (-2.0, 1.5), 0.5, 0.4)
    if name == "skewed (log-normal)":
        return np.exp(0.6 * rng.standard_normal(n)) - 1.2
    if name == "three peaks":
        k = rng.integers(0, 3, n)
        return np.array([-3.0, 0.0, 2.5])[k] + np.array([0.4, 0.6, 0.3])[k] * rng.standard_normal(n)
    raise ValueError(name)


DATA_1D = ["two peaks", "skewed (log-normal)", "three peaks"]


# ============================================================================================== VAE
class VAE:
    """Encoder q(z|x) = N(μ(x), diag σ²(x)); decoder p(x|z) = Bernoulli(sigmoid(f(z))) or N(f(z), s²I)."""

    def __init__(self, x_dim: int, z_dim: int, hidden: list[int], likelihood: str = "bernoulli",
                 obs_std: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.enc = MLP([x_dim, *hidden, 2 * z_dim], act="relu", rng=rng, out_scale=0.1)
        self.dec = MLP([z_dim, *hidden[::-1], x_dim], act="relu", rng=rng)
        self.z_dim, self.likelihood, self.obs_std = z_dim, likelihood, obs_std

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.enc(x)
        return h[:, :self.z_dim], np.clip(h[:, self.z_dim:], -12, 6)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Mean of p(x|z): pixel probabilities (Bernoulli) or the Gaussian mean."""
        out = self.dec(z)
        return sigmoid(out) if self.likelihood == "bernoulli" else out

    def params(self):
        return self.enc.params() + self.dec.params()

    def loss_and_grads(self, x: np.ndarray, rng: np.random.Generator, beta: float = 1.0) -> dict:
        """One Monte-Carlo estimate of the negative ELBO and its gradients (the reparameterisation trick)."""
        B = len(x)
        h = self.enc(x)
        mu, logvar = h[:, :self.z_dim], np.clip(h[:, self.z_dim:], -12, 6)
        std = np.exp(0.5 * logvar)
        eps = rng.standard_normal(mu.shape)
        z = mu + std * eps                                       # reparameterisation: z is a function of noise
        out = self.dec(z)
        if self.likelihood == "bernoulli":
            rec = np.sum(softplus(out) - x * out, 1)              # −log p(x|z), binary cross-entropy
            d_out = sigmoid(out) - x
        else:
            s2 = self.obs_std**2
            rec = np.sum(0.5 * (x - out) ** 2 / s2 + 0.5 * np.log(2 * np.pi * s2), 1)
            d_out = (out - x) / s2
        kl = 0.5 * np.sum(mu**2 + np.exp(logvar) - 1.0 - logvar, 1)   # KL(q(z|x) || N(0, I)), closed form
        dz = self.dec.backward(d_out / B)
        dmu = dz + beta * mu / B
        dlogvar = dz * eps * 0.5 * std + beta * 0.5 * (np.exp(logvar) - 1.0) / B
        self.enc.backward(np.c_[dmu, dlogvar])
        return {"loss": float(np.mean(rec + beta * kl)), "rec": float(np.mean(rec)), "kl": float(np.mean(kl)),
                "grads": self.enc.grads() + self.dec.grads()}


def train_vae(X: np.ndarray, z_dim: int = 2, hidden=(256,), steps: int = 3000, lr: float = 2e-3,
              beta: float = 1.0, batch: int = 128, likelihood: str = "bernoulli", obs_std: float = 0.1,
              seed: int = 0, snapshots: int = 0, X_track: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    vae = VAE(X.shape[1], z_dim, list(hidden), likelihood, obs_std, seed)
    opt = Adam(vae.params(), lr=lr)
    hist = {"loss": [], "rec": [], "kl": []}
    snaps, snap_steps = [], []
    every = max(1, steps // snapshots) if snapshots else None
    for k in range(steps):
        idx = rng.choice(len(X), batch, replace=False)
        r = vae.loss_and_grads(X[idx], rng, beta)
        opt.step(vae.params(), r["grads"], clip=50.0)
        for key in hist:
            hist[key].append(r[key])
        if every and X_track is not None and (k % every == 0 or k == steps - 1):
            snaps.append(vae.encode(X_track)[0].copy())
            snap_steps.append(k + 1)
    return {"vae": vae, **{k: np.array(v) for k, v in hist.items()}, "snaps": snaps, "snap_steps": snap_steps}


# ---- a 1-D latent model where the true posterior is computable, to see the ELBO gap -------------------------
def toy_decoder(z: np.ndarray) -> np.ndarray:
    """f(z) for the toy latent model x = f(z) + noise, z ~ N(0, 1): a parabola-shaped curve in 2-D.

    z and −z land on mirror-image points, so when the noise is large the data can't tell them apart and the
    posterior p(z | x) has two peaks.
    """
    z = np.asarray(z, float)
    return np.stack([0.6 * z**2 - 1.0, 0.35 * z], -1)


def toy_log_joint(x: np.ndarray, z: np.ndarray, noise: float) -> np.ndarray:
    """log p(x, z) = log N(z; 0, 1) + log N(x; f(z), noise² I), for a grid of z."""
    f = toy_decoder(z)
    return (-0.5 * z**2 - 0.5 * np.log(2 * np.pi)
            - 0.5 * np.sum((x - f) ** 2, -1) / noise**2 - np.log(2 * np.pi * noise**2))


def toy_elbo(x: np.ndarray, m: float, s: float, noise: float) -> dict:
    """For one observed x and q(z) = N(m, s²): the ELBO, the exact log p(x) and their gap = KL(q || p(z|x)).

    Everything is computed by numerical integration over a fine z-grid (possible because z is 1-D).
    """
    zg = np.linspace(-6, 6, 4001)
    dz = zg[1] - zg[0]
    lj = toy_log_joint(x, zg, noise)
    mx = lj.max()
    log_px = float(mx + np.log(np.sum(np.exp(lj - mx)) * dz))
    post = np.exp(lj - log_px)
    lq = -0.5 * ((zg - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
    q = np.exp(lq)
    elbo = float(np.sum(q * (lj - lq)) * dz)
    rec = float(np.sum(q * (lj + 0.5 * zg**2 + 0.5 * np.log(2 * np.pi))) * dz)      # E_q[log p(x|z)]
    kl_prior = float(np.log(1 / s) + (s**2 + m**2) / 2 - 0.5)                      # KL(q || N(0, 1))
    return {"elbo": elbo, "log_px": log_px, "gap": log_px - elbo, "z": zg, "posterior": post, "q": q,
            "rec": rec, "kl_prior": kl_prior}


def toy_best_q(x: np.ndarray, noise: float) -> tuple[float, float]:
    """The Gaussian q(z) with the highest ELBO, by a coarse-to-fine grid search."""
    best = (-np.inf, 0.0, 1.0)
    for m in np.linspace(-4, 4, 81):
        for s in np.exp(np.linspace(np.log(0.03), np.log(2), 30)):
            e = toy_elbo(x, m, s, noise)["elbo"]
            if e > best[0]:
                best = (e, m, s)
    _, m0, s0 = best
    for m in np.linspace(m0 - 0.1, m0 + 0.1, 21):
        for s in np.linspace(0.8 * s0, 1.25 * s0, 21):
            e = toy_elbo(x, m, s, noise)["elbo"]
            if e > best[0]:
                best = (e, m, s)
    return best[1], best[2]


# ============================================================================================== GAN
def train_gan(sample_data, x_dim: int, steps: int = 3000, z_dim: int = 2, hidden: int = 64, lr_g: float = 1e-3,
              lr_d: float = 1e-3, d_steps: int = 1, batch: int = 256, loss: str = "non-saturating",
              seed: int = 0, snapshots: int = 0, grid: np.ndarray | None = None, n_show: int = 600) -> dict:
    """Train a GAN. sample_data(n, rng) gives real samples. Records generator samples (fixed noise) and D on a grid.

    Discriminator: maximise log D(x_real) + log(1 − D(G(z))).
    Generator: 'minimax' minimises log(1 − D(G(z))) (the original game); 'non-saturating' maximises log D(G(z)).
    """
    rng = np.random.default_rng(seed)
    G = MLP([z_dim, hidden, hidden, x_dim], act="relu", rng=rng)
    D = MLP([x_dim, hidden, hidden, 1], act="leaky", rng=rng)
    oG, oD = Adam(G.params(), lr=lr_g, beta1=0.5), Adam(D.params(), lr=lr_d, beta1=0.5)
    z_fixed = rng.standard_normal((n_show, z_dim))
    hist = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}
    snaps, dsnaps, snap_steps = [], [], []
    every = max(1, steps // snapshots) if snapshots else None
    for k in range(steps):
        for _ in range(d_steps):
            xr = sample_data(batch, rng).reshape(batch, x_dim)
            xf = G(rng.standard_normal((batch, z_dim)))
            lr_ = D(xr)[:, 0]
            D.backward(((sigmoid(lr_) - 1.0) / batch)[:, None])
            g_real = [g.copy() for g in D.grads()]
            lf = D(xf)[:, 0]
            D.backward((sigmoid(lf) / batch)[:, None])
            oD.step(D.params(), [a + b for a, b in zip(g_real, D.grads())])
        z = rng.standard_normal((batch, z_dim))
        xg = G(z)
        lg = D(xg)[:, 0]
        if loss == "minimax":
            dl = -sigmoid(lg) / batch                       # d/dl of −softplus(l) = log(1 − D)
            g_loss = float(np.mean(-softplus(lg)))
        else:
            dl = (sigmoid(lg) - 1.0) / batch                # d/dl of softplus(−l) = −log D
            g_loss = float(np.mean(softplus(-lg)))
        dx = D.backward(dl[:, None])                        # gradient flows *through* the discriminator...
        G.backward(dx)                                      # ...into the generator
        oG.step(G.params(), G.grads())
        hist["d_loss"].append(float(np.mean(softplus(-lr_) + softplus(lf))))
        hist["g_loss"].append(g_loss)
        hist["d_real"].append(float(np.mean(sigmoid(lr_))))
        hist["d_fake"].append(float(np.mean(sigmoid(lf))))
        if every and (k % every == 0 or k == steps - 1):
            snaps.append(G(z_fixed).copy())
            if grid is not None:
                dsnaps.append(sigmoid(D(grid.reshape(len(grid), x_dim))[:, 0]).copy())
            snap_steps.append(k + 1)
    return {"G": G, "D": D, **{k: np.array(v) for k, v in hist.items()}, "snaps": snaps, "dsnaps": dsnaps,
            "snap_steps": snap_steps, "z_fixed": z_fixed}


# ============================================================================================== diffusion
def schedule(T: int, kind: str = "cosine") -> dict:
    """Noise schedule: β_t, α_t = 1 − β_t and ᾱ_t = Π α_s (the fraction of signal power left at step t)."""
    if kind == "linear":
        betas = np.linspace(1e-4, 0.02, T) * (1000 / T)
    else:                                                    # Nichol & Dhariwal (2021)
        s = 0.008
        f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
        betas = np.clip(1 - f[1:] / f[:-1], 1e-5, 0.5)   # cap the last steps so √ᾱ_T is not ~0
    alphas = 1.0 - betas
    return {"T": T, "betas": betas, "alphas": alphas, "abar": np.cumprod(alphas)}


def q_sample(x0: np.ndarray, t: np.ndarray, sch: dict, eps: np.ndarray) -> np.ndarray:
    """Forward process in one jump: x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε   (t is 0-based here)."""
    ab = sch["abar"][t].reshape(-1, *([1] * (x0.ndim - 1)))
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps


def time_embed(t: np.ndarray, T: int, n_freq: int = 8) -> np.ndarray:
    """Sinusoidal features of the step number, so one network can denoise at every noise level."""
    u = (np.asarray(t, float) + 0.5) / T
    f = 2.0 ** np.arange(n_freq) * np.pi / 2
    return np.concatenate([np.sin(u[:, None] * f), np.cos(u[:, None] * f)], 1)


class Denoiser:
    """ε̂_θ(x_t, t): predicts the noise that was added. An MLP on [x_t, embed(t)]."""

    def __init__(self, x_dim: int, T: int, hidden=(128, 128, 128), n_freq: int = 8, seed: int = 0):
        self.T, self.n_freq = T, n_freq
        self.net = MLP([x_dim + 2 * n_freq, *hidden, x_dim], act="silu", rng=np.random.default_rng(seed))

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.net(np.c_[x, time_embed(t, self.T, self.n_freq)])


def train_diffusion(X: np.ndarray, T: int = 100, kind: str = "cosine", hidden=(128, 128, 128), steps: int = 4000,
                    lr: float = 2e-3, batch: int = 256, seed: int = 0) -> dict:
    """DDPM training: pick a random step t, add noise, ask the network to predict that noise (MSE)."""
    rng = np.random.default_rng(seed)
    sch = schedule(T, kind)
    model = Denoiser(X.shape[1], T, hidden, seed=seed)
    opt = Adam(model.net.params(), lr=lr)
    losses = []
    for k in range(steps):
        x0 = X[rng.choice(len(X), batch, replace=len(X) < batch)]
        t = rng.integers(0, T, batch)
        eps = rng.standard_normal(x0.shape)
        pred = model(q_sample(x0, t, sch, eps), t)
        err = pred - eps
        losses.append(float(np.mean(err**2)))
        model.net.backward(2 * err / err.size)
        if k == int(0.66 * steps):                         # lower the learning rate for the last third
            opt.lr *= 0.3
        opt.step(model.net.params(), model.net.grads(), clip=10.0)
    return {"model": model, "sch": sch, "loss": np.array(losses)}


def sample_ddpm(eps_fn, sch: dict, n: int, x_dim: int, rng: np.random.Generator, keep: int = 0,
                clip: float | None = None, x_init: np.ndarray | None = None, t_start: int | None = None) -> dict:
    """Ancestral sampling: start from pure noise and remove a little predicted noise at every step (+ fresh noise).

    Each step forms the denoised guess x̂₀ (Tweedie), optionally clips it to the data range, and samples
    x_{t−1} from q(x_{t−1} | x_t, x̂₀), a Gaussian between the current point and the guess.
    """
    T, b, a, ab = sch["T"], sch["betas"], sch["alphas"], sch["abar"]
    t0 = T - 1 if t_start is None else t_start             # start part-way down (from a given noisy x) if asked
    x = rng.standard_normal((n, x_dim)) if x_init is None else x_init.copy()
    traj, traj_t, x0s = [x.copy()], [t0 + 1], []
    keep_at = set(np.linspace(0, t0, keep).astype(int)) if keep else set()
    for t in range(t0, -1, -1):
        e = eps_fn(x, np.full(n, t))
        x0 = (x - np.sqrt(1 - ab[t]) * e) / np.sqrt(ab[t])
        if clip is not None:
            x0 = np.clip(x0, -clip, clip)
        ab_prev = ab[t - 1] if t > 0 else 1.0
        mean = (np.sqrt(ab_prev) * b[t] * x0 + np.sqrt(a[t]) * (1 - ab_prev) * x) / (1 - ab[t])
        var = b[t] * (1 - ab_prev) / (1 - ab[t])
        x = mean + (np.sqrt(var) * rng.standard_normal(x.shape) if t > 0 else 0.0)
        if t in keep_at:
            traj.append(x.copy())
            traj_t.append(t)
            x0s.append(x0.copy())
    return {"x": x, "traj": traj, "traj_t": traj_t, "x0": x0s}


def sample_ddim(eps_fn, sch: dict, n: int, x_dim: int, rng: np.random.Generator, n_steps: int = 20,
                x_init: np.ndarray | None = None, clip: float | None = None) -> dict:
    """DDIM (deterministic): jump between a few noise levels using the predicted clean sample. Returns paths."""
    T, ab = sch["T"], sch["abar"]
    ts = np.unique(np.linspace(T - 1, 0, n_steps).round().astype(int))[::-1]
    x = rng.standard_normal((n, x_dim)) if x_init is None else x_init.copy()
    path = [x.copy()]
    for i, t in enumerate(ts):
        e = eps_fn(x, np.full(n, t))
        x0 = (x - np.sqrt(1 - ab[t]) * e) / np.sqrt(ab[t])
        if clip is not None:
            x0 = np.clip(x0, -clip, clip)
            e = (x - np.sqrt(ab[t]) * x0) / np.sqrt(1 - ab[t])     # keep ε consistent with the clipped x̂₀
        ab_prev = ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * e
        path.append(x.copy())
    return {"x": x, "path": np.array(path), "ts": ts}


# ---- exact score for Gaussian-mixture data: no network needed ---------------------------------------------
def mixture_eps(x: np.ndarray, t: np.ndarray, sch: dict, means: np.ndarray, stds: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """The ideal noise prediction for data = Σ w_k N(m_k, s_k² I):  ε* = −√(1 − ᾱ) ∇ log p_t(x)."""
    ab = sch["abar"][np.asarray(t)][:, None]
    var = ab * stds[None, :] ** 2 + (1 - ab)                  # (n, K) variance of each blurred component
    mu = np.sqrt(ab)[:, :, None] * means[None]               # (n, K, d)
    diff = x[:, None, :] - mu
    d = x.shape[1]
    logp = np.log(weights)[None] - 0.5 * np.sum(diff**2, -1) / var - 0.5 * d * np.log(var)
    r = np.exp(logp - logp.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    score = -np.sum(r[:, :, None] * diff / var[:, :, None], 1)
    return -np.sqrt(1 - ab) * score


def tweedie_x0(x: np.ndarray, t: np.ndarray, eps_hat: np.ndarray, sch: dict) -> np.ndarray:
    """Tweedie: the MMSE estimate of the clean sample, E[x₀ | x_t] = (x_t − √(1 − ᾱ) ε̂) / √ᾱ."""
    ab = sch["abar"][np.asarray(t)][:, None]
    return (x - np.sqrt(1 - ab) * eps_hat) / np.sqrt(ab)
