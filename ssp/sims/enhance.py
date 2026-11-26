"""Domain demo — single-channel speech enhancement as statistical estimation.

In the STFT domain, Y(k, l) = S(k, l) + N(k, l). Per time–frequency bin we want
an estimate of the clean amplitude |S| from the noisy observation, given

    lambda_N  noise power   (estimated by tracking it, since it is never observed)
    xi        a-priori SNR  = E|S|^2 / lambda_N          (unknown; estimated)
    gamma     a-posteriori SNR = |Y|^2 / lambda_N        (observed)

Gain rules (each is an estimator from L1):
    spectral subtraction   |S|^2 ≈ |Y|^2 - lambda_N                (moment matching)
    Wiener                 G = xi / (1 + xi)                       (MMSE of the complex S)
    MMSE-STSA              Ephraim & Malah 1984                    (MMSE of |S|)
    MMSE-LSA               Ephraim & Malah 1985                    (MMSE of log|S|)

xi is estimated with the decision-directed rule (a MAP-flavoured recursive
smoother); setting a = 0 gives the maximum-likelihood estimate, which causes
"musical noise".

Noise tracking: speech-presence-probability (SPP) MMSE estimator
(Gerkmann & Hendriks 2012), a soft-decision Bayesian update per bin.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, special

FS = 16000
NFFT = 512
HOP = 256


def stft(x: np.ndarray):
    f, t, Y = signal.stft(x, FS, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
    return f, t, Y


def istft(Y: np.ndarray, n: int) -> np.ndarray:
    _, x = signal.istft(Y, FS, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
    return x[:n]


# ---------------------------------------------------------------------------
# Noise power tracking
# ---------------------------------------------------------------------------
def noise_psd_initial(P: np.ndarray, n_frames: int = 10) -> np.ndarray:
    """Assume the first frames are speech-free; keep that estimate forever."""
    return np.tile(P[:, :n_frames].mean(axis=1, keepdims=True), (1, P.shape[1]))


def noise_psd_spp(P: np.ndarray, n_init: int = 5, xi_h1_db: float = 15.0, alpha: float = 0.8) -> dict:
    """SPP-based MMSE noise power estimator (Gerkmann & Hendriks 2012).

    Per bin: P(speech | Y) from a likelihood ratio with a fixed 'typical' SNR
    under speech presence; then E[|N|^2 | Y] = (1-P)|Y|^2 + P * lambda_N, then
    recursive smoothing. A stagnation guard prevents P getting stuck at 1.
    """
    K, L = P.shape
    xi = 10 ** (xi_h1_db / 10)
    lam = P[:, :n_init].mean(axis=1)
    out = np.zeros((K, L))
    spp = np.zeros((K, L))
    p_bar = np.full(K, 0.5)
    for l in range(L):
        post = 1.0 / (1.0 + (1 + xi) * np.exp(-P[:, l] / np.maximum(lam, 1e-12) * xi / (1 + xi)))
        p_bar = 0.9 * p_bar + 0.1 * post
        post = np.where(p_bar > 0.99, np.minimum(post, 0.99), post)
        e_n = (1 - post) * P[:, l] + post * lam
        lam = alpha * lam + (1 - alpha) * e_n
        out[:, l] = lam
        spp[:, l] = post
    return {"noise": out, "spp": spp}


# ---------------------------------------------------------------------------
# Gain rules
# ---------------------------------------------------------------------------
def gain_spectral_subtraction(xi, gamma):
    return np.sqrt(np.maximum(1 - 1 / np.maximum(gamma, 1e-12), 0))


def gain_wiener(xi, gamma):
    return xi / (1 + xi)


def gain_mmse_stsa(xi, gamma):
    v = xi / (1 + xi) * gamma
    # exponentially scaled Bessel functions keep this finite for large v
    g = (np.sqrt(np.pi) / 2) * (np.sqrt(v) / np.maximum(gamma, 1e-12)) * \
        ((1 + v) * special.i0e(v / 2) + v * special.i1e(v / 2))
    return np.minimum(g, 1.0)


def gain_mmse_lsa(xi, gamma):
    v = np.maximum(xi / (1 + xi) * gamma, 1e-10)
    return np.minimum(xi / (1 + xi) * np.exp(0.5 * special.exp1(v)), 1.0)


GAINS = {"Spectral subtraction": gain_spectral_subtraction, "Wiener": gain_wiener,
         "MMSE-STSA (Ephraim–Malah)": gain_mmse_stsa, "MMSE-LSA (log-spectral)": gain_mmse_lsa}


def enhance(x: np.ndarray, gain: str = "MMSE-LSA (log-spectral)", noise_mode: str = "spp",
            dd_alpha: float = 0.98, xi_min_db: float = -25.0, g_min_db: float = -25.0,
            noise_oracle: np.ndarray | None = None) -> dict:
    """Full enhancement chain. noise_mode: 'spp' | 'initial' | 'oracle' (needs noise_oracle)."""
    f, t, Y = stft(x)
    P = np.abs(Y) ** 2
    if noise_mode == "oracle" and noise_oracle is not None:
        _, _, Nn = stft(noise_oracle)
        lam_n = np.tile((np.abs(Nn) ** 2).mean(axis=1, keepdims=True), (1, P.shape[1]))
        spp = None
    elif noise_mode == "initial":
        lam_n, spp = noise_psd_initial(P), None
    else:
        r = noise_psd_spp(P)
        lam_n, spp = r["noise"], r["spp"]
    lam_n = np.maximum(lam_n, 1e-12)
    gamma = P / lam_n
    gfun = GAINS[gain]
    xi_min = 10 ** (xi_min_db / 10)
    g_min = 10 ** (g_min_db / 20)
    G = np.zeros_like(P)
    xi_all = np.zeros_like(P)
    prev = np.zeros(P.shape[0])                       # G^2 |Y|^2 of the previous frame
    for l in range(P.shape[1]):
        ml = np.maximum(gamma[:, l] - 1, 0)
        xi = dd_alpha * prev / lam_n[:, l] + (1 - dd_alpha) * ml       # decision-directed
        xi = np.maximum(xi, xi_min)
        g = gfun(xi, gamma[:, l])
        G[:, l] = np.maximum(g, g_min)
        xi_all[:, l] = xi
        prev = (g**2) * P[:, l]
    s_hat = istft(G * Y, len(x))
    return {"out": s_hat, "G": G, "noise_psd": lam_n, "xi": xi_all, "spp": spp, "f": f, "t": t, "P": P}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    n = min(len(clean), len(est))
    e = est[:n] - clean[:n]
    return float(10 * np.log10(np.sum(clean[:n] ** 2) / (np.sum(e**2) + 1e-12)))


def segmental_snr_db(clean: np.ndarray, est: np.ndarray, frame: int = 512, lo: float = -10, hi: float = 35) -> float:
    """Average per-frame SNR over frames that contain speech, clipped to [lo, hi] dB (standard practice)."""
    n = (min(len(clean), len(est)) // frame) * frame
    C = clean[:n].reshape(-1, frame)
    E = est[:n].reshape(-1, frame) - C
    pc = np.sum(C**2, axis=1)
    act = pc > 1e-3 * pc.max()
    seg = 10 * np.log10(pc[act] / (np.sum(E[act] ** 2, axis=1) + 1e-12))
    return float(np.mean(np.clip(seg, lo, hi)))


def noise_attenuation_db(noisy: np.ndarray, est: np.ndarray, speech_mask: np.ndarray) -> float:
    """How much quieter the speech pauses became."""
    n = min(len(noisy), len(est), len(speech_mask))
    pause = ~speech_mask[:n]
    return float(10 * np.log10(np.mean(noisy[:n][pause] ** 2) / (np.mean(est[:n][pause] ** 2) + 1e-12)))


def make_scene(duration: float, snr_db: float, noise: str, ramp_db: float, seed: int) -> dict:
    """Synthetic speech in noise. ramp_db > 0: the noise gets louder half-way (car accelerates)."""
    from ssp.sims import speech as sp
    rng = np.random.default_rng(seed)
    s, mask = sp.synth_speech(duration, rng)
    w = sp.car_noise(s.size, rng, noise)
    env = np.ones(s.size)
    if ramp_db:
        half = s.size // 2
        ramp = np.linspace(0, 1, int(0.5 * FS))
        env[half:half + ramp.size] = 10 ** (ramp_db / 20 * ramp)
        env[half + ramp.size:] = 10 ** (ramp_db / 20)
    x = sp.mix_at_snr(s, mask, w, snr_db)
    noise_part = (x - s) * env
    return {"clean": s, "noisy": s + noise_part, "noise": noise_part, "mask": mask}
