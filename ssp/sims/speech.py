"""Domain demo — voice activity detection (VAD) in car noise with an HMM.

No recordings needed: speech-like audio is synthesised (voiced syllables with a
pitch contour and vowel formants, plus short fricatives), then mixed with car
cabin noise at a chosen SNR. Because we generated it, the true speech/noise
label of every frame is known. You can also upload your own WAV.

Per-frame features (25 ms window, 10 ms hop), both restricted to 300–4000 Hz
where speech lives and car noise is weaker:
    1. band log-energy (dB)
    2. spectral flatness (dB): harmonic speech is "peaky" (low flatness),
       noise is flat (near 0 dB)

Detectors compared
    energy threshold      frame by frame, no memory          (flickers)
    HMM forward           P(speech | frames so far) > 0.5    (causal, real time)
    HMM Viterbi           best whole label sequence          (offline, cleanest)
The HMM's transition matrix is a *duration prior*: speech and pauses last
hundreds of milliseconds, so single-frame flips are unlikely. That replaces the
ad-hoc "hangover" counters used in classic VADs.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.io import wavfile

from ssp.core import hmm

FS = 16000
VOWELS = {  # first three formants (Hz)
    "a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
    "e": (530, 1840, 2480), "o": (570, 840, 2410),
}


def _resonator(x: np.ndarray, f: float, bw: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / FS)
    th = 2 * np.pi * f / FS
    return signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def _syllable(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * FS)
    t = np.arange(n) / FS
    f0 = rng.uniform(100, 210) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(1, 3) * t) - 0.1 * t / dur)
    phase = 2 * np.pi * np.cumsum(f0) / FS
    src = sum(np.sin(h * phase) / h for h in range(1, 30) if h * f0.max() < 4000)
    formants = VOWELS[rng.choice(list(VOWELS))]
    v = sum(_resonator(src, f, 80 + 40 * i) * (1.0, 0.5, 0.25)[i] for i, f in enumerate(formants))
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.6
    v = v * env
    if rng.random() < 0.4:                       # fricative onset ("s", "sh", "f")
        m = int(rng.uniform(0.04, 0.09) * FS)
        fric = signal.lfilter(*signal.butter(4, rng.uniform(2500, 4500) / (FS / 2), "high"),
                              rng.standard_normal(m)) * np.hanning(m) * 0.25
        v = np.r_[fric * np.std(v) / (np.std(fric) + 1e-9), v]
    return v


def synth_speech(duration: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Utterances of syllables separated by pauses. Returns audio and a sample-level truth mask."""
    n = int(duration * FS)
    x = np.zeros(n)
    mask = np.zeros(n, dtype=bool)
    pos = int(rng.uniform(0.5, 1.0) * FS)
    while pos < n - FS:
        start = pos
        for _ in range(rng.integers(3, 10)):
            if pos >= n:
                break
            syl = _syllable(rng.uniform(0.12, 0.28), rng)
            end = min(pos + syl.size, n)
            x[pos:end] += syl[: end - pos] * rng.uniform(0.6, 1.0)
            pos = end + int(rng.uniform(0.01, 0.06) * FS)
        mask[start:min(pos, n)] = True
        pos += int(rng.uniform(0.5, 1.6) * FS)
    x /= np.sqrt(np.mean(x[mask] ** 2)) + 1e-12
    return x, mask


def car_noise(n: int, rng: np.random.Generator, kind: str = "car") -> np.ndarray:
    """car: low-frequency rumble + engine harmonics + some road hiss; white; babble."""
    if kind == "white":
        w = rng.standard_normal(n)
    elif kind == "babble":
        w = sum(synth_speech(n / FS, np.random.default_rng(rng.integers(1 << 30)))[0] for _ in range(6))
        w = w + 0.05 * rng.standard_normal(n)
    else:
        brown = np.cumsum(rng.standard_normal(n))
        brown = signal.lfilter(*signal.butter(2, 20 / (FS / 2), "high"), brown)
        rumble = signal.lfilter(*signal.butter(4, 400 / (FS / 2)), brown)
        t = np.arange(n) / FS
        rpm_hz = 30 + 3 * np.sin(2 * np.pi * 0.1 * t)
        ph = 2 * np.pi * np.cumsum(rpm_hz) / FS
        engine = sum(np.sin(k * ph) / k for k in (1, 2, 3, 4))
        hiss = signal.lfilter(*signal.butter(2, 1500 / (FS / 2), "high"), rng.standard_normal(n))
        w = rumble / np.std(rumble) + 0.3 * engine + 0.12 * hiss
    return w / np.std(w)


def mix_at_snr(speech: np.ndarray, mask: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """SNR measured over speech-active samples (the usual 'active speech level' convention)."""
    ps = np.mean(speech[mask] ** 2)
    pn = np.mean(noise**2)
    return speech + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def frame_features(x: np.ndarray, fs: int = FS, win_ms: float = 25, hop_ms: float = 10,
                   band=(300, 4000)) -> dict:
    win = int(fs * win_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    nfr = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(nfr)[:, None]
    frames = x[idx] * np.hanning(win)
    nfft = int(2 ** np.ceil(np.log2(win)))
    P = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2 + 1e-12
    f = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    Pb = P[:, sel]
    energy_db = 10 * np.log10(Pb.sum(axis=1))
    flat_db = 10 * np.log10(np.exp(np.mean(np.log(Pb), axis=1)) / np.mean(Pb, axis=1))
    centers = (hop * np.arange(nfr) + win / 2) / fs
    return {"energy": energy_db, "flatness": flat_db, "t": centers, "hop": hop, "win": win}


def frame_labels(mask: np.ndarray, feats: dict) -> np.ndarray:
    c = (feats["t"] * FS).astype(int)
    return mask[np.clip(c, 0, len(mask) - 1)].astype(int)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def energy_vad(energy_db: np.ndarray, margin_db: float) -> np.ndarray:
    """Classic: noise floor = 10th percentile of frame energy; speech if margin dB above it."""
    return (energy_db > np.percentile(energy_db, 10) + margin_db).astype(int)


def hmm_vad(feats: dict, stay_speech: float | None, stay_noise: float | None, n_iter: int = 40,
            evidence_scale: float = 1.0) -> dict:
    """Unsupervised 2-state Gaussian HMM on standardised [energy, flatness].

    Baum–Welch learns everything from the recording itself; the state with the
    higher mean energy is called "speech". Optional stay_* override the learned
    self-transition probabilities (the duration prior) for decoding.
    """
    F = np.column_stack([feats["energy"], feats["flatness"]])
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    p0 = hmm.init_gaussian(F, 2, stay=0.95)
    fit = hmm.baum_welch(F, p0, n_iter=n_iter, var_floor=1e-3)
    p = fit["params"]
    speech = int(np.argmax(p.means[:, 0]))
    if speech == 0:                                    # reorder so state 1 = speech
        perm = [1, 0]
        p.pi, p.A = p.pi[perm], p.A[np.ix_(perm, perm)]
        p.means, p.variances = p.means[perm], p.variances[perm]
    learned_A = p.A.copy()
    if stay_noise is not None:
        p.A[0] = [stay_noise, 1 - stay_noise]
    if stay_speech is not None:
        p.A[1] = [1 - stay_speech, stay_speech]
    # Overlapping frames are not independent, so their likelihoods over-count
    # the evidence. Scaling log-likelihoods (the "acoustic scale" of speech
    # recognisers) lets the duration prior in A do its job.
    logB = evidence_scale * p.loglik_matrix(F)
    alpha, _, _ = hmm.forward(p.pi, p.A, logB)
    path, _ = hmm.viterbi(p.pi, p.A, logB)
    return {"forward": (alpha[:, 1] > 0.5).astype(int), "p_speech": alpha[:, 1], "viterbi": path,
            "params": p, "learned_A": learned_A, "loglik": fit["loglik"], "F": F}


def scores(pred: np.ndarray, truth: np.ndarray, hop_s: float) -> dict:
    sp = truth == 1
    return {"accuracy": float(np.mean(pred == truth)),
            "missed speech": float(np.mean(pred[sp] == 0)) if sp.any() else np.nan,
            "false alarm": float(np.mean(pred[~sp] == 1)) if (~sp).any() else np.nan,
            "switches / s": hmm.count_switches(pred) / (len(pred) * hop_s)}


def gate(x: np.ndarray, pred: np.ndarray, feats: dict, floor_db: float = -25.0) -> np.ndarray:
    """Attenuate non-speech regions (smoothly) — what a VAD-driven gate or codec DTX does."""
    g_frames = np.where(pred == 1, 1.0, 10 ** (floor_db / 20))
    t_s = np.arange(len(x)) / FS
    g = np.interp(t_s, feats["t"], g_frames)
    k = int(0.02 * FS)
    g = np.convolve(g, np.ones(k) / k, mode="same")
    return x * g


@dataclass
class VadScene:
    duration: float = 12.0
    snr_db: float = 5.0
    noise: str = "car"
    seed: int = 1


def make_scene(cfg: VadScene) -> dict:
    rng = np.random.default_rng(cfg.seed)
    s, mask = synth_speech(cfg.duration, rng)
    w = car_noise(s.size, rng, cfg.noise)
    x = mix_at_snr(s, mask, w, cfg.snr_db)
    return {"clean": s, "noisy": x, "mask": mask}


def load_wav(raw: bytes, max_seconds: float = 60.0) -> np.ndarray:
    """Mono float audio at 16 kHz from an uploaded WAV."""
    fs, x = wavfile.read(io.BytesIO(raw))
    if x.dtype == np.uint8:
        x = (x.astype(float) - 128) / 128
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(float) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != FS:
        from math import gcd
        g = gcd(int(fs), FS)
        x = signal.resample_poly(x, FS // g, int(fs) // g)
    peak = np.abs(x).max()
    if peak > 0.99:
        x = 0.99 * x / peak
    return x[: int(max_seconds * FS)]
