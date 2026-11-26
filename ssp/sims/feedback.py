"""Domain demo — acoustic feedback cancellation in a hearing aid.

    x[n]  incoming sound at the microphone (speech, music or noise)
    F     feedback path: receiver (loudspeaker) -> microphone, a short FIR
    y[n] = x[n] + (F * u)[n]                      microphone signal
    e[n] = y[n] - (H_hat * u)[n]                  after the adaptive canceller
    u[n] = sat( K * e[n - D] )                    amplified, delayed receiver signal

Without cancellation the loop howls once the gain K exceeds the maximum stable
gain MSG = -20 log10 max_w |F(w)|. An adaptive filter H_hat ≈ F removes the
feedback, so the effective loop is F - H_hat and much more gain is usable.

The catch is closed-loop bias. u is a delayed, amplified copy of x, so x and
u are correlated, and the least-squares assumption "regression noise is
uncorrelated with the input" is violated. The adaptive filter then partly
models the incoming sound instead of F, especially for tonal input like music,
and the margin to howling shrinks. Products add decorrelation (prediction-error
prewhitening, frequency shifting, probe noise) on top of the plain NLMS shown here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssp.sims.speech import FS, synth_speech


@dataclass
class HAConfig:
    duration: float = 5.0
    source: str = "speech"           # speech | music | noise
    gain_margin_db: float = 10.0     # forward gain above the no-canceller MSG
    delay_ms: float = 6.0            # hearing-aid processing delay
    taps: int = 64
    mu: float = 0.002
    path_change_db: float = 6.0      # feedback path jumps (phone held to the ear) half-way
    seed: int = 1


def feedback_path(L: int, rng: np.random.Generator) -> np.ndarray:
    f = np.zeros(L)
    k = np.arange(L - 4)
    f[4:] = rng.standard_normal(L - 4) * np.exp(-k / 8.0)
    return f / np.max(np.abs(np.fft.rfft(f, 1024))) * 10 ** (-25 / 20)     # peak |F| = -25 dB


def msg_db(residual_path: np.ndarray) -> float:
    """Maximum stable (broadband) gain for loop path residual_path."""
    return float(-20 * np.log10(np.max(np.abs(np.fft.rfft(residual_path, 1024))) + 1e-12))


def source_signal(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "noise":
        x = rng.standard_normal(n)
    elif kind == "music":
        t = np.arange(n) / FS
        x = np.zeros(n)
        note_len = int(0.5 * FS)
        for start in range(0, n, note_len):
            f0 = 220 * 2 ** (rng.integers(0, 12) / 12)
            seg = slice(start, min(start + note_len, n))
            tt = t[seg]
            x[seg] = sum(np.sin(2 * np.pi * h * f0 * tt * (1 + 0.002 * np.sin(2 * np.pi * 5 * tt))) / h
                         for h in (1, 2, 3))
    else:
        x, _ = synth_speech(n / FS + 0.01, rng)
        x = x[:n]
        x = x + 0.01 * rng.standard_normal(n)
    return 0.003 * x / (np.std(x) + 1e-12)


def scenario(cfg: HAConfig, method: str) -> dict:
    """Every input of one run (signal, paths, gain, delay, initial filter), so the
    Python loop and the C++ port (cpp/include/ssp/afc.hpp) can be fed identically.

    'fixed' and 'nlms' start from the feedback path measured at the fitting
    session (the true initial path plus measurement error, about -15 dB
    misalignment), which is what hearing aids actually do.
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.duration * FS)
    x = source_signal(cfg.source, n, rng)
    F1 = feedback_path(cfg.taps, np.random.default_rng(cfg.seed + 7))
    F2 = F1 * 10 ** (cfg.path_change_db / 20)
    M = cfg.taps
    meas_err = np.random.default_rng(cfg.seed + 99).standard_normal(M)
    meas_err *= np.linalg.norm(F1) / np.linalg.norm(meas_err) * 10 ** (-15 / 20)
    msg0 = msg_db(F1)
    return {"x": x, "F1": F1, "F2": F2, "change": n // 2, "msg0": msg0,
            "K": 10 ** ((msg0 + cfg.gain_margin_db) / 20), "D": int(cfg.delay_ms * FS / 1000), "M": M,
            "h0": np.zeros(M) if method == "none" else F1 + meas_err, "mu": cfg.mu}


def simulate(cfg: HAConfig, method: str) -> dict:
    """method: 'none' | 'fixed' | 'nlms'. Returns signals and diagnostics.

    'fixed' never adapts after the fitting measurement, so it fails when the
    path changes. Sample-by-sample reference; the C++ port processes blocks.
    """
    sc = scenario(cfg, method)
    x, F1, F2, change, K, D, M = (sc[k] for k in ("x", "F1", "F2", "change", "K", "D", "M"))
    n = x.size
    msg0 = sc["msg0"]
    h = sc["h0"].copy()
    u = np.zeros(n + M)                 # receiver signal, offset by M for easy history slicing
    e = np.zeros(n)
    y = np.zeros(n)
    snap_every = 160
    mis, msg_t = [], []
    for i in range(n):
        F = F1 if i < change else F2
        ui = i + M
        if i >= D:
            u[ui] = np.clip(K * e[i - D], -1.0, 1.0)          # receiver saturates when the loop howls
        hist = u[ui - M + 1:ui + 1][::-1]                       # u[i], u[i-1], ..., u[i-M+1]
        y[i] = x[i] + F @ hist
        e[i] = y[i] - (h @ hist if method != "none" else 0.0)
        if method == "nlms":
            h += cfg.mu * e[i] * hist / (1e-6 + hist @ hist)
        if i % snap_every == 0:
            mis.append(10 * np.log10(np.sum((F - h) ** 2) / np.sum(F**2)))
            msg_t.append(msg_db(F - h))
    out = u[M:]
    t_snap = np.arange(0, n, snap_every) / FS
    return {"x": x, "y": y, "e": e, "out": out, "K_db": 20 * np.log10(K), "msg0": msg0,
            "msg_after_change": msg_db(F2), "t": t_snap, "misalignment": np.array(mis), "msg": np.array(msg_t),
            "F1": F1, "F2": F2, "h": h, "change_t": change / FS, "D": D}


def howling_fraction(res: dict, frame: int = 320) -> float:
    """Fraction of 20 ms frames where the receiver is saturating (clear howling)."""
    o = res["out"]
    nfr = len(o) // frame
    peaks = np.abs(o[: nfr * frame]).reshape(nfr, frame).max(axis=1)
    return float(np.mean(peaks >= 0.999))
