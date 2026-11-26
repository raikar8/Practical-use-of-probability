"""Domain demo — predicting where another player is *right now* in an online game.

The server knows the true position. It sends snapshots at `send_hz`; each one
arrives after a random one-way delay (ping/2 + jitter) or is lost. On every
rendered frame the client must decide where to draw the remote player.

Client strategies compared
--------------------------
snap          draw the newest snapshot as-is (always late by the network delay)
extrapolate   straight-line dead reckoning from the two newest snapshots
kalman        constant-velocity Kalman filter, predicted forward to "now",
              with an uncertainty radius that grows while packets are missing
interpolate   render `interp_delay` in the past and blend between two
              snapshots — smooth and never wrong about the past, but behind
              the present by design (many engines do this for remote players)

Units: metres and seconds. Error is distance from the true *present* position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssp.core.kalman import constant_velocity


@dataclass
class NetConfig:
    duration: float = 30.0
    frame_hz: int = 60
    send_hz: int = 20
    ping_ms: float = 120.0          # round trip; one-way = ping / 2
    jitter_ms: float = 20.0
    loss: float = 0.05
    quant_std: float = 0.10         # snapshot position noise (m): quantisation, compression
    agility: float = 1.0            # direction changes per second
    max_speed: float = 7.0          # m/s
    interp_delay_ms: float = 100.0
    kf_q: float = 60.0              # KF acceleration noise density (m^2/s^3)
    seed: int = 3


ARENA = (0.0, 40.0, 0.0, 25.0)


def simulate_player(cfg: NetConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / cfg.frame_hz
    n = int(cfg.duration * cfg.frame_hz)
    xmin, xmax, ymin, ymax = ARENA
    p = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    v = np.zeros(2)
    vd = np.zeros(2)
    tau = 0.15
    P = np.zeros((n, 2))
    for k in range(n):
        if k == 0 or rng.random() < cfg.agility * dt:
            h = rng.uniform(0, 2 * np.pi)
            vd = rng.uniform(0.3, 1.0) * cfg.max_speed * np.array([np.cos(h), np.sin(h)])
        v = v + (vd - v) * (dt / tau)
        p = p + v * dt
        for i, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax))):
            if p[i] < lo or p[i] > hi:
                p[i] = 2 * lo - p[i] if p[i] < lo else 2 * hi - p[i]
                v[i], vd[i] = -v[i], -vd[i]
        P[k] = p
    return np.arange(n) * dt, P


def make_packets(t: np.ndarray, truth: np.ndarray, cfg: NetConfig, rng: np.random.Generator) -> dict:
    step = max(int(round(cfg.frame_hz / cfg.send_hz)), 1)
    idx = np.arange(0, len(t), step)
    send = t[idx]
    pos = truth[idx] + cfg.quant_std * rng.standard_normal((len(idx), 2))
    delay = cfg.ping_ms / 2000.0 + np.abs(cfg.jitter_ms / 1000.0 * rng.standard_normal(len(idx)))
    lost = rng.random(len(idx)) < cfg.loss
    return {"send": send, "arrive": send + delay, "pos": pos, "lost": lost}


def run_client(t: np.ndarray, pk: dict, cfg: NetConfig) -> dict:
    n = len(t)
    nan2 = np.full((n, 2), np.nan)
    est = {m: nan2.copy() for m in ("snap", "extrapolate", "kalman", "interpolate")}
    kf_radius = np.full(n, np.nan)
    kf_cov = np.full((n, 2, 2), np.nan)

    ok = ~pk["lost"]
    order = np.argsort(pk["arrive"][ok])
    send, arrive, pos = pk["send"][ok][order], pk["arrive"][ok][order], pk["pos"][ok][order]

    r = cfg.quant_std**2 + 1e-6
    H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    R = np.eye(2) * r
    kx, kP, kt = None, None, None          # KF state at time kt
    recv_t, recv_p = [], []                # received snapshots, in send-time order
    ptr = 0
    interp_delay = cfg.interp_delay_ms / 1000.0

    for i, now in enumerate(t):
        # deliver every packet that has arrived by this frame
        while ptr < len(arrive) and arrive[ptr] <= now:
            ts, z = send[ptr], pos[ptr]
            ptr += 1
            if recv_t and ts <= recv_t[-1]:
                continue                    # out-of-order and stale: drop
            recv_t.append(ts)
            recv_p.append(z)
            if kx is None:
                kx = np.array([z[0], 0.0, z[1], 0.0])
                kP = np.diag([r, 25.0, r, 25.0])
            else:
                F, Q = constant_velocity(ts - kt, cfg.kf_q, dims=2)
                kx = F @ kx
                kP = F @ kP @ F.T + Q
                S = H @ kP @ H.T + R
                K = kP @ H.T @ np.linalg.inv(S)
                kx = kx + K @ (z - H @ kx)
                kP = (np.eye(4) - K @ H) @ kP
            kt = ts
        if not recv_t:
            continue

        # snap
        est["snap"][i] = recv_p[-1]
        # extrapolate
        if len(recv_t) >= 2:
            vel = (recv_p[-1] - recv_p[-2]) / (recv_t[-1] - recv_t[-2])
            est["extrapolate"][i] = recv_p[-1] + vel * (now - recv_t[-1])
        else:
            est["extrapolate"][i] = recv_p[-1]
        # kalman: predict (without updating) from kt to now
        F, Q = constant_velocity(now - kt, cfg.kf_q, dims=2)
        xp = F @ kx
        Pp = F @ kP @ F.T + Q
        est["kalman"][i] = xp[[0, 2]]
        Cp = Pp[np.ix_([0, 2], [0, 2])]
        kf_cov[i] = Cp
        kf_radius[i] = 2.45 * np.sqrt(np.linalg.eigvalsh(Cp)[-1])
        # interpolate in the past: the client's estimate of server time is
        # now - ping/2, and it renders a further interp_delay behind that
        tr = now - cfg.ping_ms / 2000.0 - interp_delay
        j = np.searchsorted(recv_t, tr)
        if j == 0:
            est["interpolate"][i] = recv_p[0]
        elif j >= len(recv_t):
            est["interpolate"][i] = recv_p[-1]      # starved: hold last
        else:
            a = (tr - recv_t[j - 1]) / (recv_t[j] - recv_t[j - 1])
            est["interpolate"][i] = (1 - a) * recv_p[j - 1] + a * recv_p[j]

    return {"est": est, "kf_radius": kf_radius, "kf_cov": kf_cov}


def simulate_netcode(cfg: NetConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    t, truth = simulate_player(cfg, rng)
    pk = make_packets(t, truth, cfg, np.random.default_rng(cfg.seed + 1000))
    out = run_client(t, pk, cfg)
    err = {m: np.linalg.norm(e - truth, axis=1) for m, e in out["est"].items()}
    return {"t": t, "truth": truth, "packets": pk, **out, "err": err}


def summarize(err: dict) -> dict:
    """Mean and 95th-percentile error per method (ignoring frames before data)."""
    res = {}
    for m, e in err.items():
        e = e[np.isfinite(e)]
        res[m] = {"mean": float(np.mean(e)), "p95": float(np.percentile(e, 95))}
    return res


def sweep_ping(cfg: NetConfig, pings_ms) -> dict:
    """Mean error of each method as ping grows, same player path every time."""
    rng = np.random.default_rng(cfg.seed)
    t, truth = simulate_player(cfg, rng)
    out = {m: [] for m in ("snap", "extrapolate", "kalman", "interpolate")}
    for ping in pings_ms:
        c = NetConfig(**{**cfg.__dict__, "ping_ms": float(ping)})
        pk = make_packets(t, truth, c, np.random.default_rng(cfg.seed + 1000))
        res = run_client(t, pk, c)
        for m, e in res["est"].items():
            d = np.linalg.norm(e - truth, axis=1)
            out[m].append(float(np.nanmean(d)))
    return {"ping": np.asarray(pings_ms), **{m: np.array(v) for m, v in out.items()}}
