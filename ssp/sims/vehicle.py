"""Domain demo — car GPS + IMU fusion with an EKF, including a GPS-denied tunnel.

Truth is integrated at the IMU rate. The IMU gives yaw rate (with an unknown
constant bias) and longitudinal acceleration; GPS gives noisy position at a
low rate and nothing at all inside the tunnel.

EKF state: x = [px, py, heading, speed, gyro_bias]
    px'      = px + v cos(th) dt
    py'      = py + v sin(th) dt
    th'      = th + (omega_meas - b) dt
    v'       = v + a_meas dt
    b'       = b                      (random walk via Q)
GPS:  z = [px, py] + noise

The gyro bias is never measured directly; the filter infers it because a
biased heading makes predicted positions drift away from GPS in a consistent
direction. That is the practical power of state augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssp.core.kalman import ExtendedKalmanFilter


@dataclass
class VehicleConfig:
    duration: float = 120.0      # s
    imu_hz: float = 100.0
    gps_hz: float = 1.0
    gps_std: float = 3.0         # m
    tunnel_start: float = 50.0   # s
    tunnel_len: float = 25.0     # s (0 = no tunnel)
    gyro_bias: float = 0.01      # rad/s  (~0.57 deg/s, a cheap MEMS gyro)
    gyro_std: float = 0.02       # rad/s per sample
    accel_std: float = 0.2       # m/s^2 per sample
    estimate_bias: bool = True
    seed: int = 7


def _truth_inputs(t: np.ndarray):
    omega = 0.10 * np.sin(2 * np.pi * t / 40.0) + 0.06 * np.sin(2 * np.pi * t / 13.0 + 1.0)
    accel = 0.6 * np.sin(2 * np.pi * t / 25.0)
    return omega, accel


def _f(x, u, dt):
    px, py, th, v, b = x
    om, a = u
    return np.array([px + v * np.cos(th) * dt,
                     py + v * np.sin(th) * dt,
                     th + (om - b) * dt,
                     v + a * dt,
                     b])


def _F(x, u, dt):
    _, _, th, v, _ = x
    F = np.eye(5)
    F[0, 2] = -v * np.sin(th) * dt
    F[0, 3] = np.cos(th) * dt
    F[1, 2] = v * np.cos(th) * dt
    F[1, 3] = np.sin(th) * dt
    F[2, 4] = -dt
    return F


_H = np.array([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])


def simulate_vehicle(cfg: VehicleConfig, store_every: int = 10) -> dict:
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.imu_hz
    n = int(cfg.duration * cfg.imu_hz)
    t = np.arange(n) * dt
    omega, accel = _truth_inputs(t)

    # --- truth -------------------------------------------------------------
    X = np.zeros((n, 4))
    x = np.array([0.0, 0.0, 0.0, 12.0])
    for k in range(n):
        X[k] = x
        px, py, th, v = x
        x = np.array([px + v * np.cos(th) * dt, py + v * np.sin(th) * dt,
                      th + omega[k] * dt, max(v + accel[k] * dt, 2.0)])

    # --- sensors -------------------------------------------------------------
    om_m = omega + cfg.gyro_bias + cfg.gyro_std * rng.standard_normal(n)
    a_m = accel + cfg.accel_std * rng.standard_normal(n)
    gps_step = max(int(round(cfg.imu_hz / cfg.gps_hz)), 1)
    in_tunnel = (t >= cfg.tunnel_start) & (t < cfg.tunnel_start + cfg.tunnel_len)
    gps_idx = np.array([k for k in range(0, n, gps_step) if not in_tunnel[k]])
    gps = X[gps_idx, :2] + cfg.gps_std * rng.standard_normal((len(gps_idx), 2))
    gps_at = dict(zip(gps_idx.tolist(), gps))

    # --- dead reckoning (IMU only, no bias knowledge) -----------------------
    DR = np.zeros((n, 4))
    d = X[0].copy()
    for k in range(n):
        DR[k] = d
        px, py, th, v = d
        d = np.array([px + v * np.cos(th) * dt, py + v * np.sin(th) * dt,
                      th + om_m[k] * dt, v + a_m[k] * dt])

    # --- EKF -------------------------------------------------------------------
    q_bias = 1e-8 if cfg.estimate_bias else 0.0
    Qd = np.diag([1e-4, 1e-4, (cfg.gyro_std * dt) ** 2, (cfg.accel_std * dt) ** 2, q_bias * dt])
    P0 = np.diag([2.0**2, 2.0**2, 0.05**2, 0.5**2, (0.05**2 if cfg.estimate_bias else 1e-12)])
    x0 = np.array([X[0, 0] + 1.0, X[0, 1] - 1.0, X[0, 2] + 0.02, X[0, 3] + 0.3, 0.0])
    ekf = ExtendedKalmanFilter(_f, _F, lambda x: _H @ x, lambda x: _H, Qd,
                               np.eye(2) * cfg.gps_std**2, x0, P0)
    keep = np.arange(0, n, store_every)
    xs, Ps = np.zeros((len(keep), 5)), np.zeros((len(keep), 5, 5))
    j = 0
    for k in range(n):
        if k in gps_at:
            ekf.update(gps_at[k])
        if j < len(keep) and keep[j] == k:
            xs[j], Ps[j] = ekf.x, ekf.P
            j += 1
        ekf.predict((om_m[k], a_m[k]), dt)

    # --- errors ----------------------------------------------------------------
    tt = t[keep]
    truth = X[keep]
    dr = DR[keep]
    err_ekf = np.linalg.norm(xs[:, :2] - truth[:, :2], axis=1)
    err_dr = np.linalg.norm(dr[:, :2] - truth[:, :2], axis=1)
    lam_max = np.array([np.linalg.eigvalsh(P[:2, :2])[-1] for P in Ps])
    bound95 = 2.45 * np.sqrt(lam_max)    # sqrt(chi2_2(0.95)) = 2.45
    gps_err = np.linalg.norm(gps - X[gps_idx, :2], axis=1)
    return {
        "t": tt, "truth": truth, "dr": dr, "ekf": xs, "P": Ps,
        "gps_t": t[gps_idx], "gps": gps, "gps_err": gps_err,
        "err_ekf": err_ekf, "err_dr": err_dr, "bound95": bound95,
        "tunnel": (cfg.tunnel_start, cfg.tunnel_start + cfg.tunnel_len),
        "true_bias": cfg.gyro_bias,
    }
