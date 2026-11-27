"""A tiny neural-network library in NumPy: multilayer perceptron, backpropagation by hand, Adam.

Nothing here is magic. A layer computes  h = act(x W + b).  Backpropagation is the chain rule applied layer by
layer from the output back to the input, and Adam is gradient descent with a per-parameter step size.
The generative models (VAE, GAN, diffusion) are all built from this one class.
"""
from __future__ import annotations

import numpy as np

# activation: (function, derivative written in terms of the input a and the output h)
ACTS = {
    "tanh": (np.tanh, lambda a, h: 1.0 - h**2),
    "relu": (lambda a: np.maximum(a, 0.0), lambda a, h: (a > 0).astype(a.dtype)),
    "leaky": (lambda a: np.where(a > 0, a, 0.2 * a), lambda a, h: np.where(a > 0, 1.0, 0.2)),
    "silu": (lambda a: a * (0.5 + 0.5 * np.tanh(0.5 * a)),
             lambda a, h: (s := 0.5 + 0.5 * np.tanh(0.5 * a)) * (1.0 + a * (1.0 - s))),
    "linear": (lambda a: a, lambda a, h: np.ones_like(a)),
}


class MLP:
    """Fully connected network. sizes = [inputs, hidden..., outputs]. Hidden layers use `act`; output is linear."""

    def __init__(self, sizes: list[int], act: str = "tanh", rng: np.random.Generator | None = None,
                 out_scale: float = 1.0):
        rng = rng or np.random.default_rng(0)
        self.sizes, self.act = list(sizes), act
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = 1.0 if act in ("tanh", "linear") else np.sqrt(2.0)
            W = rng.standard_normal((n_in, n_out)) * gain / np.sqrt(n_in)
            if i == len(sizes) - 2:
                W *= out_scale
            self.W.append(W)
            self.b.append(np.zeros(n_out))
        self._cache = None

    # -- forward / backward ---------------------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        f = ACTS[self.act][0]
        hs, pre = [x], []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = h @ W + b
            pre.append(a)
            h = a if i == len(self.W) - 1 else f(a)
            hs.append(h)
        self._cache = (hs, pre)
        return h

    __call__ = forward

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Given dLoss/dOutput for the last forward pass, store dLoss/dParams and return dLoss/dInput."""
        hs, pre = self._cache
        dact = ACTS[self.act][1]
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.b)
        g = dout
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * dact(pre[i], hs[i + 1])
            self.dW[i] = hs[i].T @ g
            self.db[i] = g.sum(0)
            g = g @ self.W[i].T
        return g

    # -- parameters -----------------------------------------------------------------------------------------
    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def grads(self) -> list[np.ndarray]:
        return self.dW + self.db

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params()))

    def state(self) -> dict:
        return {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}

    def load(self, state: dict) -> "MLP":
        self.W = [np.asarray(state[f"W{i}"], float) for i in range(len(self.W))]
        self.b = [np.asarray(state[f"b{i}"], float) for i in range(len(self.b))]
        return self


class Adam:
    """Adam (Kingma & Ba, 2015): gradient descent with running averages of the gradient and its square."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray], clip: float | None = None) -> None:
        self.t += 1
        if clip is not None:                     # clip the global gradient norm (keeps early training stable)
            norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
            if norm > clip:
                grads = [g * (clip / norm) for g in grads]
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def numerical_grad(f, p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference gradient of the scalar f() with respect to array p (modified in place, then restored)."""
    g = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = p[i]
        p[i] = old + eps
        fp = f()
        p[i] = old - eps
        fm = f()
        p[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


# --------------------------------------------------------------------------------------------- small helpers
def sigmoid(a: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def softplus(a: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, a)


def fit_regression(x: np.ndarray, y: np.ndarray, hidden: list[int], steps: int, lr: float = 0.01,
                   act: str = "tanh", batch: int | None = None, seed: int = 0, snapshots: int = 0,
                   x_grid: np.ndarray | None = None) -> dict:
    """Fit y ≈ net(x) by minimising mean squared error with Adam. Optionally record the fit on x_grid over time."""
    rng = np.random.default_rng(seed)
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    net = MLP([x.shape[1], *hidden, y.shape[1]], act=act, rng=rng)
    opt = Adam(net.params(), lr=lr)
    losses, snaps, snap_steps = [], [], []
    every = max(1, steps // snapshots) if snapshots else None
    for k in range(steps):
        idx = rng.choice(len(x), batch, replace=False) if batch and batch < len(x) else slice(None)
        pred = net(x[idx])
        err = pred - y[idx]
        losses.append(float(np.mean(err**2)))
        net.backward(2 * err / err.size)
        opt.step(net.params(), net.grads())
        if every and (k % every == 0 or k == steps - 1) and x_grid is not None:
            snaps.append(net(x_grid.reshape(len(x_grid), -1))[:, 0].copy())
            snap_steps.append(k + 1)
    return {"net": net, "loss": np.array(losses), "snaps": snaps, "snap_steps": snap_steps}
