"""Pre-train the digits diffusion model shown in Level 13 (about 3 minutes on a laptop CPU).

    python -m scripts.train_digits_diffusion

Uses exactly the code the app runs (ssp.core.generative.train_diffusion). The page can also retrain a smaller
model live; this script only saves you the wait for the bigger one.
"""
from pathlib import Path

import numpy as np

from ssp.core import generative as g

T, HIDDEN, STEPS = 200, (256, 256, 256), 25000

X, _ = g.load_digits()
res = g.train_diffusion(X * 2 - 1, T=T, hidden=HIDDEN, steps=STEPS, batch=128, lr=1e-3, seed=0)
out = Path(__file__).resolve().parents[1] / "ssp" / "data" / "digits_diffusion.npz"
np.savez_compressed(out, T=T, hidden=np.array(HIDDEN), steps=STEPS,
                    **{k: v.astype(np.float32) for k, v in res["model"].net.state().items()})
print(f"saved {out}, final loss {res['loss'][-500:].mean():.4f}")
