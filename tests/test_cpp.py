"""The C++ ports must reproduce the Python implementations."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ssp.core import hmm, kalman

CPP = Path(__file__).resolve().parents[1] / "cpp"
GXX = shutil.which("g++") or shutil.which("clang++")
pytestmark = pytest.mark.skipif(GXX is None, reason="no C++ compiler")


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    out = tmp_path_factory.mktemp("cpp")
    bins = {}
    for name in ("kalman_cli", "viterbi_cli"):
        exe = out / name
        subprocess.run([GXX, "-std=c++17", "-O2", f"-I{CPP / 'include'}", str(CPP / "src" / f"{name}.cpp"),
                        "-o", str(exe)], check=True)
        bins[name] = exe
    return bins


def test_kalman_matches_python(binaries):
    dt, q, r = 0.1, 0.5, 0.3
    X, Z = kalman.simulate_cv_2d(300, dt, q, r, np.random.default_rng(0))
    csv = "\n".join(f"{a:.17g},{b:.17g}" for a, b in Z)
    out = subprocess.run([str(binaries["kalman_cli"]), str(dt), str(q), str(r)], input=csv, text=True,
                         capture_output=True, check=True).stdout
    cpp = np.array([[float(v) for v in line.split(",")] for line in out.strip().splitlines()])
    F, Q = kalman.constant_velocity(dt, q, dims=2)
    kf = kalman.KalmanFilter(F, kalman.position_observation(2), Q, np.eye(2) * r, np.zeros(4), np.eye(4) * 10)
    py = kalman.run_kalman(kf, list(Z))["x"]
    assert cpp == pytest.approx(py, abs=1e-8)


def test_viterbi_matches_python(binaries):
    s, rolls, true = hmm.sample_casino(2000, 0.02, 0.1, 0.5, np.random.default_rng(1))
    logB = true.loglik_matrix(rolls)
    lines = [f"{len(rolls)} 2", " ".join(f"{v:.17g}" for v in np.log(true.pi + 1e-300)),
             " ".join(f"{v:.17g}" for v in np.log(true.A + 1e-300).ravel())]
    lines += [" ".join(f"{v:.17g}" for v in row) for row in logB]
    out = subprocess.run([str(binaries["viterbi_cli"])], input="\n".join(lines), text=True,
                         capture_output=True, check=True).stdout.strip().splitlines()
    path_cpp = np.array([int(v) for v in out[:-1]])
    path_py, lp = hmm.viterbi(true.pi, true.A, logB)
    assert np.array_equal(path_cpp, path_py)
    assert float(out[-1].split()[1]) == pytest.approx(lp, rel=1e-9)
