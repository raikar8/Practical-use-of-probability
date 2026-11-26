# C++ ports

Header-only C++17 versions of two core algorithms, written to match the Python in `ssp/core/` line for line,
so you can read both side by side.

| File | Python original | Notes |
|---|---|---|
| `include/ssp/mat.hpp` | NumPy | fixed-size matrices on `std::array`: **no heap allocation** |
| `include/ssp/kalman.hpp` | `ssp/core/kalman.py` | `KalmanFilter<NX, NZ>`, Joseph-form update, constant-velocity model |
| `include/ssp/viterbi.hpp` | `ssp/core/hmm.py::viterbi` | log-domain dynamic programming |

## Build and run

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/kalman_cli --bench                      # ns per predict+update
./build/kalman_cli 0.1 0.5 0.3 < meas.csv       # dt q r; CSV of "x,y" measurements -> "px,vx,py,vy"
./build/viterbi_cli < problem.txt               # format documented in src/viterbi_cli.cpp
```

`pytest tests/test_cpp.py` compiles both programs with `g++` and checks that they reproduce the Python results
to 1e-9. The test is skipped if no compiler is found.

## Why allocation-free?

In an audio callback (for example 128 samples at 48 kHz = 2.67 ms) you must never call `malloc`, take a lock,
or do I/O. Any of those can block and cause an **xrun** (buffer underrun or overrun, heard as a click). `Mat<R, C>`
lives on the stack, so `KalmanFilter<4, 2>` does its maths with no allocations, in roughly 200 ns per
predict+update on a laptop core.

## Exercises (interview-style)

1. **Float vs double.** Template the filter on the scalar type, run it in `float`, and measure the drift of `P`
   from symmetry over 10⁶ steps with and without the Joseph form.
2. **Scalar fast path.** For `NZ == 1`, replace `inverse()` by a division (`if constexpr`) and measure the speed-up.
3. **Steady-state gain.** Iterate the Riccati recursion offline until `K` converges, then implement the
   constant-gain filter (an exponential smoother with optimal α). When is that good enough?
4. **NLMS echo canceller.** Port `ssp/core/adaptive.py::nlms` to process a block of 128 samples in place, with
   a circular buffer for the tap history and no allocations. Verify bit-exactness against Python with `double`.
5. **Streaming Viterbi.** Real-time VADs can't wait for the end of the file. Implement fixed-lag decoding
   (decide frame `t - L` at time `t`) and measure accuracy vs `L` against `ssp/sims/speech.py`.
6. **Thread safety.** The UI thread changes `Q` while the audio thread runs the filter. Design a lock-free
   handoff (double buffer + atomic index), and explain why a mutex is not acceptable here.
