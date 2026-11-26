# C++ ports

Header-only C++17 versions of two core algorithms, written to match the Python in `ssp/core/` line for line,
so you can read both side by side.

| File | Python original | Notes |
|---|---|---|
| `include/ssp/mat.hpp` | NumPy | fixed-size matrices on `std::array`: **no heap allocation** |
| `include/ssp/kalman.hpp` | `ssp/core/kalman.py` | `KalmanFilter<NX, NZ>`, Joseph-form update, constant-velocity model |
| `include/ssp/viterbi.hpp` | `ssp/core/hmm.py::viterbi` | log-domain dynamic programming |
| `include/ssp/afc.hpp` | `ssp/sims/feedback.py::simulate` | hearing-aid **adaptive feedback canceller** (NLMS), block-based, real-time safe |

## Build and run

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/kalman_cli --bench                      # ns per predict+update
./build/kalman_cli 0.1 0.5 0.3 < meas.csv       # dt q r; CSV of "x,y" measurements -> "px,vx,py,vy"
./build/viterbi_cli < problem.txt               # format documented in src/viterbi_cli.cpp
./build/afc_cli --bench                         # feedback canceller cost vs the real-time budget
./build/afc_cli < scenario.txt                  # replay a Python scenario (format in src/afc_cli.cpp)
```

`pytest tests/test_cpp.py` compiles all three programs with `g++` and checks that they reproduce the Python results
(Kalman to 1e-8, Viterbi exactly, feedback canceller to 1e-6 in the closed loop, even while it howls).
The tests are skipped if no compiler is found.

## The feedback canceller (`afc.hpp`)

```cpp
ssp::FeedbackCanceller<64, 96> afc;   // 64 taps, 6 ms forward delay at 16 kHz
afc.set_weights(fitting_estimate);    // feedback path measured at the fitting appointment
afc.gain = K;  afc.mu = 0.002;

// inside the audio callback, for each block of n <= 96 samples:
afc.render(receiver_out, n);          // 1. receiver samples = clip(K * e[t - D])
/* ... the block plays, leaks back through the ear canal, and is recorded ... */
afc.process(mic_in, e_out, n);        // 2. cancel + NLMS update on the recorded block
```

Design points worth being able to explain in an interview:

* **Two-phase block API.** The forward delay `D` is at least one block, so the receiver block depends only on
  errors from earlier blocks. That's what makes a closed loop computable block by block. The test
  `test_feedback_canceller_is_block_size_invariant` proves the output is bit-identical for blocks of 1, 7, 32 and 96.
* **Mirrored ring buffer.** Every input sample is written twice, `M` apart, so the newest-first tap window is a
  single contiguous slice: no modulo and no copy in the inner loop, and it vectorises.
* **One fused pass** computes the filter output `w·u` and the input power `|u|²` together, then the update is a
  second pass. That's 2M multiply-adds per sample.
* **No allocation, no locks, no I/O** inside `render`/`process`. All storage is `std::array` sized at compile time.
* **Cost:** about 70 ns per sample on a laptop core, roughly 0.1% of the 62.5 µs budget per sample at 16 kHz.
  On a hearing-aid DSP the same O(M) structure is what decides how many taps you can afford.

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
4. **NLMS echo canceller.** *Solved in `afc.hpp` for the hearing-aid loop.* Follow-ups: template it on
   `float` and measure the drift from the `double` result; replace the per-sample normalisation with a
   recursive power estimate (O(1)) and show why it needs a periodic exact refresh; add a howling detector that
   freezes adaptation when the receiver saturates.
5. **Streaming Viterbi.** Real-time VADs can't wait for the end of the file. Implement fixed-lag decoding
   (decide frame `t - L` at time `t`) and measure accuracy vs `L` against `ssp/sims/speech.py`.
6. **Thread safety.** The UI thread changes `Q` while the audio thread runs the filter. Design a lock-free
   handoff (double buffer + atomic index), and explain why a mutex is not acceptable here.
