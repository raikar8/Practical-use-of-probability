// Adaptive feedback canceller for a hearing aid: block-based, allocation-free.
// C++ port of ssp/sims/feedback.py::simulate (methods "none", "fixed", "nlms").
//
//   u[t] = clip(K * e[t - D])                    receiver (loudspeaker) sample
//   e[t] = y[t] - w . [u[t], u[t-1], ..., u[t-M+1]]      cancelled microphone sample
//   w   += mu * e[t] * u_vec / (eps + |u_vec|^2)          NLMS
//
// Why two calls per block? Because the processing delay D is at least one block,
// the receiver samples for the *next* block depend only on errors from earlier
// blocks. So a callback can (1) render the receiver block, which leaves the device
// and comes back through the acoustic path into the microphone, then (2) cancel
// and adapt on the microphone block that was recorded while it played.
//
// Real-time rules followed here: no heap allocation (all storage is std::array),
// no locks, no system calls, and O(M) work per sample.
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

namespace ssp {

template <std::size_t M, std::size_t D, typename T = double>
class FeedbackCanceller {
    static_assert(M > 0 && D > 0, "taps and delay must be positive");

public:
    T gain = T(1);        // forward-path gain K (linear)
    T mu = T(0.002);      // NLMS step size
    T eps = T(1e-6);      // regularisation of the normalisation
    T clip = T(1);        // receiver saturation level
    bool cancel = true;   // subtract the feedback estimate
    bool adapt = true;    // update the weights

    static constexpr std::size_t max_block() { return D; }

    void reset() {
        w_.fill(T(0));
        hist_.fill(T(0));
        eline_.fill(T(0));
        uq_.fill(T(0));
        pos_ = 0;
        t_render_ = t_proc_ = 0;
    }

    void set_weights(const T* w0) { std::copy(w0, w0 + M, w_.begin()); }
    const std::array<T, M>& weights() const { return w_; }

    // Phase 1: produce the next n receiver samples (n <= D). Call render(n) then
    // process(n) for each block; rendering may never run more than D samples ahead.
    void render(T* u_out, std::size_t n) {
        assert(t_render_ + n <= t_proc_ + D && "render ran too far ahead of process");
        for (std::size_t k = 0; k < n; ++k, ++t_render_) {
            // eline_[t % D] still holds e[t - D]; it is overwritten only when time t is processed.
            const T e_delayed = (t_render_ >= D) ? eline_[t_render_ % D] : T(0);
            const T u = std::clamp(gain * e_delayed, -clip, clip);
            uq_[t_render_ % D] = u;
            u_out[k] = u;
        }
    }

    // Phase 2: cancel and adapt on the n microphone samples recorded while those
    // receiver samples played. Writes the cancelled signal e to e_out.
    void process(const T* mic, T* e_out, std::size_t n) {
        for (std::size_t k = 0; k < n; ++k, ++t_proc_) {
            push(uq_[t_proc_ % D]);
            const T* h = &hist_[pos_];                 // u[t], u[t-1], ..., u[t-M+1], contiguous
            T dot = T(0), pw = T(0);
            for (std::size_t j = 0; j < M; ++j) {      // one fused pass: estimate and input power
                dot += w_[j] * h[j];
                pw += h[j] * h[j];
            }
            const T e = cancel ? mic[k] - dot : mic[k];
            if (adapt) {
                const T g = mu * e / (eps + pw);
                for (std::size_t j = 0; j < M; ++j) w_[j] += g * h[j];
            }
            eline_[t_proc_ % D] = e;
            e_out[k] = e;
        }
    }

private:
    // Mirrored ring buffer: each sample is written twice, M apart, so the newest-first
    // window of length M is always one contiguous slice starting at pos_. No modulo in the
    // inner loop, no copying.
    void push(T u) {
        pos_ = (pos_ + M - 1) % M;
        hist_[pos_] = u;
        hist_[pos_ + M] = u;
    }

    std::array<T, M> w_{};
    std::array<T, 2 * M> hist_{};
    std::array<T, D> eline_{};   // e delay line
    std::array<T, D> uq_{};      // rendered receiver samples waiting to be processed
    std::size_t pos_ = 0;
    std::size_t t_render_ = 0, t_proc_ = 0;
};

}  // namespace ssp
