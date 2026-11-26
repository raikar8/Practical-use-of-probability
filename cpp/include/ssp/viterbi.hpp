// Log-domain Viterbi decoding. Mirrors ssp/core/hmm.py::viterbi.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ssp {

struct ViterbiResult {
    std::vector<int> path;
    double log_prob;
};

// log_pi: K, log_A: K*K row-major (from, to), log_B: T*K row-major (emission log-likelihoods).
inline ViterbiResult viterbi(const std::vector<double>& log_pi, const std::vector<double>& log_A,
                             const std::vector<double>& log_B, std::size_t T, std::size_t K) {
    std::vector<double> delta(K), next(K);
    std::vector<int> back(T * K, 0);
    for (std::size_t k = 0; k < K; ++k) delta[k] = log_pi[k] + log_B[k];
    for (std::size_t t = 1; t < T; ++t) {
        for (std::size_t j = 0; j < K; ++j) {
            double best = -std::numeric_limits<double>::infinity();
            int arg = 0;
            for (std::size_t i = 0; i < K; ++i) {
                const double s = delta[i] + log_A[i * K + j];
                if (s > best) { best = s; arg = static_cast<int>(i); }
            }
            next[j] = best + log_B[t * K + j];
            back[t * K + j] = arg;
        }
        delta.swap(next);
    }
    ViterbiResult r;
    r.path.assign(T, 0);
    std::size_t last = 0;
    for (std::size_t k = 1; k < K; ++k) if (delta[k] > delta[last]) last = k;
    r.log_prob = delta[last];
    r.path[T - 1] = static_cast<int>(last);
    for (std::size_t t = T - 1; t > 0; --t) r.path[t - 1] = back[t * K + static_cast<std::size_t>(r.path[t])];
    return r;
}

}  // namespace ssp
