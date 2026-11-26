// Closed-loop hearing-aid simulation around ssp::FeedbackCanceller.
//
//   afc_cli < scenario.txt    replay a scenario exported from Python (see tests/test_cpp.py)
//       line 1: n change gain mu method block      (method: none | fixed | nlms)
//       line 2: n samples of the incoming sound x
//       line 3: M taps of the feedback path before the change (F1)
//       line 4: M taps after the change (F2)
//       line 5: M taps of the initial canceller weights (h0)
//     stdout: n lines "u e", then one line "w <M weights>"
//
//   afc_cli --bench           time the canceller alone in a 10 s closed loop
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ssp/afc.hpp"

constexpr std::size_t M = 64;   // canceller taps (matches HAConfig.taps)
constexpr std::size_t D = 96;   // 6 ms at 16 kHz (matches HAConfig.delay_ms)
using Canceller = ssp::FeedbackCanceller<M, D, double>;

// The "room": convolves the receiver signal with the feedback path. This part is the
// simulated world, not device code, so it may use std::vector.
struct AcousticPath {
    std::vector<double> hist = std::vector<double>(M, 0.0);   // newest first
    double step(double u, const std::vector<double>& F) {
        for (std::size_t j = M - 1; j > 0; --j) hist[j] = hist[j - 1];
        hist[0] = u;
        double y = 0.0;
        for (std::size_t j = 0; j < M; ++j) y += F[j] * hist[j];
        return y;
    }
};

static int bench() {
    const double fs = 16000.0;
    const std::size_t n = static_cast<std::size_t>(10 * fs), B = 32;
    std::mt19937 rng(1);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> F(M, 0.0);
    for (std::size_t j = 4; j < M; ++j) F[j] = 0.01 * nd(rng) * std::exp(-(double)(j - 4) / 8.0);
    Canceller afc;
    afc.reset();
    afc.gain = 10.0;
    double u[B], y[B], e[B];
    AcousticPath room;
    double busy_ns = 0.0;
    for (std::size_t t = 0; t < n; t += B) {
        const auto t0 = std::chrono::steady_clock::now();
        afc.render(u, B);
        const auto t1 = std::chrono::steady_clock::now();
        for (std::size_t k = 0; k < B; ++k) y[k] = 0.003 * nd(rng) + room.step(u[k], F);
        const auto t2 = std::chrono::steady_clock::now();
        afc.process(y, e, B);
        const auto t3 = std::chrono::steady_clock::now();
        busy_ns += std::chrono::duration<double, std::nano>((t1 - t0) + (t3 - t2)).count();
    }
    const double per_sample = busy_ns / n;
    const double budget = 1e9 / fs;
    std::printf("M=%zu-tap NLMS feedback canceller, blocks of %zu at 16 kHz\n", M, B);
    std::printf("%.1f ns per sample = %.2f%% of the real-time budget (%.0f ns per sample)\n",
                per_sample, 100.0 * per_sample / budget, budget);
    std::printf("block of %zu samples: %.2f us of work in a %.0f us callback\n", B, per_sample * B / 1e3,
                B * budget / 1e3);
    return 0;
}

static std::vector<double> read_vec(std::size_t k) {
    std::vector<double> v(k);
    for (auto& x : v) std::cin >> x;
    return v;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench();

    std::size_t n = 0, change = 0, B = 0;
    double gain = 0, mu = 0;
    std::string method;
    if (!(std::cin >> n >> change >> gain >> mu >> method >> B) || B == 0 || B > D) {
        std::cerr << "bad header (block must be 1.." << D << ")\n";
        return 1;
    }
    const auto x = read_vec(n), F1 = read_vec(M), F2 = read_vec(M), h0 = read_vec(M);

    Canceller afc;
    afc.reset();
    afc.set_weights(h0.data());
    afc.gain = gain;
    afc.mu = mu;
    afc.cancel = method != "none";
    afc.adapt = method == "nlms";

    AcousticPath room;
    std::vector<double> u(n), e(n), y(B);
    for (std::size_t t = 0; t < n; t += B) {
        const std::size_t b = std::min(B, n - t);
        afc.render(&u[t], b);                                   // device -> receiver
        for (std::size_t k = 0; k < b; ++k)                     // receiver -> room -> microphone
            y[k] = x[t + k] + room.step(u[t + k], (t + k) < change ? F1 : F2);
        afc.process(y.data(), &e[t], b);                        // microphone -> device
    }
    for (std::size_t i = 0; i < n; ++i) std::printf("%.17g %.17g\n", u[i], e[i]);
    std::printf("w");
    for (double v : afc.weights()) std::printf(" %.17g", v);
    std::printf("\n");
    return 0;
}
