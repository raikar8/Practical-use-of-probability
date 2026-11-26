// Viterbi decoder.
//   stdin line 1: T K
//   line 2: K values of log pi
//   line 3: K*K values of log A (row-major, from -> to)
//   next T lines: K emission log-likelihoods each
//   stdout: the decoded state path, one integer per line, then "logp <value>"
#include <cstdio>
#include <iostream>
#include <vector>

#include "ssp/viterbi.hpp"

int main() {
    std::size_t T = 0, K = 0;
    if (!(std::cin >> T >> K) || T == 0 || K == 0) {
        std::cerr << "bad header\n";
        return 1;
    }
    std::vector<double> log_pi(K), log_A(K * K), log_B(T * K);
    for (auto& v : log_pi) std::cin >> v;
    for (auto& v : log_A) std::cin >> v;
    for (auto& v : log_B) std::cin >> v;
    const auto r = ssp::viterbi(log_pi, log_A, log_B, T, K);
    for (int s : r.path) std::printf("%d\n", s);
    std::printf("logp %.12g\n", r.log_prob);
    return 0;
}
