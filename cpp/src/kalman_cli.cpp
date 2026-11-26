// 2-D constant-velocity Kalman tracker.
//   kalman_cli dt q r < measurements.csv   ->  prints "px,vx,py,vy" per line
//   kalman_cli --bench                      ->  timing of one predict+update
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "ssp/kalman.hpp"

using KF = ssp::KalmanFilter<4, 2>;

static KF make_filter(double dt, double q, double r) {
    KF kf;
    ssp::constant_velocity<2>(dt, q, kf.F, kf.Q);
    kf.H = ssp::Mat<2, 4>::zeros();
    kf.H(0, 0) = 1.0;
    kf.H(1, 2) = 1.0;
    kf.R = ssp::Mat<2, 2>::identity();
    kf.R(0, 0) = r;
    kf.R(1, 1) = r;
    kf.P = ssp::Mat<4, 4>::identity();
    for (double& v : kf.P.a) v *= 10.0;
    kf.x = ssp::Mat<4, 1>::zeros();
    return kf;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        KF kf = make_filter(0.01, 1.0, 0.1);
        ssp::Mat<2, 1> z;
        const int n = 2000000;
        double sink = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) {
            z(0, 0) = 0.001 * i;
            z(1, 0) = -0.002 * i;
            kf.predict();
            sink += kf.update(z);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / n;
        std::printf("4-state / 2-measurement Kalman: %.1f ns per predict+update (checksum %.3g)\n", ns, sink);
        std::printf("A 48 kHz audio block of 128 samples lasts 2.67 ms, room for ~%.0f updates.\n", 2.67e6 / ns);
        return 0;
    }
    if (argc < 4) {
        std::cerr << "usage: kalman_cli dt q r < measurements.csv  |  kalman_cli --bench\n";
        return 1;
    }
    KF kf = make_filter(std::atof(argv[1]), std::atof(argv[2]), std::atof(argv[3]));
    std::string line;
    ssp::Mat<2, 1> z;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        std::string a, b;
        if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',')) continue;
        z(0, 0) = std::stod(a);
        z(1, 0) = std::stod(b);
        kf.predict();
        kf.update(z);
        std::printf("%.12g,%.12g,%.12g,%.12g\n", kf.x(0, 0), kf.x(1, 0), kf.x(2, 0), kf.x(3, 0));
    }
    return 0;
}
