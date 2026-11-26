// Linear Kalman filter, header-only and allocation-free.
// Mirrors ssp/core/kalman.py (KalmanFilter.predict / update, Joseph form).
#pragma once

#include "mat.hpp"

namespace ssp {

template <std::size_t NX, std::size_t NZ>
class KalmanFilter {
public:
    using StateVec = Mat<NX, 1>;
    using MeasVec = Mat<NZ, 1>;

    Mat<NX, NX> F, Q, P;
    Mat<NZ, NX> H;
    Mat<NZ, NZ> R;
    StateVec x;

    // x <- F x ;  P <- F P F^T + Q
    void predict() {
        x = F * x;
        P = F * P * F.T() + Q;
    }

    // Fuse measurement z. Returns the NIS (normalised innovation squared).
    double update(const MeasVec& z) {
        const MeasVec y = z - H * x;                       // innovation
        const Mat<NZ, NZ> S = H * P * H.T() + R;           // innovation covariance
        const Mat<NZ, NZ> Si = inverse(S);
        const Mat<NX, NZ> K = P * H.T() * Si;              // Kalman gain
        x = x + K * y;
        const Mat<NX, NX> I_KH = Mat<NX, NX>::identity() - K * H;
        P = I_KH * P * I_KH.T() + K * R * K.T();           // Joseph form: stays symmetric PD
        return (y.T() * Si * y)(0, 0);
    }
};

// Constant-velocity model for `dims` axes, state [p, v] per axis (matches constant_velocity in Python).
template <std::size_t DIMS>
void constant_velocity(double dt, double q, Mat<2 * DIMS, 2 * DIMS>& F, Mat<2 * DIMS, 2 * DIMS>& Q) {
    F = Mat<2 * DIMS, 2 * DIMS>::zeros();
    Q = Mat<2 * DIMS, 2 * DIMS>::zeros();
    for (std::size_t d = 0; d < DIMS; ++d) {
        const std::size_t i = 2 * d;
        F(i, i) = 1.0; F(i, i + 1) = dt; F(i + 1, i + 1) = 1.0;
        Q(i, i) = q * dt * dt * dt / 3.0; Q(i, i + 1) = q * dt * dt / 2.0;
        Q(i + 1, i) = q * dt * dt / 2.0;   Q(i + 1, i + 1) = q * dt;
    }
}

}  // namespace ssp
