// Fixed-size matrices for small filters: no heap allocation, so they are safe
// to use inside a real-time audio or control callback.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ssp {

template <std::size_t R, std::size_t C>
struct Mat {
    std::array<double, R * C> a{};

    double& operator()(std::size_t i, std::size_t j) { return a[i * C + j]; }
    double operator()(std::size_t i, std::size_t j) const { return a[i * C + j]; }

    static Mat zeros() { return Mat{}; }
    static Mat identity() {
        static_assert(R == C, "identity needs a square matrix");
        Mat m{};
        for (std::size_t i = 0; i < R; ++i) m(i, i) = 1.0;
        return m;
    }
    Mat<C, R> T() const {
        Mat<C, R> t{};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j) t(j, i) = (*this)(i, j);
        return t;
    }
};

template <std::size_t R, std::size_t K, std::size_t C>
Mat<R, C> operator*(const Mat<R, K>& x, const Mat<K, C>& y) {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t k = 0; k < K; ++k) {
            const double xik = x(i, k);
            for (std::size_t j = 0; j < C; ++j) out(i, j) += xik * y(k, j);
        }
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator+(Mat<R, C> x, const Mat<R, C>& y) {
    for (std::size_t i = 0; i < R * C; ++i) x.a[i] += y.a[i];
    return x;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator-(Mat<R, C> x, const Mat<R, C>& y) {
    for (std::size_t i = 0; i < R * C; ++i) x.a[i] -= y.a[i];
    return x;
}

// Gauss–Jordan inverse with partial pivoting. Fine for the 1x1–4x4 innovation
// covariances of typical filters; use a Cholesky solve for anything larger.
template <std::size_t N>
Mat<N, N> inverse(Mat<N, N> m) {
    Mat<N, N> inv = Mat<N, N>::identity();
    for (std::size_t c = 0; c < N; ++c) {
        std::size_t p = c;
        for (std::size_t r = c + 1; r < N; ++r)
            if (std::fabs(m(r, c)) > std::fabs(m(p, c))) p = r;
        if (std::fabs(m(p, c)) < 1e-300) throw std::runtime_error("singular matrix");
        for (std::size_t j = 0; j < N; ++j) {
            std::swap(m(c, j), m(p, j));
            std::swap(inv(c, j), inv(p, j));
        }
        const double d = m(c, c);
        for (std::size_t j = 0; j < N; ++j) { m(c, j) /= d; inv(c, j) /= d; }
        for (std::size_t r = 0; r < N; ++r) {
            if (r == c) continue;
            const double f = m(r, c);
            for (std::size_t j = 0; j < N; ++j) { m(r, j) -= f * m(c, j); inv(r, j) -= f * inv(c, j); }
        }
    }
    return inv;
}

}  // namespace ssp
