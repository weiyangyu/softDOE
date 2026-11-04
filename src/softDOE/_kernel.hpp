#pragma once
#include <cmath>
#include <limits>

namespace softdoe {

// log(1 + (x-y)^2 / beta), retaining finite logarithms when an
// intermediate subtraction, square, or reciprocal exceeds double range.
inline double log_kernel_term(double x, double y, double beta) {
    const double difference = x - y;
    const double ratio = difference * difference * (1.0 / beta);
    if (std::isfinite(ratio) && (ratio > 0 || x == y))
        return std::log1p(ratio);
    if (x == y) return 0;
    double log_distance;
    if (std::isfinite(difference)) {
        log_distance = std::log(std::abs(difference));
    } else {
        // Overflow in subtraction requires opposite signs; halving both
        // operands makes the difference representable.
        log_distance = std::log(std::abs(x / 2 - y / 2)) + std::log(2.0);
    }
    const double log_ratio = 2 * log_distance - std::log(beta);
    return log_ratio > 0 ? log_ratio + std::log1p(std::exp(-log_ratio))
                         : std::log1p(std::exp(log_ratio));
}

inline double kernel_log_sum(const double* x, const double* y, int p,
                             double alpha, double beta) {
    double value = 0;
    for (int d = 0; d < p; ++d)
        value -= alpha * log_kernel_term(x[d], y[d], beta);
    return value;
}

inline double kernel_correlation(const double* x, const double* y, int p,
                                 double alpha, double beta) {
    if (alpha != 1) return std::exp(kernel_log_sum(x, y, p, alpha, beta));
    // Preserve the normal product path and its ordering for MOFAT parity.
    const double invbeta = 1.0 / beta;
    double value = 1;
    for (int d = 0; d < p; ++d) {
        const double difference = x[d] - y[d];
        value *= 1.0 / (1.0 + difference * difference * invbeta);
    }
    if (value > 0 && std::isfinite(value)) return value;
    return std::exp(kernel_log_sum(x, y, p, alpha, beta));
}

inline double kernel_log_correlation(const double* x, const double* y, int p,
                                     double alpha, double beta) {
    if (alpha == 1) {
        const double value = kernel_correlation(x, y, p, alpha, beta);
        if (value >= std::numeric_limits<double>::min()) return std::log(value);
    }
    return kernel_log_sum(x, y, p, alpha, beta);
}

}  // namespace softdoe
