#include <metal_stdlib>
using namespace metal;

// Matrices are row-major. Evaluation columns are contiguous so neighboring
// threads read and write neighboring entries throughout the triangular solve.
inline float log_correlation(device const float* row,
                             device const float* points,
                             uint q, uint p, uint m, float beta) {
    float value = 0.0f;
    for (uint d = 0; d < p; ++d) {
        float diff = row[d] - points[d * m + q];
        value -= log(1.0f + diff * diff / beta);
    }
    return value;
}

kernel void initialize(
    device float* y, device float* scales, device float* norms, device float* logs,
    device const float* points, device const float* x, device const float* L,
    constant uint& n, constant uint& p, constant uint& m, constant float& beta,
    constant float& alpha,
    uint q [[thread_position_in_grid]]) {
    float scale = -INFINITY;
    for (uint r = 0; r < n; ++r) {
        float value = alpha * log_correlation(x + r * p, points, q, p, m, beta);
        y[r * m + q] = value;
        scale = max(scale, value);
    }
    float norm = 0.0f;
    for (uint r = 0; r < n; ++r) {
        float value = exp(y[r * m + q] - scale);
        for (uint c = 0; c < r; ++c)
            value -= L[r * n + c] * y[c * m + q];
        value /= L[r * n + r];
        y[r * m + q] = value;
        norm = fma(value, value, norm);
    }
    scales[q] = scale;
    norms[q] = norm;
    logs[q] = 2.0f * scale + log(max(norm, FLT_MIN));
}

kernel void evaluate_pair(
    device float* logs, device float* z, device float* new_scales,
    device float* new_norms, device const float* y,
    device const float* scales, device const float* norms,
    device const float* points, device const float* packed,
    constant uint& n, constant uint& k, constant uint& p, constant uint& m,
    constant float& beta, uint tid [[thread_position_in_grid]]) {
    uint choice = tid / m, q = tid % m;
    device const float* block = packed + choice * k * p;
    device const float* W = packed + 2 * k * p + choice * k * n;
    device const float* G = packed + 2 * k * (p + n) + choice * k * k;
    device float* candidate = z + choice * k * m;
    float scale = scales[q];
    for (uint r = 0; r < k; ++r) {
        float value = log_correlation(block + r * p, points, q, p, m, beta);
        candidate[r * m + q] = value;
        scale = max(scale, value);
    }
    float rescale = exp(scales[q] - scale);
    float norm = norms[q] * rescale * rescale;
    for (uint r = 0; r < k; ++r) {
        float value = exp(candidate[r * m + q] - scale);
        for (uint c = 0; c < n; ++c)
            value -= W[r * n + c] * y[c * m + q] * rescale;
        for (uint c = 0; c < r; ++c)
            value -= G[r * k + c] * candidate[c * m + q];
        value /= G[r * k + r];
        candidate[r * m + q] = value;
        norm = fma(value, value, norm);
    }
    new_scales[tid] = scale;
    new_norms[tid] = norm;
    logs[tid] = 2.0f * scale + log(max(norm, FLT_MIN));
}

kernel void accept(
    device float* y, device float* scales, device float* norms,
    device const float* z, device const float* new_scales,
    device const float* new_norms, constant uint& n, constant uint& k,
    constant uint& m, constant uint& choice,
    uint q [[thread_position_in_grid]]) {
    uint selected = choice * m + q;
    float rescale = exp(scales[q] - new_scales[selected]);
    for (uint r = 0; r < n; ++r) y[r * m + q] *= rescale;
    for (uint r = 0; r < k; ++r)
        y[(n + r) * m + q] = z[(choice * k + r) * m + q];
    scales[q] = new_scales[selected];
    norms[q] = new_norms[selected];
}
