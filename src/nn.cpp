// M2 Stage 1 CPU tensor core implementation — see include/diar/nn.hpp
// for the upstream pins. Everything here is plain FP32 loops with no
// dependencies, mirroring upstream inference semantics (dropouts are
// inference no-ops everywhere, so they never appear).
#include "diar/nn.hpp"

#include <cmath>
#include <cstddef>

namespace diar::nn {

void linear_forward(const float* x, const float* weight, const float* bias, float* y,
    std::size_t t, std::size_t in, std::size_t out) {
    for (std::size_t r = 0; r < t; ++r) {
        const float* xr = x + r * in;
        float* yr = y + r * out;
        for (std::size_t o = 0; o < out; ++o) {
            const float* w = weight + o * in;
            float acc = 0.0F;
            for (std::size_t i = 0; i < in; ++i) acc += xr[i] * w[i];
            yr[o] = bias != nullptr ? acc + bias[o] : acc;
        }
    }
}

void layernorm_forward(const float* x, const float* gamma, const float* beta, float* y,
    std::size_t t, std::size_t c, float eps) {
    // ggml_norm semantics: biased variance (divide by n), eps INSIDE the
    // sqrt; output is normalized COPY (input untouched), then affine.
    for (std::size_t r = 0; r < t; ++r) {
        const float* xr = x + r * c;
        float* yr = y + r * c;
        float sum = 0.0F;
        for (std::size_t i = 0; i < c; ++i) sum += xr[i];
        const float mean = c > 0 ? sum / c : 0.0F;
        float var = 0.0F;
        for (std::size_t i = 0; i < c; ++i) {
            const float d = xr[i] - mean;
            var += d * d;
        }
        var = c > 0 ? var / c : 0.0F;
        const float inv = 1.0F / std::sqrt(var + eps);
        for (std::size_t i = 0; i < c; ++i) {
            const float g = gamma != nullptr ? gamma[i] : 1.0F;
            const float b = beta != nullptr ? beta[i] : 0.0F;
            yr[i] = (xr[i] - mean) * inv * g + b;
        }
    }
}

void batchnorm1d_infer_forward(const float* x, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, float* y, std::size_t t,
    std::size_t c, float eps) {
    // Upstream BatchNorm1d::build_graph: (x-mean)/sqrt(var+eps)*w+b (infer).
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t i = 0; i < c; ++i) {
            const float w = weight != nullptr ? weight[i] : 1.0F;
            const float b = bias != nullptr ? bias[i] : 0.0F;
            const float m = running_mean != nullptr ? running_mean[i] : 0.0F;
            const float v = running_var != nullptr ? running_var[i] : 0.0F;
            y[r * c + i] = (x[r * c + i] - m) / std::sqrt(v + eps) * w + b;
        }
    }
}

void softmax_last_dim(const float* x, float* y, std::size_t rows, std::size_t cols) {
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float m = xr[0];
        for (std::size_t i = 1; i < cols; ++i)
            if (xr[i] > m) m = xr[i];
        float sum = 0.0F;
        for (std::size_t i = 0; i < cols; ++i) {
            yr[i] = std::exp(xr[i] - m);
            sum += yr[i];
        }
        const float inv = 1.0F / sum;
        for (std::size_t i = 0; i < cols; ++i) yr[i] *= inv;
    }
}

void relu_inplace(float* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (x[i] < 0.0F) x[i] = 0.0F;
}

void silu_forward(const float* x, float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] / (1.0F + std::exp(-x[i]));
}

void sigmoid_forward(const float* x, float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = 1.0F / (1.0F + std::exp(-x[i]));
}

void glu_forward(const float* x, float* y, std::size_t t, std::size_t c) {
    // Upstream: part_a * sigmoid(part_b), a = FIRST half (ggml_glu swapped=true
    // keeps the same halves convention; see header note).
    for (std::size_t r = 0; r < t; ++r) {
        const float* a = x + r * 2 * c;
        const float* b = a + c;
        float* yr = y + r * c;
        for (std::size_t i = 0; i < c; ++i)
            yr[i] = a[i] / (1.0F + std::exp(-b[i]));
    }
}

void conv1d_forward(const float* x, const float* weight, const float* bias, float* y, int t_in,
    int c_in, int c_out, int kernel, int stride, int padding) {
    const int t_out = (t_in + 2 * padding - kernel) / stride + 1;
    for (int t = 0; t < t_out; ++t) {
        for (int o = 0; o < c_out; ++o) {
            float acc = 0.0F;
            for (int i = 0; i < c_in; ++i) {
                for (int k = 0; k < kernel; ++k) {
                    const int ti = t * stride - padding + k;
                    if (ti < 0 || ti >= t_in) continue;
                    acc += x[ti * c_in + i] * weight[(o * c_in + i) * kernel + k];
                }
            }
            y[t * c_out + o] = bias != nullptr ? acc + bias[o] : acc;
        }
    }
}

void conv1d_depthwise_forward(const float* x, const float* weight, const float* bias, float* y,
    int t_in, int channels, int kernel, int stride, int padding) {
    // Depthwise: channel o sees only input channel o (weight [C, K]).
    const int t_out = (t_in + 2 * padding - kernel) / stride + 1;
    for (int t = 0; t < t_out; ++t) {
        for (int o = 0; o < channels; ++o) {
            float acc = 0.0F;
            for (int k = 0; k < kernel; ++k) {
                const int ti = t * stride - padding + k;
                if (ti < 0 || ti >= t_in) continue;
                acc += x[ti * channels + o] * weight[o * kernel + k];
            }
            y[t * channels + o] = bias != nullptr ? acc + bias[o] : acc;
        }
    }
}

void conv2d_forward(const float* x, const float* weight, const float* bias, float* y, int c_in,
    int h_in, int w_in, int c_out, int kh, int kw, int stride_h, int stride_w, int pad_h,
    int pad_w) {
    const int h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    const int w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    auto at_x = [&](int c, int h, int w) -> float {
        if (h < 0 || h >= h_in || w < 0 || w >= w_in) return 0.0F;
        return x[(c * h_in + h) * w_in + w];
    };
    for (int o = 0; o < c_out; ++o) {
        for (int ho = 0; ho < h_out; ++ho) {
            for (int wo = 0; wo < w_out; ++wo) {
                float acc = 0.0F;
                for (int i = 0; i < c_in; ++i) {
                    for (int kh_i = 0; kh_i < kh; ++kh_i) {
                        for (int kw_i = 0; kw_i < kw; ++kw_i) {
                            const float v =
                                at_x(i, ho * stride_h - pad_h + kh_i, wo * stride_w - pad_w + kw_i);
                            acc +=
                                v * weight[((o * c_in + i) * kh + kh_i) * kw + kw_i];
                        }
                    }
                }
                y[(o * h_out + ho) * w_out + wo] =
                    bias != nullptr ? acc + bias[o] : acc;
            }
        }
    }
}

void rel_shift_forward(const float* bd, float* out, std::size_t t) {
    // Proven mapping (index simulation over the ggml op chain, T=2..5 green):
    // S[k, j] = BD[k - j + T - 1, j], BD rows = relpos 0..2T-2.
    for (std::size_t k = 0; k < t; ++k) {
        for (std::size_t j = 0; j < t; ++j) {
            out[k * t + j] = bd[(k - j + t - 1) * t + j];
        }
    }
}

void xscale_forward(const float* x, float* y, std::size_t n, float scale) {
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

}  // namespace diar::nn
