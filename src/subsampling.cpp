// M2 Stage 2 pre_encode stem — see include/diar/subsampling.hpp for pins.
#include "diar/subsampling.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

#include "diar/nn.hpp"

namespace diar {
namespace {

// out[t] = floor((in + 2*pad - k)/s) + 1 (torch floor-mode conv formula).
int conv_out_len(int in, int k, int s, int pad) { return (in + 2 * pad - k) / s + 1; }

// Plain Conv2d [out,in,KH,KW], stride 2, pad 1, +bias, +ReLU optional.
// x: [C,H,W] contiguous (((c*H)+h)*W+w).
void conv2d_relu(const float* x, const float* w, const float* b, float* y, int c_in, int h_in,
    int w_in, int c_out, bool relu) {
    const int k = 3, s = 2, p = 1;
    const int h_out = conv_out_len(h_in, k, s, p);
    const int w_out = conv_out_len(w_in, k, s, p);
    for (int o = 0; o < c_out; ++o) {
        for (int ho = 0; ho < h_out; ++ho) {
            for (int wo = 0; wo < w_out; ++wo) {
                float acc = b != nullptr ? b[o] : 0.0F;
                for (int i = 0; i < c_in; ++i)
                    for (int kh = 0; kh < k; ++kh)
                        for (int kw = 0; kw < k; ++kw) {
                            const int hi = ho * s - p + kh;
                            const int wi = wo * s - p + kw;
                            if (hi < 0 || hi >= h_in || wi < 0 || wi >= w_in) continue;
                            acc += x[(i * h_in + hi) * w_in + wi] *
                                   w[((o * c_in + i) * k + kh) * k + kw];
                        }
                if (relu && acc < 0.0F) acc = 0.0F;
                y[(o * h_out + ho) * w_out + wo] = acc;
            }
        }
    }
}

// Grouped depthwise Conv2d [C,KH,KW] (channel c reads only channel c).
void conv2d_dw_relu(const float* x, const float* w, const float* b, float* y, int c, int h_in,
    int w_in, bool relu) {
    const int k = 3, s = 2, p = 1;
    const int h_out = conv_out_len(h_in, k, s, p);
    const int w_out = conv_out_len(w_in, k, s, p);
    for (int ch = 0; ch < c; ++ch) {
        for (int ho = 0; ho < h_out; ++ho) {
            for (int wo = 0; wo < w_out; ++wo) {
                float acc = b != nullptr ? b[ch] : 0.0F;
                for (int kh = 0; kh < k; ++kh)
                    for (int kw = 0; kw < k; ++kw) {
                        const int hi = ho * s - p + kh;
                        const int wi = wo * s - p + kw;
                        if (hi < 0 || hi >= h_in || wi < 0 || wi >= w_in) continue;
                        acc += x[(ch * h_in + hi) * w_in + wi] * w[(ch * k + kh) * k + kw];
                    }
                if (relu && acc < 0.0F) acc = 0.0F;
                y[(ch * h_out + ho) * w_out + wo] = acc;
            }
        }
    }
}

// Pointwise 1x1 Conv2d [C,C,1,1] == per-position Linear, +bias, +ReLU.
void pointwise_relu(const float* x, const float* w, const float* b, float* y, int c, int hw) {
    for (int o = 0; o < c; ++o) {
        for (int q = 0; q < hw; ++q) {
            float acc = b != nullptr ? b[o] : 0.0F;
            for (int i = 0; i < c; ++i) acc += x[i * hw + q] * w[o * c + i];
            if (acc < 0.0F) acc = 0.0F;
            y[o * hw + q] = acc;
        }
    }
}

}  // namespace

void subsampling_forward(const float* mel_in, const SubsamplingWeights& w, float* y, int t_mel,
    int feat_in, int conv_channels, int d_model) {
    if (t_mel <= 0) return;
    const int C = conv_channels;
    // Stage 0: (1, T, F) -> (C, T1, F1).
    const int t1 = conv_out_len(t_mel, 3, 2, 1);
    const int f1 = conv_out_len(feat_in, 3, 2, 1);
    // torch input (B,T,F) channel-first (1,T,F): index (t*F+f).
    std::vector<float> s0(C * t1 * f1);
    conv2d_relu(mel_in, w.c0_w, w.c0_b, s0.data(), 1, t_mel, feat_in, C, /*relu=*/true);
    // Stage 1: DW + pointwise.
    const int t2 = conv_out_len(t1, 3, 2, 1);
    const int f2 = conv_out_len(f1, 3, 2, 1);
    std::vector<float> d1(C * t2 * f2), s1(C * t2 * f2);
    conv2d_dw_relu(s0.data(), w.dw1_w, w.dw1_b, d1.data(), C, t1, f1, /*relu=*/false);
    pointwise_relu(d1.data(), w.pw1_w, w.pw1_b, s1.data(), C, t2 * f2);
    // Stage 2: DW + pointwise.
    const int t3 = conv_out_len(t2, 3, 2, 1);
    const int f3 = conv_out_len(f2, 3, 2, 1);
    std::vector<float> d2(C * t3 * f3), s2(C * t3 * f3);
    conv2d_dw_relu(s1.data(), w.dw2_w, w.dw2_b, d2.data(), C, t2, f2, /*relu=*/false);
    pointwise_relu(d2.data(), w.pw2_w, w.pw2_b, s2.data(), C, t3 * f3);
    // Flatten (C,F) per time step, Linear -> d_model. s2 layout [C,T3,F3]:
    // frame t is C*F3 values strided by F3 (matches torch reshape(b,t,-1)
    // of (b,c,t,f).transpose(1,2)).
    std::vector<float> flat(static_cast<std::size_t>(t3) * C * f3);
    for (int t = 0; t < t3; ++t)
        for (int c = 0; c < C; ++c)
            for (int f = 0; f < f3; ++f) flat[(t * C + c) * f3 + f] = s2[(c * t3 + t) * f3 + f];
    nn::linear_forward(flat.data(), w.out_w, w.out_b, y, static_cast<std::size_t>(t3),
        static_cast<std::size_t>(C * f3), static_cast<std::size_t>(d_model));
}

}  // namespace diar
