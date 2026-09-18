#pragma once

// M2 Stage 2 pre_encode stem (8x dw-striding conv, non-causal / symmetric).
// Reference FP32 implementation; inference semantics.
//
// Upstream pins (NOT assumptions):
//   NeMo torch ConvSubsampling dw_striding branch + forward (subsampling.py):
//     sampling_num = log2(8) = 3 stages. Layer 0: Conv2d(1->256, k=3, s=2,
//     pad=1) + ReLU; stages 1-2: depthwise Conv2d(256 groups, k=3, s=2,
//     pad=1) + pointwise Conv2d(256->256, k=1) + ReLU. Then flatten (C,F)
//     and Linear(C*F -> 512). Input x is (B,T,128) frame-major; torch runs
//     channel-first (B,1,T,128) — values identical, layout differs.
//   C++ ggml port SubSampling ctor + symmetric build_graph (fastconformer.cpp
//     + nn.h Conv2D/Conv2DDW/ReLU/SequenceModule): same topology, Conv2D
//     k=3 s=2 pad=1, DW groups, 1x1 pointwise, ReLU between, out Linear over
//     (C*F). Causal path (pad_ext) explicitly NOT implemented here.
//   ceil_mode=False with symmetric pad=1: out = floor((L+1)/2). Stage 1
//     Conv2d oracle already pins this geometry on synthetic data; Stage 0
//     chunk000 real-data check pins 160 -> 20 frames (mel 160x128 in,
//     pre_encode 20x512 out).
//   v13 MASKED-TAIL pin (subsampling.py MaskedConvSequential, NeMo 3.0.0,
//     pinned 2026-09-18 from v12 P6 probe evidence): the conv stack runs on
//     the FULL window grid (t_mel rows, T_full = t_mel/8 out rows), but a
//     multiplicative time mask is applied around every layer with per-stage
//     floor lengths iterated from feat_len (calculate_conv_output_size):
//     L_{k+1} = (L_k + 2 - 3)/2 + 1; 86 -> 43 -> 22 -> 11. Rows >= Lk are
//     zeroed after level k's activation (bit-equivalent to per-layer
//     masking; only zero-vs-nonzero matters downstream). Final rows >= L3
//     flatten to zero features so the out Linear yields out.bias EXACTLY —
//     the bit-identical fill rows observed on tail chunks (rms 0.337,
//     identical across audios). L3 == calc_length(repeat_num=3) ==
//     state_lens_before[2] == the encoder's reported valid length.
//
// Diar config: feat_in=128, conv_channels=256, d_model=512, factor 8.
//
// Layout contract: mel_in [T_mel,128] frame-major; weights Conv2d
// [out,in,KH,KW]; DW [C,KH,KW] with in==out==C (nn.hpp conv2d DW branch is
// C,H,W single-channel — NOT reused here; this file has its own explicit
// grouped loop); pointwise 1x1 [C,C,1,1]; out Linear [512, C*F]+b. All
// biases present (torch Conv2d default bias=True). T <= 0 no-op.
// feat_len <= 0 or > t_mel means full window (masks become no-ops).

namespace diar {

struct SubsamplingWeights {
    const float* c0_w = nullptr;  // [256,1,3,3] + b[256]
    const float* c0_b = nullptr;
    const float* dw1_w = nullptr;  // [256,3,3] grouped + b[256]
    const float* dw1_b = nullptr;
    const float* pw1_w = nullptr;  // [256,256,1,1] + b[256]
    const float* pw1_b = nullptr;
    const float* dw2_w = nullptr;  // [256,3,3] grouped + b[256]
    const float* dw2_b = nullptr;
    const float* pw2_w = nullptr;  // [256,256,1,1] + b[256]
    const float* pw2_b = nullptr;
    const float* out_w = nullptr;  // [512, 256*F] + b[512]
    const float* out_b = nullptr;
};

// mel [T_mel,128] -> y [T_enc,512]. f_out is the surviving freq bins after
// 3x ceil-div (16 for feat 128); asserted against computed geometry.
// feat_len: valid mel rows (NeMo processed_signal_length); rows >= feat_len
// are masked and tail output rows become out.bias (see pin above).
// feat_len <= 0 || feat_len > t_mel selects the full window.
void subsampling_forward(const float* mel_in, const SubsamplingWeights& w, float* y, int t_mel,
    int feat_in = 128, int conv_channels = 256, int d_model = 512, int feat_len = -1);

}  // namespace diar
